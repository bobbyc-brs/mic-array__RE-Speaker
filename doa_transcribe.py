#!/usr/bin/env python3
"""
doa_transcribe.py — Multi-speaker transcription via ReSpeaker 4 Mic Array DOA.

Replaces separate doctor.py / patient.py processes. Audio from the single array
is routed to per-speaker buffers based on the XMOS onboard Direction of Arrival
angle. Segments are cut on natural boundaries: a pause in speech (webrtcvad) or
a DOA switch to a different speaker, whichever comes first.

Usage example:
    python doa_transcribe.py --speaker doctor:0 --speaker patient:180 --tolerance 45
"""

import argparse
import json
import queue
import re
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import usb.core
import usb.util
import webrtcvad

try:
    import whisper_timestamped as whisper
except ImportError as exc:
    raise SystemExit("Missing dependency 'whisper-timestamped'. Install requirements first.") from exc

try:
    from pyrpc.RPCClient import RPCClient
except Exception:
    RPCClient = None

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE              = 16000
BLOCK_SECONDS            = 0.05        # 50 ms audio callback blocks
ARRAY_CHANNELS           = 6           # ReSpeaker: ch0-3 raw mics, ch4 AEC+NS, ch5 ref
AUDIO_CHANNEL            = 4           # ch4 = XMOS-processed output (best for ASR)
VAD_FRAME_MS             = 30          # webrtcvad supports 10/20/30 ms
VAD_FRAME_SAMPLES        = SAMPLE_RATE * VAD_FRAME_MS // 1000   # 480
DEFAULT_PAUSE_SECONDS    = 0.6         # silence duration that closes a segment
DEFAULT_MAX_SECONDS      = 30.0        # hard cap per segment
DEFAULT_TOLERANCE        = 45          # degrees either side of speaker center

VENDOR_ID       = 0x2886
PRODUCT_ID      = 0x0018
PARAM_DOA_ANGLE = 21
CTRL_IN = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE


# ── DOA polling ───────────────────────────────────────────────────────────────

class DoaPoller:
    """Reads XMOS onboard DOA angle in a daemon thread."""

    def __init__(self, poll_interval: float = 0.1):
        self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if self.dev is None:
            raise RuntimeError("ReSpeaker not found. Check USB connection and udev rule.")
        self._angle = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = poll_interval
        self._thread = threading.Thread(target=self._loop, daemon=True, name="doa-poller")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    @property
    def angle(self) -> int:
        with self._lock:
            return self._angle

    def _loop(self):
        while not self._stop.is_set():
            try:
                data = self.dev.ctrl_transfer(CTRL_IN, 0, 0xC0, PARAM_DOA_ANGLE, 8, timeout=500)
                with self._lock:
                    self._angle = int.from_bytes(data[0:4], "little", signed=True)
            except Exception:
                pass
            time.sleep(self._interval)


# ── speaker zones ─────────────────────────────────────────────────────────────

@dataclass
class SpeakerZone:
    role: str
    center: int  # 0-359 degrees


def angle_diff(a: int, b: int) -> int:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def resolve_speaker(
    doa: int, zones: List[SpeakerZone], tolerance: int
) -> Optional[SpeakerZone]:
    """Return the closest zone within tolerance, or None if no match."""
    best: Optional[SpeakerZone] = None
    best_diff = tolerance + 1
    for zone in zones:
        d = angle_diff(doa, zone.center)
        if d < best_diff:
            best_diff = d
            best = zone
    return best


# ── segment manager ───────────────────────────────────────────────────────────

# Items placed on the transcription queue: (zone, audio_float32, wall_clock_start)
TranscriptionItem = Tuple[SpeakerZone, np.ndarray, float]


class SegmentManager:
    """
    Receives raw audio blocks tagged with the current speaker zone.
    Classifies each 30 ms frame with webrtcvad and emits complete segments
    to a queue whenever a natural boundary is detected:

      1. Pause  — silence >= pause_seconds closes the current segment.
      2. Switch — DOA moves to a different speaker; the in-progress segment
                  for the old speaker is flushed immediately.
      3. Cap    — segment exceeds max_seconds regardless of VAD.

    Trailing silence is trimmed before the segment is enqueued.
    """

    def __init__(
        self,
        out_q: "queue.Queue[TranscriptionItem]",
        pause_seconds: float,
        max_seconds: float,
        vad_aggressiveness: int = 2,
    ):
        self._q              = out_q
        self._pause_frames   = int(pause_seconds * 1000 / VAD_FRAME_MS)
        self._max_frames     = int(max_seconds   * 1000 / VAD_FRAME_MS)
        self._vad            = webrtcvad.Vad(vad_aggressiveness)

        self._speaker: Optional[SpeakerZone] = None
        self._frames: List[bytes]             = []   # 30 ms PCM16 frames
        self._speech_mask: List[bool]         = []   # True = speech frame
        self._silence_run: int                = 0
        self._seg_start: float                = 0.0  # wall clock

        # Leftover PCM bytes from a partial frame at the previous callback
        self._remainder: bytes = b""

    def feed(self, block: np.ndarray, speaker: Optional[SpeakerZone], wall_clock: float):
        """
        block   : float32 mono, any length
        speaker : zone resolved from current DOA, or None for dead-zone audio
        """
        # Speaker change → flush whatever we have for the old speaker
        if speaker != self._speaker:
            self._flush()
            self._speaker   = speaker
            self._seg_start = wall_clock

        if speaker is None:
            return  # dead zone — discard audio

        # Convert float32 → int16 PCM bytes and prepend any leftover
        pcm = (block * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        pcm = self._remainder + pcm
        frame_bytes = VAD_FRAME_SAMPLES * 2  # 2 bytes per int16 sample

        offset = 0
        while offset + frame_bytes <= len(pcm):
            frame = pcm[offset : offset + frame_bytes]
            offset += frame_bytes

            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
            self._frames.append(frame)
            self._speech_mask.append(is_speech)

            if is_speech:
                self._silence_run = 0
            else:
                self._silence_run += 1

            # Flush on pause
            if self._silence_run >= self._pause_frames:
                self._flush()
                self._seg_start = wall_clock
                continue

            # Flush on max duration
            if len(self._frames) >= self._max_frames:
                self._flush()
                self._seg_start = wall_clock

        self._remainder = pcm[offset:]

    def flush_final(self):
        self._flush()

    def _flush(self):
        if not self._frames:
            return

        # Find last speech frame and trim trailing silence
        last_speech = len(self._speech_mask) - 1
        while last_speech >= 0 and not self._speech_mask[last_speech]:
            last_speech -= 1

        if last_speech < 0:
            # All silence — nothing to transcribe
            self._reset()
            return

        speech_frames = self._frames[: last_speech + 1]
        pcm_bytes = b"".join(speech_frames)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

        self._q.put((self._speaker, audio, self._seg_start))
        self._reset()

    def _reset(self):
        self._frames      = []
        self._speech_mask = []
        self._silence_run = 0
        self._remainder   = b""


# ── transcription ─────────────────────────────────────────────────────────────

class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: Optional[str],
        language: Optional[str],
        accurate: bool,
    ):
        print(f"Loading Whisper model '{model_name}'…")
        self.model    = whisper.load_model(model_name, device=device)
        self.language = language
        self.accurate = accurate

    def transcribe(self, audio: np.ndarray, offset_seconds: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "language":                    self.language,
            "vad":                         True,
            "compute_word_confidence":     True,
            "remove_punctuation_from_words": False,
            "condition_on_previous_text":  False,
        }
        if self.accurate:
            kwargs.update({"beam_size": 5, "best_of": 5, "temperature": (0.0, 0.2, 0.4, 0.6)})
        else:
            kwargs.update({"temperature": 0.0})
        result = whisper.transcribe(self.model, audio, **kwargs)
        result["_offset_seconds"] = offset_seconds
        return result


# ── forwarding ────────────────────────────────────────────────────────────────

class Forwarder:
    def __init__(self, url: str):
        self.url = url

    def send(self, payload: dict):
        try:
            import requests
            requests.post(self.url, json=payload, timeout=5).raise_for_status()
        except Exception as e:
            print(f"[HTTP forward failed] {e}", file=sys.stderr)


class RPCForwarder:
    def __init__(self, host: str, port: int, method: str):
        if RPCClient is None:
            raise RuntimeError("pyrpc not installed; cannot use --rpc-host")
        self.client = RPCClient((host, port))
        self.method = method

    def send(self, payload: dict):
        try:
            getattr(self.client, self.method)(payload)
        except Exception as e:
            print(f"[RPC forward failed] {e}", file=sys.stderr)


# ── result normalization (same structure as doctor/patient) ───────────────────

@dataclass
class TranscriptAlternative:
    text: str
    score: Optional[float] = None

@dataclass
class WordResult:
    text: str
    start: float
    end: float
    confidence: Optional[float]
    alternatives: List[TranscriptAlternative]

@dataclass
class SegmentResult:
    segment_id: int
    start: float
    end: float
    text: str
    confidence: Optional[float]
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    alternatives: List[TranscriptAlternative]
    words: List[WordResult]


def _seg_alts(seg: Dict) -> List[TranscriptAlternative]:
    alts = []
    primary = seg.get("text", "").strip()
    if primary:
        alts.append(TranscriptAlternative(text=primary, score=seg.get("confidence")))
    uncertain = [
        w.get("text", "").strip()
        for w in (seg.get("words") or [])
        if (w.get("confidence") or 1.0) < 0.6 and w.get("text")
    ]
    if uncertain:
        alts.append(TranscriptAlternative(text=f"Possible uncertainty around: {' '.join(uncertain)}"))
    if seg.get("no_speech_prob", 0) > 0.5:
        alts.append(TranscriptAlternative(text="Possible silence or non-speech segment",
                                          score=seg.get("no_speech_prob")))
    return alts[:3]


def _word_alts(word: Dict) -> List[TranscriptAlternative]:
    conf = word.get("confidence")
    text = word.get("text", "").strip()
    alts = [TranscriptAlternative(text=text, score=conf)] if text else []
    if conf is not None and conf < 0.5 and text:
        alts.append(TranscriptAlternative(text=f"{text} (?)", score=conf))
    return alts[:2]


def normalize_result(raw: Dict, next_segment_id: int) -> List[SegmentResult]:
    offset = raw.get("_offset_seconds", 0.0)
    out = []
    for idx, seg in enumerate(raw.get("segments") or []):
        words = [
            WordResult(
                text=w.get("text", "").strip(),
                start=offset + float(w.get("start", 0)),
                end=offset + float(w.get("end", 0)),
                confidence=w.get("confidence"),
                alternatives=_word_alts(w),
            )
            for w in (seg.get("words") or [])
        ]
        out.append(SegmentResult(
            segment_id=next_segment_id + idx,
            start=offset + float(seg.get("start", 0)),
            end=offset + float(seg.get("end", 0)),
            text=seg.get("text", "").strip(),
            confidence=seg.get("confidence"),
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
            alternatives=_seg_alts(seg),
            words=words,
        ))
    return out


# ── transcription worker ──────────────────────────────────────────────────────

def _transcription_worker(
    in_q: "queue.Queue[Optional[TranscriptionItem]]",
    transcriber: Transcriber,
    forwarder: Any,
    args: argparse.Namespace,
):
    """Single thread: consumes (zone, audio, wall_start) from the queue."""
    # Per-speaker offset and segment counters, keyed by role
    offsets:  Dict[str, float] = {}
    counters: Dict[str, int]   = {}

    while True:
        item = in_q.get()
        if item is None:
            break  # sentinel

        zone, audio, wall_start = item
        role = zone.role

        offset = offsets.get(role, 0.0)
        seg_id = counters.get(role, 0)

        raw      = transcriber.transcribe(audio, offset)
        segments = normalize_result(raw, seg_id)

        offsets[role]  = offset + len(audio) / SAMPLE_RATE
        counters[role] = seg_id + len(segments)

        if not segments:
            continue

        payload = {
            "event":         "transcript_batch",
            "session_id":    args.session_id,
            "device_id":     args.device_id,
            "speaker_role":  role,
            "doa_center":    zone.center,
            "captured_at":   wall_start,
            "language":      raw.get("language"),
            "language_probs": raw.get("language_probs", {}),
            "segments":      [asdict(s) for s in segments],
        }
        for seg in segments:
            print(f"[{seg.start:8.2f}-{seg.end:8.2f}] {role:12s} "
                  f"conf={seg.confidence!s:>5}  {seg.text}")

        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        try:
            forwarder.send(payload)
        except Exception as exc:
            print(f"[forward] {role}: {exc}", file=sys.stderr)


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    zones: List[SpeakerZone] = []
    for spec in args.speaker:
        m = re.match(r"^([^:]+):(\d+)$", spec.strip())
        if not m:
            raise SystemExit(f"Bad --speaker spec '{spec}'. Expected role:angle, e.g. doctor:0")
        zones.append(SpeakerZone(role=m.group(1), center=int(m.group(2)) % 360))

    print(f"Speakers:  {', '.join(f'{z.role}@{z.center}°' for z in zones)}")
    print(f"Tolerance: ±{args.tolerance}°  |  "
          f"Pause: {args.pause_seconds}s  |  "
          f"Max segment: {args.max_segment_seconds}s")

    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())

    doa = DoaPoller(poll_interval=0.1)
    doa.start()

    transcriber = Transcriber(args.model, args.whisper_device, args.language, args.accurate)

    if args.rpc_host and args.rpc_port:
        forwarder: Any = RPCForwarder(args.rpc_host, args.rpc_port, args.rpc_method)
    else:
        forwarder = Forwarder("http://127.0.0.1:8080/api/ingest")

    tx_queue: "queue.Queue[Optional[TranscriptionItem]]" = queue.Queue()
    tx_thread = threading.Thread(
        target=_transcription_worker,
        args=(tx_queue, transcriber, forwarder, args),
        daemon=True,
        name="transcriber",
    )
    tx_thread.start()

    seg_mgr = SegmentManager(
        out_q=tx_queue,
        pause_seconds=args.pause_seconds,
        max_seconds=args.max_segment_seconds,
        vad_aggressiveness=args.vad_aggressiveness,
    )

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono    = indata[:, AUDIO_CHANNEL].copy()
        speaker = resolve_speaker(doa.angle, zones, args.tolerance)
        seg_mgr.feed(mono, speaker, time.time())

    try:
        dev_idx = next(
            i for i, d in enumerate(sd.query_devices())
            if "respeaker" in d["name"].lower() and d["max_input_channels"] > 0
        )
    except StopIteration:
        raise SystemExit("ReSpeaker audio device not found.")

    blocksize = int(SAMPLE_RATE * BLOCK_SECONDS)
    print("Listening… Press Ctrl+C to stop.\n")
    with sd.InputStream(
        device=dev_idx,
        samplerate=SAMPLE_RATE,
        channels=ARRAY_CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        stop_event.wait()

    seg_mgr.flush_final()
    tx_queue.put(None)   # sentinel to stop transcriber thread
    tx_thread.join(timeout=10)
    doa.stop()
    print("\nStopped.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-speaker transcription using ReSpeaker 4 Mic Array DOA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Two speakers facing each other:
    %(prog)s --speaker doctor:0 --speaker patient:180

  Three speakers, tighter zones:
    %(prog)s --speaker alice:0 --speaker bob:120 --speaker carol:240 --tolerance 30

  Quicker cuts (short pause threshold):
    %(prog)s --speaker doctor:0 --speaker patient:180 --pause-seconds 0.4
""",
    )
    p.add_argument(
        "--speaker", action="append", metavar="ROLE:ANGLE", required=True,
        help="Speaker as role:angle (degrees 0-359). Repeat for each speaker.",
    )
    p.add_argument(
        "--tolerance", type=int, default=DEFAULT_TOLERANCE,
        help=f"DOA acceptance window in degrees either side of center (default {DEFAULT_TOLERANCE})",
    )
    p.add_argument(
        "--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS,
        help=f"Silence duration that closes a segment (default {DEFAULT_PAUSE_SECONDS}s)",
    )
    p.add_argument(
        "--max-segment-seconds", type=float, default=DEFAULT_MAX_SECONDS,
        help=f"Hard cap on segment length (default {DEFAULT_MAX_SECONDS}s)",
    )
    p.add_argument(
        "--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
        help="webrtcvad aggressiveness 0 (least) – 3 (most) (default 2)",
    )
    p.add_argument("--session-id",          default="session-001")
    p.add_argument("--device-id",           default="respeaker-array")
    p.add_argument("--model",               default="base", help="Whisper model name")
    p.add_argument("--whisper-device",      default=None,   help="Torch device (cpu / cuda)")
    p.add_argument("--language",            default=None,   help="Force language (e.g. en)")
    p.add_argument("--accurate",            action="store_true")
    p.add_argument("--jsonl",               default="transcripts.jsonl")
    p.add_argument("--rpc-host",            default=None)
    p.add_argument("--rpc-port",            type=int, default=None)
    p.add_argument("--rpc-method",          default="push_transcript")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
