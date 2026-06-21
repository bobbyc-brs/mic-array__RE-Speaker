# mic-array / ReSpeaker 4 Mic Array

Python tools for the [Seeed ReSpeaker 4 Mic Array](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) (USB `2886:0018`).

Reads audio and onboard Direction of Arrival (DOA) from the XMOS XVF3000 chip, routes audio to per-speaker buffers, transcribes with Whisper, and extracts structured medical form fields via Claude AI.

---

## Hardware

| Item | Detail |
|---|---|
| Device | Seeed ReSpeaker 4 Mic Array v2 |
| USB | VID `0x2886` / PID `0x0018` (UAC1.0) |
| Chip | XMOS XVF3000 |
| Channels | 6 (ch0–3 raw mics, **ch4** AEC+NS processed, ch5 ref) |
| DOA | Onboard, 0–359°, polled via USB vendor control transfer |

---

## Setup

### 1. udev rule (non-root USB access)

```bash
sudo cp 99-respeaker.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# Re-plug the device, then add your user to plugdev if not already there:
sudo usermod -aG plugdev $USER
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. API key (for `ai_client.py`)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Tools

### `mic_array.py` — Live DOA + audio level monitor

Displays real-time Direction of Arrival angle and audio level in the terminal. Useful for checking speaker positions before running the transcriber.

```bash
python mic_array.py
```

---

### `doa_transcribe.py` — Multi-speaker transcription

Routes audio to per-speaker buffers based on DOA angle. Segments are cut on natural conversation boundaries: a pause in speech (webrtcvad) or a DOA switch to a different speaker. Each segment is transcribed with Whisper and forwarded as a `transcript_batch` JSON event.

```bash
# Two speakers facing each other
python doa_transcribe.py --speaker paramedic:0 --speaker patient:180

# Three speakers, tighter zones
python doa_transcribe.py --speaker alice:0 --speaker bob:120 --speaker carol:240 --tolerance 30

# Save transcripts locally and forward to AI client
python doa_transcribe.py --speaker paramedic:0 --speaker patient:180 --jsonl transcripts.jsonl
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--speaker ROLE:ANGLE` | required | Repeat per speaker; angle 0–359° |
| `--tolerance` | 45° | DOA acceptance window either side of center |
| `--pause-seconds` | 0.6s | Silence duration that closes a segment |
| `--max-segment-seconds` | 30s | Hard cap per segment |
| `--min-speech-seconds` | 0.5s | Minimum speech to send to Whisper |
| `--vad-aggressiveness` | 2 | webrtcvad aggressiveness 0–3 |
| `--model` | `base` | Whisper model (`tiny`, `base`, `small`, `medium`, `large`) |
| `--accurate` | off | Beam search + temperature fallback for higher accuracy |
| `--jsonl FILE` | `transcripts.jsonl` | Append compact payloads here |

**Forwarding:** by default posts to `http://127.0.0.1:8080/api/ingest` — the `ai_client.py` endpoint.

---

### `ai_client.py` — AI-powered ePCR field extraction

Listens for `transcript_batch` events, accumulates conversation context per session, and calls an LLM to extract structured field values from `schemas/epcr.yaml`. Updated fields are printed to the screen and written to `<session_id>_epcr.json`.

Supports three providers — whichever key is present is used automatically:

| Provider | Env var | Key file |
|---|---|---|
| Anthropic (Claude) | `$ANTHROPIC_API_KEY` | `AnthropicAPI.key` |
| Perplexity | `$PERPLEXITY_KEY` | `PerplexityAPI.key` |
| Cohere | `$COHERE_API_KEY` | `CohereAPI.key` |

```bash
# Start the AI client first (default port 8080)
python ai_client.py

# Explicit provider
python ai_client.py --provider perplexity
python ai_client.py --provider cohere

# With options
python ai_client.py --output-dir /tmp/pcr --debounce 5
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--port` | 8080 | HTTP listen port |
| `--schema` | `schemas/epcr.yaml` | Form schema file |
| `--output-dir` | `.` | Where to write `*_epcr.json` files |
| `--provider` | auto | `anthropic`, `perplexity`, or `cohere` |
| `--model` | provider default | Override the provider's default model |
| `--debounce` | 4s | Seconds between extraction passes |

**Typical two-terminal workflow:**

```bash
# Terminal 1
python ai_client.py --output-dir /tmp/pcr

# Terminal 2
python doa_transcribe.py --speaker paramedic:0 --speaker patient:180 --tolerance 60
```

---

---

### `tests/` — Offline testing without live audio

Test extraction quality without needing the mic array or a live call.

**Convert a script to JSON:**

Write a plain-text script in `Speaker Name: Text` format, then convert it:

```bash
python tests/convert_script.py my_script.txt -o tests/conversations/my_case.json

# With explicit role IDs and metadata
python tests/convert_script.py my_script.txt \
    --map "Paramedic 1:paramedic_1" \
    --map "Patient:patient" \
    --title "Allergic reaction call" \
    -o tests/conversations/allergic_reaction.json
```

**Replay against a running ai_client:**

```bash
# Terminal 1 — start the AI client
python ai_client.py --provider cohere --output-dir /tmp/pcr

# Terminal 2 — replay a scenario
python tests/test_replay.py tests/conversations/concussion_assessment.json
python tests/test_replay.py tests/conversations/chest_pain.json --wait 20
```

**Bundled scenarios:**

| File | Description |
|---|---|
| `conversations/concussion_assessment.json` | Cyclist with retrograde amnesia; cognitive screening, c-collar applied |
| `conversations/chest_pain.json` | 47F with acute chest pain, prior MI, on warfarin, aspirin given pre-arrival |

---

### `schemas/epcr.yaml` — ePCR form schema

YAML schema describing all fields of an Ontario paramedic ePCR (BR SEPCR). Each field is tagged by `source` so the AI client knows what to attempt from audio:

| Source | Meaning |
|---|---|
| `conversation` | Extracted from what people say |
| `paramedic` | Observed/documented by paramedic, may be spoken aloud |
| `scan` | Populated by health card / drivers licence scanner |
| `vitals` | Measured values entered by paramedic |
| `derived` | Computed from other fields |

Fields with `optional: true` are only activated when `unlock_keywords` appear in the transcript (e.g. cardiac arrest fields unlock on "collapsed", "CPR", "no pulse").

---

## Transcript payload format

Each `transcript_batch` event:

```json
{
  "event": "transcript_batch",
  "session_id": "session-001",
  "speaker_role": "paramedic",
  "captured_at": 1750000000.0,
  "language": "en",
  "language_changed": false,
  "segments": [
    {
      "segment_id": 0,
      "start": 0.0,
      "end": 3.4,
      "text": "What brings you in today?",
      "confidence": 0.97,
      "alternatives": [],
      "words": [
        {"text": "What", "start": 0.0, "end": 0.2, "confidence": 0.99}
      ]
    }
  ]
}
```

---

## Design notes

See [`design.md`](design.md) for the full USB interface map, DOA protocol reverse-engineering, and AudioControl topology.

---

## License

MIT — see [LICENSE](LICENSE).
