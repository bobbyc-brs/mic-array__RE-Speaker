#!/usr/bin/env python3
"""
ai_client.py — AI-powered ePCR field extraction from live transcript batches.

Listens for transcript_batch events from doa_transcribe.py (HTTP POST /api/ingest),
accumulates conversation context per session, and calls Claude to extract structured
ePCR field values. Updated fields are printed to the screen and the full form state
is written to a JSON file after each extraction pass.

Usage:
    python ai_client.py                            # listen on 0.0.0.0:8080
    python ai_client.py --port 9090
    python ai_client.py --output-dir /tmp/pcr
    python ai_client.py --model claude-haiku-4-5-20251001 --debounce 5
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import yaml
from flask import Flask, jsonify, request

DEFAULT_SCHEMA = Path(__file__).parent / "schemas" / "epcr.yaml"
DEFAULT_MODEL  = "claude-haiku-4-5-20251001"
DEFAULT_PORT   = 8080
DEFAULT_DEBOUNCE = 4.0   # seconds between Claude calls

EXTRACTABLE_SOURCES = {"conversation", "paramedic"}


# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_field_index(schema: Dict) -> Dict[str, Dict]:
    """Fields that can be extracted from audio. Excludes scan/vitals/derived and tables."""
    return {
        f["key"]: f
        for f in schema["fields"]
        if f.get("source") in EXTRACTABLE_SOURCES and f.get("type") != "table"
    }


# ── System prompt construction ─────────────────────────────────────────────────

def build_system_prompt(field_index: Dict[str, Dict], schema: Dict) -> str:
    # Group fields by section, preserving schema order
    by_section: Dict[str, List[Dict]] = {}
    for f in schema["fields"]:
        if f["key"] in field_index:
            by_section.setdefault(f["section"], []).append(f)

    section_labels = {s["id"]: s["label"] for s in schema.get("sections", [])}

    lines = [
        "You are an ePCR (Electronic Patient Care Report) extraction assistant for paramedic use.",
        "Extract structured field values from a spoken conversation between paramedics, patients, and bystanders.",
        "",
        "RULES:",
        "- Only extract values that are clearly stated or strongly implied.",
        "- Never hallucinate values not supported by the transcript.",
        "- For multicheck fields, return a JSON list of applicable option strings from those listed.",
        "- For radio fields, return exactly one of the listed option strings.",
        "- For boolean fields, return true or false.",
        "- Omit a field (or set value to null) if it has not been mentioned.",
        "- Prefer quoting the patient or paramedic directly as evidence.",
        "",
        "CONFIDENCE SCALE:",
        "- 0.9-1.0 : directly and explicitly stated",
        "- 0.7-0.9 : clearly implied or paraphrased",
        "- 0.5-0.7 : inferred but uncertain",
        "- < 0.5   : do not include",
        "",
        "FIELDS TO EXTRACT:",
        "",
    ]

    for section_id, fields in by_section.items():
        label = section_labels.get(section_id, section_id.replace("_", " ").title())
        lines.append(f"### {label}")
        for f in fields:
            type_info = f["type"]
            if f.get("options"):
                type_info += f"  options={f['options']}"
            optional_tag = "  [OPTIONAL]" if f.get("optional") else ""
            unlock_info = ""
            if f.get("unlock_keywords"):
                kws = f["unlock_keywords"][:6]
                unlock_info = f"  unlocked-by={kws}"
            syn_info = ""
            if f.get("synonyms"):
                syns = f["synonyms"][:8]
                syn_info = f"  synonyms={syns}"

            lines.append(f"  {f['key']}  ({f['label']}){optional_tag}")
            lines.append(f"    type: {type_info}{syn_info}{unlock_info}")
            if f.get("prompt_if_missing"):
                lines.append(f"    if-missing: {f['prompt_if_missing']}")
        lines.append("")

    lines += [
        "RESPONSE FORMAT — return only valid JSON, no markdown fences:",
        "{",
        '  "fields": {',
        '    "<field_key>": {',
        '      "value": "<extracted value, list for multicheck, null if absent>",',
        '      "confidence": 0.0,',
        '      "evidence": "<brief direct quote or phrase from transcript>"',
        "    }",
        "  },",
        '  "follow_up_needed": ["<field_key>", ...],',
        '  "notes": "<clinical observations not captured in any field, or null>"',
        "}",
    ]

    return "\n".join(lines)


# ── Session state ──────────────────────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str, field_index: Dict[str, Dict]):
        self.session_id = session_id
        self.utterances: List[Dict] = []
        self.fields: Dict[str, Dict] = {
            key: {
                "key":          key,
                "label":        spec["label"],
                "section":      spec.get("section", ""),
                "value":        None,
                "confidence":   0.0,
                "evidence":     [],
                "status":       "missing",
                "last_updated": None,
            }
            for key, spec in field_index.items()
        }
        self._lock = threading.Lock()
        self.new_since_last_extraction = 0

    def add_utterances(self, speaker_role: str, captured_at: float, segments: List[Dict]):
        with self._lock:
            for seg in segments:
                text = seg.get("text", "").strip()
                if text:
                    self.utterances.append({
                        "time":       captured_at,
                        "speaker":    speaker_role,
                        "text":       text,
                        "confidence": seg.get("confidence"),
                    })
                    self.new_since_last_extraction += 1

    def take_new_count(self) -> int:
        """Return and reset the new-utterance counter atomically."""
        with self._lock:
            n = self.new_since_last_extraction
            self.new_since_last_extraction = 0
            return n

    def build_transcript(self) -> str:
        with self._lock:
            lines = []
            for u in self.utterances:
                ts = datetime.fromtimestamp(u["time"]).strftime("%H:%M:%S")
                lines.append(f"[{ts}] {u['speaker']}: {u['text']}")
            return "\n".join(lines)

    def update_fields(self, extracted: Dict[str, Dict]) -> List[str]:
        """Merge Claude's response into field state. Returns list of changed keys."""
        changed = []
        with self._lock:
            for key, data in extracted.items():
                if key not in self.fields:
                    continue
                value = data.get("value")
                if value is None:
                    continue
                conf      = float(data.get("confidence") or 0.0)
                evidence  = str(data.get("evidence") or "").strip()
                current   = self.fields[key]

                # Only update if new value is more confident than current
                if current["value"] is None or conf > current["confidence"]:
                    self.fields[key].update({
                        "value":        value,
                        "confidence":   conf,
                        "evidence":     [evidence] if evidence else [],
                        "status":       "filled" if conf >= 0.88 else "review",
                        "last_updated": time.time(),
                    })
                    changed.append(key)
        return changed

    def metrics(self) -> Dict[str, int]:
        with self._lock:
            statuses = [f["status"] for f in self.fields.values()]
        return {
            "filled":  statuses.count("filled"),
            "review":  statuses.count("review"),
            "missing": statuses.count("missing"),
            "total":   len(statuses),
        }

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "session_id":      self.session_id,
                "updated_at":      time.time(),
                "utterance_count": len(self.utterances),
                "fields":          {k: dict(v) for k, v in self.fields.items()},
                "metrics":         self.metrics(),
            }


# ── Claude extraction ──────────────────────────────────────────────────────────

def extract_fields(
    session: Session,
    system_prompt: str,
    ai: anthropic.Anthropic,
    model: str,
) -> Optional[Dict]:
    transcript = session.build_transcript()
    if not transcript.strip():
        return None

    user_msg = (
        f"Here is the conversation so far:\n\n{transcript}\n\n"
        "Extract all available ePCR field values from this transcript."
    )

    try:
        response = ai.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        print(f"[ai_client] No JSON in response: {text[:120]}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"[ai_client] JSON parse error: {e}", file=sys.stderr)
    except anthropic.APIError as e:
        print(f"[ai_client] Anthropic API error: {e}", file=sys.stderr)
    return None


# ── Output ─────────────────────────────────────────────────────────────────────

def print_changed_fields(changed_keys: List[str], session: Session, follow_ups: List[str], notes: Optional[str]):
    ts = datetime.now().strftime("%H:%M:%S")
    m  = session.metrics()
    print(f"\n[{ts}] ePCR  {m['filled']} filled  {m['review']} review  {m['missing']} missing")

    for key in changed_keys:
        f     = session.fields[key]
        label = f["label"]
        value = f["value"]
        conf  = f["confidence"]
        status = f["status"]
        evid  = f["evidence"][0] if f.get("evidence") else ""

        icon     = "+" if status == "filled" else "~"
        conf_str = f"{conf * 100:.0f}%"

        # Format value for display
        if isinstance(value, list):
            disp = ", ".join(str(v) for v in value)
        else:
            disp = str(value)

        print(f"  {icon} {label:<35} {conf_str:>4}  {disp}")
        if evid:
            print(f"      \"{evid[:90]}\"")

    if follow_ups:
        print(f"\n  Still needed: {', '.join(follow_ups[:5])}")
    if notes:
        print(f"  Notes: {notes}")


def save_form_state(session: Session, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session.session_id}_epcr.json"
    data = session.snapshot()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


# ── Background extraction loop ─────────────────────────────────────────────────

def extraction_loop(
    sessions: Dict[str, Session],
    sessions_lock: threading.Lock,
    system_prompt: str,
    ai: anthropic.Anthropic,
    model: str,
    output_dir: Path,
    debounce: float,
):
    """Polls for sessions with new utterances and runs Claude extraction."""
    while True:
        time.sleep(debounce)

        with sessions_lock:
            active = list(sessions.values())

        for session in active:
            new_count = session.take_new_count()
            if new_count == 0:
                continue

            result = extract_fields(session, system_prompt, ai, model)
            if result is None:
                continue

            extracted  = result.get("fields") or {}
            follow_ups = result.get("follow_up_needed") or []
            notes      = result.get("notes")

            changed = session.update_fields(extracted)

            if changed:
                print_changed_fields(changed, session, follow_ups, notes)
                path = save_form_state(session, output_dir)
                print(f"  -> {path}")
            else:
                # Nothing new — still save so utterance_count is current
                save_form_state(session, output_dir)


# ── Flask HTTP server ──────────────────────────────────────────────────────────

def make_app(sessions: Dict, sessions_lock: threading.Lock, field_index: Dict) -> Flask:
    app = Flask(__name__)
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)   # silence per-request logs

    @app.route("/api/ingest", methods=["POST"])
    def ingest():
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "no JSON body"}), 400

        session_id   = payload.get("session_id", "session-001")
        speaker_role = payload.get("speaker_role", "unknown")
        captured_at  = float(payload.get("captured_at") or time.time())
        segments     = payload.get("segments") or []

        with sessions_lock:
            if session_id not in sessions:
                sessions[session_id] = Session(session_id, field_index)
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] New session: {session_id}")
            session = sessions[session_id]

        session.add_utterances(speaker_role, captured_at, segments)

        return jsonify({
            "ok":             True,
            "session_id":     session_id,
            "utterances":     len(session.utterances),
        })

    @app.route("/api/session/<session_id>", methods=["GET"])
    def get_session(session_id):
        with sessions_lock:
            session = sessions.get(session_id)
        if session is None:
            return jsonify({"error": "session not found"}), 404
        return jsonify(session.snapshot())

    @app.route("/health", methods=["GET"])
    def health():
        with sessions_lock:
            n = len(sessions)
        return jsonify({"status": "ok", "sessions": n})

    return app


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-powered ePCR extraction from doa_transcribe.py transcript batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Start alongside doa_transcribe (which posts to http://127.0.0.1:8080/api/ingest by default):

    python ai_client.py &
    python doa_transcribe.py --speaker paramedic:0 --speaker patient:180
""",
    )
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--port",        type=int, default=DEFAULT_PORT)
    p.add_argument("--schema",      type=Path, default=DEFAULT_SCHEMA,
                   help="Path to epcr.yaml schema (default: schemas/epcr.yaml)")
    p.add_argument("--output-dir",  type=Path, default=Path("."),
                   help="Directory to write <session_id>_epcr.json files (default: .)")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help=f"Claude model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--debounce",    type=float, default=DEFAULT_DEBOUNCE,
                   help=f"Seconds between Claude extraction passes (default: {DEFAULT_DEBOUNCE})")
    p.add_argument("--api-key",     default=None,
                   help="Anthropic API key (default: $ANTHROPIC_API_KEY env var)")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Anthropic API key found. Set $ANTHROPIC_API_KEY or pass --api-key."
        )

    if not args.schema.exists():
        raise SystemExit(f"Schema not found: {args.schema}")

    print(f"Loading schema: {args.schema}")
    schema      = load_schema(args.schema)
    field_index = build_field_index(schema)
    print(f"  {len(field_index)} extractable fields loaded "
          f"({sum(1 for f in field_index.values() if not f.get('optional'))} required, "
          f"{sum(1 for f in field_index.values() if f.get('optional'))} optional)")

    system_prompt = build_system_prompt(field_index, schema)

    ai = anthropic.Anthropic(api_key=api_key)
    print(f"Claude model: {args.model}")
    print(f"Extraction debounce: {args.debounce}s")
    print(f"Output dir: {args.output_dir.resolve()}")

    sessions: Dict[str, Session] = {}
    sessions_lock = threading.Lock()

    extractor = threading.Thread(
        target=extraction_loop,
        args=(sessions, sessions_lock, system_prompt, ai, args.model, args.output_dir, args.debounce),
        daemon=True,
        name="extractor",
    )
    extractor.start()

    app = make_app(sessions, sessions_lock, field_index)
    print(f"\nListening on http://{args.host}:{args.port}/api/ingest\n")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
