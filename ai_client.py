#!/usr/bin/env python3
"""
ai_client.py — AI-powered ePCR field extraction from live transcript batches.

Listens for transcript_batch events from doa_transcribe.py (HTTP POST /api/ingest),
accumulates conversation context per session, and calls an AI provider to extract
structured ePCR field values. Updated fields are printed to the screen and the full
form state is written to a JSON file after each extraction pass.

Provider priority (first key found wins):
  1. Anthropic  — $ANTHROPIC_API_KEY  or  AnthropicAPI.key
  2. Perplexity — $PERPLEXITY_KEY     or  PerplexityAPI.key
  3. Cohere     — $COHERE_API_KEY     or  CohereAPI.key

Usage:
    python ai_client.py                            # auto-detect provider, port 8080
    python ai_client.py --port 9090
    python ai_client.py --output-dir /tmp/pcr
    python ai_client.py --model sonar              # override model for detected provider
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

import yaml
from flask import Flask, jsonify, request

DEFAULT_SCHEMA   = Path(__file__).parent / "schemas" / "epcr.yaml"
DEFAULT_PORT     = 8080
DEFAULT_DEBOUNCE = 4.0

EXTRACTABLE_SOURCES = {"conversation", "paramedic"}

# ── Provider catalogue ─────────────────────────────────────────────────────────
# Checked in order; first one with a key wins.

PROVIDERS = [
    {
        "name":     "Anthropic",
        "env":      "ANTHROPIC_API_KEY",
        "file":     "AnthropicAPI.key",
        "type":     "anthropic",
        "model":    "claude-haiku-4-5-20251001",
    },
    {
        "name":     "Perplexity",
        "env":      "PERPLEXITY_KEY",
        "file":     "PerplexityAPI.key",
        "type":     "openai_compat",
        "base_url": "https://api.perplexity.ai",
        "model":    "sonar",
    },
    {
        "name":     "Cohere",
        "env":      "COHERE_API_KEY",
        "file":     "CohereAPI.key",
        "type":     "openai_compat",
        "base_url": "https://api.cohere.com/compatibility/v1",
        "model":    "command-r-plus",
    },
]


def _read_key(env: str, file: str) -> Optional[str]:
    val = os.environ.get(env, "").strip()
    if val:
        return val
    p = Path(file)
    if p.exists():
        val = p.read_text().strip()
        if val:
            return val
    return None


def resolve_provider(name: Optional[str] = None) -> Optional[Dict]:
    """Return first available provider dict (with 'key' added), or None.
    If name is given, only that provider is tried."""
    candidates = PROVIDERS
    if name:
        name_lower = name.lower()
        candidates = [p for p in PROVIDERS if p["name"].lower() == name_lower]
        if not candidates:
            names = ", ".join(p["name"].lower() for p in PROVIDERS)
            raise SystemExit(f"Unknown provider '{name}'. Choose from: {names}")
    for spec in candidates:
        key = _read_key(spec["env"], spec["file"])
        if key:
            return {**spec, "key": key}
    return None


# ── LLM client abstraction ─────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, provider: Dict, model_override: Optional[str] = None):
        self.provider_name = provider["name"]
        self.model         = model_override or provider["model"]
        self._type         = provider["type"]

        if self._type == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=provider["key"])
        else:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=provider["key"],
                base_url=provider["base_url"],
            )

    def chat(self, system: str, user: str) -> str:
        if self._type == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content


# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_field_index(schema: Dict) -> Dict[str, Dict]:
    """Fields extractable from audio. Excludes scan/vitals/derived and tables."""
    return {
        f["key"]: f
        for f in schema["fields"]
        if f.get("source") in EXTRACTABLE_SOURCES and f.get("type") != "table"
    }


# ── System prompt construction ─────────────────────────────────────────────────

def build_system_prompt(field_index: Dict[str, Dict], schema: Dict) -> str:
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
            unlock_info  = (f"  unlocked-by={f['unlock_keywords'][:6]}"
                            if f.get("unlock_keywords") else "")
            syn_info     = (f"  synonyms={f['synonyms'][:8]}"
                            if f.get("synonyms") else "")
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
        changed = []
        with self._lock:
            for key, data in extracted.items():
                if key not in self.fields:
                    continue
                value = data.get("value")
                if value is None:
                    continue
                conf     = float(data.get("confidence") or 0.0)
                evidence = str(data.get("evidence") or "").strip()
                current  = self.fields[key]
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

    def _metrics_locked(self) -> Dict[str, int]:
        """Compute metrics — caller must already hold self._lock."""
        statuses = [f["status"] for f in self.fields.values()]
        return {
            "filled":  statuses.count("filled"),
            "review":  statuses.count("review"),
            "missing": statuses.count("missing"),
            "total":   len(statuses),
        }

    def metrics(self) -> Dict[str, int]:
        with self._lock:
            return self._metrics_locked()

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "session_id":      self.session_id,
                "updated_at":      time.time(),
                "utterance_count": len(self.utterances),
                "fields":          {k: dict(v) for k, v in self.fields.items()},
                "metrics":         self._metrics_locked(),
            }


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_fields(session: Session, system_prompt: str, llm: LLMClient) -> Optional[Dict]:
    transcript = session.build_transcript()
    if not transcript.strip():
        return None

    user_msg = (
        f"Here is the conversation so far:\n\n{transcript}\n\n"
        "Extract all available ePCR field values from this transcript."
    )

    try:
        text = llm.chat(system_prompt, user_msg).strip()
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
    except Exception as e:
        print(f"[ai_client] API error: {e}", file=sys.stderr)
    return None


# ── Output ─────────────────────────────────────────────────────────────────────

def print_changed_fields(
    changed_keys: List[str],
    session: Session,
    follow_ups: List[str],
    notes: Optional[str],
):
    ts = datetime.now().strftime("%H:%M:%S")
    m  = session.metrics()
    print(f"\n[{ts}] ePCR  {m['filled']} filled  {m['review']} review  {m['missing']} missing")

    for key in changed_keys:
        f        = session.fields[key]
        conf_str = f"{f['confidence'] * 100:.0f}%"
        icon     = "+" if f["status"] == "filled" else "~"
        evid     = f["evidence"][0] if f.get("evidence") else ""
        value    = f["value"]
        disp     = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)

        print(f"  {icon} {f['label']:<35} {conf_str:>4}  {disp}")
        if evid:
            print(f"      \"{evid[:90]}\"")

    if follow_ups:
        print(f"\n  Still needed: {', '.join(follow_ups[:5])}")
    if notes:
        print(f"  Notes: {notes}")


def save_form_state(session: Session, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session.session_id}_epcr.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.snapshot(), f, indent=2, ensure_ascii=False, default=str)
    return path


# ── Background extraction loop ─────────────────────────────────────────────────

def extraction_loop(
    sessions: Dict[str, "Session"],
    sessions_lock: threading.Lock,
    system_prompt: str,
    llm: LLMClient,
    output_dir: Path,
    debounce: float,
):
    while True:
        time.sleep(debounce)
        with sessions_lock:
            active = list(sessions.values())
        for session in active:
            if session.take_new_count() == 0:
                continue
            result = extract_fields(session, system_prompt, llm)
            if result is None:
                continue
            extracted  = result.get("fields") or {}
            follow_ups = result.get("follow_up_needed") or []
            notes      = result.get("notes")
            changed    = session.update_fields(extracted)
            if changed:
                print_changed_fields(changed, session, follow_ups, notes)
                path = save_form_state(session, output_dir)
                print(f"  -> {path}")
            else:
                save_form_state(session, output_dir)


# ── Flask HTTP server ──────────────────────────────────────────────────────────

def make_app(sessions: Dict, sessions_lock: threading.Lock, field_index: Dict) -> Flask:
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

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
                print(f"[{datetime.now():%H:%M:%S}] New session: {session_id}")
            session = sessions[session_id]

        session.add_utterances(speaker_role, captured_at, segments)
        return jsonify({"ok": True, "session_id": session_id,
                        "utterances": len(session.utterances)})

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

Provider key lookup order (env var then file):
    Anthropic  : $ANTHROPIC_API_KEY  /  AnthropicAPI.key
    Perplexity : $PERPLEXITY_KEY     /  PerplexityAPI.key
    Cohere     : $COHERE_API_KEY     /  CohereAPI.key
""",
    )
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--port",       type=int, default=DEFAULT_PORT)
    p.add_argument("--schema",     type=Path, default=DEFAULT_SCHEMA,
                   help="Path to epcr.yaml schema (default: schemas/epcr.yaml)")
    p.add_argument("--output-dir", type=Path, default=Path("."),
                   help="Directory to write <session_id>_epcr.json (default: .)")
    p.add_argument("--provider",   default=None,
                   metavar="NAME",
                   help="Force a specific provider: anthropic, perplexity, cohere (default: auto)")
    p.add_argument("--model",      default=None,
                   help="Override the provider's default model")
    p.add_argument("--debounce",   type=float, default=DEFAULT_DEBOUNCE,
                   help=f"Seconds between extraction passes (default: {DEFAULT_DEBOUNCE})")
    return p.parse_args()


def main():
    args = parse_args()

    provider = resolve_provider(args.provider)
    if provider is None:
        raise SystemExit(
            "No API key found. Provide one of:\n"
            "  $ANTHROPIC_API_KEY  or  AnthropicAPI.key\n"
            "  $PERPLEXITY_KEY     or  PerplexityAPI.key\n"
            "  $COHERE_API_KEY     or  CohereAPI.key"
        )

    if not args.schema.exists():
        raise SystemExit(f"Schema not found: {args.schema}")

    schema      = load_schema(args.schema)
    field_index = build_field_index(schema)
    req_count   = sum(1 for f in field_index.values() if not f.get("optional"))
    opt_count   = sum(1 for f in field_index.values() if f.get("optional"))

    llm           = LLMClient(provider, model_override=args.model)
    system_prompt = build_system_prompt(field_index, schema)

    print(f"Provider : {llm.provider_name}  (model: {llm.model})")
    print(f"Schema   : {args.schema}  ({len(field_index)} fields: {req_count} required, {opt_count} optional)")
    print(f"Debounce : {args.debounce}s")
    print(f"Output   : {args.output_dir.resolve()}")

    sessions: Dict[str, Session] = {}
    sessions_lock = threading.Lock()

    threading.Thread(
        target=extraction_loop,
        args=(sessions, sessions_lock, system_prompt, llm, args.output_dir, args.debounce),
        daemon=True,
        name="extractor",
    ).start()

    app = make_app(sessions, sessions_lock, field_index)
    print(f"\nListening on http://{args.host}:{args.port}/api/ingest\n")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
