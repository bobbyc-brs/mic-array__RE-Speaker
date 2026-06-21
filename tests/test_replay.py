#!/usr/bin/env python3
"""
test_replay.py — Replay a JSON conversation file against a running ai_client instance.

Sends all utterances as HTTP batches, then polls until extraction completes and
prints the resulting ePCR field values.

Usage:
    python tests/test_replay.py tests/conversations/concussion_assessment.json
    python tests/test_replay.py tests/conversations/chest_pain.json --provider cohere --wait 20
    python tests/test_replay.py tests/conversations/concussion_assessment.json --session my-run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. pip install requests")

BATCH_SIZE = 5   # utterances per POST


# ── sending ────────────────────────────────────────────────────────────────────

def send_conversation(url: str, session_id: str, utterances: list):
    base_time = time.time()
    n = len(utterances)
    for i in range(0, n, BATCH_SIZE):
        batch = utterances[i : i + BATCH_SIZE]
        # Merge consecutive same-speaker lines into one payload
        current_role: str = None
        segments = []
        captured_at = base_time
        for u in batch:
            if u["speaker"] != current_role:
                if segments and current_role:
                    _post(url, session_id, current_role, captured_at, segments)
                segments = []
                current_role = u["speaker"]
            segments.append({"text": u["text"], "confidence": 0.95})
            captured_at = base_time + float(u.get("offset", 0))
        if segments and current_role:
            _post(url, session_id, current_role, captured_at, segments)
        print(f"  sent {i + 1}–{min(i + BATCH_SIZE, n)}/{n}")


def _post(url, session_id, role, captured_at, segments):
    requests.post(
        f"{url}/api/ingest",
        json={"session_id": session_id, "speaker_role": role,
              "captured_at": captured_at, "segments": segments},
        timeout=5,
    ).raise_for_status()


# ── polling & display ──────────────────────────────────────────────────────────

def wait_for_extraction(url: str, session_id: str, max_wait: int) -> dict:
    print(f"\nWaiting up to {max_wait}s for extraction…", end="", flush=True)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(2)
        try:
            d = requests.get(f"{url}/api/session/{session_id}", timeout=5).json()
            if d["metrics"]["filled"] > 0:
                print(" done.")
                return d
        except Exception:
            pass
        print(".", end="", flush=True)
    print(" timed out.")
    return requests.get(f"{url}/api/session/{session_id}", timeout=5).json()


def print_results(d: dict):
    m = d["metrics"]
    roles = d.get("speaker_roles", {})
    print(f"\nUtterances : {d['utterance_count']}  |  "
          f"{m['filled']} filled  {m['review']} review  {m['missing']} missing / {m['total']} total")
    if roles:
        print(f"Roles      : {roles}")
    print()
    for f in d["fields"].values():
        if f["status"] not in ("filled", "review"):
            continue
        val  = f["value"]
        disp = ", ".join(val) if isinstance(val, list) else str(val)
        icon = "+" if f["status"] == "filled" else "~"
        ev   = f["evidence"][0][:75] if f.get("evidence") else ""
        print(f"  {icon} {f['label']:<35} {f['confidence']*100:.0f}%  {disp}")
        if ev:
            print(f'       "{ev}"')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Replay a JSON conversation file against a running ai_client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_replay.py tests/conversations/concussion_assessment.json
  python tests/test_replay.py tests/conversations/chest_pain.json --wait 20
  python tests/test_replay.py tests/conversations/concussion_assessment.json --session custom-id
""",
    )
    p.add_argument("conversation",        help="Path to a conversation .json file")
    p.add_argument("--session", default=None,
                   help="Session ID (default: slugified title from the JSON)")
    p.add_argument("--url",     default="http://127.0.0.1:8080",
                   help="ai_client base URL (default: http://127.0.0.1:8080)")
    p.add_argument("--wait",    type=int, default=15,
                   help="Max seconds to wait for extraction result (default 15)")
    args = p.parse_args()

    path = Path(args.conversation)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    conv       = json.loads(path.read_text(encoding="utf-8"))
    utterances = conv.get("utterances", [])
    title      = conv.get("title", path.stem)
    session_id = args.session or re.sub(r"[^\w-]", "-", title.lower())[:40]

    print(f"Scenario : {title}")
    if conv.get("description"):
        print(f"           {conv['description']}")
    print(f"Session  : {session_id}  →  {args.url}")
    print(f"Speakers : {sorted({u['speaker'] for u in utterances})}")
    print()

    send_conversation(args.url, session_id, utterances)
    result = wait_for_extraction(args.url, session_id, args.wait)
    print_results(result)


if __name__ == "__main__":
    main()
