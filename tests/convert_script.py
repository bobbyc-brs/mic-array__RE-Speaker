#!/usr/bin/env python3
"""
convert_script.py — Convert a plain-text conversation script to a JSON file
for use with test_replay.py.

Input format (one utterance per line, blank lines ignored):

    Paramedic 1: Hey there, I'm a paramedic. Can I talk to you for a minute?

    Patient: Uh… yeah. Okay.

    Paramedic 2: Over here — there's a bicycle down.

Usage:
    # Auto-generate role IDs from display names
    python tests/convert_script.py script.txt -o tests/conversations/my_case.json

    # Explicit name → role mapping
    python tests/convert_script.py script.txt \\
        --map "Paramedic 1:paramedic_1" \\
        --map "Paramedic 2:paramedic_2" \\
        --map "Patient:patient" \\
        --title "Chest pain call" \\
        -o tests/conversations/chest_pain.json

    # From stdin
    cat script.txt | python tests/convert_script.py - -o tests/conversations/out.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

WORDS_PER_SECOND = 2.5   # average conversational speech rate
GAP_SECONDS      = 1.0   # pause between speaker turns


def name_to_role(name: str) -> str:
    """Auto-generate a role ID from a display name: 'Paramedic 1' → 'paramedic_1'."""
    return re.sub(r"\s+", "_", name.strip().lower())


def estimate_duration(text: str) -> float:
    return max(1.0, len(text.split()) / WORDS_PER_SECOND)


def parse_script(raw: str, name_map: dict) -> list:
    utterances = []
    t = 0.0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]+):\s*(.+)$", line)
        if not m:
            continue
        display = m.group(1).strip()
        text    = m.group(2).strip()
        role    = name_map.get(display) or name_to_role(display)
        utterances.append({"speaker": role, "text": text, "offset": round(t, 1)})
        t += estimate_duration(text) + GAP_SECONDS
    return utterances


def main():
    p = argparse.ArgumentParser(
        description="Convert a Speaker: Text script to a JSON conversation file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    p.add_argument("input",          help="Input .txt script file (or - for stdin)")
    p.add_argument("-o", "--output", help="Output .json file (default: print to stdout)")
    p.add_argument("--title",        default="", help="Conversation title")
    p.add_argument("--description",  default="", help="Brief description of the scenario")
    p.add_argument(
        "--map", action="append", metavar="DISPLAY:ROLE",
        help="Map a display name to a role ID (repeatable). "
             "Without this, names are auto-converted: 'Paramedic 1' → 'paramedic_1'.",
    )
    args = p.parse_args()

    name_map = {}
    for entry in (args.map or []):
        parts = entry.split(":", 1)
        if len(parts) == 2:
            name_map[parts[0].strip()] = parts[1].strip()

    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")

    utterances = parse_script(raw, name_map)
    if not utterances:
        sys.exit("No utterances parsed. Check that lines follow 'Speaker: Text' format.")

    title = args.title or (
        Path(args.input).stem.replace("_", " ").replace("-", " ").title()
        if args.input != "-" else "Conversation"
    )

    doc = {"title": title, "description": args.description, "utterances": utterances}
    out = json.dumps(doc, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(f"Written {len(utterances)} utterances → {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
