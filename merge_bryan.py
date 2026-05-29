"""
merge_bryan.py
Builds summaries-bryan.json by pairing generated summaries from summaries.json
with reference summaries from summaries-bryan-clean.json (matched by ID).

Usage:
    python merge_bryan.py
    python merge_bryan.py --generated summaries.json --annotations annotated-data/summaries-bryan-clean.json --output summaries-bryan.json
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", default="summaries.json")
    parser.add_argument("--annotations", default="annotated-data/summaries-bryan-clean.json")
    parser.add_argument("--output", default="summaries-bryan.json")
    args = parser.parse_args()

    generated = {e["id"]: e for e in json.loads(Path(args.generated).read_text(encoding="utf-8"))}
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))

    results = []
    skipped = []

    for ann in annotations:
        fid = ann["id"]
        ref = ann.get("reference_summary", "").strip()
        if not ref:
            skipped.append(fid)
            continue
        if fid not in generated:
            skipped.append(fid)
            print(f"  WARNING: {fid} not found in generated summaries, skipping.")
            continue
        results.append({
            "id": fid,
            "title": ann["title"],
            "course": ann["course"],
            "generated_summary": generated[fid]["generated_summary"],
            "reference_summary": ref,
        })

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {len(results)} entries to '{args.output}'.")
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
