"""
generate_for_annotated.py
Generates model summaries only for forums that have human annotations,
then merges them into a single file ready for evaluate.py.

Reads:
  - annotated-data/summaries-fathir-clean.json  (reference summaries)
  - parsed/{id}.json                             (forum post trees)

Writes:
  - summaries.json  (both generated_summary + reference_summary filled)

Usage:
    # Make sure app.py is running first, then:
    python generate_for_annotated.py
    python generate_for_annotated.py --annotations annotated-data/summaries-fathir-clean.json
    python generate_for_annotated.py --url http://localhost:5000 --output summaries.json
"""

import json
import time
import argparse
from pathlib import Path

import requests


def call_summarizer(url: str, posts: list, timeout: int = 0, retries: int = 2) -> str | None:
    # timeout=0 means no limit (None in requests)
    effective_timeout = None if timeout == 0 else timeout
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/summarize/forum",
                json={"posts": posts},
                timeout=effective_timeout,
            )
            resp.raise_for_status()
            return resp.json().get("summary", "")
        except requests.exceptions.ConnectionError:
            print("    ERROR: Cannot connect to summarizer. Is app.py running?")
            return None
        except requests.exceptions.Timeout:
            if attempt <= retries:
                print(f"    Timeout on attempt {attempt}/{retries + 1}, retrying...")
            else:
                print(f"    ERROR: Timed out after {retries + 1} attempts.")
                return None
        except Exception as e:
            print(f"    ERROR: {e}")
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate model summaries for annotated forums and merge with reference summaries"
    )
    parser.add_argument(
        "--annotations",
        default="annotated-data/summaries-fathir-clean.json",
        help="Path to cleaned annotations JSON (default: annotated-data/summaries-fathir-clean.json)",
    )
    parser.add_argument(
        "--parsed",
        default="parsed",
        help="Directory of parsed forum JSON files (default: parsed)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Base URL of the summarizer Flask app (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--output",
        default="summaries.json",
        help="Output file for evaluate.py (default: summaries.json)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entries that already have a generated_summary in the output file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Seconds to wait per request before retrying (default: 0 = no limit)",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    parsed_dir = Path(args.parsed)
    out_path = Path(args.output)

    if not annotations_path.exists():
        print(f"ERROR: Annotations file not found: {annotations_path}")
        raise SystemExit(1)
    if not parsed_dir.exists():
        print(f"ERROR: Parsed directory not found: {parsed_dir}")
        raise SystemExit(1)

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(annotations)} annotated entries from '{annotations_path}'.")

    # Always load existing output so --skip-existing can resume
    existing = {}
    if out_path.exists():
        try:
            for entry in json.loads(out_path.read_text(encoding="utf-8")):
                if entry.get("generated_summary", "").strip():
                    existing[entry["id"]] = entry
        except Exception:
            pass
    if args.skip_existing and existing:
        print(f"Resuming — {len(existing)} entries already done.\n")

    results = list(existing.values()) if args.skip_existing else []
    skipped = 0
    missing_parsed = []

    for i, ann in enumerate(annotations, 1):
        forum_id = ann["id"]
        ref_summary = ann.get("reference_summary", "").strip()

        if not ref_summary:
            print(f"[{i}/{len(annotations)}] {forum_id}  → SKIP (no reference_summary)")
            skipped += 1
            continue

        # Skip if already generated (only when --skip-existing is active)
        if args.skip_existing and forum_id in existing:
            print(f"[{i}/{len(annotations)}] {forum_id}  → already done")
            continue

        # Find the parsed JSON — IDs use backslashes on Windows, convert to path separators
        parsed_path = parsed_dir / Path(forum_id.replace("\\", "/")).with_suffix(".json")
        if not parsed_path.exists():
            print(f"[{i}/{len(annotations)}] {forum_id}  → SKIP (parsed file not found: {parsed_path})")
            missing_parsed.append(forum_id)
            skipped += 1
            continue

        forum_data = json.loads(parsed_path.read_text(encoding="utf-8"))
        posts = forum_data.get("posts", [])

        if not posts or not posts[0].get("replies"):
            print(f"[{i}/{len(annotations)}] {forum_id}  → SKIP (no replies in parsed data)")
            skipped += 1
            continue

        print(f"[{i}/{len(annotations)}] {forum_id}")
        t0 = time.time()
        summary = call_summarizer(args.url, posts, timeout=args.timeout)
        elapsed = time.time() - t0

        if summary is None:
            print("  Stopping — progress saved. Re-run with --skip-existing to resume.")
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit(1)

        print(f"  Done in {elapsed:.1f}s — {len(summary.split())} words")

        results.append({
            "id": forum_id,
            "title": ann["title"],
            "course": ann["course"],
            "generated_summary": summary,
            "reference_summary": ref_summary,
        })

        # Save after every entry so --skip-existing can always resume
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done. {len(results)} entries written to '{out_path}'.")
    if skipped:
        print(f"  Skipped: {skipped} entries (no annotation or no parsed data)")
    if missing_parsed:
        print(f"  Missing parsed files for: {missing_parsed}")
    print(f"\nNext step: python evaluate.py --input {out_path}")


if __name__ == "__main__":
    main()
