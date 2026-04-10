"""
generate_summaries.py
Calls the summarizer Flask API on every parsed JSON and saves the results.

Reads from:  parsed/   (output of parse_html.py)
Writes to:   summaries.json   (flat list of {id, title, course, source_json, generated_summary})

Usage:
    python generate_summaries.py                          # default http://localhost:5000
    python generate_summaries.py --url http://localhost:5000
    python generate_summaries.py --url http://localhost:5000 --input parsed/
"""

import json
import time
import argparse
from pathlib import Path

import requests


def load_parsed_jsons(parsed_dir: Path) -> list[dict]:
    """Walk parsed_dir and return all non-empty forum JSONs."""
    entries = []
    for path in sorted(parsed_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        posts = data.get("posts", [])
        if not posts or not posts[0].get("replies"):
            # Skip forums with no replies — nothing to summarise
            rel = path.relative_to(parsed_dir)
            print(f"  [skip] {rel}  (no replies)")
            continue
        entries.append({
            "id": str(path.relative_to(parsed_dir).with_suffix("")),
            "title": data.get("title", ""),
            "course": data.get("course", ""),
            "posts": posts,
        })
    return entries


def call_summarizer(url: str, posts: list, timeout: int = 300) -> str | None:
    """POST to /summarize/forum and return the summary string, or None on error."""
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/summarize/forum",
            json={"posts": posts},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("summary", "")
    except requests.exceptions.ConnectionError:
        print("    ERROR: Cannot connect to summarizer. Is the Flask app running?")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for all parsed forums")
    parser.add_argument("--url", default="http://localhost:5000",
                        help="Base URL of the summarizer Flask app (default: http://localhost:5000)")
    parser.add_argument("--input", default="parsed",
                        help="Directory of parsed JSON files (default: parsed)")
    parser.add_argument("--output", default="summaries.json",
                        help="Output file (default: summaries.json)")
    args = parser.parse_args()

    parsed_dir = Path(args.input)
    if not parsed_dir.exists():
        print(f"Error: '{parsed_dir}' not found. Run parse_html.py first.")
        raise SystemExit(1)

    entries = load_parsed_jsons(parsed_dir)
    print(f"\nFound {len(entries)} forum(s) to summarise.\n")

    results = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['id']}")
        t0 = time.time()
        summary = call_summarizer(args.url, entry["posts"])
        elapsed = time.time() - t0

        if summary is None:
            print("  Stopping — fix the connection issue and re-run.")
            raise SystemExit(1)

        print(f"  Done in {elapsed:.1f}s — {len(summary.split())} words")
        results.append({
            "id": entry["id"],
            "title": entry["title"],
            "course": entry["course"],
            "generated_summary": summary,
            "reference_summary": "",   # filled in by human annotation
        })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} result(s) to '{out_path}'.")
    print("Next step: fill in 'reference_summary' fields in that file, then run evaluate.py")


if __name__ == "__main__":
    main()
