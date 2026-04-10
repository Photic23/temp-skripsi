"""
prepare_annotation.py
Creates an annotation template from parsed forum JSONs.

Each entry contains the forum title, course, the full post thread (so you can
read it), and an empty 'reference_summary' field for you to fill in.

The generated summaries are intentionally excluded to avoid annotation bias.

Reads from:  parsed/   (output of parse_html.py)
Writes to:   annotation.json

Usage:
    python prepare_annotation.py
    python prepare_annotation.py --input parsed/ --output annotation.json
"""

import json
import argparse
from pathlib import Path


def build_readable_thread(posts: list[dict]) -> list[dict]:
    """Return a nested list retaining parent-child structure with only readable fields."""
    result = []
    for post in posts:
        result.append({
            "author": post.get("author", ""),
            "date": post.get("date", "")[:10],   # date only, drop time
            "content": post.get("content", "").strip(),
            "replies": build_readable_thread(post.get("replies", [])),
        })
    return result


def count_posts(posts: list[dict]) -> int:
    total = 0
    for p in posts:
        total += 1 + count_posts(p.get("replies", []))
    return total


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation template from parsed forums")
    parser.add_argument("--input", default="parsed", help="Parsed JSON directory (default: parsed)")
    parser.add_argument("--output", default="annotation.json", help="Output file (default: annotation.json)")
    args = parser.parse_args()

    parsed_dir = Path(args.input)
    if not parsed_dir.exists():
        print(f"Error: '{parsed_dir}' not found. Run parse_html.py first.")
        raise SystemExit(1)

    entries = []
    skipped = []

    for path in sorted(parsed_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        posts = data.get("posts", [])

        # Skip forums with no student replies
        has_replies = posts and posts[0].get("replies")
        if not has_replies:
            rel = path.relative_to(parsed_dir)
            skipped.append(str(rel))
            continue

        rel_id = str(path.relative_to(parsed_dir).with_suffix(""))
        total = count_posts(posts)

        entries.append({
            "id": rel_id,
            "title": data.get("title", ""),
            "course": data.get("course", ""),
            "total_posts": total,
            "thread": build_readable_thread(posts),
            "reference_summary": ""     # ← fill this in
        })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Annotation template created: '{out_path}'")
    print(f"  {len(entries)} forum(s) to annotate")
    if skipped:
        print(f"  {len(skipped)} skipped (no replies): {', '.join(skipped)}")
    print(f"\nOpen '{out_path}' and fill in each 'reference_summary' field.")
    print("When done, run generate_summaries.py then evaluate.py.")


if __name__ == "__main__":
    main()
