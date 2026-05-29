"""
migrate_annotation.py
One-time script: converts annotation.json -> web_annotation.json
Adds an empty per-annotator 'annotations' dict to each entry.
LLM-generated reference_summary values are intentionally NOT carried over
so annotators remain unbiased.

Usage:
    python migrate_annotation.py
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_annotators() -> list[str]:
    names = []
    for i in range(1, 10):
        name = os.getenv(f"ANNOTATOR_{i}_NAME")
        if name:
            names.append(name)
    return names


def main():
    src = Path("annotation.json")
    dst = Path("web_annotation.json")

    if not src.exists():
        print("annotation.json not found. Run prepare_annotation.py first.")
        raise SystemExit(1)

    annotators = get_annotators()
    if not annotators:
        print("No annotators found in .env. Add ANNOTATOR_1_NAME, ANNOTATOR_2_NAME, etc.")
        raise SystemExit(1)

    data = json.loads(src.read_text(encoding="utf-8"))

    for entry in data:
        entry.pop("reference_summary", None)   # remove single-annotator field
        entry["annotations"] = {name: "" for name in annotators}

    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created '{dst}' with {len(data)} forums and {len(annotators)} annotators: {annotators}")


if __name__ == "__main__":
    main()
