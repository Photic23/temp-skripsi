"""
build_mt5_rev.py
Derives summaries-annotated-rev/summaries-bryan.json from the already-generated
summaries-annotated-rev/summaries-fathir.json by swapping in Bryan's references.

Usage:
    python build_mt5_rev.py
"""

import json
from pathlib import Path

FATHIR_REV  = Path("summaries-annotated-rev/summaries-fathir.json")
BRYAN_SRC   = Path("summaries-annotated/summaries-bryan.json")
BRYAN_OUT   = Path("summaries-annotated-rev/summaries-bryan.json")


def main():
    for p in [FATHIR_REV, BRYAN_SRC]:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            raise SystemExit(1)

    fathir_rev = json.loads(FATHIR_REV.read_text(encoding="utf-8"))
    bryan_src  = json.loads(BRYAN_SRC.read_text(encoding="utf-8"))

    bryan_ref = {e["id"]: e.get("reference_summary", "").strip() for e in bryan_src}

    result = []
    skipped = []
    for entry in fathir_rev:
        id_ = entry["id"]
        ref = bryan_ref.get(id_, "")
        if not ref:
            skipped.append(id_)
            continue
        result.append({
            "id": id_,
            "title": entry.get("title", ""),
            "course": entry.get("course", ""),
            "generated_summary": entry["generated_summary"],
            "reference_summary": ref,
        })

    BRYAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    BRYAN_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {len(result)} entries → {BRYAN_OUT}")
    if skipped:
        print(f"Skipped {len(skipped)} (no Bryan reference): {skipped}")


if __name__ == "__main__":
    main()
