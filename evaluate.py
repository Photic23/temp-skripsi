"""
evaluate.py
Runs ROUGE and BERTScore on the generated vs. reference summaries.

Reads:   summaries.json  (output of generate_for_annotated.py)
Writes:  results.json    (per-entry scores + aggregate averages)

Usage:
    python evaluate.py
    python evaluate.py --input summaries.json --output results.json
    python evaluate.py --lang id
"""

import json
import argparse
from pathlib import Path


def compute_rouge(hypotheses: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    scores_per_entry = []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        entry = {k: round(s[k].fmeasure, 4) for k in agg}
        scores_per_entry.append(entry)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    avg = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
    return {"per_entry": scores_per_entry, "average": avg}


def compute_bertscore(hypotheses: list[str], references: list[str], lang: str = "id") -> dict:
    from bert_score import score as bert_score
    P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
    scores_per_entry = [
        {"precision": round(p.item(), 4), "recall": round(r.item(), 4), "f1": round(f.item(), 4)}
        for p, r, f in zip(P, R, F1)
    ]
    avg = {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }
    return {"per_entry": scores_per_entry, "average": avg}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries with ROUGE and BERTScore")
    parser.add_argument("--input", default="summaries.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--lang", default="id",
                        help="Language code for BERTScore (default: id for Indonesian)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    missing = [e["id"] for e in data if not e.get("reference_summary", "").strip()]
    if missing:
        print(f"ERROR: {len(missing)} entry/entries are missing 'reference_summary':")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)

    hypotheses = [e["generated_summary"] for e in data]
    references = [e["reference_summary"] for e in data]

    # --- ROUGE ---
    print("Computing ROUGE...", flush=True)
    rouge_results = compute_rouge(hypotheses, references)
    print(f"  ROUGE-1 avg : {rouge_results['average']['rouge1']}")
    print(f"  ROUGE-2 avg : {rouge_results['average']['rouge2']}")
    print(f"  ROUGE-L avg : {rouge_results['average']['rougeL']}")

    # --- BERTScore ---
    print(f"\nComputing BERTScore (lang={args.lang})...", flush=True)
    bert_results = compute_bertscore(hypotheses, references, lang=args.lang)
    print(f"  Precision avg : {bert_results['average']['precision']}")
    print(f"  Recall avg    : {bert_results['average']['recall']}")
    print(f"  F1 avg        : {bert_results['average']['f1']}")

    # --- Assemble output ---
    output = []
    for i, entry in enumerate(data):
        output.append({
            "id": entry["id"],
            "title": entry["title"],
            "course": entry["course"],
            "generated_summary": entry["generated_summary"],
            "reference_summary": entry["reference_summary"],
            "rouge": rouge_results["per_entry"][i],
            "bertscore": bert_results["per_entry"][i],
        })

    final = {
        "averages": {
            "rouge": rouge_results["average"],
            "bertscore": bert_results["average"],
        },
        "entries": output,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to '{out_path}'.")


if __name__ == "__main__":
    main()
