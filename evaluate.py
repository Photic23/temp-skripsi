"""
evaluate.py
Runs ROUGE, BERTScore, and TIGERScore on the generated vs. reference summaries.

Reads:   summaries.json  (output of generate_summaries.py, with reference_summary filled in)
Writes:  results.json    (per-entry scores + aggregate averages)

Usage:
    python evaluate.py
    python evaluate.py --input summaries.json --output results.json
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


def compute_tigerscore(sources: list[str], hypotheses: list[str]) -> dict:
    """
    TIGERScore is reference-free: evaluates (source, hypothesis) pairs.
    source here is the concatenated forum post text.
    """
    from tigerscore import TIGERScorer
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-V1.0")
    instruction = "Summarize the following forum discussion into a concise paragraph."
    results_raw = scorer.score(
        instruction=[instruction] * len(hypotheses),
        input_context=sources,
        hypo_output=hypotheses,
    )
    scores_per_entry = []
    raw_scores = []
    for r in results_raw:
        s = r.get("score", None)
        scores_per_entry.append({"score": round(s, 4) if s is not None else None,
                                  "explanation": r.get("explanation", "")})
        if s is not None:
            raw_scores.append(s)
    avg = round(sum(raw_scores) / len(raw_scores), 4) if raw_scores else None
    return {"per_entry": scores_per_entry, "average": {"score": avg}}


def flatten_posts_to_text(posts: list[dict]) -> str:
    """Flatten nested post tree into a single plain-text string for TIGERScore source."""
    lines = []

    def _walk(post):
        author = post.get("author", "")
        content = post.get("content", "").strip()
        if content:
            lines.append(f"[{author}]: {content}")
        for r in post.get("replies", []):
            _walk(r)

    for p in posts:
        _walk(p)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument("--input", default="summaries.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--lang", default="id",
                        help="Language code for BERTScore (default: id for Indonesian)")
    parser.add_argument("--skip-tigerscore", action="store_true",
                        help="Skip TIGERScore (requires large GPU model)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    # Validate all entries have both summaries filled in
    missing = [e["id"] for e in data if not e.get("reference_summary", "").strip()]
    if missing:
        print(f"ERROR: {len(missing)} entry/entries are missing 'reference_summary':")
        for m in missing:
            print(f"  - {m}")
        print("\nFill in the 'reference_summary' fields in summaries.json, then re-run.")
        raise SystemExit(1)

    hypotheses = [e["generated_summary"] for e in data]
    references = [e["reference_summary"] for e in data]

    # --- ROUGE ---
    print("Computing ROUGE...", flush=True)
    rouge_results = compute_rouge(hypotheses, references)
    print(f"  ROUGE-1 avg: {rouge_results['average']['rouge1']}")
    print(f"  ROUGE-2 avg: {rouge_results['average']['rouge2']}")
    print(f"  ROUGE-L avg: {rouge_results['average']['rougeL']}")

    # --- BERTScore ---
    print(f"\nComputing BERTScore (lang={args.lang})...", flush=True)
    bert_results = compute_bertscore(hypotheses, references, lang=args.lang)
    print(f"  BERTScore F1 avg: {bert_results['average']['f1']}")

    # --- TIGERScore ---
    tiger_results = None
    if not args.skip_tigerscore:
        print("\nComputing TIGERScore (this may take a while)...", flush=True)

        # Rebuild source texts from parsed JSONs (if available), else use reference as proxy
        parsed_dir = Path("parsed")
        sources = []
        for entry in data:
            json_path = parsed_dir / (entry["id"] + ".json")
            if json_path.exists():
                forum = json.loads(json_path.read_text(encoding="utf-8"))
                sources.append(flatten_posts_to_text(forum.get("posts", [])))
            else:
                # Fallback: use reference summary as source proxy
                sources.append(entry["reference_summary"])

        tiger_results = compute_tigerscore(sources, hypotheses)
        print(f"  TIGERScore avg: {tiger_results['average']['score']}")

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
            "tigerscore": tiger_results["per_entry"][i] if tiger_results else None,
        })

    final = {
        "averages": {
            "rouge": rouge_results["average"],
            "bertscore": bert_results["average"],
            "tigerscore": tiger_results["average"] if tiger_results else None,
        },
        "entries": output,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to '{out_path}'.")


if __name__ == "__main__":
    main()
