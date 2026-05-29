"""
visualize.py
Generates comparison charts for two annotator evaluation results.

Produces:
  - comparison_averages.png  : grouped bar chart of average scores (Fathir vs Bryan)
  - comparison_boxplot.png   : box plots of per-entry score distributions

Usage:
    python visualize.py
    python visualize.py --fathir results.json --bryan results-bryan.json --out charts/
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


METRIC_INFO = {
    "ROUGE-1": {
        "label": "ROUGE-1",
        "description": "Unigram overlap between generated\nand reference summary.",
        "justification": "Measures basic vocabulary coverage;\nwidely used as a baseline lexical metric.",
    },
    "ROUGE-2": {
        "label": "ROUGE-2",
        "description": "Bigram overlap between generated\nand reference summary.",
        "justification": "Captures phrase-level similarity and\nword order, stricter than ROUGE-1.",
    },
    "ROUGE-L": {
        "label": "ROUGE-L",
        "description": "Longest Common Subsequence (LCS)\nbetween generated and reference.",
        "justification": "Reflects sentence-level fluency and\nstructural similarity without requiring\nconsecutive matches.",
    },
    "BERTScore\nPrecision": {
        "label": "BERTScore\nPrecision",
        "description": "Fraction of generated tokens\nsemantically matched in the reference.",
        "justification": "Penalises hallucinated content not\npresent in the reference summary.",
    },
    "BERTScore\nRecall": {
        "label": "BERTScore\nRecall",
        "description": "Fraction of reference tokens\ncovered by the generated summary.",
        "justification": "Measures how completely the model\ncaptures the reference's key content.",
    },
    "BERTScore\nF1": {
        "label": "BERTScore\nF1",
        "description": "Harmonic mean of BERTScore\nPrecision and Recall.",
        "justification": "Primary semantic metric; uses\nmultilingual BERT embeddings to\nhandle Indonesian paraphrases.",
    },
}

BOXPLOT_METRIC_INFO = {
    "ROUGE-1":      METRIC_INFO["ROUGE-1"],
    "ROUGE-2":      METRIC_INFO["ROUGE-2"],
    "ROUGE-L":      METRIC_INFO["ROUGE-L"],
    "BERTScore F1": METRIC_INFO["BERTScore\nF1"],
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def per_entry_scores(data: dict) -> dict:
    return {
        "ROUGE-1":  [e["rouge"]["rouge1"]            for e in data["entries"]],
        "ROUGE-2":  [e["rouge"]["rouge2"]            for e in data["entries"]],
        "ROUGE-L":  [e["rouge"]["rougeL"]            for e in data["entries"]],
        "BERTScore F1": [e["bertscore"]["f1"]        for e in data["entries"]],
    }


def averages(data: dict) -> dict:
    r = data["averages"]["rouge"]
    b = data["averages"]["bertscore"]
    return {
        "ROUGE-1":      r["rouge1"],
        "ROUGE-2":      r["rouge2"],
        "ROUGE-L":      r["rougeL"],
        "BERTScore\nPrecision": b["precision"],
        "BERTScore\nRecall":    b["recall"],
        "BERTScore\nF1":        b["f1"],
    }


def plot_averages(fathir: dict, bryan: dict, out_path: Path):
    metrics = list(fathir.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    bars1 = ax.bar(x - width / 2, list(fathir.values()), width, label="Fathir", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, list(bryan.values()),  width, label="Bryan",  color="#DD8452")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_INFO[m]["label"] for m in metrics], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Average Evaluation Scores — Fathir vs Bryan", fontsize=13, pad=10)
    ax.legend()
    ax.set_ylim(0, max(max(fathir.values()), max(bryan.values())) + 0.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Description table
    ax_table.axis("off")
    col_labels = ["Metric", "Description", "Justification"]
    table_data = [
        [
            METRIC_INFO[m]["label"].replace("\n", " "),
            METRIC_INFO[m]["description"],
            METRIC_INFO[m]["justification"],
        ]
        for m in metrics
    ]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.15, 0.42, 0.43],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#BBBBBB")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_boxplots(fathir: dict, bryan: dict, out_path: Path):
    metrics = list(fathir.keys())
    n = len(metrics)

    fig = plt.figure(figsize=(4 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1.8], hspace=0.5)
    axes      = [fig.add_subplot(gs[0, i]) for i in range(n)]
    axes_desc = [fig.add_subplot(gs[1, i]) for i in range(n)]

    colors = ["#4C72B0", "#DD8452"]

    for ax, ax_desc, metric in zip(axes, axes_desc, metrics):
        data = [fathir[metric], bryan[metric]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric, fontsize=11, pad=6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Fathir", "Bryan"], fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Description box below each plot
        info = BOXPLOT_METRIC_INFO[metric]
        desc_text = f"Description:\n{info['description']}\n\nJustification:\n{info['justification']}"
        ax_desc.axis("off")
        ax_desc.text(
            0.5, 0.95, desc_text,
            transform=ax_desc.transAxes,
            fontsize=7.5, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BBBBBB"),
            wrap=True,
        )

    fig.suptitle("Per-Entry Score Distribution — Fathir vs Bryan", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def avg_dicts(a: dict, b: dict) -> dict:
    """Average two metric dicts with the same keys."""
    return {k: round((a[k] + b[k]) / 2, 4) for k in a}


def avg_lists(a: dict, b: dict) -> dict:
    """Element-wise average two per-entry score dicts."""
    return {k: [(x + y) / 2 for x, y in zip(a[k], b[k])] for k in a}


def plot_model_comparison(mt5_avg: dict, claude_avg: dict, out_path: Path):
    """Grouped bar chart comparing mT5 vs Claude, scores averaged across annotators."""
    metrics = list(mt5_avg.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    bars1 = ax.bar(x - width / 2, list(mt5_avg.values()),    width, label="mT5 (default)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, list(claude_avg.values()), width, label="Claude",         color="#55A868")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_INFO[m]["label"] for m in metrics], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("mT5 vs Claude — Average Scores (Fathir + Bryan)", fontsize=13, pad=10)
    ax.legend()
    ax.set_ylim(0, max(max(mt5_avg.values()), max(claude_avg.values())) + 0.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax_table.axis("off")
    col_labels = ["Metric", "Description", "Justification"]
    table_data = [
        [
            METRIC_INFO[m]["label"].replace("\n", " "),
            METRIC_INFO[m]["description"],
            METRIC_INFO[m]["justification"],
        ]
        for m in metrics
    ]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.15, 0.42, 0.43],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#BBBBBB")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_model_boxplots(mt5_scores: dict, claude_scores: dict, out_path: Path):
    """Box plots comparing mT5 vs Claude per-entry distributions."""
    metrics = list(mt5_scores.keys())
    n = len(metrics)

    fig = plt.figure(figsize=(4 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1.8], hspace=0.5)
    axes      = [fig.add_subplot(gs[0, i]) for i in range(n)]
    axes_desc = [fig.add_subplot(gs[1, i]) for i in range(n)]

    colors = ["#4C72B0", "#55A868"]

    for ax, ax_desc, metric in zip(axes, axes_desc, metrics):
        data = [mt5_scores[metric], claude_scores[metric]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric, fontsize=11, pad=6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["mT5", "Claude"], fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        info = BOXPLOT_METRIC_INFO[metric]
        desc_text = f"Description:\n{info['description']}\n\nJustification:\n{info['justification']}"
        ax_desc.axis("off")
        ax_desc.text(
            0.5, 0.95, desc_text,
            transform=ax_desc.transAxes,
            fontsize=7.5, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BBBBBB"),
        )

    fig.suptitle("Per-Entry Score Distribution — mT5 vs Claude", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fathir",        default="results.json")
    parser.add_argument("--bryan",         default="results-bryan.json")
    parser.add_argument("--claude-fathir", default="results-claude.json")
    parser.add_argument("--claude-bryan",  default="results-claude-bryan.json")
    parser.add_argument("--out",           default=".")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fathir_data = load(args.fathir)
    bryan_data  = load(args.bryan)

    # --- Annotator comparison (mT5 only) ---
    plot_averages(
        averages(fathir_data),
        averages(bryan_data),
        out_dir / "comparison_averages.png",
    )
    plot_boxplots(
        per_entry_scores(fathir_data),
        per_entry_scores(bryan_data),
        out_dir / "comparison_boxplot.png",
    )

    # --- Model comparison (mT5 vs Claude) ---
    cf_path = Path(args.claude_fathir)
    cb_path = Path(args.claude_bryan)

    if not cf_path.exists() or not cb_path.exists():
        missing = [str(p) for p in [cf_path, cb_path] if not p.exists()]
        print(f"\nSkipping model comparison charts — missing: {', '.join(missing)}")
        print("Run the Claude pipeline first, then re-run visualize.py.")
    else:
        claude_fathir_data = load(args.claude_fathir)
        claude_bryan_data  = load(args.claude_bryan)

        mt5_avg    = avg_dicts(averages(fathir_data),        averages(bryan_data))
        claude_avg = avg_dicts(averages(claude_fathir_data), averages(claude_bryan_data))

        mt5_scores    = avg_lists(per_entry_scores(fathir_data),        per_entry_scores(bryan_data))
        claude_scores = avg_lists(per_entry_scores(claude_fathir_data), per_entry_scores(claude_bryan_data))

        plot_model_comparison(mt5_avg, claude_avg, out_dir / "model_comparison_averages.png")
        plot_model_boxplots(mt5_scores, claude_scores, out_dir / "model_comparison_boxplot.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
