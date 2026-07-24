"""Compare the baseline (TF-IDF) and candidate (MiniLM) models trained by
train_baseline.py / train_candidate.py and state whether the candidate
should be promoted.

Promotion rule (primary metric = test F1, matching the Kaggle competition's
own scoring metric and giving a balanced view for a binary triage task):
the candidate is promoted only if it beats the baseline's test F1 by at
least F1_PROMOTION_MARGIN. A small, un-adjusted-for-significance sample
(1,124 test rows) means a marginal win could just be noise, so the bar is
intentionally not "any improvement at all".

Run from the repo root:
    python -m scripts.compare_models
"""
import json
import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.promotion import F1_PROMOTION_MARGIN, should_promote

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "metrics"

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main() -> None:
    baseline = json.loads((METRICS_DIR / "baseline.json").read_text())
    candidate = json.loads((METRICS_DIR / "candidate.json").read_text())

    for split in ["val", "test"]:
        print(f"\n{split} set:")
        header = f"{'metric':<10}{'baseline':>12}{'candidate':>12}{'delta':>12}"
        print(header)
        print("-" * len(header))
        for metric in METRIC_ORDER:
            b = baseline[split][metric]
            c = candidate[split][metric]
            print(f"{metric:<10}{b:>12.4f}{c:>12.4f}{c - b:>+12.4f}")

    baseline_f1 = baseline["test"]["f1"]
    candidate_f1 = candidate["test"]["f1"]
    delta = candidate_f1 - baseline_f1

    print(f"\ntest F1: baseline={baseline_f1:.4f} candidate={candidate_f1:.4f} "
          f"delta={delta:+.4f} (promotion margin={F1_PROMOTION_MARGIN})")

    if should_promote(candidate_f1, baseline_f1):
        verdict = "PROMOTE candidate"
    else:
        verdict = "KEEP baseline"
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
