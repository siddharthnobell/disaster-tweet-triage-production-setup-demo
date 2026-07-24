"""Check a batch of tweets (raw, or already scored by scripts/ingest.py)
for feature drift against the training distribution, and for
predicted-positive-rate drift if the batch has already been scored.

Exits with status 1 if drift is flagged, so this can be wired into a cron
job or CI check that alerts someone or blocks auto-promotion.

Run from the repo root:
    python -m scripts.monitor_batch data/triaged/some_batch.csv
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.feature_pipeline import build_features
from src.monitoring import DriftReport, check_positive_rate_drift, detect_feature_drift

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "splits" / "train.csv"
REPORT_DIR = REPO_ROOT / "monitoring" / "reports"


def _print_report(report: DriftReport) -> None:
    print(f"reference n={report.n_reference}, current n={report.n_current}")
    if report.insufficient_sample:
        print(
            f"WARNING: batch has fewer than {30} rows - "
            "drift flags below are low-confidence, treat as informational only"
        )

    for feature, result in report.feature_flags.items():
        flag = "DRIFT" if result["flagged"] else "ok"
        print(f"  {feature:<15}[{flag:5}] {result}")

    if report.positive_rate_reference is not None:
        flag = "DRIFT" if report.positive_rate_flagged else "ok"
        print(
            f"  {'positive_rate':<15}[{flag:5}] reference={report.positive_rate_reference:.3f} "
            f"current={report.positive_rate_current:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_path", type=Path, help="CSV of tweets, e.g. from data/triaged/")
    args = parser.parse_args()

    reference = pd.read_csv(TRAIN_SPLIT_PATH)
    current = pd.read_csv(args.batch_path)

    if "word_count" not in current.columns:
        current = build_features(current)

    report = detect_feature_drift(reference, current)

    if "label" in current.columns:
        report = check_positive_rate_drift(
            report,
            reference_positive_rate=reference["target"].mean(),
            current_predicted_positive_rate=current["label"].mean(),
        )

    _print_report(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"drift_{timestamp}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nsaved report -> {report_path}")

    if report.any_drift:
        print("\nVERDICT: drift detected")
        sys.exit(1)
    print("\nVERDICT: no drift detected")


if __name__ == "__main__":
    main()
