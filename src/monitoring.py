"""Data-drift detection for live traffic, using the 5 handcrafted features
from feature_pipeline.py as the drift signal.

These are used (rather than the raw text itself) because they're cheap to
compute and summarize, and don't need ground-truth labels - unlike model
*performance* metrics (precision/recall/F1), which need someone to review
and label live tweets before they can be computed at all. This only tells
us the incoming tweet stream looks statistically different from training;
it does not tell us whether the model is still accurate on it.

We also track the model's *predicted* positive rate against the training
set's true positive rate as a cheap proxy for concept drift: a big swing
either means the model is behaving differently on this traffic, or the
traffic itself has genuinely shifted (e.g. an actual disaster surge) - both
are worth a human look.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd
from scipy.stats import ks_2samp

NUMERIC_FEATURES = ["word_count", "hashtag_count", "mention_count"]
RATE_FEATURES = ["has_url", "has_keyword"]

KS_P_VALUE_THRESHOLD = 0.05
RATE_DIFF_THRESHOLD = 0.15
POSITIVE_RATE_DIFF_THRESHOLD = 0.15
MIN_BATCH_SIZE = 30


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    insufficient_sample: bool
    feature_flags: dict = field(default_factory=dict)
    positive_rate_reference: Optional[float] = None
    positive_rate_current: Optional[float] = None
    positive_rate_flagged: bool = False

    @property
    def any_drift(self) -> bool:
        return any(f["flagged"] for f in self.feature_flags.values()) or self.positive_rate_flagged

    def to_dict(self) -> dict:
        return asdict(self)


def detect_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
    """Compare `current` (a batch of scored/featurized tweets) against
    `reference` (the training split) feature-by-feature.

    Numeric features use a two-sample KS test (distribution shape);
    binary/rate features use a simple absolute rate-difference threshold,
    since a KS test on a 0/1 column is just a proportion test in disguise.
    """
    report = DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        insufficient_sample=len(current) < MIN_BATCH_SIZE,
    )

    for col in NUMERIC_FEATURES:
        stat, p_value = ks_2samp(reference[col], current[col])
        report.feature_flags[col] = {
            "test": "ks_2samp",
            "statistic": float(stat),
            "p_value": float(p_value),
            "flagged": bool(p_value < KS_P_VALUE_THRESHOLD),
        }

    for col in RATE_FEATURES:
        ref_rate = float(reference[col].mean())
        cur_rate = float(current[col].mean())
        diff = cur_rate - ref_rate
        report.feature_flags[col] = {
            "test": "rate_diff",
            "reference_rate": ref_rate,
            "current_rate": cur_rate,
            "diff": diff,
            "flagged": bool(abs(diff) > RATE_DIFF_THRESHOLD),
        }

    return report


def check_positive_rate_drift(
    report: DriftReport, reference_positive_rate: float, current_predicted_positive_rate: float
) -> DriftReport:
    report.positive_rate_reference = float(reference_positive_rate)
    report.positive_rate_current = float(current_predicted_positive_rate)
    report.positive_rate_flagged = bool(
        abs(report.positive_rate_current - report.positive_rate_reference) > POSITIVE_RATE_DIFF_THRESHOLD
    )
    return report
