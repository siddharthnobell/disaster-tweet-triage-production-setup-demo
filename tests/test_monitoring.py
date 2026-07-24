import json

import numpy as np
import pandas as pd

from src.monitoring import (
    MIN_BATCH_SIZE,
    check_positive_rate_drift,
    detect_feature_drift,
)


def _reference_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "word_count": rng.integers(5, 25, size=n),
        "hashtag_count": rng.integers(0, 3, size=n),
        "mention_count": rng.integers(0, 3, size=n),
        "has_url": rng.integers(0, 2, size=n),
        "has_keyword": np.ones(n, dtype=int),
    })


def test_no_drift_when_current_matches_reference_distribution():
    reference = _reference_df(seed=1)
    current = _reference_df(seed=2)
    report = detect_feature_drift(reference, current)
    assert not report.any_drift


def test_numeric_drift_is_flagged_for_shifted_distribution():
    reference = _reference_df(seed=1)
    shifted = reference.copy()
    shifted["word_count"] = shifted["word_count"] + 100
    report = detect_feature_drift(reference, shifted)
    assert report.feature_flags["word_count"]["flagged"] is True
    assert report.any_drift


def test_rate_feature_drift_is_flagged_beyond_threshold():
    reference = _reference_df(seed=1)
    current = reference.copy()
    current["has_url"] = 0
    report = detect_feature_drift(reference, current)
    assert report.feature_flags["has_url"]["flagged"] is True


def test_insufficient_sample_flag():
    reference = _reference_df(seed=1)
    small_current = _reference_df(n=MIN_BATCH_SIZE - 1, seed=2)
    report = detect_feature_drift(reference, small_current)
    assert report.insufficient_sample is True

    large_current = _reference_df(n=MIN_BATCH_SIZE, seed=2)
    report2 = detect_feature_drift(reference, large_current)
    assert report2.insufficient_sample is False


def test_positive_rate_drift_flagging():
    reference = _reference_df(seed=1)
    report = detect_feature_drift(reference, reference.copy())

    not_flagged = check_positive_rate_drift(report, reference_positive_rate=0.43, current_predicted_positive_rate=0.45)
    assert not_flagged.positive_rate_flagged is False

    flagged = check_positive_rate_drift(report, reference_positive_rate=0.43, current_predicted_positive_rate=0.90)
    assert flagged.positive_rate_flagged is True


def test_drift_report_is_json_serializable():
    reference = _reference_df(seed=1)
    current = _reference_df(seed=2)
    report = detect_feature_drift(reference, current)
    report = check_positive_rate_drift(report, 0.4, 0.6)
    # regression test: numpy bool_/float64 leaking into the report used to
    # break json.dumps with "Object of type bool is not JSON serializable"
    json.dumps(report.to_dict())
