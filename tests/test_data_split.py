import pandas as pd

from src.data_split import make_splits


def _toy_df(n=200, positive_rate=0.4):
    n_pos = int(n * positive_rate)
    target = [1] * n_pos + [0] * (n - n_pos)
    return pd.DataFrame({
        "text": [f"tweet {i}" for i in range(n)],
        "target": target,
    })


def test_make_splits_sizes_roughly_match_fractions():
    df = _toy_df(n=1000)
    train, val, test = make_splits(df, val_fraction=0.15, test_fraction=0.15)
    assert len(train) + len(val) + len(test) == len(df)
    assert 690 <= len(train) <= 710
    assert 140 <= len(val) <= 160
    assert 140 <= len(test) <= 160


def test_make_splits_no_overlap_and_full_coverage():
    df = _toy_df(n=500)
    train, val, test = make_splits(df)
    all_text = set(train["text"]) | set(val["text"]) | set(test["text"])
    assert all_text == set(df["text"])
    assert len(set(train["text"]) & set(val["text"])) == 0
    assert len(set(train["text"]) & set(test["text"])) == 0
    assert len(set(val["text"]) & set(test["text"])) == 0


def test_make_splits_preserves_class_balance():
    df = _toy_df(n=1000, positive_rate=0.4)
    train, val, test = make_splits(df)
    for split in [train, val, test]:
        assert abs(split["target"].mean() - 0.4) < 0.03


def test_make_splits_is_deterministic():
    df = _toy_df(n=300)
    train1, val1, test1 = make_splits(df, random_state=42)
    train2, val2, test2 = make_splits(df, random_state=42)
    assert list(train1["text"]) == list(train2["text"])
    assert list(val1["text"]) == list(val2["text"])
    assert list(test1["text"]) == list(test2["text"])
