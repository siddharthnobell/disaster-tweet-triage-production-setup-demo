import pandas as pd
import pytest

from src.feature_pipeline import (
    FEATURE_COLUMNS,
    build_features,
    clean_and_featurize,
    clean_text,
    dedupe_labels,
    has_keyword,
    has_url,
    hashtag_count,
    mention_count,
    word_count,
)


# --- clean_text -------------------------------------------------------

def test_clean_text_strips_urls():
    assert clean_text("Fire downtown http://t.co/abc123 stay safe") == "Fire downtown stay safe"


def test_clean_text_unescapes_html_entities():
    assert clean_text("Rock &amp; roll") == "Rock & roll"


def test_clean_text_fixes_known_mojibake():
    assert clean_text("can\x89\xdb\xaat believe it") == "can't believe it"


def test_clean_text_strips_leftover_mojibake_bytes():
    cleaned = clean_text("weird\xe5\xa3byte junk")
    assert all(ord(c) < 128 for c in cleaned)


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many\n\nspaces") == "too many spaces"


def test_clean_text_preserves_casing_and_punctuation():
    assert clean_text("This is BAD!!!") == "This is BAD!!!"


def test_clean_text_non_string_input_returns_empty():
    assert clean_text(float("nan")) == ""


# --- individual feature functions -------------------------------------

def test_word_count():
    assert word_count("this has four words") == 4
    assert word_count("") == 0
    assert word_count(float("nan")) == 0


def test_has_url():
    assert has_url("check this http://t.co/abc") == 1
    assert has_url("check this www.example.com") == 1
    assert has_url("no link here") == 0
    assert has_url(float("nan")) == 0


def test_hashtag_count():
    assert hashtag_count("#fire #smoke near me") == 2
    assert hashtag_count("no hashtags") == 0
    assert hashtag_count("price is $5, not a #tag#broken") == 1


def test_mention_count():
    assert mention_count("cc @alice @bob") == 2
    assert mention_count("no mentions") == 0
    assert mention_count("email me at foo@bar.com") == 0


def test_has_keyword():
    assert has_keyword("earthquake") == 1
    assert has_keyword("") == 0
    assert has_keyword(float("nan")) == 0


# --- build_features / dedupe_labels / clean_and_featurize -------------

def _toy_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "keyword": ["fire", None, ""],
        "location": ["NYC", None, "LA"],
        "text": [
            "Fire near #downtown http://t.co/x cc @alice",
            "just chilling today",
            "",
        ],
        "target": [1, 0, 0],
    })


def test_build_features_adds_all_expected_columns():
    out = build_features(_toy_df())
    for col in ["cleaned_text"] + FEATURE_COLUMNS:
        assert col in out.columns
    assert list(out["word_count"]) == [6, 3, 0]
    assert list(out["has_url"]) == [1, 0, 0]
    assert list(out["hashtag_count"]) == [1, 0, 0]
    assert list(out["mention_count"]) == [1, 0, 0]
    assert list(out["has_keyword"]) == [1, 0, 0]


def test_build_features_preserves_row_count_and_original_columns():
    df = _toy_df()
    out = build_features(df)
    assert len(out) == len(df)
    for col in df.columns:
        assert col in out.columns


def test_build_features_works_without_target_column():
    df = _toy_df().drop(columns=["target"])
    out = build_features(df)
    assert "target" not in out.columns
    assert len(out) == len(df)


def test_dedupe_labels_majority_vote_resolves_conflict():
    df = pd.DataFrame({
        "text": ["same tweet", "same tweet", "same tweet", "unique"],
        "target": [1, 1, 0, 0],
    })
    out = dedupe_labels(df)
    assert len(out) == 2
    row = out[out["text"] == "same tweet"].iloc[0]
    assert row["target"] == 1


def test_dedupe_labels_drops_exact_ties():
    df = pd.DataFrame({
        "text": ["tied tweet", "tied tweet", "unique"],
        "target": [1, 0, 0],
    })
    out = dedupe_labels(df)
    assert list(out["text"]) == ["unique"]


def test_dedupe_labels_no_duplicates_is_noop():
    df = pd.DataFrame({"text": ["a", "b", "c"], "target": [1, 0, 1]})
    out = dedupe_labels(df)
    assert len(out) == 3
    assert set(out["target"]) == {0, 1}


def test_clean_and_featurize_full_pipeline():
    df = pd.DataFrame({
        "keyword": ["fire", "fire", None],
        "text": ["duplicate text", "duplicate text", "other text"],
        "target": [1, 1, 0],
    })
    out = clean_and_featurize(df)
    assert len(out) == 2
    for col in ["cleaned_text"] + FEATURE_COLUMNS:
        assert col in out.columns


def test_real_dataset_no_missing_features():
    pytest.importorskip("pandas")
    from pathlib import Path

    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"
    if not raw_path.exists():
        pytest.skip("raw dataset not present")
    df = pd.read_csv(raw_path)
    out = clean_and_featurize(df)
    assert out[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert out["cleaned_text"].apply(lambda t: any(ord(c) > 127 for c in t)).sum() == 0
