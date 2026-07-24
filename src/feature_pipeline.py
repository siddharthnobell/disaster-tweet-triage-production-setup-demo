"""Shared data cleaning + feature construction for disaster tweet triage.

This module is imported by both the training pipeline and the inference
service so that train-time and serve-time feature computation can never
drift apart (no re-implementing cleaning logic twice).

Two kinds of output are produced per row:
  - `cleaned_text`: normalized tweet text, handed to the text representation
    stage (TF-IDF baseline or sentence-embedding candidate).
  - 5 handcrafted metadata features (see FEATURE_COLUMNS) that are computed
    from the *raw* text/keyword fields and concatenated alongside whichever
    text representation is in use.
"""
from __future__ import annotations

import html
import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")

# This dataset was exported through at least one lossy encoding hop (looks
# like UTF-8 bytes re-decoded as Windows-1252), so smart quotes/apostrophes
# show up as byte-garbage instead of real characters. Found by scanning
# data/raw/train.csv for non-ASCII runs: ~700 of 7,613 rows are affected,
# concentrated in ~8 recurring sequences that map back to ordinary
# punctuation. The table below restores those; anything left over (long
# tail of one-off sequences, ~1% of rows) is dropped by a catch-all strip
# in clean_text rather than guessed at.
_MOJIBAKE_REPLACEMENTS = {
    "\x89\xdb\xaa": "'",   # right single quote, e.g. "can\x89\xdb\xaat" -> "can't"
    "\x89\xdb\xcf": '"',   # opening double quote
    "\x89\xdb\xf7": "'",   # right single quote (variant)
    "\x89\xdb\x9d": '"',   # closing double quote
    "\x89\xdb\xd2": '"',   # opening double quote (variant)
    "\x89\xdb\xd3": '"',   # closing double quote (variant)
    "\x89\xdb_": "...",    # truncation marker used by the scraper
    "\xe5\xca": "-",       # mis-decoded dash
}
_MOJIBAKE_CATCHALL = re.compile(r"[\x80-\xff]+")

FEATURE_COLUMNS = [
    "word_count",
    "has_url",
    "hashtag_count",
    "mention_count",
    "has_keyword",
]


def clean_text(text: object) -> str:
    """Normalize raw tweet text ahead of vectorization.

    Fixes HTML entities and known mojibake, strips URLs, and collapses
    whitespace. Deliberately keeps casing and punctuation intact: the
    embedding candidate model (a transformer) is sensitive to both, so
    lowercasing/stripping punctuation here would bias the comparison in
    favor of the TF-IDF baseline, which lowercases on its own anyway.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = html.unescape(cleaned)
    cleaned = _MOJIBAKE_CATCHALL.sub("", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def word_count(text: object) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def has_url(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return int(bool(URL_PATTERN.search(text)))


def hashtag_count(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return len(HASHTAG_PATTERN.findall(text))


def mention_count(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return len(MENTION_PATTERN.findall(text))


def has_keyword(keyword: object) -> int:
    return int(isinstance(keyword, str) and keyword.strip() != "")


def dedupe_labels(
    df: pd.DataFrame, text_col: str = "text", target_col: str = "target"
) -> pd.DataFrame:
    """Collapse rows sharing identical text but conflicting labels.

    This dataset has ~18 texts (out of 7,613 rows) that appear more than
    once with a different `target` each time - a labeling inconsistency in
    the source data, not a modeling choice we can fix. Rows are collapsed
    to one per unique text via majority vote; the rare exact tie (equal
    counts on both sides) is dropped rather than guessed at, since a
    coin-flip label would inject noise directly into train/val/test splits.
    """
    majority = df.groupby(text_col)[target_col].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) == 1 else pd.NA
    )
    resolved = df.drop_duplicates(subset=text_col, keep="first").copy()
    resolved[target_col] = resolved[text_col].map(majority)

    before = len(resolved)
    resolved = resolved.dropna(subset=[target_col])
    dropped = before - len(resolved)
    if dropped:
        print(f"dedupe_labels: dropped {dropped} texts with tied conflicting labels")
    resolved[target_col] = resolved[target_col].astype(int)
    return resolved.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `cleaned_text` and the 5 handcrafted features to a dataframe.

    Expects `text` and `keyword` columns; `target` is untouched if present
    and not required otherwise (this same function runs at inference time,
    where no label exists).
    """
    out = df.copy()
    out["cleaned_text"] = out["text"].apply(clean_text)
    out["word_count"] = out["text"].apply(word_count)
    out["has_url"] = out["text"].apply(has_url)
    out["hashtag_count"] = out["text"].apply(hashtag_count)
    out["mention_count"] = out["text"].apply(mention_count)
    out["has_keyword"] = out["keyword"].apply(has_keyword)
    return out


def clean_and_featurize(df: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """Full Section-A pipeline: optional label dedupe, then featurization.

    `dedupe` only applies when `target` is present (i.e. training data);
    it's a no-op at inference time.
    """
    working = df.copy()
    if dedupe and "target" in working.columns:
        working = dedupe_labels(working)
    return build_features(working)
