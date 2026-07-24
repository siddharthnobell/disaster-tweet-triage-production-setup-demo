"""Stratified train/val/test split, shared by both models in the Section B
comparison so that baseline and candidate are scored on identical data.

Split once here and persist to disk (rather than re-splitting inside each
training script) so the split can't silently drift between runs.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42


def make_splits(
    df: pd.DataFrame,
    target_col: str = "target",
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into stratified train/val/test sets.

    Stratifies on `target_col` so the ~57/43 class balance is preserved in
    every split, which keeps val/test metrics comparable to each other and
    to train.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df[target_col],
        random_state=random_state,
    )
    relative_val_fraction = val_fraction / (1 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_fraction,
        stratify=train_val[target_col],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
