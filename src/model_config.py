"""Single source of truth for the classifier used by every representation.

Kept identical across baseline and candidate on purpose: the Section B
comparison is only valid if the text representation is the one thing that
changes between runs.
"""
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def make_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
