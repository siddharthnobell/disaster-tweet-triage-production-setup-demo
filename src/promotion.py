"""Shared promotion rule: is a new model good enough to replace the current
one? Used by scripts/compare_models.py (baseline vs. candidate, Section B)
and scripts/retrain.py (retrained vs. currently-deployed, Section D), so
"should we ship this" is decided identically everywhere.

Primary metric = test F1, matching the Kaggle competition's own scoring
metric and giving a balanced view for a binary triage task. The margin
exists because these test sets are small (~1,100 rows): a marginal win
could just be noise, so the bar is intentionally not "any improvement".
"""
F1_PROMOTION_MARGIN = 0.01


def should_promote(new_f1: float, current_f1: float, margin: float = F1_PROMOTION_MARGIN) -> bool:
    return (new_f1 - current_f1) >= margin
