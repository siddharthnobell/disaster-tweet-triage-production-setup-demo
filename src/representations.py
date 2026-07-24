"""Text representation builders for the Section B model comparison.

The whole point of this comparison is to isolate ONE variable - how the
tweet text is turned into a feature vector - while holding everything else
fixed: the same 5 handcrafted features (src.feature_pipeline) are
concatenated onto both representations, and both feed the same classifier
type (see scripts/train_baseline.py / train_candidate.py).

Both classes expose fit/transform so they can be fit on the training split
only and then reused, unchanged, on val/test/inference input - the same
train-serve consistency goal as feature_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.feature_pipeline import FEATURE_COLUMNS

TEXT_COLUMN = "cleaned_text"


class BaselineRepresentation:
    """TF-IDF over cleaned_text, concatenated with the 5 handcrafted features."""

    name = "baseline_tfidf"

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=min_df
        )
        self.scaler = StandardScaler()

    def fit(self, df) -> "BaselineRepresentation":
        self.vectorizer.fit(df[TEXT_COLUMN])
        self.scaler.fit(df[FEATURE_COLUMNS])
        return self

    def transform(self, df):
        text_matrix = self.vectorizer.transform(df[TEXT_COLUMN])
        handcrafted = sp.csr_matrix(self.scaler.transform(df[FEATURE_COLUMNS]))
        return sp.hstack([text_matrix, handcrafted]).tocsr()

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class CandidateRepresentation:
    """Frozen MiniLM sentence embeddings, concatenated with handcrafted features.

    The encoder is pretrained and not fine-tuned here (only the downstream
    LogisticRegression is trained) - this is the standard cheap way to try
    an embedding model before committing to fine-tuning it.
    """

    name = "candidate_minilm"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME)
        self.scaler = StandardScaler()

    def fit(self, df) -> "CandidateRepresentation":
        self.scaler.fit(df[FEATURE_COLUMNS])
        return self

    def _embed(self, texts) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )

    def transform(self, df):
        embeddings = self._embed(df[TEXT_COLUMN])
        handcrafted = self.scaler.transform(df[FEATURE_COLUMNS])
        return np.hstack([embeddings, handcrafted])

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def __getstate__(self):
        # SentenceTransformer reloads cheaply from its model name; avoid
        # pickling the full torch module graph into our joblib artifact.
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        from sentence_transformers import SentenceTransformer

        self.__dict__.update(state)
        self.model = SentenceTransformer(self.MODEL_NAME)
