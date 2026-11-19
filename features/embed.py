"""Sentence embeddings of the listing's own ad, compressed inside the fold.

v3 tried a fixed lexicon and an eight-component TF-IDF. Both missed the
bar. This is the same eight columns, from a small static embedding model
instead of word counts, so the comparison is the representation and not
the width. The embedding model is frozen. Only the PCA is fit on the
training fold, which keeps a held-out ad from defining the axes.

An empty ad is missing, not a vector of zeros. Zeros would say the host
wrote a blank listing on purpose.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from evaluation.protocol import EMBED_COMPONENTS, EMBED_MODEL
from sklearn.decomposition import PCA

from features.columns import embed_component_names
from features.text import clean_texts


@lru_cache(maxsize=1)
def _static_model(name: str):
    """The weights are downloaded once per process. They are not trained here."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained(name)


class FoldEmbed:
    """PCA of a frozen sentence embedding. `transform` always has `n_components` columns."""

    def __init__(
        self,
        n_components: int = EMBED_COMPONENTS,
        model_name: str = EMBED_MODEL,
        encoder=None,
    ):
        self.n_components = int(n_components)
        self.model_name = model_name
        # A test passes a callable so it does not download a model.
        # It is not part of the saved artifact's contract; the name is.
        self._encoder = encoder
        self._pca: PCA | None = None
        self.n_components_used = 0

    @property
    def column_names(self) -> list[str]:
        return embed_component_names(self.n_components)

    def _vectors(self, texts) -> np.ndarray:
        cleaned = clean_texts(texts)
        if self._encoder is not None:
            vectors = np.asarray(self._encoder(cleaned), dtype=float)
        else:
            vectors = np.asarray(_static_model(self.model_name).encode(cleaned), dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors, cleaned

    def fit(self, texts) -> FoldEmbed:
        vectors, cleaned = self._vectors(texts)
        empty = np.array([text.strip() == "" for text in cleaned])
        usable = vectors[~empty] if empty.any() else vectors
        n_docs, n_dim = usable.shape if len(usable) else (0, 0)
        n_comp = min(self.n_components, n_dim, max(n_docs - 1, 0))
        if n_comp < 1:
            self._pca = None
            self.n_components_used = 0
            return self
        self.n_components_used = int(n_comp)
        self._pca = PCA(n_components=n_comp, random_state=0)
        self._pca.fit(usable)
        return self

    def transform(self, texts) -> np.ndarray:
        cleaned_len = len(list(texts))
        out = np.full((cleaned_len, self.n_components), np.nan, dtype=float)
        if self._pca is None:
            return out
        vectors, cleaned = self._vectors(texts)
        empty = np.array([text.strip() == "" for text in cleaned])
        if (~empty).any():
            reduced = self._pca.transform(vectors[~empty])
            block = np.full((int((~empty).sum()), self.n_components), np.nan)
            block[:, : reduced.shape[1]] = reduced
            out[~empty] = block
        return out
