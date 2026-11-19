"""Listing text that does not use the price.

Two representations:

* A fixed lexicon. The phrases were written down from the way hosts
  describe a place (a view, a renovation, a walk downtown). They are not
  a vocabulary that was searched on a test fold.
* A TF-IDF + SVD compression fit inside the training fold only. The
  held-out listings do not get to add words to the vocabulary. Eight
  components is the protocol's number, not a tuned one. Columns are padded
  to that width so every fold has the same features.

Photos are not read. The picture URL points at a host's image, and that
image is not part of the CC BY listing table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Word boundaries keep "review" from counting as "view" and "baby" from
# counting as "bay". "new" alone is too common to mean a new building.
LEXICON_PATTERNS = {
    "text_mentions_view": r"\bviews?\b|\bskyline\b",
    "text_mentions_luxury": r"\bluxur(?:y|ious)\b|\bupscale\b|\bpremium\b",
    "text_mentions_renovated": r"\brenovat|\bremodel|\bupdated\b",
    "text_mentions_new": r"\bbrand new\b|\bnewly built\b|\bnew construction\b",
    "text_mentions_quiet": r"\bquiet\b|\bpeaceful\b|\btranquil\b",
    "text_mentions_walk": r"\bwalkable\b|\bwalking distance\b|\bsteps away\b|\bwalk to\b",
    "text_mentions_downtown": r"\bdowntown\b|\bcity cent(?:er|re)\b",
    "text_mentions_water": r"\bbeach(?:es)?\b|\bocean\b|\bwaterfront\b|\blakefront\b|\bbay\b",
}

# A pasted essay should not own the vocabulary. 5,000 characters is far
# past the median description in these files (about 460).
TEXT_CHAR_CAP = 5000


def clean_texts(texts) -> list[str]:
    """Strip HTML and collapse whitespace. Does not look at the price."""
    series = pd.Series(texts, dtype="string").fillna("")
    cleaned = series.str.replace(r"<[^>]+>", " ", regex=True)
    cleaned = cleaned.str.replace(r"&\w+;", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return [text[:TEXT_CHAR_CAP] for text in cleaned.tolist()]


def compose_listing_text(df: pd.DataFrame) -> pd.Series:
    """Title, description, and the neighbourhood blurb. Not the host bio.

    The host bio describes the person, and it repeats across every listing
    a host runs. The ad copy describes the place being priced.
    """
    pieces = []
    for column in ("name", "description", "neighborhood_overview"):
        if column in df.columns:
            pieces.append(df[column].fillna("").astype(str))
        else:
            pieces.append(pd.Series("", index=df.index))
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined.str.cat(piece, sep=" ")
    return pd.Series(clean_texts(combined), index=df.index)


def lexicon_frame(texts) -> pd.DataFrame:
    """One row of lexicon features per text. An empty ad is missing."""
    cleaned = pd.Series(clean_texts(texts))
    lower = cleaned.str.lower()
    data: dict[str, np.ndarray] = {
        "log_text_chars": np.log1p(lower.str.len().to_numpy(dtype=float))
    }
    for name, pattern in LEXICON_PATTERNS.items():
        data[name] = lower.str.contains(pattern, regex=True, na=False).astype(float).to_numpy()
    frame = pd.DataFrame(data)
    # Empty copy is unknown. A real ad that simply never says "luxury"
    # is a zero, and that zero is allowed to be informative.
    empty = lower.str.strip().eq("")
    frame.loc[empty.to_numpy(), :] = np.nan
    return frame


class FoldText:
    """TF-IDF and SVD fit on one training fold.

    `transform` always returns `n_components` columns. A fold whose
    vocabulary is smaller pads the rest with zeros, so the matrix the
    trees see does not change width from fold to fold.
    """

    def __init__(self, n_components: int = 8, max_features: int = 4000, min_df: int = 5):
        self.n_components = int(n_components)
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.n_components_used = 0

    @property
    def column_names(self) -> list[str]:
        return [f"text_svd_{index:02d}" for index in range(self.n_components)]

    def _vectorizer(self, min_df: int) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float64,
        )

    def fit(self, texts) -> FoldText:
        cleaned = clean_texts(texts)
        self.vectorizer = self._vectorizer(self.min_df)
        matrix = self.vectorizer.fit_transform(cleaned)
        # A tiny fold (the smoke test) can have fewer than min_df repeats.
        if matrix.shape[1] < 2 and self.min_df > 1:
            self.vectorizer = self._vectorizer(1)
            matrix = self.vectorizer.fit_transform(cleaned)
        n_features = int(matrix.shape[1])
        n_docs = int(matrix.shape[0])
        # arpack needs a component count strictly below both dimensions.
        n_comp = min(self.n_components, n_features - 1, n_docs - 1)
        if n_comp < 1:
            self.svd = None
            self.n_components_used = 0
            return self
        self.n_components_used = int(n_comp)
        self.svd = TruncatedSVD(n_components=n_comp, algorithm="arpack", random_state=0)
        self.svd.fit(matrix)
        return self

    def transform(self, texts) -> np.ndarray:
        out = np.zeros((len(list(texts)), self.n_components), dtype=float)
        if self.vectorizer is None or self.svd is None:
            return out
        reduced = self.svd.transform(self.vectorizer.transform(clean_texts(texts)))
        out[:, : reduced.shape[1]] = reduced
        return out
