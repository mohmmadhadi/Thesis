"""Feature extraction helpers from notebook 06 user clustering."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd


EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE_TWEETS = 20
HDBSCAN_MIN_SAMPLES_TWEETS = 3


class FeatureExtractor:
    """Extract deterministic text features used by user clustering."""

    def __init__(
        self,
        stop_words: set[str] | None = None,
        lemmatizer: Any | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        readability_func: Callable[[str], float] | None = None,
    ) -> None:
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.readability_func = readability_func

    def _ensure_text_tools(self) -> None:
        if self.stop_words is None:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words("english"))
        if self.lemmatizer is None:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()
        if self.tokenizer is None:
            from nltk.tokenize import word_tokenize

            self.tokenizer = word_tokenize

    def _ensure_readability(self) -> None:
        if self.readability_func is None:
            from textstat import flesch_reading_ease

            self.readability_func = flesch_reading_ease

    @staticmethod
    def clean_tweet(text: str) -> str:
        """Strip URLs, mentions, hashtags, punctuation; lowercase and collapse whitespace."""
        cleaned = re.sub(r"http\S+|www\S+", "", str(text))
        cleaned = re.sub(r"@\w+", "", cleaned)
        cleaned = re.sub(r"#\w+", "", cleaned)
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def preprocess_for_lexical_features(self, text: str) -> str:
        """Lowercase, strip punctuation, remove stopwords, and lemmatise."""
        if not isinstance(text, str):
            return ""
        self._ensure_text_tools()
        cleaned = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower().strip())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in self.tokenizer(cleaned)
            if token not in self.stop_words
        ]
        return " ".join(tokens)

    @staticmethod
    def mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
        """Compute Measure of Textual Lexical Diversity."""

        def _one_pass(toks: list[str]) -> float:
            factors, types, n = 0, set(), 0
            for token in toks:
                n += 1
                types.add(token)
                if len(types) / n <= ttr_threshold:
                    factors += 1
                    types, n = set(), 0
            if n:
                factors += (1 - len(types) / n) / (1 - ttr_threshold)
            return len(toks) / factors if factors else 0

        if len(tokens) < 2:
            return 0.0
        return (_one_pass(tokens) + _one_pass(tokens[::-1])) / 2

    @staticmethod
    def shannon_entropy(tokens: list[str]) -> float:
        """Compute Shannon entropy over token frequencies."""
        counts = Counter(tokens)
        total = len(tokens)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def lexical_features(self, text: str) -> dict[str, float | int | None]:
        """Return MTLD, Shannon entropy, and token count for a text string."""
        self._ensure_text_tools()
        tokens = self.tokenizer(text)
        if len(tokens) < 5:
            return {"mtld": None, "entropy": None, "token_count": len(tokens)}
        return {
            "mtld": self.mtld(tokens),
            "entropy": self.shannon_entropy(tokens),
            "token_count": len(tokens),
        }

    def liwc_features(self, text: str) -> pd.Series:
        """Return the notebook's approximate LIWC-style features."""
        index = [
            "pronoun_ratio",
            "avg_sent_length",
            "readability_score",
            "exclamation_freq",
            "ellipsis_freq",
        ]
        if not isinstance(text, str) or not text.strip():
            return pd.Series([0.0] * 5, index=index)

        self._ensure_text_tools()
        self._ensure_readability()
        tokens = self.tokenizer(text.lower())
        total = len(tokens) or 1
        sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]

        i_pron = sum(1 for token in tokens if token in {"i", "me", "my", "mine", "myself"})
        we_pron = sum(
            1
            for token in tokens
            if token in {"we", "us", "our", "ours", "you", "your", "yours"}
        )
        return pd.Series(
            {
                "pronoun_ratio": i_pron / (we_pron + 1e-5),
                "avg_sent_length": total / max(len(sentences), 1),
                "readability_score": self.readability_func(text),
                "exclamation_freq": text.count("!") / total,
                "ellipsis_freq": text.count("...") / total,
            }
        )

    @staticmethod
    def emoji_rate(text: str) -> float:
        """Return emoji matches per character."""
        length = len(str(text))
        return len(EMOJI_RE.findall(str(text))) / length if length else 0.0

    @staticmethod
    def punctuation_rate(text: str) -> float:
        """Return punctuation characters per character."""
        length = len(str(text))
        return (
            sum(1 for character in str(text) if character in string.punctuation) / length
            if length
            else 0.0
        )

    def transform(self, df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
        """Add reusable tweet-level feature columns without clustering."""
        result = df.copy()
        result["clean_tweet"] = result[text_col].apply(self.clean_tweet)
        result["cleaned_tweet"] = result[text_col].apply(self.preprocess_for_lexical_features)
        self._ensure_text_tools()
        result["tokens"] = result["cleaned_tweet"].apply(self.tokenizer)
        lexical = result[text_col].apply(self.lexical_features).apply(pd.Series)
        liwc = result[text_col].apply(self.liwc_features)
        result = pd.concat([result, lexical, liwc], axis=1)
        result["emoji_rate"] = result[text_col].apply(self.emoji_rate)
        result["punctuation_rate"] = result[text_col].apply(self.punctuation_rate)
        result["length"] = result[text_col].apply(len)
        return result

    def aggregate_user_rates(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
    ) -> pd.DataFrame:
        """Aggregate notebook emoji and punctuation rates per user."""
        return (
            df.groupby(user_col)
            .agg(
                mean_emoji_rate=("emoji_rate", "mean"),
                mean_punct_rate=("punctuation_rate", "mean"),
            )
            .fillna(0)
            .reset_index()
        )


class TweetClusterer:
    """Run notebook-style tweet embedding, UMAP reduction, and HDBSCAN clustering."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        umap_n_neighbors: int = UMAP_N_NEIGHBORS,
        umap_min_dist: float = UMAP_MIN_DIST,
        hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE_TWEETS,
        hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES_TWEETS,
        embedding_model: Any | None = None,
        reducer: Any | None = None,
        clusterer: Any | None = None,
        vectorizer_cls: Any | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.embedding_model = embedding_model
        self.reducer = reducer
        self.clusterer = clusterer
        self.vectorizer_cls = vectorizer_cls
        self.embeddings: np.ndarray | None = None
        self.embeddings_2d: np.ndarray | None = None
        self.cluster_labels_: dict[int, str] | None = None
        self.n_clusters_: int | None = None
        self.n_noise_: int | None = None

    def _load_embedding_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_model_name)

    def _build_reducer(self) -> Any:
        import umap

        return umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=2,
            min_dist=self.umap_min_dist,
            metric="cosine",
            random_state=UMAP_RANDOM_STATE,
        )

    def _build_clusterer(self) -> Any:
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )

    def prepare_tweets(self, df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
        """Add clean_tweet and drop rows empty after notebook cleaning."""
        result = df.copy()
        result["clean_tweet"] = result[text_col].apply(FeatureExtractor.clean_tweet)
        return result[result["clean_tweet"].str.strip() != ""].reset_index(drop=True)

    def compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode cleaned tweets with normalized sentence embeddings."""
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings)
        return self.embeddings

    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 2D with notebook UMAP settings."""
        if self.reducer is None:
            self.reducer = self._build_reducer()
        reduced = self.reducer.fit_transform(embeddings)
        self.embeddings_2d = np.asarray(reduced)
        return self.embeddings_2d

    def cluster_embeddings(self, embeddings_2d: np.ndarray) -> np.ndarray:
        """Cluster 2D embeddings with notebook HDBSCAN settings."""
        if self.clusterer is None:
            self.clusterer = self._build_clusterer()
        return np.asarray(self.clusterer.fit_predict(embeddings_2d))

    def build_cluster_labels(
        self,
        df: pd.DataFrame,
        cluster_col: str = "cluster",
        text_col: str = "clean_tweet",
    ) -> dict[int, str]:
        """Build TF-IDF labels using lowest-IDF words per cluster."""
        if self.vectorizer_cls is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer_cls = TfidfVectorizer

        labels: dict[int, str] = {}
        for cluster_id in sorted(df[cluster_col].unique()):
            if cluster_id == -1:
                labels[int(cluster_id)] = "Noise"
                continue
            cluster_tweets = df[df[cluster_col] == cluster_id][text_col].tolist()
            tfidf = self.vectorizer_cls(max_features=200, stop_words="english")
            tfidf.fit(cluster_tweets)
            scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
            top_words = sorted(scores, key=scores.get)[:4]
            labels[int(cluster_id)] = f"Cluster {cluster_id}: {', '.join(top_words)}"
        self.cluster_labels_ = labels
        return labels

    def fit_transform(self, df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
        """Return tweets with clean text, UMAP coordinates, clusters, and labels."""
        result = self.prepare_tweets(df, text_col=text_col)
        embeddings = self.compute_embeddings(result["clean_tweet"].tolist())
        embeddings_2d = self.reduce_embeddings(embeddings)
        result["x"] = embeddings_2d[:, 0]
        result["y"] = embeddings_2d[:, 1]
        result["cluster"] = self.cluster_embeddings(embeddings_2d)

        self.n_clusters_ = result["cluster"].nunique() - (
            1 if -1 in result["cluster"].values else 0
        )
        self.n_noise_ = int((result["cluster"] == -1).sum())

        labels = self.build_cluster_labels(result)
        result["cluster_label"] = result["cluster"].map(labels)
        return result
