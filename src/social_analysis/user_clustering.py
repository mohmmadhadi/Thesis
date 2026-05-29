"""Feature extraction helpers from notebook 06 user clustering."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
PCA_VARIANCE_THRESHOLD = 0.80
GMM_N_COMPONENTS = 4
HDBSCAN_MIN_CLUSTER_SIZE_AGENTS = 6
HDBSCAN_MIN_SAMPLES_AGENTS = 2
CLUSTER_FEATURES = [
    "mean_sent",
    "std_emo",
    "mtld",
    "pronoun_ratio",
    "mean_emoji_rate",
    "avg_reply_coherence",
    "has_parent",
]
PERSONALITY_TRAITS = ["ex", "oe", "co", "ag", "ne"]
DEMOGRAPHIC_TRAITS = ["gender", "leaning", "age", "education_level"]
USER_METADATA_COLUMNS = ["id"] + PERSONALITY_TRAITS + DEMOGRAPHIC_TRAITS


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

    @staticmethod
    def cluster_count_summary(
        tweets: pd.DataFrame,
        cluster_col: str = "cluster",
    ) -> dict[str, int]:
        """Return notebook tweet cluster and noise counts."""
        n_clusters = tweets[cluster_col].nunique() - (
            1 if -1 in tweets[cluster_col].values else 0
        )
        n_noise = int((tweets[cluster_col] == -1).sum())
        return {"n_clusters": int(n_clusters), "n_noise": n_noise}

    @staticmethod
    def cluster_trait_crosstab(
        tweets: pd.DataFrame,
        trait: str,
        cluster_label_col: str = "cluster_label",
        cluster_col: str = "cluster",
        include_noise: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Return notebook cluster-by-trait counts, row percentages, and labels."""
        if trait not in tweets.columns:
            return {}
        data = tweets.copy()
        if not include_noise:
            data = data[data[cluster_col] != -1].copy()

        counts = pd.crosstab(data[cluster_label_col], data[trait])
        row_percent = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        annotation = counts.copy().astype(str)
        for row_label in counts.index:
            total = counts.loc[row_label].sum()
            for column in counts.columns:
                value = counts.loc[row_label, column]
                annotation.loc[row_label, column] = (
                    f"{value}<br>({value / total:.0%})" if total > 0 else "0"
                )
        return {
            "counts": counts,
            "row_percent": row_percent,
            "annotation": annotation,
        }

    @staticmethod
    def add_tweet_short(
        tweets: pd.DataFrame,
        text_col: str = "tweet",
    ) -> pd.DataFrame:
        """Add notebook 90-character hover preview text."""
        result = tweets.copy()
        if text_col in result.columns:
            result["tweet_short"] = result[text_col].astype(str).str[:90] + "..."
        return result

    @staticmethod
    def trait_symbol_map(
        tweets: pd.DataFrame,
        trait: str,
    ) -> dict[Any, str]:
        """Return notebook Plotly symbol mapping for one trait."""
        default_symbols = [
            "circle",
            "x",
            "diamond",
            "cross",
            "square",
            "triangle-up",
            "triangle-down",
            "pentagon",
            "hexagon",
            "star",
        ]
        if trait not in tweets.columns:
            return {}
        unique_vals = sorted(tweets[trait].dropna().unique())
        return {
            value: default_symbols[index % len(default_symbols)]
            for index, value in enumerate(unique_vals)
        }

    @staticmethod
    def trait_scatter_filename(trait: str) -> str:
        """Return notebook trait-specific tweet cluster HTML filename."""
        return f"tweet_clusters_{trait}.html"

    @staticmethod
    def trait_heatmap_filename(trait: str) -> str:
        """Return notebook cluster-trait heatmap HTML filename."""
        return f"heatmap_cluster_{trait}.html"

    @staticmethod
    def cluster_trait_overlap_diagnostics(
        tweets: pd.DataFrame,
        traits: list[str] | None = None,
        cluster_col: str = "cluster",
        cluster_label_col: str = "cluster_label",
        include_noise: bool = False,
    ) -> pd.DataFrame:
        """Compute notebook chi-square, ARI, and NMI diagnostics per trait."""
        from scipy.stats import chi2_contingency
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        selected_traits = traits or PERSONALITY_TRAITS
        data = tweets.copy()
        if not include_noise:
            data = data[data[cluster_col] != -1].copy()

        rows = []
        for trait in selected_traits:
            if trait not in data.columns:
                rows.append(
                    {
                        "trait": trait,
                        "chi2": np.nan,
                        "p": np.nan,
                        "dof": np.nan,
                        "significance": "missing",
                        "ari": np.nan,
                        "nmi": np.nan,
                        "ari_nmi_status": "missing",
                        "n_values": 0,
                    }
                )
                continue

            trait_data = data[[cluster_label_col, cluster_col, trait]].dropna()
            unique_vals = trait_data[trait].unique()
            row = {
                "trait": trait,
                "chi2": np.nan,
                "p": np.nan,
                "dof": np.nan,
                "significance": "not computed",
                "ari": np.nan,
                "nmi": np.nan,
                "ari_nmi_status": (
                    f"skipped (trait has {len(unique_vals)} values, need 2)"
                ),
                "n_values": len(unique_vals),
            }

            crosstab = pd.crosstab(trait_data[cluster_label_col], trait_data[trait])
            if crosstab.shape[0] > 0 and crosstab.shape[1] > 0:
                chi2, p_value, dof, _ = chi2_contingency(crosstab)
                row.update(
                    {
                        "chi2": float(chi2),
                        "p": float(p_value),
                        "dof": int(dof),
                        "significance": (
                            "SIGNIFICANT (p < 0.05)"
                            if p_value < 0.05
                            else "not significant (p >= 0.05)"
                        ),
                    }
                )

            if len(unique_vals) == 2:
                label_map = {value: idx for idx, value in enumerate(unique_vals)}
                encoded = trait_data[trait].map(label_map)
                row["ari"] = float(adjusted_rand_score(encoded, trait_data[cluster_col]))
                row["nmi"] = float(
                    normalized_mutual_info_score(encoded, trait_data[cluster_col])
                )
                row["ari_nmi_status"] = "computed"
            rows.append(row)

        return pd.DataFrame(
            rows,
            columns=[
                "trait",
                "chi2",
                "p",
                "dof",
                "significance",
                "ari",
                "nmi",
                "ari_nmi_status",
                "n_values",
            ],
        )


class UserClusterer:
    """Build agent feature matrices and run notebook-style user clustering."""

    def __init__(
        self,
        pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD,
        gmm_n_components: int = GMM_N_COMPONENTS,
        hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE_AGENTS,
        hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES_AGENTS,
        cluster_features: list[str] | None = None,
        feature_extractor: FeatureExtractor | None = None,
        scaler: Any | None = None,
        pca: Any | None = None,
        pca_vis: Any | None = None,
        gmm: Any | None = None,
        hdbscan_clusterer: Any | None = None,
    ) -> None:
        self.pca_variance_threshold = pca_variance_threshold
        self.gmm_n_components = gmm_n_components
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.cluster_features = cluster_features or CLUSTER_FEATURES
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.scaler = scaler
        self.pca = pca
        self.pca_vis = pca_vis
        self.gmm = gmm
        self.hdbscan_clusterer = hdbscan_clusterer
        self.selected_features_: list[str] | None = None
        self.agents_cluster_df_: pd.DataFrame | None = None
        self.x_scaled_: np.ndarray | None = None
        self.x_pca_: np.ndarray | None = None
        self.x_2d_: np.ndarray | None = None
        self.labels_gmm_: np.ndarray | None = None
        self.labels_hdbscan_: np.ndarray | None = None

    @staticmethod
    def initialize_agents(user_info: pd.DataFrame) -> pd.DataFrame:
        """Copy user demographic/personality rows as the agent base table."""
        return user_info.copy()

    @staticmethod
    def prepare_user_metadata(
        user_info: pd.DataFrame,
        columns: list[str] | None = None,
        drop_header_artifact: bool = True,
    ) -> pd.DataFrame:
        """Clean user metadata using the notebook's row-drop and column selection."""
        if "id" not in user_info.columns:
            raise ValueError("user_info must contain an 'id' column.")

        result = user_info.copy()
        if drop_header_artifact and len(result) > 0:
            result = result.drop(result.index[0]).reset_index(drop=True)

        selected_columns = [
            column for column in (columns or USER_METADATA_COLUMNS) if column in result.columns
        ]
        if "id" not in selected_columns:
            selected_columns = ["id"] + selected_columns
        return result[selected_columns].copy()

    @staticmethod
    def merge_user_metadata_into_tweets(
        tweets: pd.DataFrame,
        user_info: pd.DataFrame,
        user_col: str = "user_id",
        tweet_id_col: str = "id",
        prepared_user_info: bool = False,
    ) -> pd.DataFrame:
        """Merge notebook user metadata onto tweets before tweet clustering."""
        if user_col not in tweets.columns:
            raise ValueError(f"tweets must contain a '{user_col}' column.")

        metadata = (
            user_info.copy()
            if prepared_user_info
            else UserClusterer.prepare_user_metadata(user_info)
        )
        raw = tweets.rename(columns={tweet_id_col: "tweet_id"})
        return raw.merge(metadata.set_index("id"), left_on=user_col, right_index=True)

    @staticmethod
    def add_length_features(
        agents: pd.DataFrame,
        tweets: pd.DataFrame,
        text_col: str = "tweet",
    ) -> pd.DataFrame:
        """Merge mean and std tweet length per user."""
        tweet_data = tweets.copy()
        tweet_data["length"] = tweet_data[text_col].apply(len)
        len_stats = (
            tweet_data.groupby("user_id")["length"]
            .agg(mean_len="mean", std_len="std")
            .fillna(0)
            .reset_index()
        )
        return (
            agents.merge(len_stats, left_on="id", right_on="user_id", how="left")
            .drop(columns="user_id", errors="ignore")
        )

    @staticmethod
    def add_sentiment_features(
        agents: pd.DataFrame,
        sentiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge notebook sentiment and emotion score aggregates per user."""
        sent_stats = (
            sentiments.groupby("user_id")
            .agg(
                mean_sent=("roberta_score", "mean"),
                std_sent=("roberta_score", "std"),
                mean_emo=("emotion_score", "mean"),
                std_emo=("emotion_score", "std"),
            )
            .fillna(0)
            .reset_index()
        )
        result = (
            agents.merge(sent_stats, left_on="id", right_on="user_id", how="left")
            .drop(columns="user_id", errors="ignore")
        )
        result[["mean_sent", "std_sent", "mean_emo", "std_emo"]] = result[
            ["mean_sent", "std_sent", "mean_emo", "std_emo"]
        ].fillna(0)
        return result

    def add_text_feature_aggregates(
        self,
        agents: pd.DataFrame,
        tweets: pd.DataFrame,
        text_col: str = "tweet",
    ) -> pd.DataFrame:
        """Merge lexical diversity and LIWC-style user averages."""
        user_text = tweets.groupby("user_id")[text_col].apply(" ".join)
        diversity_feats = (
            user_text.apply(self.feature_extractor.lexical_features)
            .apply(pd.Series)
            .reset_index()
        )

        liwc_raw = tweets[text_col].apply(self.feature_extractor.liwc_features)
        liwc_df = pd.concat([tweets[["user_id"]], liwc_raw], axis=1)
        user_fp = liwc_df.groupby("user_id").mean().reset_index()
        user_fp = user_fp.merge(diversity_feats, on="user_id", how="left")

        return (
            agents.merge(user_fp, left_on="id", right_on="user_id", how="left")
            .drop(columns="user_id", errors="ignore")
        )

    @staticmethod
    def compute_user_tweet_cosine(
        tweets: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Compute mean pairwise tweet cosine similarity per user."""
        agent_cosine: dict[Any, float] = {}
        for user_id in tweets["user_id"].unique():
            idx = tweets[tweets["user_id"] == user_id].index.tolist()
            user_embeddings = embeddings[idx]
            if len(user_embeddings) > 1:
                sim_mat = cosine_similarity(user_embeddings)
                upper = np.triu_indices(len(user_embeddings), k=1)
                agent_cosine[user_id] = float(np.mean(sim_mat[upper]))
            else:
                agent_cosine[user_id] = 1.0 if len(user_embeddings) == 1 else 0.0
        return pd.DataFrame(
            agent_cosine.items(),
            columns=["id", "avg_tweet_cosine_similarity"],
        )

    @staticmethod
    def add_user_tweet_cosine(
        agents: pd.DataFrame,
        tweets: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Merge per-user average tweet cosine similarity."""
        cosine_df = UserClusterer.compute_user_tweet_cosine(tweets, embeddings)
        result = agents.merge(cosine_df, on="id", how="left")
        result["avg_tweet_cosine_similarity"] = result[
            "avg_tweet_cosine_similarity"
        ].fillna(0)
        return result

    def add_rate_features(
        self,
        agents: pd.DataFrame,
        tweets: pd.DataFrame,
        text_col: str = "tweet",
    ) -> pd.DataFrame:
        """Merge mean emoji and punctuation rates per user."""
        tweet_data = tweets.copy()
        tweet_data["emoji_rate"] = tweet_data[text_col].apply(self.feature_extractor.emoji_rate)
        tweet_data["punctuation_rate"] = tweet_data[text_col].apply(
            self.feature_extractor.punctuation_rate
        )
        rate_stats = self.feature_extractor.aggregate_user_rates(tweet_data)
        return (
            agents.merge(rate_stats, left_on="id", right_on="user_id", how="left")
            .drop(columns="user_id", errors="ignore")
        )

    @staticmethod
    def add_reply_coherence(
        agents: pd.DataFrame,
        reply_sim: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge weighted average reply coherence and missingness indicator."""
        reply_data = reply_sim.copy()
        reply_data["coherence_score"] = (
            0.20 * reply_data["cosine_similarity"]
            + 0.35 * reply_data["bs_f1"]
            + 0.45 * reply_data["cross_encoder_score"]
        )
        avg_coh = (
            reply_data.groupby("user_id")["coherence_score"]
            .mean()
            .reset_index()
            .rename(columns={"coherence_score": "avg_reply_coherence"})
        )
        result = (
            agents.merge(avg_coh, left_on="id", right_on="user_id", how="left")
            .drop(columns="user_id", errors="ignore")
        )
        result["has_parent"] = result["avg_reply_coherence"].notna().astype(int)
        mean_coh = result["avg_reply_coherence"].mean()
        result["avg_reply_coherence"] = result["avg_reply_coherence"].fillna(mean_coh)
        return result

    def prepare_clustering_matrix(self, agents: pd.DataFrame) -> pd.DataFrame:
        """Select notebook cluster features and drop rows with missing values."""
        selected = [feature for feature in self.cluster_features if feature in agents.columns]
        self.selected_features_ = selected
        matrix = agents[selected].copy().dropna()
        self.agents_cluster_df_ = matrix
        return matrix

    def clustering_matrix_summary(
        self,
        agents: pd.DataFrame,
    ) -> dict[str, Any]:
        """Return notebook feature matrix shape and dropped-agent count."""
        matrix = self.prepare_clustering_matrix(agents)
        return {
            "matrix": matrix,
            "shape": matrix.shape,
            "dropped_agents": len(agents) - len(matrix),
            "cluster_features": self.selected_features_,
        }

    def _build_scaler(self) -> Any:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()

    def _build_pca(self) -> Any:
        from sklearn.decomposition import PCA

        return PCA(n_components=self.pca_variance_threshold)

    def _build_pca_vis(self) -> Any:
        from sklearn.decomposition import PCA

        return PCA(n_components=2)

    def _build_gmm(self) -> Any:
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=self.gmm_n_components,
            covariance_type="full",
            random_state=42,
        )

    def _build_hdbscan_clusterer(self) -> Any:
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )

    def fit_clusters(
        self,
        agents_cluster_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale features, apply PCA, GMM, HDBSCAN, and 2D PCA projection."""
        if self.scaler is None:
            self.scaler = self._build_scaler()
        self.x_scaled_ = self.scaler.fit_transform(agents_cluster_df)

        if self.pca is None:
            self.pca = self._build_pca()
        self.x_pca_ = np.asarray(self.pca.fit_transform(self.x_scaled_))

        if self.gmm is None:
            self.gmm = self._build_gmm()
        self.labels_gmm_ = np.asarray(self.gmm.fit_predict(self.x_pca_))

        if self.hdbscan_clusterer is None:
            self.hdbscan_clusterer = self._build_hdbscan_clusterer()
        self.labels_hdbscan_ = np.asarray(self.hdbscan_clusterer.fit_predict(self.x_pca_))

        if self.pca_vis is None:
            self.pca_vis = self._build_pca_vis()
        self.x_2d_ = np.asarray(self.pca_vis.fit_transform(self.x_scaled_))

        return self.labels_gmm_, self.labels_hdbscan_, self.x_pca_, self.x_2d_

    @staticmethod
    def agent_cluster_count_summary(
        labels_gmm: np.ndarray,
        labels_hdbscan: np.ndarray,
    ) -> dict[str, int]:
        """Return notebook GMM/HDBSCAN cluster and noise counts."""
        labels_hdbscan_array = np.asarray(labels_hdbscan)
        return {
            "gmm_clusters": int(len(set(labels_gmm))),
            "hdbscan_clusters": int(
                len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
            ),
            "hdbscan_noise": int((labels_hdbscan_array == -1).sum()),
        }

    @staticmethod
    def assign_cluster_labels(
        agents: pd.DataFrame,
        agents_cluster_df: pd.DataFrame,
        labels_gmm: np.ndarray,
        labels_hdbscan: np.ndarray,
    ) -> pd.DataFrame:
        """Return agent rows with GMM and HDBSCAN cluster labels."""
        result = agents.loc[agents_cluster_df.index].copy()
        result["gmm_cluster"] = labels_gmm
        result["hdbscan_cluster"] = labels_hdbscan
        return result

    @staticmethod
    def profile_clusters(
        agents_with_clusters: pd.DataFrame,
        trait: str,
        cluster_col: str = "Agent_Cluster",
    ) -> dict[str, Any]:
        """Compute notebook-style trait profile data for a cluster assignment."""
        if trait not in agents_with_clusters.columns:
            return {}
        column = agents_with_clusters[trait]
        if column.dtype == object or column.nunique() <= 10:
            profile: dict[str, Any] = {
                "crosstab": pd.crosstab(
                    agents_with_clusters[cluster_col],
                    column,
                    normalize="index",
                ).round(3),
                "raw_crosstab": pd.crosstab(agents_with_clusters[cluster_col], column),
            }
            raw = profile["raw_crosstab"]
            if raw.shape[0] > 1 and raw.shape[1] > 1:
                from scipy.stats import chi2_contingency

                chi2, p_value, dof, _ = chi2_contingency(raw)
                profile.update(
                    {
                        "chi2": float(chi2),
                        "p": float(p_value),
                        "dof": int(dof),
                        "significance": (
                            "SIGNIFICANT" if p_value < 0.05 else "not significant"
                        ),
                    }
                )
            unique_vals = column.dropna().unique()
            if len(unique_vals) == 2:
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

                label_map = {value: idx for idx, value in enumerate(unique_vals)}
                encoded = column.dropna().map(label_map)
                cluster_labels = agents_with_clusters.loc[encoded.index, cluster_col]
                profile["ari"] = float(adjusted_rand_score(encoded, cluster_labels))
                profile["nmi"] = float(normalized_mutual_info_score(encoded, cluster_labels))
            return profile

        return {
            "describe": agents_with_clusters.groupby(cluster_col)[trait].describe().round(3)
        }

    @staticmethod
    def profile_held_out_traits(
        agents_with_clusters: pd.DataFrame,
        traits: list[str] | None = None,
        cluster_col: str = "Agent_Cluster",
    ) -> dict[str, dict[str, Any]]:
        """Profile notebook held-out personality and demographic traits."""
        selected_traits = traits or (PERSONALITY_TRAITS + DEMOGRAPHIC_TRAITS)
        return {
            trait: UserClusterer.profile_clusters(
                agents_with_clusters,
                trait,
                cluster_col=cluster_col,
            )
            for trait in selected_traits
            if trait in agents_with_clusters.columns
        }

    @staticmethod
    def behavioral_correlation_matrix(
        agents: pd.DataFrame,
        behavior_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return notebook behavioural feature correlation matrix."""
        default_cols = [
            "mean_len",
            "std_len",
            "mean_sent",
            "std_sent",
            "mean_emo",
            "std_emo",
            "mtld",
            "entropy",
            "pronoun_ratio",
            "avg_sent_length",
            "readability_score",
            "exclamation_freq",
            "ellipsis_freq",
            "avg_tweet_cosine_similarity",
            "mean_emoji_rate",
            "mean_punct_rate",
            "avg_reply_coherence",
        ]
        selected_cols = [
            column for column in (behavior_cols or default_cols) if column in agents.columns
        ]
        return agents[selected_cols].corr()

    @staticmethod
    def cluster_profile_summary(
        agents_with_clusters: pd.DataFrame,
        feature_cols: list[str],
        cluster_col: str = "Agent_Cluster",
    ) -> pd.DataFrame:
        """Summarize behavioural features by agent cluster."""
        selected_features = [
            feature for feature in feature_cols if feature in agents_with_clusters.columns
        ]
        if not selected_features:
            return pd.DataFrame()
        return agents_with_clusters.groupby(cluster_col)[selected_features].mean().round(3)

    def fit_transform_agents(
        self,
        agents: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare feature matrix, cluster agents, and return labeled agents."""
        matrix = self.prepare_clustering_matrix(agents)
        labels_gmm, labels_hdbscan, _, _ = self.fit_clusters(matrix)
        return self.assign_cluster_labels(agents, matrix, labels_gmm, labels_hdbscan)
