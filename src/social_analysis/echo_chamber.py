"""Echo chamber stance-estimation helpers extracted from notebook 07."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-base"
EMBEDDING_BATCH_SIZE = 32
PRO_STANCE_THRESHOLD = 0.3
ANTI_STANCE_THRESHOLD = -0.3
BETA = 0.5

PRO_ANCHORS = [
    "Artificial intelligence will revolutionize healthcare by accelerating drug discovery, curing complex diseases, and ultimately saving countless lives.",
    "The widespread adoption of AI will trigger an unprecedented economic boom, supercharging global productivity and creating entirely new industries.",
    "AI is humanity's most powerful tool for solving intractable global challenges, from managing the climate crisis to ending resource scarcity.",
]

ANTI_ANCHORS = [
    "The rapid deployment of automation and AI will inevitably cause mass global unemployment, rendering millions of human workers obsolete.",
    "Unchecked advancement toward artificial general intelligence (AGI) represents a literal existential threat that could lead to human extinction.",
    "Delegating critical decisions to opaque AI algorithms will result in a dystopian nightmare of surveillance, automated bias, and loss of human agency.",
]


class StanceEstimator:
    """Estimate stance scores using embedding prototypes and cross-encoder anchors."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        cross_encoder_model_name: str = CROSS_ENCODER_MODEL,
        embedding_model: Any | None = None,
        cross_encoder: Any | None = None,
        pro_anchors: list[str] | None = None,
        anti_anchors: list[str] | None = None,
        load_embedding_model: bool = True,
        load_cross_encoder: bool = True,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.pro_anchors = pro_anchors or PRO_ANCHORS
        self.anti_anchors = anti_anchors or ANTI_ANCHORS
        self.prototype_pro: np.ndarray | None = None
        self.prototype_anti: np.ndarray | None = None

        if self.embedding_model is None and load_embedding_model:
            self.embedding_model = self._load_embedding_model()
        if self.cross_encoder is None and load_cross_encoder:
            self.cross_encoder = self._load_cross_encoder()

    def _load_embedding_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_model_name)

    def _load_cross_encoder(self) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.cross_encoder_model_name)

    def compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode texts with the notebook sentence-transformer settings."""
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return np.asarray(embeddings)

    @staticmethod
    def build_stance_prototypes(
        embeddings: np.ndarray,
        stance_score_initial: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build pro/anti prototype embeddings from initial sentiment thresholds."""
        scores = np.asarray(stance_score_initial)
        pro_mask = scores > PRO_STANCE_THRESHOLD
        anti_mask = scores < ANTI_STANCE_THRESHOLD

        prototype_pro = (
            np.mean(embeddings[pro_mask], axis=0)
            if pro_mask.sum() > 0
            else np.zeros(embeddings.shape[1])
        )
        prototype_anti = (
            np.mean(embeddings[anti_mask], axis=0)
            if anti_mask.sum() > 0
            else np.zeros(embeddings.shape[1])
        )
        return prototype_pro, prototype_anti

    @staticmethod
    def calculate_stance_similarity(
        embedding: np.ndarray,
        proto_pro: np.ndarray,
        proto_anti: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return stance score and pro/anti cosine similarities."""
        sim_pro = cosine_similarity([embedding], [proto_pro])[0][0]
        sim_anti = cosine_similarity([embedding], [proto_anti])[0][0]
        stance_score = sim_pro - sim_anti
        return float(stance_score), float(sim_pro), float(sim_anti)

    def compute_embedding_stance(
        self,
        embeddings: np.ndarray,
        prototype_pro: np.ndarray | None = None,
        prototype_anti: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute stance_similarity, sim_pro, and sim_anti for embeddings."""
        proto_pro = prototype_pro if prototype_pro is not None else self.prototype_pro
        proto_anti = prototype_anti if prototype_anti is not None else self.prototype_anti
        if proto_pro is None or proto_anti is None:
            raise ValueError("Stance prototypes must be built before similarity scoring.")

        results = [
            self.calculate_stance_similarity(embedding, proto_pro, proto_anti)
            for embedding in embeddings
        ]
        return pd.DataFrame(
            {
                "stance_similarity": [result[0] for result in results],
                "sim_pro": [result[1] for result in results],
                "sim_anti": [result[2] for result in results],
            }
        )

    def compute_cross_encoder_score(self, text: str) -> float:
        """Score one text against pro and anti anchors using mean score difference."""
        if self.cross_encoder is None:
            self.cross_encoder = self._load_cross_encoder()

        pro_pairs = [(text, anchor) for anchor in self.pro_anchors]
        scores_pro = self.cross_encoder.predict(pro_pairs)
        score_pro_aggregated = np.mean(scores_pro)

        anti_pairs = [(text, anchor) for anchor in self.anti_anchors]
        scores_anti = self.cross_encoder.predict(anti_pairs)
        score_anti_aggregated = np.mean(scores_anti)

        return float(score_pro_aggregated - score_anti_aggregated)

    def compute_cross_encoder_scores(self, texts: list[str]) -> pd.Series:
        """Compute cross_encoder_score for each text."""
        return pd.Series(
            [self.compute_cross_encoder_score(text) for text in texts],
            name="cross_encoder_score",
        )

    @staticmethod
    def fuse_stance_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and fuse stance similarity with cross-encoder scores."""
        result = df.copy()
        scaler = MinMaxScaler(feature_range=(-1, 1))

        result["stance_similarity_norm"] = scaler.fit_transform(
            result[["stance_similarity"]]
        )

        result["cross_encoder_norm"] = result["cross_encoder_score"]
        non_null_mask = result["cross_encoder_norm"].notna()
        if non_null_mask.sum() > 0:
            result.loc[non_null_mask, "cross_encoder_norm"] = scaler.fit_transform(
                result.loc[non_null_mask, ["cross_encoder_score"]]
            )

        result["stance_ensemble"] = result["stance_similarity_norm"].copy()
        ce_available = result["cross_encoder_norm"].notna()
        result.loc[ce_available, "stance_ensemble"] = (
            0.6 * result.loc[ce_available, "stance_similarity_norm"]
            + 0.4 * result.loc[ce_available, "cross_encoder_norm"]
        )
        return result

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        initial_score_col: str = "roberta_compound",
    ) -> pd.DataFrame:
        """Run the notebook stance-estimation workflow on a DataFrame."""
        result = df.copy()
        result["stance_score_initial"] = result[initial_score_col]
        embeddings = self.compute_embeddings(result[text_col].tolist())
        self.prototype_pro, self.prototype_anti = self.build_stance_prototypes(
            embeddings,
            result["stance_score_initial"],
        )

        stance = self.compute_embedding_stance(embeddings)
        stance.index = result.index
        for column in stance.columns:
            result[column] = stance[column]

        result["cross_encoder_score"] = self.compute_cross_encoder_scores(
            result[text_col].tolist()
        ).values
        return self.fuse_stance_scores(result)


class AttitudeScorer:
    """Compute tweet-level and user-level attitude scores."""

    def __init__(self, beta: float = BETA) -> None:
        self.beta = beta

    def calculate_attitude_score(self, stance: float, sentiment: float) -> float:
        """Calculate attitude score from stance and sentiment."""
        amplifier = 1 + (self.beta * abs(sentiment))
        attitude = np.tanh(stance * amplifier)
        return float(np.clip(attitude, -1, 1))

    def add_sentiment_score(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "roberta_compound",
        output_col: str = "sentiment_score",
    ) -> pd.DataFrame:
        """Add the notebook sentiment_score column."""
        result = df.copy()
        result[output_col] = result[sentiment_col]
        return result

    def add_attitude_scores(
        self,
        df: pd.DataFrame,
        stance_col: str = "stance_ensemble",
        sentiment_col: str = "sentiment_score",
        output_col: str = "attitude_score",
    ) -> pd.DataFrame:
        """Add tweet-level attitude scores."""
        result = df.copy()
        result[output_col] = result.apply(
            lambda row: self.calculate_attitude_score(
                row[stance_col],
                row[sentiment_col],
            ),
            axis=1,
        )
        return result

    @staticmethod
    def aggregate_user_attitudes(user_df: pd.DataFrame) -> pd.Series:
        """Aggregate attitudes for one user using notebook formulas."""
        attitudes = user_df["attitude_score"].values
        days = user_df["day"].values

        mean_attitude = np.mean(attitudes)
        weights = np.exp(days - days.max())
        weighted_attitude = np.average(attitudes, weights=weights)
        engagement = user_df["like"].values + 1
        engagement_weighted = np.average(attitudes, weights=engagement)

        return pd.Series(
            {
                "mean_attitude": mean_attitude,
                "weighted_attitude": weighted_attitude,
                "engagement_weighted_attitude": engagement_weighted,
                "attitude_std": np.std(attitudes),
                "num_posts": len(attitudes),
                "total_likes": user_df["like"].sum(),
                "total_reactions": user_df["reaction_count"].sum(),
            }
        )

    def compute_user_attitudes(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
    ) -> pd.DataFrame:
        """Aggregate tweet attitudes to user-level attitude summaries."""
        user_attitudes = (
            df.groupby(user_col)
            .apply(self.aggregate_user_attitudes, include_groups=False)
            .reset_index()
        )
        user_attitudes.loc[user_attitudes["num_posts"] < 2, "attitude_std"] = np.nan
        return user_attitudes

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "roberta_compound",
        stance_col: str = "stance_ensemble",
    ) -> pd.DataFrame:
        """Add sentiment_score and attitude_score columns."""
        result = self.add_sentiment_score(df, sentiment_col=sentiment_col)
        return self.add_attitude_scores(result, stance_col=stance_col)


class InteractionNetworkBuilder:
    """Build the directed social interaction graph from notebook 07."""

    def __init__(
        self,
        user_col: str = "user_id",
        tweet_id_col: str = "id",
        reply_col: str = "comment_to",
        share_col: str = "shared_from",
    ) -> None:
        self.user_col = user_col
        self.tweet_id_col = tweet_id_col
        self.reply_col = reply_col
        self.share_col = share_col

    def add_user_nodes(
        self,
        graph: nx.DiGraph,
        user_attitudes: pd.DataFrame,
    ) -> nx.DiGraph:
        """Add user nodes with attitude attributes."""
        for _, user in user_attitudes.iterrows():
            graph.add_node(
                user[self.user_col],
                attitude=user["mean_attitude"],
                num_posts=user["num_posts"],
                attitude_std=user["attitude_std"],
            )
        return graph

    @staticmethod
    def _add_or_increment_edge(
        graph: nx.DiGraph,
        source: Any,
        target: Any,
        edge_type: str,
    ) -> None:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1, type=edge_type)

    def _tweet_author_lookup(self, df: pd.DataFrame) -> dict[Any, Any]:
        return df.set_index(self.tweet_id_col)[self.user_col].to_dict()

    def add_reply_edges(self, graph: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
        """Add directed reply edges using comment_to."""
        tweet_authors = self._tweet_author_lookup(df)
        reply_interactions = df[df[self.reply_col] != -1][
            [self.user_col, self.reply_col]
        ].copy()

        for _, row in reply_interactions.iterrows():
            source = row[self.user_col]
            target = tweet_authors.get(row[self.reply_col])
            if (
                target is not None
                and source != target
                and source in graph.nodes()
                and target in graph.nodes()
            ):
                self._add_or_increment_edge(graph, source, target, "reply")
        return graph

    def add_share_edges(self, graph: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
        """Add directed retweet/share edges using shared_from."""
        tweet_authors = self._tweet_author_lookup(df)
        retweet_interactions = df[df[self.share_col] != -1][
            [self.user_col, self.share_col]
        ].copy()

        for _, row in retweet_interactions.iterrows():
            source = row[self.user_col]
            target = tweet_authors.get(row[self.share_col])
            if (
                target is not None
                and source != target
                and source in graph.nodes()
                and target in graph.nodes()
            ):
                self._add_or_increment_edge(graph, source, target, "retweet")
        return graph

    def build_graph(
        self,
        df: pd.DataFrame,
        user_attitudes: pd.DataFrame,
    ) -> nx.DiGraph:
        """Build the directed interaction graph."""
        graph = nx.DiGraph()
        self.add_user_nodes(graph, user_attitudes)
        self.add_reply_edges(graph, df)
        self.add_share_edges(graph, df)
        return graph

    @staticmethod
    def to_undirected(graph: nx.DiGraph) -> nx.Graph:
        """Convert the directed graph to undirected form."""
        return graph.to_undirected()

    @staticmethod
    def graph_stats(graph: nx.DiGraph) -> dict[str, float | int]:
        """Return the basic graph construction statistics printed in the notebook."""
        stats: dict[str, float | int] = {
            "number_of_nodes": graph.number_of_nodes(),
            "number_of_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "number_of_connected_components": nx.number_weakly_connected_components(graph),
        }
        if graph.number_of_edges() > 0:
            stats["average_degree"] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        return stats

    def build(
        self,
        df: pd.DataFrame,
        user_attitudes: pd.DataFrame,
    ) -> tuple[nx.DiGraph, nx.Graph]:
        """Return directed and undirected notebook-compatible interaction graphs."""
        graph = self.build_graph(df, user_attitudes)
        return graph, self.to_undirected(graph)


class EchoChamberAnalyzer:
    """Compute echo-chamber metrics from attitudes and an interaction graph."""

    @staticmethod
    def _neighbor_attitudes_for_user(
        user_id: Any,
        user_df: pd.DataFrame,
        graph: nx.Graph,
        user_col: str = "user_id",
        attitude_col: str = "propagated_attitude",
    ) -> list[float]:
        neighbors = list(graph.neighbors(user_id))
        if len(neighbors) == 0:
            return []
        return [
            user_df[user_df[user_col] == neighbor][attitude_col].values[0]
            for neighbor in neighbors
            if neighbor in user_df[user_col].values
        ]

    @classmethod
    def build_pca_weight_metrics(
        cls,
        user_df: pd.DataFrame,
        community_stats: pd.DataFrame,
        graph: nx.Graph,
        diversity_col_name: str = "low_diversity",
        user_col: str = "user_id",
    ) -> pd.DataFrame:
        """Build per-user metrics used by the notebook PCA weighting cells."""
        metrics_list = []

        for _, user in user_df.iterrows():
            user_id = user[user_col]

            local_polarization = abs(user["propagated_attitude"])

            neighbor_attitudes = cls._neighbor_attitudes_for_user(
                user_id,
                user_df,
                graph,
                user_col=user_col,
            )
            if len(neighbor_attitudes) > 0:
                local_homophily = np.mean(
                    [
                        1 if user["propagated_attitude"] * neighbor_attitude > 0 else 0
                        for neighbor_attitude in neighbor_attitudes
                    ]
                )
            else:
                local_homophily = 0

            local_diversity = 1 - user["exposure_diversity"]

            user_comm = user["community"]
            comm_attitude = community_stats.loc[user_comm, "avg_attitude"]
            opposite_attitudes = community_stats[
                community_stats["avg_attitude"] * comm_attitude < 0
            ]["avg_attitude"].values

            if len(opposite_attitudes) > 0:
                local_separation = abs(
                    user["propagated_attitude"] - np.mean(opposite_attitudes)
                )
            else:
                local_separation = 0

            metrics_list.append(
                {
                    "user_id": user_id,
                    "polarization": local_polarization,
                    "homophily": local_homophily,
                    diversity_col_name: local_diversity,
                    "separation": local_separation,
                }
            )

        return pd.DataFrame(metrics_list)

    @classmethod
    def optimize_echo_chamber_weights(
        cls,
        user_df: pd.DataFrame,
        community_stats: pd.DataFrame,
        graph: nx.Graph,
    ) -> tuple[dict[str, float], Any]:
        """Use one-component PCA to find notebook data-driven weights."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        metrics_df = cls.build_pca_weight_metrics(
            user_df,
            community_stats,
            graph,
            diversity_col_name="low_diversity",
        )

        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(
            metrics_df[["polarization", "homophily", "low_diversity", "separation"]]
        )

        pca = PCA(n_components=1)
        pca.fit(metrics_scaled)

        weights_raw = np.abs(pca.components_[0])
        weights_normalized = weights_raw / weights_raw.sum()

        weight_dict = {
            "polarization": float(weights_normalized[0]),
            "homophily": float(weights_normalized[1]),
            "diversity": float(weights_normalized[2]),
            "separation": float(weights_normalized[3]),
        }

        return weight_dict, pca

    @staticmethod
    def compute_optimized_component_scores(
        daily_polarization: dict[Any, dict[str, float]],
        daily_homophily: dict[Any, dict[str, float]],
        user_attitudes: pd.DataFrame,
        community_stats: pd.DataFrame,
        final_day: Any | None = None,
    ) -> dict[str, float]:
        """Compute the notebook's optimized component-score table."""
        day = max(daily_polarization.keys()) if final_day is None else final_day
        echo_chamber_metrics_optimized = {
            "polarization_score": daily_polarization[day]["variance"] / 0.5,
            "homophily_score": daily_homophily[day]["homophily_ratio"],
            "diversity_score": 1 - user_attitudes["exposure_diversity"].mean(),
            "separation_score": community_stats["avg_attitude"].var() / 0.3,
        }

        for key in echo_chamber_metrics_optimized:
            echo_chamber_metrics_optimized[key] = min(
                1.0,
                echo_chamber_metrics_optimized[key],
            )

        return {
            key: float(value) for key, value in echo_chamber_metrics_optimized.items()
        }

    @staticmethod
    def compute_optimized_echo_chamber_score(
        metrics: dict[str, float],
        optimized_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply one-component PCA weights to optimized component scores."""
        overall_score_optimized = (
            optimized_weights["polarization"] * metrics["polarization_score"]
            + optimized_weights["homophily"] * metrics["homophily_score"]
            + optimized_weights["diversity"] * metrics["diversity_score"]
            + optimized_weights["separation"] * metrics["separation_score"]
        )

        result = dict(metrics)
        result["overall_score"] = float(overall_score_optimized)
        return result

    @staticmethod
    def sensitivity_analysis(
        metrics: dict[str, float],
        optimized_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Test how sensitive the overall score is to notebook weight schemes."""
        weight_sets = {
            "Equal": [0.25, 0.25, 0.25, 0.25],
            "Polarization-heavy": [0.5, 0.2, 0.15, 0.15],
            "Homophily-heavy": [0.2, 0.5, 0.15, 0.15],
            "Diversity-heavy": [0.2, 0.15, 0.5, 0.15],
            "Separation-heavy": [0.2, 0.15, 0.15, 0.5],
            "Optimized": list(optimized_weights.values()),
        }

        results = []
        for name, weights in weight_sets.items():
            score = (
                weights[0] * metrics["polarization_score"]
                + weights[1] * metrics["homophily_score"]
                + weights[2] * metrics["diversity_score"]
                + weights[3] * metrics["separation_score"]
            )
            results.append({"scheme": name, "score": float(score)})

        return pd.DataFrame(results)

    @classmethod
    def optimize_echo_chamber_weights_2d(
        cls,
        user_df: pd.DataFrame,
        community_stats: pd.DataFrame,
        graph: nx.Graph,
    ) -> tuple[dict[str, Any], Any, pd.DataFrame, np.ndarray]:
        """Use two-component PCA to find structural and ideological weights."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        metrics_df = cls.build_pca_weight_metrics(
            user_df,
            community_stats,
            graph,
            diversity_col_name="diversity",
        )

        scaler = StandardScaler()
        feature_cols = ["polarization", "homophily", "diversity", "separation"]
        metrics_scaled = scaler.fit_transform(metrics_df[feature_cols])

        pca = PCA(n_components=2)
        pca.fit(metrics_scaled)

        weights_pc1_raw = np.abs(pca.components_[0])
        weights_pc1_norm = weights_pc1_raw / weights_pc1_raw.sum()

        weights_pc2_raw = np.abs(pca.components_[1])
        weights_pc2_norm = weights_pc2_raw / weights_pc2_raw.sum()

        var_ratios = pca.explained_variance_ratio_

        weight_dicts: dict[str, Any] = {
            "pc1": {
                key: float(value)
                for key, value in zip(feature_cols, weights_pc1_norm)
            },
            "pc2": {
                key: float(value)
                for key, value in zip(feature_cols, weights_pc2_norm)
            },
            "variances": var_ratios,
        }

        return weight_dicts, pca, metrics_df, metrics_scaled

    @staticmethod
    def compute_2d_optimized_score(
        metrics: dict[str, float],
        weights_2d: dict[str, Any],
    ) -> dict[str, float]:
        """Compute PC1, PC2, and variance-weighted 2D optimized scores."""
        score_pc1 = (
            weights_2d["pc1"]["polarization"] * metrics["polarization_score"]
            + weights_2d["pc1"]["homophily"] * metrics["homophily_score"]
            + weights_2d["pc1"]["diversity"] * metrics["diversity_score"]
            + weights_2d["pc1"]["separation"] * metrics["separation_score"]
        )

        score_pc2 = (
            weights_2d["pc2"]["polarization"] * metrics["polarization_score"]
            + weights_2d["pc2"]["homophily"] * metrics["homophily_score"]
            + weights_2d["pc2"]["diversity"] * metrics["diversity_score"]
            + weights_2d["pc2"]["separation"] * metrics["separation_score"]
        )

        var1, var2 = weights_2d["variances"][0], weights_2d["variances"][1]
        final_combined_score = (var1 * score_pc1 + var2 * score_pc2) / (var1 + var2)

        return {
            "score_pc1": float(score_pc1),
            "score_pc2": float(score_pc2),
            "final_combined_score": float(final_combined_score),
        }

    @staticmethod
    def build_echo_chamber_landscape_dataframe(
        metrics_df: pd.DataFrame,
        metrics_scaled: np.ndarray,
        pca_model: Any,
        user_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the notebook's 2D PCA landscape dataframe for plotting."""
        pca_scores = pca_model.transform(metrics_scaled)

        plot_df = metrics_df.copy()
        plot_df["PC1_Ideological"] = pca_scores[:, 0]
        plot_df["PC2_Structural"] = pca_scores[:, 1]

        return plot_df.merge(
            user_df[["user_id", "community", "propagated_attitude"]],
            on="user_id",
        )

    @classmethod
    def build_echo_chamber_landscape_data(
        cls,
        metrics_df: pd.DataFrame,
        metrics_scaled: np.ndarray,
        pca_model: Any,
        user_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Return plot-ready data and metadata for the notebook 2D PCA landscape."""
        landscape_df = cls.build_echo_chamber_landscape_dataframe(
            metrics_df,
            metrics_scaled,
            pca_model,
            user_df,
        )

        return {
            "landscape_df": landscape_df,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "pca_components": pca_model.components_,
            "x_col": "PC1_Ideological",
            "y_col": "PC2_Structural",
            "hue_col": "community",
            "size_col": "polarization",
            "title": "Echo Chamber Landscape: Ideological vs. Structural Bias",
            "xlabel": (
                "PC1: Ideological Extremism & Separation \n"
                "<-- Moderate --- Radical -->"
            ),
            "ylabel": (
                "PC2: Structural Homophily & Isolation \n"
                "<-- Diverse Networks --- Isolated Networks -->"
            ),
            "size_range": (30, 300),
            "alpha": 0.7,
            "palette": "Set1",
            "edgecolor": "black",
            "origin_lines": {"x": 0, "y": 0},
            "quadrant_annotations": [
                {
                    "label": "Double Trap\n(Radical & Isolated)",
                    "x_multiplier": 0.7,
                    "y_multiplier": 0.85,
                    "color": "darkred",
                },
                {
                    "label": "Bridge Agents\n(Moderate & Diverse)",
                    "x_multiplier": 0.8,
                    "y_multiplier": 0.8,
                    "color": "darkgreen",
                },
            ],
        }

    @staticmethod
    def improved_bimodality_test(
        attitudes_array: list[float] | np.ndarray,
        min_sample_size: int = 50,
    ) -> dict[str, Any]:
        """Run the notebook robust bimodality detection procedure."""
        n = len(attitudes_array)

        if n < min_sample_size:
            return {
                "sample_size": n,
                "bimodal": False,
                "reason": "insufficient_data",
                "dip_statistic": None,
                "separation_index": None,
                "variance": np.var(attitudes_array),
            }

        attitudes_array_np = np.array(attitudes_array)

        try:
            from diptest import diptest

            dip_stat, pval = diptest(attitudes_array_np)
            is_bimodal_dip = pval < 0.05
        except ImportError:
            dip_stat, pval, is_bimodal_dip = None, None, None

        hist, bin_edges = np.histogram(attitudes_array_np, bins=20, density=True)
        _ = (bin_edges[:-1] + bin_edges[1:]) / 2

        from scipy.signal import find_peaks

        peaks, _ = find_peaks(hist, height=0.1)

        if len(peaks) >= 2:
            top_peaks = sorted(peaks, key=lambda peak: hist[peak], reverse=True)[:2]
            left_peak, right_peak = sorted(top_peaks)

            valley_region = hist[left_peak : right_peak + 1]
            valley_depth = (hist[left_peak] + hist[right_peak]) / 2 - np.min(
                valley_region
            )
            peak_height = (hist[left_peak] + hist[right_peak]) / 2

            separation_index = valley_depth / (peak_height + 1e-6)
            is_bimodal_separation = separation_index > 0.3
        else:
            separation_index = 0
            is_bimodal_separation = False

        from scipy.stats import kurtosis, skew

        sk = skew(attitudes_array_np)
        kurt = kurtosis(attitudes_array_np)
        bc = (
            (sk**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            if n > 3
            else 0
        )
        is_bimodal_bc = bc > 0.555

        votes = sum(
            [
                is_bimodal_dip if is_bimodal_dip is not None else False,
                is_bimodal_separation,
                is_bimodal_bc,
            ]
        )

        return {
            "sample_size": n,
            "bimodal": bool(votes >= 2),
            "dip_statistic": dip_stat,
            "dip_pvalue": pval,
            "separation_index": separation_index,
            "bimodality_coefficient": bc,
            "num_peaks": len(peaks),
            "variance": np.var(attitudes_array_np),
        }

    @staticmethod
    def calculate_bimodality(attitudes_dict: dict[Any, float]) -> float:
        """Calculate the notebook bimodality coefficient for user attitudes."""
        vals = np.array(list(attitudes_dict.values()))
        n = len(vals)
        if n <= 3:
            return 0

        from scipy.stats import kurtosis, skew

        sk = skew(vals)
        kurt = kurtosis(vals)
        bc = (sk**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return float(bc)

    @classmethod
    def calculate_bimodality_trend(
        cls,
        propagation_history: dict[Any, dict[Any, float]],
    ) -> pd.DataFrame:
        """Return daily bimodality coefficients from propagation history."""
        days = sorted(propagation_history.keys())
        return pd.DataFrame(
            {
                "day": days,
                "bimodality": [
                    cls.calculate_bimodality(propagation_history[day])
                    for day in days
                ],
            }
        )

    @classmethod
    def build_bimodality_plot_data(
        cls,
        attitudes_array: list[float] | np.ndarray,
        min_sample_size: int = 50,
    ) -> dict[str, Any]:
        """Return histogram, peaks, and KDE data used by the notebook plot."""
        attitudes_array_np = np.array(attitudes_array)
        result = cls.improved_bimodality_test(
            attitudes_array_np,
            min_sample_size=min_sample_size,
        )

        hist, bin_edges = np.histogram(
            attitudes_array_np,
            bins=30,
            density=True,
        )

        from scipy.signal import find_peaks

        peaks, _ = find_peaks(hist, height=0.05)
        peak_positions = [
            (bin_edges[peak] + bin_edges[peak + 1]) / 2 for peak in peaks
        ]

        kde_density = None
        x_range = None
        try:
            from sklearn.neighbors import KernelDensity

            kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
            kde.fit(attitudes_array_np.reshape(-1, 1))
            x_range = np.linspace(-1, 1, 200).reshape(-1, 1)
            log_density = kde.score_samples(x_range)
            kde_density = np.exp(log_density)
        except Exception:
            pass

        return {
            "bimodality_result": result,
            "hist": hist,
            "bin_edges": bin_edges,
            "peaks": peaks,
            "peak_positions": peak_positions,
            "x_range": x_range,
            "kde_density": kde_density,
        }

    @staticmethod
    def measure_polarization(attitudes: dict[Any, float]) -> dict[str, float]:
        """Measure variance, bimodality, and inter-group attitude distance."""
        attitudes_array = np.array(list(attitudes.values()))
        n = len(attitudes_array)
        if n == 0:
            return {"variance": 0, "bimodality": 0, "inter_group_distance": 0}

        variance = np.var(attitudes_array)

        from scipy.stats import kurtosis, skew

        sk = skew(attitudes_array)
        kurt = kurtosis(attitudes_array)
        bimodality = (
            (sk**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            if n > 3
            else 0
        )

        pos = attitudes_array[attitudes_array > 0]
        neg = attitudes_array[attitudes_array < 0]
        inter_group_distance = np.mean(pos) - np.mean(neg) if len(pos) > 0 and len(neg) > 0 else 0

        return {
            "variance": float(variance),
            "bimodality": float(bimodality),
            "inter_group_distance": float(inter_group_distance),
        }

    @staticmethod
    def measure_homophily(graph: nx.Graph, attitudes: dict[Any, float]) -> dict[str, float]:
        """Measure network assortativity and same-sign edge ratio."""
        if graph.number_of_edges() == 0:
            return {"assortativity": 0, "homophily_ratio": 0}

        valid_attitudes = {node: attitudes[node] for node in graph.nodes() if node in attitudes}
        nx.set_node_attributes(graph, valid_attitudes, "attitude")

        try:
            assortativity = nx.numeric_assortativity_coefficient(graph, "attitude")
        except Exception:
            assortativity = 0

        similar, total = 0, 0
        for source, target in graph.edges():
            if source in valid_attitudes and target in valid_attitudes:
                if valid_attitudes[source] * valid_attitudes[target] > 0:
                    similar += 1
                total += 1
        return {
            "assortativity": float(assortativity),
            "homophily_ratio": float(similar / total if total > 0 else 0),
        }

    @staticmethod
    def calculate_exposure_diversity(
        graph: nx.Graph,
        attitudes: dict[Any, float],
    ) -> dict[Any, float]:
        """Calculate exposure diversity score for each graph node."""
        diversity_scores: dict[Any, float] = {}

        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 0:
                diversity_scores[node] = 0
                continue

            neighbor_attitudes = [attitudes[neighbor] for neighbor in neighbors if neighbor in attitudes]
            if len(neighbor_attitudes) == 0:
                diversity_scores[node] = 0
                continue

            std_diversity = np.std(neighbor_attitudes)
            bins = np.array([-1, -0.5, 0, 0.5, 1])
            hist, _ = np.histogram(neighbor_attitudes, bins=bins)
            probs = hist / hist.sum() if hist.sum() > 0 else hist
            entropy_diversity = entropy(probs + 1e-10)
            range_diversity = np.max(neighbor_attitudes) - np.min(neighbor_attitudes)

            diversity_scores[node] = float(
                (std_diversity + entropy_diversity / 2 + range_diversity) / 3
            )

        return diversity_scores

    @staticmethod
    def detect_communities(graph: nx.Graph) -> dict[Any, int]:
        """Detect communities with Louvain, matching notebook behavior."""
        if graph.number_of_edges() == 0:
            return {node: 0 for node in graph.nodes()}

        import community as community_louvain

        return community_louvain.best_partition(graph)

    @staticmethod
    def compute_modularity(communities: dict[Any, int], graph: nx.Graph) -> float:
        """Compute Louvain modularity for a community assignment."""
        import community as community_louvain

        return float(community_louvain.modularity(communities, graph))

    @staticmethod
    def add_community_labels(
        user_attitudes: pd.DataFrame,
        communities: dict[Any, int],
        user_col: str = "user_id",
    ) -> pd.DataFrame:
        """Add community labels to user attitude rows."""
        result = user_attitudes.copy()
        result["community"] = result[user_col].map(communities)
        return result

    @staticmethod
    def compute_community_stats(user_attitudes: pd.DataFrame) -> pd.DataFrame:
        """Compute notebook community statistics sorted by size."""
        community_stats = user_attitudes.groupby("community").agg(
            {
                "user_id": "count",
                "propagated_attitude": ["mean", "std"],
                "exposure_diversity": "mean",
            }
        ).round(4)

        community_stats.columns = ["size", "avg_attitude", "attitude_std", "avg_diversity"]
        return community_stats.sort_values("size", ascending=False)

    @staticmethod
    def compute_echo_chamber_score(
        polarization: dict[str, float],
        homophily: dict[str, float],
        diversity_stats: pd.Series,
        community_stats: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute the notebook's final echo chamber component scores."""
        polarization_score = min(
            1.0,
            (
                0.4 * min(1.0, polarization["variance"] / 0.5)
                + 0.3 * min(1.0, polarization["bimodality"] / 0.7)
                + 0.3 * min(1.0, abs(polarization["inter_group_distance"]) / 1.5)
            ),
        )

        homophily_score = min(
            1.0,
            (
                0.5 * max(0, homophily["assortativity"])
                + 0.5 * homophily["homophily_ratio"]
            ),
        )

        avg_diversity = diversity_stats.mean()
        diversity_score = max(0, 1 - avg_diversity)

        between_community_var = community_stats["avg_attitude"].var()
        within_community_var = community_stats["attitude_std"].mean()

        if within_community_var > 0:
            separation_score = min(1.0, between_community_var / (within_community_var + 0.1))
        else:
            separation_score = 0

        echo_chamber_score = (
            0.3 * polarization_score
            + 0.3 * homophily_score
            + 0.2 * diversity_score
            + 0.2 * separation_score
        )

        return {
            "overall_score": float(echo_chamber_score),
            "polarization_score": float(polarization_score),
            "homophily_score": float(homophily_score),
            "diversity_score": float(diversity_score),
            "separation_score": float(separation_score),
        }


class EchoChamberPipeline:
    """Coordinate the extracted echo-chamber analysis components."""

    def __init__(
        self,
        config: Any | None = None,
        stance_estimator: StanceEstimator | None = None,
        attitude_scorer: AttitudeScorer | None = None,
        network_builder: InteractionNetworkBuilder | None = None,
        analyzer: EchoChamberAnalyzer | None = None,
    ) -> None:
        self.config = config
        self.stance_estimator = stance_estimator
        self.attitude_scorer = attitude_scorer or AttitudeScorer()
        self.network_builder = network_builder or InteractionNetworkBuilder()
        self.analyzer = analyzer or EchoChamberAnalyzer()

    @staticmethod
    def analyze_real_propagation(
        dataframe: pd.DataFrame,
        user_col: str = "user_id",
        day_col: str = "day",
        attitude_col: str = "attitude_score",
    ) -> dict[Any, dict[Any, float]]:
        """Build the notebook's cumulative daily user-attitude snapshots."""
        days = sorted(dataframe[day_col].unique())
        real_propagation_history = {}

        for day in days:
            day_df = dataframe[dataframe[day_col] <= day]
            day_attitudes = day_df.groupby(user_col)[attitude_col].mean().to_dict()
            real_propagation_history[day] = day_attitudes

        return real_propagation_history

    @staticmethod
    def _final_propagated_attitudes(
        propagation_history: dict[Any, dict[Any, float]],
    ) -> tuple[Any | None, dict[Any, float]]:
        if not propagation_history:
            return None, {}
        final_day = max(propagation_history.keys())
        return final_day, propagation_history[final_day]

    def _ensure_stance_columns(
        self,
        df: pd.DataFrame,
        text_col: str,
        initial_score_col: str,
        stance_col: str,
    ) -> pd.DataFrame:
        if stance_col in df.columns:
            return df.copy()

        estimator = self.stance_estimator or StanceEstimator()
        return estimator.transform_dataframe(
            df,
            text_col=text_col,
            initial_score_col=initial_score_col,
        )

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        user_col: str = "user_id",
        day_col: str = "day",
        initial_score_col: str = "roberta_compound",
        sentiment_col: str = "roberta_compound",
        stance_col: str = "stance_ensemble",
    ) -> dict[str, Any]:
        """Run the notebook-compatible echo-chamber workflow."""
        tweet_df = self._ensure_stance_columns(
            df,
            text_col=text_col,
            initial_score_col=initial_score_col,
            stance_col=stance_col,
        )
        tweet_df = self.attitude_scorer.transform_dataframe(
            tweet_df,
            sentiment_col=sentiment_col,
            stance_col=stance_col,
        )

        user_attitudes = self.attitude_scorer.compute_user_attitudes(
            tweet_df,
            user_col=user_col,
        )
        graph = self.network_builder.build_graph(tweet_df, user_attitudes)
        graph_undirected = self.network_builder.to_undirected(graph)
        graph_stats = self.network_builder.graph_stats(graph)

        propagation_history = self.analyze_real_propagation(
            tweet_df,
            user_col=user_col,
            day_col=day_col,
            attitude_col="attitude_score",
        )
        final_day, propagated_attitudes = self._final_propagated_attitudes(
            propagation_history
        )
        user_attitudes["propagated_attitude"] = user_attitudes[user_col].map(
            propagated_attitudes
        )

        polarization = self.analyzer.measure_polarization(propagated_attitudes)
        homophily = self.analyzer.measure_homophily(
            graph_undirected,
            propagated_attitudes,
        )
        diversity_scores = self.analyzer.calculate_exposure_diversity(
            graph_undirected,
            propagated_attitudes,
        )
        user_attitudes["exposure_diversity"] = user_attitudes[user_col].map(
            diversity_scores
        )

        communities = self.analyzer.detect_communities(graph_undirected)
        modularity = (
            self.analyzer.compute_modularity(communities, graph_undirected)
            if graph_undirected.number_of_edges() > 0
            else 0
        )
        user_attitudes = self.analyzer.add_community_labels(
            user_attitudes,
            communities,
            user_col=user_col,
        )
        community_stats = self.analyzer.compute_community_stats(user_attitudes)
        echo_chamber_metrics = self.analyzer.compute_echo_chamber_score(
            polarization,
            homophily,
            user_attitudes["exposure_diversity"].dropna(),
            community_stats,
        )
        daily_polarization = {
            day: self.analyzer.measure_polarization(attitudes)
            for day, attitudes in propagation_history.items()
        }
        daily_homophily = {
            day: self.analyzer.measure_homophily(graph_undirected, attitudes)
            for day, attitudes in propagation_history.items()
        }
        optimized_weights, pca_model = self.analyzer.optimize_echo_chamber_weights(
            user_attitudes,
            community_stats,
            graph_undirected,
        )
        optimized_component_scores = self.analyzer.compute_optimized_component_scores(
            daily_polarization,
            daily_homophily,
            user_attitudes,
            community_stats,
            final_day=final_day,
        )
        echo_chamber_metrics_optimized = (
            self.analyzer.compute_optimized_echo_chamber_score(
                optimized_component_scores,
                optimized_weights,
            )
        )
        sensitivity_results = self.analyzer.sensitivity_analysis(
            echo_chamber_metrics_optimized,
            optimized_weights,
        )
        weights_2d, pca_model_2d, pca_metrics_df, pca_metrics_scaled = (
            self.analyzer.optimize_echo_chamber_weights_2d(
                user_attitudes,
                community_stats,
                graph_undirected,
            )
        )
        echo_chamber_metrics_optimized_2d = self.analyzer.compute_2d_optimized_score(
            echo_chamber_metrics_optimized,
            weights_2d,
        )
        landscape_df = self.analyzer.build_echo_chamber_landscape_dataframe(
            pca_metrics_df,
            pca_metrics_scaled,
            pca_model_2d,
            user_attitudes,
        )
        landscape_data = self.analyzer.build_echo_chamber_landscape_data(
            pca_metrics_df,
            pca_metrics_scaled,
            pca_model_2d,
            user_attitudes,
        )

        return {
            "tweets": tweet_df,
            "user_attitudes": user_attitudes,
            "graph": graph,
            "graph_undirected": graph_undirected,
            "graph_stats": graph_stats,
            "propagation_history": propagation_history,
            "final_day": final_day,
            "propagated_attitudes": propagated_attitudes,
            "polarization": polarization,
            "homophily": homophily,
            "diversity_scores": diversity_scores,
            "communities": communities,
            "modularity": modularity,
            "community_stats": community_stats,
            "echo_chamber_metrics": echo_chamber_metrics,
            "daily_polarization": daily_polarization,
            "daily_homophily": daily_homophily,
            "optimized_weights": optimized_weights,
            "pca_model": pca_model,
            "optimized_component_scores": optimized_component_scores,
            "echo_chamber_metrics_optimized": echo_chamber_metrics_optimized,
            "sensitivity_results": sensitivity_results,
            "weights_2d": weights_2d,
            "pca_model_2d": pca_model_2d,
            "pca_metrics_df": pca_metrics_df,
            "pca_metrics_scaled": pca_metrics_scaled,
            "echo_chamber_metrics_optimized_2d": echo_chamber_metrics_optimized_2d,
            "landscape_df": landscape_df,
            "landscape_data": landscape_data,
        }
