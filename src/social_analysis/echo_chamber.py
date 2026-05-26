"""Echo chamber stance-estimation helpers extracted from notebook 07."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


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
