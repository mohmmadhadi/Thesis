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
