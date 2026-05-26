"""Response coherence scoring helpers extracted from notebook 04."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from social_analysis.config import Config


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERTSCORE_MODEL = "roberta-large"
CROSS_ENC_MODEL = "cross-encoder/stsb-roberta-base"

COSINE_LOW_THRESHOLD = 0.25
BERTSCORE_LOW_THRESHOLD = 0.85
CROSS_ENC_LOW_THRESHOLD = 0.50

BATCH_SIZE = 64
DEVICE = "cpu"

KEEP_COLUMNS = [
    "id",
    "thread_id",
    "round",
    "user_id",
    "root_id",
    "root_tweet",
    "tweet",
    "cosine_similarity",
    "bs_precision",
    "bs_recall",
    "bs_f1",
    "cross_encoder_score",
    "flag_low_cosine",
    "flag_low_bertscore",
    "flag_low_crossenc",
    "flag_consensus",
]


class CoherenceScorer:
    """Score reply coherence with embeddings, BERTScore, and a cross-encoder."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        bertscore_model_name: str = BERTSCORE_MODEL,
        cross_encoder_model_name: str = CROSS_ENC_MODEL,
        cosine_low_threshold: float = COSINE_LOW_THRESHOLD,
        bertscore_low_threshold: float = BERTSCORE_LOW_THRESHOLD,
        cross_encoder_low_threshold: float = CROSS_ENC_LOW_THRESHOLD,
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE,
        embedding_model: Any | None = None,
        cross_encoder_model: Any | None = None,
        load_embedding_model: bool = True,
        load_cross_encoder_model: bool = True,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.bertscore_model_name = bertscore_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cosine_low_threshold = cosine_low_threshold
        self.bertscore_low_threshold = bertscore_low_threshold
        self.cross_encoder_low_threshold = cross_encoder_low_threshold
        self.batch_size = batch_size
        self.device = device

        self.embedding_model = embedding_model
        if self.embedding_model is None and load_embedding_model:
            self.embedding_model = self._load_embedding_model()

        self.cross_encoder_model = cross_encoder_model
        if self.cross_encoder_model is None and load_cross_encoder_model:
            self.cross_encoder_model = self._load_cross_encoder_model()

    def _load_embedding_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_model_name, device=self.device)

    def _load_cross_encoder_model(self) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.cross_encoder_model_name, device=self.device)

    def compute_cosine_similarity(self, replies: pd.DataFrame) -> pd.Series:
        """Compute row-wise cosine similarity for root/reply text pairs."""
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()

        emb_reply = self.embedding_model.encode(
            replies["tweet"].tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb_root = self.embedding_model.encode(
            replies["root_tweet"].tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        scores = (np.asarray(emb_reply) * np.asarray(emb_root)).sum(axis=1)
        return pd.Series(scores, index=replies.index, name="cosine_similarity")

    def compute_bertscore(self, replies: pd.DataFrame) -> pd.DataFrame:
        """Compute BERTScore precision, recall, and F1 columns."""
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            replies["tweet"].tolist(),
            replies["root_tweet"].tolist(),
            model_type=self.bertscore_model_name,
            lang="en",
            device=self.device,
            batch_size=max(1, self.batch_size // 4),
            verbose=True,
        )
        return pd.DataFrame(
            {
                "bs_precision": precision.numpy(),
                "bs_recall": recall.numpy(),
                "bs_f1": f1.numpy(),
            },
            index=replies.index,
        )

    def compute_cross_encoder(self, replies: pd.DataFrame) -> pd.Series:
        """Compute sigmoid-normalized cross-encoder scores."""
        if self.cross_encoder_model is None:
            self.cross_encoder_model = self._load_cross_encoder_model()

        pairs = list(zip(replies["root_tweet"], replies["tweet"]))
        raw: list[float] = []
        for i in tqdm(range(0, len(pairs), self.batch_size)):
            predictions = self.cross_encoder_model.predict(pairs[i : i + self.batch_size])
            raw.extend(np.asarray(predictions).tolist())

        scores = 1 / (1 + np.exp(-np.array(raw)))
        return pd.Series(scores, index=replies.index, name="cross_encoder_score")

    @staticmethod
    def compute_composite_coherence(summary: pd.DataFrame) -> pd.Series:
        """Compute the notebook thread-level composite coherence score."""
        return (
            summary["mean_cosine"]
            + (summary["mean_bertscore_f1"] - 0.8) / 0.2
            + summary["mean_cross_encoder"]
        ) / 3

    @staticmethod
    def compute_row_composite(results: pd.DataFrame) -> pd.Series:
        """Compute the notebook plot-level row composite score."""
        bs_norm = ((results["bs_f1"] - 0.7) / 0.3).clip(0, 1)
        return (results["cosine_similarity"] + bs_norm + results["cross_encoder_score"]) / 3

    def flag_low_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-method low-coherence flags and consensus flag."""
        result = df.copy()
        result["flag_low_cosine"] = (
            result["cosine_similarity"] < self.cosine_low_threshold
        )
        result["flag_low_bertscore"] = result["bs_f1"] < self.bertscore_low_threshold
        result["flag_low_crossenc"] = (
            result["cross_encoder_score"] < self.cross_encoder_low_threshold
        )
        result["flag_consensus"] = (
            result["flag_low_cosine"].astype(int)
            + result["flag_low_bertscore"].astype(int)
            + result["flag_low_crossenc"].astype(int)
        ) >= 2
        return result

    def summarise_by_thread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate scored replies by thread and sort least coherent first."""
        summary = (
            df.groupby("thread_id")
            .agg(
                n_replies=("id", "count"),
                mean_cosine=("cosine_similarity", "mean"),
                min_cosine=("cosine_similarity", "min"),
                mean_bertscore_f1=("bs_f1", "mean"),
                min_bertscore_f1=("bs_f1", "min"),
                mean_cross_encoder=("cross_encoder_score", "mean"),
                min_cross_encoder=("cross_encoder_score", "min"),
                pct_flagged_consensus=("flag_consensus", "mean"),
            )
            .reset_index()
        )
        summary["composite_coherence"] = self.compute_composite_coherence(summary)
        summary.sort_values("composite_coherence", ascending=True, inplace=True)
        return summary

    def score_dataframe(self, replies: pd.DataFrame) -> pd.DataFrame:
        """Run all notebook coherence scorers and flags on prepared pairs."""
        result = replies.copy()
        result = result.assign(cosine_similarity=self.compute_cosine_similarity(result))
        result = result.join(self.compute_bertscore(result))
        result = result.assign(cross_encoder_score=self.compute_cross_encoder(result))
        return self.flag_low_coherence(result)


class ResponseCoherencePipeline:
    """Prepare reply pairs and run the response coherence workflow."""

    def __init__(
        self,
        config: Config | dict[str, Any] | None = None,
        scorer: CoherenceScorer | None = None,
        pairing_strategy: Literal["thread_root", "direct_parent"] = "thread_root",
        device: str = DEVICE,
    ) -> None:
        self.config = config or Config()
        self.pairing_strategy = pairing_strategy

        if scorer is not None:
            self.scorer = scorer
        else:
            models = self.config.get("models", {}) if hasattr(self.config, "get") else {}
            coherence = self.config.get("coherence", {}) if hasattr(self.config, "get") else {}
            self.scorer = CoherenceScorer(
                embedding_model_name=models.get("sentence_transformer", EMBEDDING_MODEL),
                bertscore_model_name=models.get("bertscore_model", BERTSCORE_MODEL),
                cross_encoder_model_name=models.get("cross_encoder", CROSS_ENC_MODEL),
                cosine_low_threshold=coherence.get(
                    "cosine_low_threshold", COSINE_LOW_THRESHOLD
                ),
                bertscore_low_threshold=coherence.get(
                    "bertscore_low_threshold", BERTSCORE_LOW_THRESHOLD
                ),
                cross_encoder_low_threshold=coherence.get(
                    "cross_encoder_low_threshold", CROSS_ENC_LOW_THRESHOLD
                ),
                device=device,
            )

    @staticmethod
    def prepare_thread_root_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pair each reply with the original root tweet of its thread."""
        data = df.copy()
        data["is_root"] = data["comment_to"] == -1
        if data["is_root"].sum() == 0:
            min_rounds = data.groupby("thread_id")["round"].transform("min")
            data["is_root"] = data["round"] == min_rounds

        roots = (
            data[data["is_root"]]
            .set_index("thread_id")[["id", "tweet"]]
            .rename(columns={"id": "root_id", "tweet": "root_tweet"})
        )
        replies = data[~data["is_root"] & data["thread_id"].notna()].copy()
        replies = replies.merge(roots, on="thread_id", how="inner")
        return roots, replies

    @staticmethod
    def prepare_direct_parent_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pair each reply with its direct parent tweet."""
        data = df.copy()
        data["is_root"] = data["comment_to"] == -1
        if data["is_root"].sum() == 0:
            min_rounds = data.groupby("thread_id")["round"].transform("min")
            data["is_root"] = data["round"] == min_rounds

        parent_ids = set(data.loc[data["comment_to"] != -1, "comment_to"])
        data["is_local_root"] = data["is_root"] | data["id"].isin(parent_ids)

        roots = (
            data[data["is_local_root"]]
            .set_index("id")[["thread_id", "tweet"]]
            .rename(columns={"tweet": "root_tweet"})
            .reset_index()
            .rename(columns={"id": "root_id"})
        )

        replies = data[~data["is_root"] & data["thread_id"].notna()].copy()
        replies = replies.merge(
            roots[["root_id", "root_tweet"]],
            left_on="comment_to",
            right_on="root_id",
            how="inner",
        )
        return roots, replies

    def prepare_pairs(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare reply/root pairs using the configured notebook strategy."""
        if self.pairing_strategy == "direct_parent":
            return self.prepare_direct_parent_pairs(df)
        return self.prepare_thread_root_pairs(df)

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full coherence workflow and return result columns."""
        _, replies = self.prepare_pairs(df)
        results = self.scorer.score_dataframe(replies)
        return results[[column for column in KEEP_COLUMNS if column in results.columns]]

    def summarize(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return the notebook thread-level coherence summary."""
        return self.scorer.summarise_by_thread(results)

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return response-level results and thread-level summary."""
        results = self.transform_dataframe(df)
        summary = self.summarize(results)
        return results, summary
