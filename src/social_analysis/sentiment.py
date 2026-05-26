"""Sentiment and emotion analysis helpers extracted from notebook 02."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from social_analysis.config import Config
from social_analysis.preprocessing import TextPreprocessor


SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
TRANSFORMER_BATCH = 32
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

SENTIMENT_LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "neg": "negative",
    "neu": "neutral",
    "pos": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


class VaderSentimentAnalyzer:
    """Add VADER sentiment scores and labels to text."""

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError("pip install vaderSentiment") from exc

        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def label(compound: float) -> str | float:
        """Map VADER compound score to the notebook label."""
        if pd.isna(compound):
            return np.nan
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    def predict(self, text: str) -> dict[str, float]:
        """Return raw VADER polarity scores for one text."""
        if text:
            return self.analyzer.polarity_scores(text)
        return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "compound": np.nan}

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
    ) -> pd.DataFrame:
        """Add VADER score and label columns to a DataFrame."""
        result = df.copy()
        scores = result[text_col].apply(self.predict)
        scores_df = pd.DataFrame(scores.tolist(), index=result.index)
        scores_df.columns = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
        result = pd.concat([result, scores_df], axis=1)
        result["vader_label"] = result["vader_compound"].apply(self.label)
        return result


class _TransformerClassifier:
    """Base class for notebook-style HuggingFace classifiers."""

    desc = "classifier"

    def __init__(
        self,
        model_name: str,
        batch_size: int = TRANSFORMER_BATCH,
        device: int = -1,
        classifier: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.classifier = classifier or self._load_classifier(model_name, device)

    @staticmethod
    def _load_classifier(model_name: str, device: int) -> Any:
        from transformers import AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=128,
            device=device,
        )

    def _run_classifier(self, texts: list[str]) -> list[Any | None]:
        valid_mask = [bool(t and t.strip()) for t in texts]
        valid_texts = [t for t, valid in zip(texts, valid_mask) if valid]

        preds: list[Any] = []
        for i in tqdm(
            range(0, len(valid_texts), self.batch_size),
            unit="batch",
            desc=self.desc,
        ):
            preds.extend(self.classifier(valid_texts[i : i + self.batch_size]))

        results: list[Any | None] = [None] * len(texts)
        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                results[i] = preds[j]
                j += 1
        return results


class TransformerSentimentAnalyzer(_TransformerClassifier):
    """Add RoBERTa sentiment probabilities, labels, and compound scores."""

    desc = "sentiment"

    def __init__(
        self,
        model_name: str = SENTIMENT_MODEL,
        batch_size: int = TRANSFORMER_BATCH,
        device: int = -1,
        classifier: Any | None = None,
    ) -> None:
        super().__init__(model_name, batch_size, device, classifier)

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """Predict notebook-style RoBERTa sentiment columns."""
        results = self._run_classifier(texts)
        rows: dict[str, list[Any]] = {
            key: []
            for key in [
                "roberta_neg",
                "roberta_neu",
                "roberta_pos",
                "roberta_label",
                "roberta_score",
            ]
        }

        for result in results:
            if result is None:
                for key in rows:
                    rows[key].append(np.nan)
                continue

            score_dict = {
                SENTIMENT_LABEL_MAP.get(item["label"].lower(), item["label"].lower()): item[
                    "score"
                ]
                for item in result
            }
            best = max(result, key=lambda item: item["score"])
            rows["roberta_neg"].append(score_dict.get("negative", np.nan))
            rows["roberta_neu"].append(score_dict.get("neutral", np.nan))
            rows["roberta_pos"].append(score_dict.get("positive", np.nan))
            rows["roberta_label"].append(
                SENTIMENT_LABEL_MAP.get(best["label"].lower(), best["label"].lower())
            )
            rows["roberta_score"].append(best["score"])

        predictions = pd.DataFrame(rows)
        predictions["roberta_compound"] = (
            predictions["roberta_pos"] - predictions["roberta_neg"]
        )
        return predictions

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
    ) -> pd.DataFrame:
        """Add transformer sentiment columns to a DataFrame."""
        result = df.copy()
        predictions = self.predict(result[text_col].tolist())
        predictions.index = result.index
        for column in predictions.columns:
            result[column] = predictions[column]
        return result


class EmotionAnalyzer(_TransformerClassifier):
    """Add emotion probabilities, labels, valence, and arousal."""

    desc = "emotion"

    def __init__(
        self,
        model_name: str = EMOTION_MODEL,
        batch_size: int = TRANSFORMER_BATCH,
        device: int = -1,
        classifier: Any | None = None,
        emotion_labels: list[str] | None = None,
    ) -> None:
        super().__init__(model_name, batch_size, device, classifier)
        self.emotion_labels = emotion_labels or EMOTION_LABELS

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """Predict notebook-style emotion columns."""
        results = self._run_classifier(texts)
        rows: dict[str, list[Any]] = {f"emotion_{label}": [] for label in self.emotion_labels}
        rows["emotion_label"] = []
        rows["emotion_score"] = []

        for result in results:
            if result is None:
                for key in rows:
                    rows[key].append(np.nan)
                continue

            score_dict = {item["label"].lower(): item["score"] for item in result}
            best = max(result, key=lambda item: item["score"])
            for label in self.emotion_labels:
                rows[f"emotion_{label}"].append(score_dict.get(label, np.nan))
            rows["emotion_label"].append(best["label"].lower())
            rows["emotion_score"].append(best["score"])

        predictions = pd.DataFrame(rows)
        unpleasant = predictions[
            ["emotion_sadness", "emotion_fear", "emotion_disgust", "emotion_anger"]
        ].mean(axis=1)
        predictions["emotion_valence"] = predictions["emotion_joy"] - unpleasant

        activated = predictions[
            ["emotion_anger", "emotion_fear", "emotion_surprise", "emotion_joy"]
        ].mean(axis=1)
        deactivated = predictions[
            ["emotion_sadness", "emotion_disgust", "emotion_neutral"]
        ].mean(axis=1)
        predictions["emotion_arousal"] = activated - deactivated
        return predictions

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
    ) -> pd.DataFrame:
        """Add transformer emotion columns to a DataFrame."""
        result = df.copy()
        predictions = self.predict(result[text_col].tolist())
        predictions.index = result.index
        for column in predictions.columns:
            result[column] = predictions[column]
        return result


class SentimentEmotionPipeline:
    """Run preprocessing, VADER, transformer sentiment, and emotion analysis."""

    def __init__(
        self,
        config: Config | dict[str, Any] | None = None,
        preprocessor: TextPreprocessor | None = None,
        vader_analyzer: VaderSentimentAnalyzer | None = None,
        sentiment_analyzer: TransformerSentimentAnalyzer | None = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
        device: int = -1,
    ) -> None:
        self.config = config or Config()
        self.preprocessor = preprocessor or TextPreprocessor(keep_emojis=True)
        self.vader_analyzer = vader_analyzer or VaderSentimentAnalyzer()

        models = self.config.get("models", {}) if hasattr(self.config, "get") else {}
        self.sentiment_analyzer = sentiment_analyzer or TransformerSentimentAnalyzer(
            model_name=models.get("sentiment_model", SENTIMENT_MODEL),
            batch_size=TRANSFORMER_BATCH,
            device=device,
        )
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer(
            model_name=models.get("emotion_model", EMOTION_MODEL),
            batch_size=TRANSFORMER_BATCH,
            device=device,
        )

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "tweet",
    ) -> pd.DataFrame:
        """Run the full notebook sentiment/emotion workflow."""
        result = self.preprocessor.transform_dataframe(df, text_col, "clean_text")
        result["is_empty"] = result["clean_text"].str.strip() == ""
        result = self.vader_analyzer.transform_dataframe(result, "clean_text")
        result = self.sentiment_analyzer.transform_dataframe(result, "clean_text")
        result = self.emotion_analyzer.transform_dataframe(result, "clean_text")
        return result
