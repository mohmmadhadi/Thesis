"""Conversation dynamics analysis helpers extracted from notebook 03."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from social_analysis.config import Config


DEFAULT_SIGNAL = "roberta_compound"
SECONDARY_SIGNAL = "vader_compound"
EMOTION_SIGNALS = ["emotion_valence", "emotion_arousal"]
EMOTION_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

MIN_TURNS_DYNAMICS = 3
SLOPE_THRESH = 0.05
VOL_THRESH = 0.25
CPD_MODEL = "rbf"
CPD_PENALTY = 3


class ConversationDynamicsAnalyzer:
    """Compute tweet-level and thread-level conversation dynamics."""

    def __init__(
        self,
        signal: str = DEFAULT_SIGNAL,
        secondary_signal: str = SECONDARY_SIGNAL,
        emotion_signals: list[str] | None = None,
        emotion_classes: list[str] | None = None,
        min_turns_dynamics: int = MIN_TURNS_DYNAMICS,
        slope_thresh: float = SLOPE_THRESH,
        vol_thresh: float = VOL_THRESH,
        cpd_model: str = CPD_MODEL,
        cpd_penalty: float = CPD_PENALTY,
        config: Config | dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.signal = signal
        self.secondary_signal = secondary_signal
        self.emotion_signals = emotion_signals or EMOTION_SIGNALS
        self.emotion_classes = emotion_classes or EMOTION_CLASSES
        self.min_turns_dynamics = min_turns_dynamics
        self.slope_thresh = slope_thresh
        self.vol_thresh = vol_thresh
        self.cpd_model = cpd_model
        self.cpd_penalty = cpd_penalty

        if config is not None and hasattr(config, "get"):
            conversation = config.get("conversation", {})
            self.cpd_penalty = conversation.get("changepoint_penalty", self.cpd_penalty)

    def validate_required_columns(self, df: pd.DataFrame, sort_col: str) -> None:
        """Raise if required thread and sort columns are missing."""
        required = {"thread_id", sort_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def load_and_validate(self, df: pd.DataFrame, sort_col: str = "round") -> pd.DataFrame:
        """Sort by thread/order and add turn, thread length, and relative turn."""
        self.validate_required_columns(df, sort_col)
        result = df.sort_values(["thread_id", sort_col]).reset_index(drop=True).copy()
        result["turn"] = result.groupby("thread_id").cumcount()
        result["thread_len"] = result.groupby("thread_id")["turn"].transform("max") + 1
        result["turn_rel"] = result["turn"] / (result["thread_len"] - 1).clip(lower=1)
        return result

    def compute_tweet_level_features(
        self,
        df: pd.DataFrame,
        signal: str | None = None,
    ) -> pd.DataFrame:
        """Add rolling features for a sentiment signal."""
        signal = signal or self.signal
        if signal not in df.columns:
            return df

        result = df.copy().sort_values(["thread_id", "turn"])
        grouped = result.groupby("thread_id")[signal]

        result[f"{signal}_delta"] = grouped.diff()
        result[f"{signal}_roll3_mean"] = grouped.transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        result[f"{signal}_roll3_std"] = grouped.transform(
            lambda series: series.rolling(3, min_periods=2).std()
        )

        if self.signal in result.columns and self.secondary_signal in result.columns:
            result["sentiment_agreement"] = (
                result[self.signal] - result[self.secondary_signal]
            ).abs()
            result["low_agreement"] = result["sentiment_agreement"] > 0.4

        return result

    @staticmethod
    def linear_trend(values: np.ndarray) -> tuple[float, float]:
        """Return slope and R-squared of an OLS fit over turn indices."""
        if len(values) < 2:
            return np.nan, np.nan
        x = np.arange(len(values), dtype=float)
        result = stats.linregress(x, values)
        return result.slope, result.rvalue**2

    def detect_changepoints(self, values: np.ndarray, penalty: float | None = None) -> list[int]:
        """Detect structural break turns with ruptures PELT."""
        try:
            import ruptures as rpt
        except ImportError:
            return []
        if len(values) < 4:
            return []
        try:
            algo = rpt.Pelt(model=self.cpd_model, min_size=2, jump=1).fit(
                values.reshape(-1, 1).astype(float)
            )
            breaks = algo.predict(pen=self.cpd_penalty if penalty is None else penalty)
            return [breakpoint - 1 for breakpoint in breaks if breakpoint < len(values)]
        except Exception:
            return []

    def classify_arc_type(
        self,
        slope: float,
        volatility: float,
        n_changepoints: int,
    ) -> str:
        """Classify a thread's emotional arc using notebook thresholds."""
        if pd.isna(slope) or pd.isna(volatility):
            return "unknown"
        if volatility > self.vol_thresh:
            return "chaotic"
        if n_changepoints == 1:
            return "V-shape" if slope > 0 else "inverted-V"
        if abs(slope) < self.slope_thresh:
            return "stable"
        return "escalating" if slope > 0 else "de-escalating"

    def compute_thread_summaries(
        self,
        df: pd.DataFrame,
        signal: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate one row per thread with sentiment, emotion, and arc metrics."""
        signal = signal or self.signal
        records: list[dict[str, Any]] = []

        for thread_id, group in df.groupby("thread_id"):
            group = group.sort_values("turn")
            n_turns = len(group)
            record: dict[str, Any] = {
                "thread_id": thread_id,
                "thread_len": n_turns,
                "short_thread": n_turns < self.min_turns_dynamics,
            }

            if signal in group.columns and group[signal].notna().sum() >= 2:
                values = group[signal].dropna().values
                slope, r2 = self.linear_trend(values)
                changepoints_for_arc = self.detect_changepoints(values)
                record.update(
                    {
                        "mean_sentiment": float(np.mean(values)),
                        "std_sentiment": float(np.std(values)),
                        "trend_slope": float(slope),
                        "trend_r2": float(r2),
                        "sentiment_start": float(values[0]),
                        "sentiment_end": float(values[-1]),
                        "sentiment_delta": float(values[-1] - values[0]),
                        "arc_type": self.classify_arc_type(
                            slope,
                            float(np.std(values)),
                            len(changepoints_for_arc),
                        ),
                    }
                )
                if n_turns >= self.min_turns_dynamics:
                    changepoints = self.detect_changepoints(values)
                    record["n_changepoints"] = len(changepoints)
                    record["changepoint_turns"] = str(changepoints)
                    record["first_changepoint_turn"] = (
                        changepoints[0] if changepoints else np.nan
                    )
                else:
                    record.update(
                        {
                            "n_changepoints": np.nan,
                            "changepoint_turns": np.nan,
                            "first_changepoint_turn": np.nan,
                        }
                    )
            else:
                record.update(
                    {
                        key: np.nan
                        for key in [
                            "mean_sentiment",
                            "std_sentiment",
                            "trend_slope",
                            "trend_r2",
                            "sentiment_start",
                            "sentiment_end",
                            "sentiment_delta",
                            "arc_type",
                            "n_changepoints",
                            "changepoint_turns",
                            "first_changepoint_turn",
                        ]
                    }
                )

            emotion_cols = [
                f"emotion_{emotion}"
                for emotion in self.emotion_classes
                if f"emotion_{emotion}" in group.columns
            ]
            if emotion_cols:
                emotion_means = group[emotion_cols].mean()
                dominant = emotion_means.idxmax().replace("emotion_", "")
                probs = emotion_means.values
                probs = probs / probs.sum() if probs.sum() > 0 else probs
                record["dominant_emotion"] = dominant
                record["emotion_entropy"] = float(-np.sum(probs * np.log(probs + 1e-9)))
                for col in emotion_cols:
                    record[f"{col}_mean"] = float(group[col].mean())

            for axis in self.emotion_signals:
                if axis in group.columns:
                    record[f"{axis}_mean"] = float(group[axis].mean())

            if "low_agreement" in group.columns:
                record["mean_low_agreement"] = float(group["low_agreement"].mean())

            records.append(record)

        return pd.DataFrame(records)


class ConversationDynamicsPipeline:
    """Run the full conversation dynamics workflow."""

    def __init__(
        self,
        analyzer: ConversationDynamicsAnalyzer | None = None,
        config: Config | dict[str, Any] | None = None,
        sort_col: str = "round",
        signal: str = DEFAULT_SIGNAL,
    ) -> None:
        self.config = config
        self.sort_col = sort_col
        self.signal = signal
        self.analyzer = analyzer or ConversationDynamicsAnalyzer(
            signal=signal,
            config=config,
        )

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return tweet-level dynamics and thread-level summaries."""
        tweet_level = self.analyzer.load_and_validate(df, sort_col=self.sort_col)
        tweet_level = self.analyzer.compute_tweet_level_features(
            tweet_level,
            signal=self.signal,
        )
        summaries = self.analyzer.compute_thread_summaries(tweet_level, signal=self.signal)
        return tweet_level, summaries
