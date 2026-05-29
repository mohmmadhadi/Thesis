"""Reusable plotting helpers extracted from notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from social_analysis.conversation_dynamics import (
    DEFAULT_SIGNAL,
    EMOTION_CLASSES,
    MIN_TURNS_DYNAMICS,
)


PLOT_STYLE = "seaborn-v0_8-whitegrid"
CMAP_EMOTION = "YlOrRd"
FIG_DPI = 150

ARC_COLORS = {
    "stable": "#4C9BE8",
    "escalating": "#2ECC71",
    "de-escalating": "#E74C3C",
    "chaotic": "#F39C12",
    "V-shape": "#9B59B6",
    "inverted-V": "#1ABC9C",
    "unknown": "#CCCCCC",
}


class ConversationDynamicsVisualizer:
    """Reusable plots for conversation dynamics outputs."""

    def __init__(
        self,
        plot_style: str = PLOT_STYLE,
        fig_dpi: int = FIG_DPI,
        arc_colors: dict[str, str] | None = None,
    ) -> None:
        self.plot_style = plot_style
        self.fig_dpi = fig_dpi
        self.arc_colors = arc_colors or ARC_COLORS

    def _safe_style(self) -> None:
        import matplotlib.pyplot as plt

        try:
            plt.style.use(self.plot_style)
        except Exception:
            plt.style.use("ggplot")

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plot_sample_threads(
        self,
        df: pd.DataFrame,
        summary: pd.DataFrame,
        signal: str = DEFAULT_SIGNAL,
        out_dir: str | Path = ".",
        n_per_arc: int = 2,
    ) -> Path | None:
        """Plot sample thread trajectories, grouped by arc type."""
        if signal not in df.columns:
            return None

        import matplotlib.pyplot as plt

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)

        arc_types = [arc for arc in summary["arc_type"].dropna().unique() if arc != "unknown"]
        sample_ids: list[Any] = []
        for arc in arc_types:
            sample_ids.extend(
                summary[summary["arc_type"] == arc]["thread_id"].head(n_per_arc).tolist()
            )
        if not sample_ids:
            return None

        ncols = 3
        nrows = int(np.ceil(len(sample_ids) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"Sample Thread Sentiment Trajectories  [{signal}]",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        for idx, thread_id in enumerate(sample_ids):
            ax = axes[idx // ncols][idx % ncols]
            group = df[df["thread_id"] == thread_id].sort_values("turn")
            row = summary[summary["thread_id"] == thread_id].iloc[0]
            arc = row.get("arc_type", "unknown")
            color = self.arc_colors.get(arc, "#888888")

            turns = group["turn"].values
            raw = group[signal].values
            roll_col = f"{signal}_roll3_mean"

            ax.plot(
                turns,
                raw,
                "o--",
                color=color,
                alpha=0.6,
                linewidth=1.2,
                markersize=4,
                label="raw",
            )
            if roll_col in group.columns:
                ax.plot(
                    turns,
                    group[roll_col].values,
                    "-",
                    color=color,
                    linewidth=2.2,
                    label="roll-3",
                )

            try:
                changepoints = eval(str(row.get("changepoint_turns", "[]")))
            except Exception:
                changepoints = []
            for changepoint in changepoints:
                ax.axvline(changepoint, color="black", linestyle=":", linewidth=1, alpha=0.7)

            if len(turns) >= 2 and not pd.isna(row.get("trend_slope")):
                x_fit = np.array([turns[0], turns[-1]], dtype=float)
                y_fit = row["sentiment_start"] + row["trend_slope"] * x_fit
                ax.plot(x_fit, y_fit, "--", color="grey", linewidth=1, alpha=0.5)

            ax.axhline(0, color="grey", linewidth=0.8, alpha=0.4)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"Thread {thread_id}  [{arc}]  n={len(group)}", fontsize=9)
            ax.set_xlabel("Turn", fontsize=8)
            ax.set_ylabel(signal, fontsize=8)
            ax.legend(fontsize=7, loc="lower right")

        for idx in range(len(sample_ids), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        output_path = output_dir / "sample_thread_trajectories.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path

    def plot_arc_distribution(
        self,
        summary: pd.DataFrame,
        out_dir: str | Path = ".",
    ) -> Path | None:
        """Plot arc type counts."""
        if "arc_type" not in summary.columns:
            return None

        import matplotlib.pyplot as plt

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)
        counts = summary["arc_type"].value_counts()
        colors = list(self.arc_colors.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            counts.index,
            counts.values,
            color=colors[: len(counts)],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Thread Arc Type Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Arc Type")
        ax.set_ylabel("Number of Threads")
        plt.tight_layout()
        output_path = output_dir / "arc_distribution.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path

    def plot_emotion_heatmap(
        self,
        summary: pd.DataFrame,
        out_dir: str | Path = ".",
    ) -> Path | None:
        """Plot average emotion probability per arc type."""
        emotion_cols = [f"emotion_{emotion}_mean" for emotion in EMOTION_CLASSES]
        available = [col for col in emotion_cols if col in summary.columns]
        if not available or "arc_type" not in summary.columns:
            return None

        import matplotlib.pyplot as plt
        import seaborn as sns

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)
        heat = (
            summary.groupby("arc_type")[available]
            .mean()
            .rename(columns=lambda col: col.replace("emotion_", "").replace("_mean", ""))
        )

        fig, ax = plt.subplots(figsize=(10, max(3, len(heat) * 0.8 + 1)))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap=CMAP_EMOTION,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "mean prob"},
        )
        ax.set_title("Average Emotion Profile per Arc Type", fontsize=13, fontweight="bold")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Arc Type")
        plt.tight_layout()
        output_path = output_dir / "emotion_heatmap_by_arc.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path

    def plot_sentiment_over_relative_position(
        self,
        df: pd.DataFrame,
        signal: str = DEFAULT_SIGNAL,
        out_dir: str | Path = ".",
    ) -> Path | None:
        """Plot average sentiment across relative thread position bins."""
        if signal not in df.columns or "turn_rel" not in df.columns:
            return None

        import matplotlib.pyplot as plt

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)
        data = df[df["thread_len"] >= MIN_TURNS_DYNAMICS].copy()
        data["pos_bin"] = pd.cut(
            data["turn_rel"],
            bins=10,
            labels=[f"{i / 10:.1f}" for i in range(10)],
        )
        agg = data.groupby("pos_bin")[signal].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(agg))
        ax.plot(x, agg["mean"], "o-", color="#4C9BE8", linewidth=2, markersize=6)
        ax.fill_between(
            x,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color="#4C9BE8",
            alpha=0.15,
            label="+-1 std",
        )
        ax.axhline(0, color="grey", linewidth=0.8, alpha=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["Start", "", "", "", "", "", "", "", "", "End"], fontsize=9)
        ax.set_ylim(-1, 1)
        ax.set_title(
            "Average Sentiment Across Relative Thread Position\n"
            "(0 = first turn, 1 = last turn)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Relative position in thread")
        ax.set_ylabel(signal)
        ax.legend(fontsize=9)
        plt.tight_layout()
        output_path = output_dir / "sentiment_over_thread_position.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path

    def plot_valence_arousal_by_arc(
        self,
        summary: pd.DataFrame,
        out_dir: str | Path = ".",
    ) -> Path | None:
        """Plot mean valence versus mean arousal, coloured by arc type."""
        x_col = "emotion_valence_mean"
        y_col = "emotion_arousal_mean"
        if x_col not in summary.columns or y_col not in summary.columns:
            return None

        import matplotlib.pyplot as plt

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)
        fig, ax = plt.subplots(figsize=(7, 6))
        for arc, group in summary.groupby("arc_type"):
            ax.scatter(
                group[x_col],
                group[y_col],
                label=arc,
                alpha=0.55,
                s=20,
                color=self.arc_colors.get(arc, "#888"),
            )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        for x_pos, y_pos, label in [
            (0.5, 0.5, "joy/excitement"),
            (-0.5, 0.5, "anger/fear"),
            (-0.5, -0.5, "sadness/boredom"),
            (0.5, -0.5, "calm/content"),
        ]:
            ax.text(
                x_pos,
                y_pos,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="grey",
                alpha=0.7,
            )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Valence  (joy <-> negative affect)", fontsize=10)
        ax.set_ylabel("Arousal  (activated <-> deactivated)", fontsize=10)
        ax.set_title(
            "Valence-Arousal Space by Arc Type\n(Russell's Circumplex)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        output_path = output_dir / "valence_arousal_by_arc.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path

    def plot_changepoint_timing(
        self,
        summary: pd.DataFrame,
        out_dir: str | Path = ".",
    ) -> Path | None:
        """Plot first changepoint relative position."""
        col = "first_changepoint_turn"
        if col not in summary.columns:
            return None
        data = summary[summary[col].notna() & summary["thread_len"].notna()].copy()
        if len(data) == 0:
            return None

        import matplotlib.pyplot as plt

        self._safe_style()
        output_dir = self._prepare_out_dir(out_dir)
        data["cp_rel"] = data[col] / (data["thread_len"] - 1).clip(lower=1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            data["cp_rel"],
            bins=20,
            color="#E74C3C",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
        )
        ax.axvline(
            data["cp_rel"].median(),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"median={data['cp_rel'].median():.2f}",
        )
        ax.set_xlabel("Relative position of first changepoint", fontsize=10)
        ax.set_ylabel("Number of threads", fontsize=10)
        ax.set_title(
            "When Do Tone Shifts Happen?\n"
            "(First changepoint, relative position)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        output_path = output_dir / "changepoint_timing.png"
        fig.savefig(output_path, dpi=self.fig_dpi, bbox_inches="tight")
        return output_path


class EDAVisualizer:
    """Reusable EDA plots from notebook 01."""

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _save(fig: Any, out_dir: str | Path, filename: str, dpi: int = FIG_DPI) -> Path:
        import matplotlib.pyplot as plt

        output_path = EDAVisualizer._prepare_out_dir(out_dir) / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_tweet_volume_engagement(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot daily tweet volume and total reactions."""
        if not {"day", "reaction_count"}.issubset(tweets.columns):
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        daily_volume = tweets.groupby("day").size().reset_index(name="tweet_count")
        daily_engagement = tweets.groupby("day")["reaction_count"].sum().reset_index(name="total_reactions")
        daily_stats = pd.merge(daily_volume, daily_engagement, on="day")

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        sns.lineplot(data=daily_stats, x="day", y="tweet_count", marker="o", label="Tweet volume", color="steelblue", ax=ax1)
        sns.lineplot(data=daily_stats, x="day", y="total_reactions", marker="x", label="Total reactions", color="tomato", ax=ax2)
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Number of tweets")
        ax2.set_ylabel("Total reactions")
        ax1.set_title("Tweet volume and engagement over time")
        fig.autofmt_xdate()
        fig.tight_layout()
        return self._save(fig, out_dir, "tweet_volume_engagement_over_time.png")

    def plot_anomaly_detection_daily(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot rolling mean spike detection for daily tweet volume."""
        if "day" not in tweets.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        daily_stats = tweets.groupby("day").size().reset_index(name="tweet_count")
        df_roll = daily_stats.sort_values("day").reset_index(drop=True)
        window = 3
        df_roll["rolling_mean"] = df_roll["tweet_count"].rolling(window, min_periods=1).mean()
        df_roll["rolling_std"] = df_roll["tweet_count"].rolling(window, min_periods=1).std().fillna(0)
        df_roll["upper_bound"] = df_roll["rolling_mean"] + 3 * df_roll["rolling_std"]
        df_roll["is_spike"] = df_roll["tweet_count"] > df_roll["upper_bound"]
        spikes = df_roll[df_roll["is_spike"]]

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=df_roll, x="day", y="tweet_count", label="Actual volume", color="steelblue", alpha=0.7, ax=ax)
        sns.lineplot(data=df_roll, x="day", y="rolling_mean", label="Rolling mean", color="green", linestyle="--", ax=ax)
        sns.lineplot(data=df_roll, x="day", y="upper_bound", label="Threshold (mean + 3 std)", color="orange", alpha=0.5, ax=ax)
        ax.scatter(spikes["day"], spikes["tweet_count"], color="red", s=100, zorder=5, label="Detected spike")
        ax.set_title("Anomaly detection - tweet volume (daily)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Number of tweets")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, out_dir, "anomaly_detection_daily.png")

    def plot_activity_by_hour(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot global activity by hour."""
        if "hour" not in tweets.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        hourly_counts = tweets.groupby("hour").size().reset_index(name="tweet_count")
        hourly_counts = pd.DataFrame({"hour": range(24)}).merge(hourly_counts, on="hour", how="left").fillna(0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=hourly_counts, x="hour", y="tweet_count", color="steelblue", ax=ax)
        ax.set_title("Global activity by hour of day")
        ax.set_xlabel("Hour (0-23)")
        ax.set_ylabel("Total tweets")
        fig.tight_layout()
        return self._save(fig, out_dir, "activity_by_hour.png")

    def plot_activity_by_topic(self, topic_tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot global activity by topic."""
        if "topic" not in topic_tweets.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        count_by_topic = topic_tweets.groupby("topic").size().reset_index(name="tweet_count")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(data=count_by_topic, x="topic", y="tweet_count", color="steelblue", ax=ax)
        ax.set_title("Global activity by topic")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Total tweets")
        fig.tight_layout()
        return self._save(fig, out_dir, "activity_by_topic.png")

    def plot_topic_trends(self, topic_tweets: pd.DataFrame, out_dir: str | Path) -> list[Path]:
        """Plot topic trends by day and by round."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        outputs: list[Path] = []
        if "topic" not in topic_tweets.columns:
            return outputs
        for time_col, filename in [("day", "topic_trends_by_day.png"), ("round", "topic_trends_by_round.png")]:
            if time_col not in topic_tweets.columns:
                continue
            trend = topic_tweets.groupby([time_col, "topic"]).size().reset_index(name="tweet_count")
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.lineplot(data=trend, x=time_col, y="tweet_count", hue="topic", marker="o", ax=ax)
            ax.set_title(f"Topic choice trend over time (by {time_col})")
            ax.set_xlabel(time_col.capitalize())
            ax.set_ylabel("Total tweets on topic")
            ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            outputs.append(self._save(fig, out_dir, filename))
        return outputs

    def plot_user_topic_trends(self, topic_tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot per-user topic trends."""
        if not {"user_id", "day", "topic"}.issubset(topic_tweets.columns):
            return None
        import math
        import matplotlib.pyplot as plt
        import seaborn as sns

        topic_trends = topic_tweets.groupby(["user_id", "day", "topic"]).size().reset_index(name="topic_count")
        unique_users = topic_trends["user_id"].dropna().unique()
        if len(unique_users) == 0:
            return None
        num_cols = 4
        num_rows = math.ceil(len(unique_users) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows), squeeze=False)
        axes_flat = axes.flatten()
        for i, user_id in enumerate(unique_users):
            user_data = topic_trends[topic_trends["user_id"] == user_id]
            sns.lineplot(data=user_data, x="day", y="topic_count", hue="topic", marker="o", ax=axes_flat[i])
            axes_flat[i].set_title(f"User {user_id}")
            axes_flat[i].set_xlabel("Day")
            axes_flat[i].set_ylabel("Posts")
            axes_flat[i].legend(title="Topic", loc="upper left", bbox_to_anchor=(1, 1))
            axes_flat[i].grid(True, linestyle="--", alpha=0.7)
        for j in range(len(unique_users), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.tight_layout()
        return self._save(fig, out_dir, "user_topic_trends.png")

    def plot_length_distributions(self, tweets: pd.DataFrame, topic_tweets: pd.DataFrame, out_dir: str | Path) -> list[Path]:
        """Plot notebook tweet length distributions."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        outputs: list[Path] = []
        if "tweet" not in tweets.columns:
            return outputs
        tweets_data = tweets.copy()
        topic_data = topic_tweets.copy()
        tweets_data["length"] = tweets_data["tweet"].astype(str).apply(len)
        if "tweet" in topic_data.columns:
            topic_data["length"] = topic_data["tweet"].fillna("").astype(str).apply(len)

        configs = []
        if "user_id" in tweets_data.columns:
            configs.append((tweets_data.groupby("user_id")["length"].mean().reset_index(name="avg_tweet_length"), "avg_tweet_length", "Average tweet length by user", "tweet_length_by_user.png"))
        if {"user_id", "thread_id"}.issubset(tweets_data.columns):
            configs.append((tweets_data.groupby(["user_id", "thread_id"])["length"].mean().reset_index(), "length", "Average tweet length by user and thread", "tweet_length_by_user_thread.png"))
        if {"user_id", "topic"}.issubset(topic_data.columns):
            configs.append((topic_data.groupby(["user_id", "topic"])["length"].mean().reset_index(), "length", "Average tweet length by user and topic", "tweet_length_by_user_topic.png"))

        for df_plot, col, title, filename in configs:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_plot[col], kde=True, bins=30, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Average tweet length (characters)")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.7)
            fig.tight_layout()
            outputs.append(self._save(fig, out_dir, filename))
        return outputs

    def plot_post_count_distribution(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot distribution of post counts by user."""
        if "user_id" not in tweets.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        post_dist = tweets.groupby("user_id").size().reset_index(name="post_count")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(post_dist["post_count"], kde=True, bins=30, ax=ax)
        ax.set_title("Distribution of post counts among users")
        ax.set_xlabel("Post count")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        return self._save(fig, out_dir, "post_count_distribution.png")

    def plot_all(self, tweets: pd.DataFrame, topic_tweets: pd.DataFrame, out_dir: str | Path) -> list[Path]:
        """Save all reusable EDA plots."""
        outputs = [
            self.plot_tweet_volume_engagement(tweets, out_dir),
            self.plot_anomaly_detection_daily(tweets, out_dir),
            self.plot_activity_by_hour(tweets, out_dir),
            self.plot_activity_by_topic(topic_tweets, out_dir),
            self.plot_user_topic_trends(topic_tweets, out_dir),
            self.plot_post_count_distribution(tweets, out_dir),
        ]
        paths = [path for path in outputs if path is not None]
        paths.extend(self.plot_topic_trends(topic_tweets, out_dir))
        paths.extend(self.plot_length_distributions(tweets, topic_tweets, out_dir))
        return paths


class SentimentEmotionVisualizer:
    """Reusable sentiment and emotion plots from notebooks 02 and 07."""

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plot_sentiment_distribution(self, df: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot sentiment score and label distributions."""
        if "roberta_compound" not in df.columns:
            return None
        import matplotlib.pyplot as plt

        output_path = self._prepare_out_dir(out_dir) / "sentiment_distribution.png"
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(df["roberta_compound"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Sentiment Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Sentiment Scores")
        ax1.axvline(x=0, color="r", linestyle="--", label="Neutral")
        ax1.legend()
        if "roberta_label" in df.columns:
            ax2 = fig.add_subplot(1, 2, 2)
            sentiment_labels = df["roberta_label"].value_counts()
            ax2.bar(sentiment_labels.index, sentiment_labels.values)
            ax2.set_xlabel("Sentiment Label")
            ax2.set_ylabel("Count")
            ax2.set_title("Sentiment Label Distribution")
            ax2.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_emotion_distribution(self, df: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot emotion label counts if emotion columns are available."""
        if "emotion_label" not in df.columns:
            return None
        import matplotlib.pyplot as plt

        output_path = self._prepare_out_dir(out_dir) / "emotion_label_distribution.png"
        counts = df["emotion_label"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index, counts.values, color="steelblue")
        ax.set_title("Emotion Label Distribution")
        ax.set_xlabel("Emotion Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_all(self, df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
        """Save all sentiment/emotion plots available from columns."""
        return [path for path in [self.plot_sentiment_distribution(df, out_dir), self.plot_emotion_distribution(df, out_dir)] if path is not None]


class CoherenceVisualizer:
    """Reusable response coherence plots from notebook 04."""

    C_TEAL = "#1D9E75"
    C_PURPLE = "#7F77DD"
    C_CORAL = "#D85A30"
    C_TEAL_L = "#9FE1CB"
    C_PURPLE_L = "#AFA9EC"
    C_CORAL_L = "#F0997B"
    C_RED = "#E24B4A"
    C_GRAY = "#888780"
    C_BG = "#FAFAF8"
    C_GRID = "#EEECEA"

    METRIC_COLORS = {
        "cosine_similarity": (C_TEAL, C_TEAL_L, 0.25, (0.0, 1.0)),
        "bs_f1": (C_PURPLE, C_PURPLE_L, 0.85, (0.7, 1.0)),
        "cross_encoder_score": (C_CORAL, C_CORAL_L, 0.50, (0.0, 1.0)),
    }
    METRIC_LABELS = {
        "cosine_similarity": "Cosine similarity",
        "bs_f1": "BERTScore F1",
        "cross_encoder_score": "Cross-encoder",
    }

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _score_columns(df: pd.DataFrame) -> list[str]:
        return [col for col in ["cosine_similarity", "bs_f1", "cross_encoder_score"] if col in df.columns]

    @classmethod
    def _apply_style(cls) -> None:
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "axes.facecolor": cls.C_BG,
                "figure.facecolor": "white",
                "axes.edgecolor": "#D3D1C7",
                "axes.linewidth": 0.6,
                "axes.grid": True,
                "grid.color": cls.C_GRID,
                "grid.linewidth": 0.5,
                "xtick.color": cls.C_GRAY,
                "ytick.color": cls.C_GRAY,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.titlepad": 10,
                "legend.fontsize": 9,
                "legend.frameon": False,
            }
        )

    @staticmethod
    def _add_stats_box(ax: Any, data: pd.Series) -> None:
        txt = f"mean  {data.mean():.3f}\nmedian {data.median():.3f}\nstd    {data.std():.3f}"
        ax.text(
            0.97,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#444441",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec="#D3D1C7",
                lw=0.5,
                alpha=0.85,
            ),
        )

    def plot_score_distributions(self, results: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot notebook score distributions with KDE and threshold shading."""
        if not set(self.METRIC_COLORS).issubset(results.columns):
            return None
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy.stats import gaussian_kde

        self._apply_style()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        fig.suptitle(
            "Response coherence -- score distributions",
            fontsize=13,
            fontweight="bold",
            color="#2C2C2A",
            y=1.01,
        )

        for ax, (col, (color, color_l, thresh, (lo, hi))) in zip(
            axes,
            self.METRIC_COLORS.items(),
        ):
            data = results[col].dropna()
            if data.empty:
                continue
            bins = np.linspace(lo, hi, 26)
            counts, edges = np.histogram(data, bins=bins)
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                ax.bar(
                    left,
                    count,
                    width=(right - left) * 0.92,
                    align="edge",
                    color=color_l if right <= thresh else color,
                    zorder=2,
                )
            try:
                kde = gaussian_kde(data, bw_method=0.12)
                kde_x = np.linspace(lo, hi, 300)
                ax.plot(
                    kde_x,
                    kde(kde_x) * len(data) * (bins[1] - bins[0]),
                    color=color,
                    lw=2,
                    zorder=4,
                )
            except Exception:
                pass

            ax.axvspan(lo, thresh, color=color_l, alpha=0.15, zorder=1)
            ax.axvline(thresh, color=self.C_RED, lw=1.5, ls="--", zorder=5)
            ax.axvline(data.mean(), color=color, lw=1.2, ls=":", zorder=5)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(self.METRIC_LABELS[col])
            ax.set_ylabel("Count" if ax is axes[0] else "")
            ax.set_title(self.METRIC_LABELS[col])
            self._add_stats_box(ax, data)

            n_flagged = int((data < thresh).sum())
            ax.text(
                0.03,
                0.97,
                f"{n_flagged} flagged ({n_flagged / len(data) * 100:.1f}%)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=self.C_RED,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="#F09595",
                    lw=0.5,
                    alpha=0.9,
                ),
            )
            ax.legend(
                handles=[
                    mpatches.Patch(color=color_l, label="below threshold"),
                    mpatches.Patch(color=color, label="above threshold"),
                    plt.Line2D([0], [0], color=self.C_RED, lw=1.5, ls="--", label=f"threshold {thresh}"),
                ],
                loc="upper left",
                fontsize=7.5,
            )

        output_path = self._prepare_out_dir(out_dir) / "fig1_score_distributions.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_score_correlations(self, results: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot notebook 3x3 score correlation matrix."""
        metrics = list(self.METRIC_COLORS.keys())
        if not set(metrics).issubset(results.columns):
            return None
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        self._apply_style()
        flag = results["flag_consensus"] if "flag_consensus" in results.columns else pd.Series(False, index=results.index)
        flagged = flag.astype(bool)
        not_flagged = ~flagged
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(
            "Score correlation matrix -- coloured by consensus flag",
            fontsize=12,
            fontweight="bold",
            color="#2C2C2A",
            y=1.01,
        )
        for i, row_m in enumerate(metrics):
            for j, col_m in enumerate(metrics):
                ax = axes[i][j]
                ax.set_facecolor(self.C_BG)
                if i == j:
                    color, _, _, (lo, hi) = self.METRIC_COLORS[row_m]
                    xs = np.linspace(lo, hi, 300)
                    for mask, line_color, line_style in [
                        (not_flagged, color, "-"),
                        (flagged, self.C_RED, "--"),
                    ]:
                        data = results.loc[mask, row_m].dropna()
                        if len(data) > 5:
                            density = gaussian_kde(data, bw_method=0.15)(xs)
                            ax.fill_between(xs, density, color=line_color, alpha=0.2)
                            ax.plot(xs, density, color=line_color, lw=1.5, ls=line_style)
                    ax.set_xlim(lo, hi)
                    ax.set_title(self.METRIC_LABELS[row_m], fontsize=9, pad=6)
                else:
                    ax.scatter(results.loc[not_flagged, col_m], results.loc[not_flagged, row_m], s=6, color=self.C_GRAY, alpha=0.35, linewidths=0, rasterized=True)
                    ax.scatter(results.loc[flagged, col_m], results.loc[flagged, row_m], s=8, color=self.C_RED, alpha=0.65, linewidths=0, rasterized=True)
                    corr = results[[col_m, row_m]].dropna().corr().iloc[0, 1]
                    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=8, va="top", color="#2C2C2A", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D3D1C7", lw=0.5, alpha=0.85))
                    ax.axvline(self.METRIC_COLORS[col_m][2], color=self.C_RED, lw=0.8, ls="--", alpha=0.6)
                    ax.axhline(self.METRIC_COLORS[row_m][2], color=self.C_RED, lw=0.8, ls="--", alpha=0.6)
                    lox, hix = self.METRIC_COLORS[col_m][3]
                    loy, hiy = self.METRIC_COLORS[row_m][3]
                    ax.set_xlim(lox, hix)
                    ax.set_ylim(loy, hiy)
                if i == 2:
                    ax.set_xlabel(self.METRIC_LABELS[col_m], fontsize=8)
                if j == 0:
                    ax.set_ylabel(self.METRIC_LABELS[row_m], fontsize=8)
                ax.tick_params(labelsize=7)
        fig.legend(
            handles=[
                plt.scatter([], [], s=20, color=self.C_GRAY, alpha=0.6, label="coherent"),
                plt.scatter([], [], s=20, color=self.C_RED, alpha=0.8, label="flagged"),
            ],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
        )
        output_path = self._prepare_out_dir(out_dir) / "fig2_score_correlations.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_decay(self, results: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot notebook mean +/- std coherence decay by conversation round."""
        metrics = list(self.METRIC_COLORS.keys())
        if "round" not in results.columns or not set(metrics).issubset(results.columns):
            return None
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        self._apply_style()
        agg = results.groupby("round")[metrics].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["round"] + [f"{metric}_{stat}" for metric in metrics for stat in ["mean", "std", "count"]]
        rounds = agg["round"].values
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        fig.suptitle(
            "Coherence decay by conversation round",
            fontsize=13,
            fontweight="bold",
            color="#2C2C2A",
            y=1.01,
        )
        for ax, (col, (color, _, thresh, _)) in zip(axes, self.METRIC_COLORS.items()):
            mu = agg[f"{col}_mean"].values
            sd = agg[f"{col}_std"].fillna(0).values
            ax.fill_between(rounds, mu - sd, mu + sd, color=color, alpha=0.12, label="+/-1 std")
            ax.plot(rounds, mu, color=color, lw=2.5, marker="o", markersize=5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5, zorder=4, label="mean")
            ax.axhline(thresh, color=self.C_RED, lw=1.2, ls="--", alpha=0.8, label=f"threshold {thresh}")
            if len(rounds) >= 3:
                z = np.polyfit(rounds, mu, 1)
                ax.plot(rounds, np.poly1d(z)(rounds), color=color, lw=1, ls=":", alpha=0.6, label=f"trend {z[0]:+.3f}/round")
            ax.set_xlim(rounds[0] - 0.3, rounds[-1] + 0.3)
            ax.set_xlabel("Round")
            ax.set_ylabel(self.METRIC_LABELS[col] if ax is axes[0] else "")
            ax.set_title(self.METRIC_LABELS[col])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=7.5, loc="upper right")
        output_path = self._prepare_out_dir(out_dir) / "fig3_coherence_decay.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_thread_heatmap(self, results: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot notebook composite coherence heatmap by thread and round."""
        required = {"thread_id", "round", "cosine_similarity", "bs_f1", "cross_encoder_score"}
        if not required.issubset(results.columns):
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._apply_style()
        data = results.copy()
        data["bs_norm"] = ((data["bs_f1"] - 0.7) / 0.3).clip(0, 1)
        data["composite"] = (data["cosine_similarity"] + data["bs_norm"] + data["cross_encoder_score"]) / 3
        pivot = (
            data[data["round"] <= 10]
            .groupby(["thread_id", "round"])[["composite"]]
            .mean()
            .unstack("round")
        )
        pivot.columns = pivot.columns.droplevel(0)
        if pivot.empty:
            return None
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        if len(pivot) > 50:
            half = 25
            pivot = pd.concat([pivot.head(half), pivot.tail(half)])
        fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.22 + 2)), constrained_layout=True)
        fig.suptitle(
            "Thread coherence heatmap (composite score per round)",
            fontsize=12,
            fontweight="bold",
            color="#2C2C2A",
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=sns.diverging_palette(10, 150, s=70, l=45, as_cmap=True),
            center=0.5,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.3,
            linecolor="#EEECEA",
            cbar_kws={"label": "composite coherence", "shrink": 0.6},
            yticklabels=False,
        )
        ax.set_xlabel("Conversation round")
        ax.set_ylabel(f"Threads (n={len(pivot)}, sorted by mean coherence)")
        worst_n = min(5, len(pivot) // 4)
        ax.axhline(worst_n, color=self.C_RED, lw=1.2, ls="--", alpha=0.7)
        ax.axhline(len(pivot) - worst_n, color=self.C_TEAL, lw=1.2, ls="--", alpha=0.7)
        ax.text(-0.5, worst_n / 2, "least\ncoherent", ha="right", va="center", fontsize=8, color=self.C_RED, transform=ax.get_yaxis_transform())
        ax.text(-0.5, len(pivot) - worst_n / 2, "most\ncoherent", ha="right", va="center", fontsize=8, color=self.C_TEAL, transform=ax.get_yaxis_transform())
        output_path = self._prepare_out_dir(out_dir) / "fig4_thread_heatmap.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_all(self, results: pd.DataFrame, out_dir: str | Path) -> list[Path]:
        """Save all response coherence plots."""
        return [path for path in [self.plot_score_distributions(results, out_dir), self.plot_score_correlations(results, out_dir), self.plot_decay(results, out_dir), self.plot_thread_heatmap(results, out_dir)] if path is not None]


class TopicModelingVisualizer:
    """Reusable topic modeling visualizations from notebook 05."""

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_pyldavis(self, lda_model: Any, corpus: Any, dictionary: Any, out_dir: str | Path) -> Path | None:
        """Save the notebook pyLDAvis visualization as HTML."""
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
        except ImportError:
            return None
        output_path = self._prepare_out_dir(out_dir) / "lda_pyldavis.html"
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, str(output_path))
        return output_path

    def plot_topic_counts(self, topic_df: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot BERTopic topic counts if topic assignments exist."""
        topic_col = "bertopic_topic" if "bertopic_topic" in topic_df.columns else "lda_topic"
        if topic_col not in topic_df.columns:
            return None
        import matplotlib.pyplot as plt

        counts = topic_df[topic_col].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index.astype(str), counts.values, color="steelblue")
        ax.set_title("Topic Distribution")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Number of Tweets")
        fig.tight_layout()
        output_path = self._prepare_out_dir(out_dir) / "topic_distribution.png"
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_lda_topic_distribution(
        self,
        topic_df: pd.DataFrame,
        out_dir: str | Path,
        topic_col: str = "lda_topic",
    ) -> Path | None:
        """Plot notebook LDA topic distribution across documents."""
        if topic_col not in topic_df.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_path = self._prepare_out_dir(out_dir) / "lda_topic_distribution.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=topic_df[topic_col], color="steelblue", ax=ax)
        ax.set_title("LDA -- Topic Distribution Across Documents")
        ax.set_xlabel("Topic ID")
        ax.set_ylabel("Number of Documents")
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_token_length_distribution(
        self,
        topic_df: pd.DataFrame,
        out_dir: str | Path,
        length_col: str = "length",
    ) -> Path | None:
        """Plot notebook token count distribution after preprocessing."""
        if length_col not in topic_df.columns:
            return None
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_path = self._prepare_out_dir(out_dir) / "token_length_distribution.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(topic_df[length_col], bins=30, kde=True, color="steelblue", ax=ax)
        ax.set_title("Distribution of Token Counts (after preprocessing)")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def save_bertopic_barchart_html(
        self,
        topic_model: Any,
        out_dir: str | Path,
        top_n_topics: int = 10,
    ) -> Path | None:
        """Save BERTopic top-word barchart HTML from notebook cell 20."""
        if topic_model is None or not hasattr(topic_model, "visualize_barchart"):
            return None
        output_path = self._prepare_out_dir(out_dir) / "bertopic_barchart.html"
        fig = topic_model.visualize_barchart(top_n_topics=top_n_topics)
        fig.write_html(output_path)
        return output_path

    def save_bertopic_topics_html(
        self,
        topic_model: Any,
        out_dir: str | Path,
    ) -> Path | None:
        """Save BERTopic 2D inter-topic distance map HTML from notebook cell 21."""
        if topic_model is None or not hasattr(topic_model, "visualize_topics"):
            return None
        output_path = self._prepare_out_dir(out_dir) / "bertopic_topics.html"
        fig = topic_model.visualize_topics()
        fig.write_html(output_path)
        return output_path


class UserClusteringVisualizer:
    """Reusable Plotly visualizations from notebook 06."""

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_tweet_clusters_html(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Save the tweet UMAP cluster scatter plot."""
        if not {"x", "y", "cluster_label"}.issubset(tweets.columns):
            return None
        import plotly.express as px

        data = tweets.copy()
        if "tweet" in data.columns:
            data["tweet_short"] = data["tweet"].astype(str).str[:90] + "..."
        output_path = self._prepare_out_dir(out_dir) / "tweet_clusters.html"
        hover_data = {"x": False, "y": False, "cluster_label": False}
        if "tweet_short" in data.columns:
            hover_data["tweet_short"] = True
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="cluster_label",
            hover_data=hover_data,
            title="Tweet Clusters (UMAP + HDBSCAN)",
            labels={"cluster_label": "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white",
            width=950,
            height=650,
        )
        fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))
        fig.update_layout(title_font_size=16, legend=dict(itemsizing="constant"))
        fig.write_html(output_path)
        return output_path

    def save_tweet_trait_scatter_html(
        self,
        tweets: pd.DataFrame,
        trait: str,
        out_dir: str | Path,
    ) -> Path | None:
        """Save notebook trait-specific tweet cluster scatter HTML."""
        required = {"x", "y", "cluster_label", trait}
        if not required.issubset(tweets.columns):
            return None
        import plotly.express as px

        data = tweets.copy()
        if "tweet_short" not in data.columns and "tweet" in data.columns:
            data["tweet_short"] = data["tweet"].astype(str).str[:90] + "..."
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
        unique_vals = sorted(data[trait].dropna().unique())
        symbol_map = {
            value: default_symbols[index % len(default_symbols)]
            for index, value in enumerate(unique_vals)
        }
        hover_data = {trait: True, "x": False, "y": False}
        if "tweet_short" in data.columns:
            hover_data["tweet_short"] = True

        output_path = self._prepare_out_dir(out_dir) / f"tweet_clusters_{trait}.html"
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="cluster_label",
            symbol=trait,
            symbol_map=symbol_map,
            hover_data=hover_data,
            title=f"Tweet Clusters -- colour: cluster | shape: {trait.upper()}",
            labels={"cluster_label": "Cluster", trait: trait.upper()},
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white",
            width=950,
            height=650,
        )
        fig.update_traces(
            marker=dict(
                size=9,
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            )
        )
        fig.write_html(output_path)
        return output_path

    def save_cluster_trait_heatmap_html(
        self,
        tweets: pd.DataFrame,
        trait: str,
        out_dir: str | Path,
    ) -> Path | None:
        """Save notebook cluster-by-trait annotated heatmap HTML."""
        required = {"cluster", "cluster_label", trait}
        if not required.issubset(tweets.columns):
            return None
        import plotly.graph_objects as go

        df_clean_tweets = tweets[tweets["cluster"] != -1].copy()
        ct = pd.crosstab(df_clean_tweets["cluster_label"], df_clean_tweets[trait])
        if ct.empty:
            return None

        z_text = []
        for row in ct.values:
            total = row.sum()
            z_text.append(
                [f"{value}<br>({value / total:.0%})" if total > 0 else "0" for value in row]
            )

        fig = go.Figure(
            go.Heatmap(
                z=ct.values,
                x=ct.columns.tolist(),
                y=ct.index.tolist(),
                colorscale="Blues",
                colorbar=dict(title="Count"),
            )
        )
        for i, row in enumerate(z_text):
            for j, text in enumerate(row):
                fig.add_annotation(
                    x=ct.columns[j],
                    y=ct.index[i],
                    text=text,
                    showarrow=False,
                    font=dict(size=11, color="black"),
                )
        fig.update_layout(
            title=f"Cluster x {trait.upper()} -- count + row %",
            xaxis_title=f"{trait.upper()}",
            yaxis_title="Cluster",
            template="plotly_white",
            width=750,
            height=500,
            font=dict(family="Inter, Arial, sans-serif"),
            margin=dict(l=120),
        )
        output_path = self._prepare_out_dir(out_dir) / f"heatmap_cluster_{trait}.html"
        fig.write_html(output_path)
        return output_path

    def save_trait_visualizations_html(
        self,
        tweets: pd.DataFrame,
        traits: list[str],
        out_dir: str | Path,
    ) -> list[Path]:
        """Save all notebook trait-specific tweet scatter and heatmap HTMLs."""
        outputs: list[Path] = []
        for trait in traits:
            scatter = self.save_tweet_trait_scatter_html(tweets, trait, out_dir)
            if scatter is not None:
                outputs.append(scatter)
            heatmap = self.save_cluster_trait_heatmap_html(tweets, trait, out_dir)
            if heatmap is not None:
                outputs.append(heatmap)
        return outputs

    def save_agent_clusters_html(self, agents: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Save the agent cluster scatter plot when PCA coordinates exist."""
        x_col = "PC1" if "PC1" in agents.columns else None
        y_col = "PC2" if "PC2" in agents.columns else None
        cluster_col = "Agent_Cluster" if "Agent_Cluster" in agents.columns else "hdbscan_cluster"
        if x_col is None or y_col is None or cluster_col not in agents.columns:
            return None
        import plotly.express as px

        output_path = self._prepare_out_dir(out_dir) / "agent_clusters.html"
        fig = px.scatter(
            agents,
            x=x_col,
            y=y_col,
            color=cluster_col,
            title="Agent Clusters (HDBSCAN on PCA-reduced behavioural features)",
            labels={x_col: "PC 1", y_col: "PC 2", cluster_col: "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="plotly_white",
            width=950,
            height=650,
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color="white")))
        fig.update_layout(title_font_size=16, legend=dict(itemsizing="constant"))
        fig.write_html(output_path)
        return output_path


class EchoChamberVisualizer:
    """Reusable echo chamber plots from notebook 07."""

    @staticmethod
    def _prepare_out_dir(out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _save(fig: Any, out_dir: str | Path, filename: str, dpi: int = FIG_DPI) -> Path:
        import matplotlib.pyplot as plt

        output_path = EchoChamberVisualizer._prepare_out_dir(out_dir) / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_attitude_scores(self, tweets: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot attitude distribution and stance/sentiment scatter."""
        if not {"attitude_score", "stance_ensemble", "sentiment_score"}.issubset(tweets.columns):
            return None
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(tweets["attitude_score"].dropna(), bins=50, edgecolor="black", alpha=0.7, color="purple")
        ax1.set_xlabel("Attitude Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Attitude Scores")
        ax1.axvline(x=0, color="r", linestyle="--", label="Neutral")
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        scatter = ax2.scatter(tweets["stance_ensemble"], tweets["sentiment_score"], c=tweets["attitude_score"], cmap="RdYlGn", alpha=0.6, s=10)
        ax2.set_xlabel("Stance Score")
        ax2.set_ylabel("Sentiment Score")
        ax2.set_title("Stance vs Sentiment (colored by Attitude)")
        fig.colorbar(scatter, ax=ax2, label="Attitude Score")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return self._save(fig, out_dir, "attitude_scores_distribution_vs_stance_sentiment.png")

    def plot_user_attitudes(self, user_attitudes: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot user-level attitude summaries."""
        if not {"mean_attitude", "num_posts", "attitude_std"}.issubset(user_attitudes.columns):
            return None
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.hist(user_attitudes["mean_attitude"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Mean Attitude")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of User Mean Attitudes")
        ax1.axvline(x=0, color="r", linestyle="--")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(user_attitudes["num_posts"], user_attitudes["mean_attitude"], alpha=0.6, s=50)
        ax2.set_xlabel("Number of Posts")
        ax2.set_ylabel("Mean Attitude")
        ax2.set_title("Posts vs Mean Attitude")
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        ax3 = fig.add_subplot(1, 3, 3)
        points = ax3.scatter(user_attitudes["attitude_std"], user_attitudes["mean_attitude"], alpha=0.6, s=50, c=user_attitudes["num_posts"], cmap="viridis")
        ax3.set_xlabel("Attitude Std Dev")
        ax3.set_ylabel("Mean Attitude")
        ax3.set_title("Attitude Consistency vs Position")
        fig.colorbar(points, ax=ax3, label="Num Posts")
        ax3.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, out_dir, "user_attitude_distribution.png")

    def plot_exposure_diversity(self, user_attitudes: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot exposure diversity distribution and relationship to attitude."""
        if "exposure_diversity" not in user_attitudes.columns:
            return None
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.hist(user_attitudes["exposure_diversity"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Exposure Diversity Score")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of Exposure Diversity")
        if "propagated_attitude" in user_attitudes.columns:
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.scatter(user_attitudes["propagated_attitude"], user_attitudes["exposure_diversity"], alpha=0.6, s=50)
            ax2.set_xlabel("User Attitude")
            ax2.set_ylabel("Exposure Diversity")
            ax2.set_title("Attitude vs Exposure Diversity")
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax3 = fig.add_subplot(1, 3, 3)
        categories = pd.cut(user_attitudes["exposure_diversity"], bins=3, labels=["Low", "Medium", "High"])
        counts = categories.value_counts()
        ax3.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax3.set_title("Users by Exposure Diversity Level")
        fig.tight_layout()
        return self._save(fig, out_dir, "exposure_diversity.png")

    def plot_community_stats(self, community_stats: pd.DataFrame, out_dir: str | Path) -> Path | None:
        """Plot community size, attitude, and diversity summaries."""
        if community_stats.empty:
            return None
        import matplotlib.pyplot as plt

        data = community_stats.copy()
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        top = data.head(10)
        ax1.barh(range(len(top)), top["size"])
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels([f"C{i}" for i in top.index])
        ax1.set_xlabel("Number of Users")
        ax1.set_ylabel("Community")
        ax1.set_title("Top 10 Communities by Size")
        if {"avg_attitude", "attitude_std"}.issubset(data.columns):
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.scatter(data["avg_attitude"], data["size"], s=data["attitude_std"].fillna(0) * 100, alpha=0.6, c=range(len(data)), cmap="tab20")
            ax2.set_xlabel("Average Community Attitude")
            ax2.set_ylabel("Community Size")
            ax2.set_title("Community Attitude vs Size\n(bubble size = attitude std)")
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        if {"avg_attitude", "avg_diversity"}.issubset(data.columns):
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.scatter(data["avg_attitude"], data["avg_diversity"], s=data["size"] * 2, alpha=0.6, c=range(len(data)), cmap="tab20")
            ax3.set_xlabel("Average Community Attitude")
            ax3.set_ylabel("Average Exposure Diversity")
            ax3.set_title("Community Attitude vs Diversity\n(bubble size = community size)")
            ax3.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return self._save(fig, out_dir, "community_stats.png")

    def plot_echo_chamber_scores(self, metrics: dict[str, float], out_dir: str | Path) -> Path | None:
        """Plot final echo chamber component scores."""
        if not metrics:
            return None
        import matplotlib.pyplot as plt

        components = ["polarization_score", "homophily_score", "diversity_score", "separation_score", "overall_score"]
        labels = ["Polarization", "Homophily", "Low Diversity", "Separation", "Overall"]
        scores = [metrics.get(component, np.nan) for component in components]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(labels, scores, color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#6c5ce7"], alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_title("Echo Chamber Component Scores")
        for bar, score in zip(bars, scores):
            if not pd.isna(score):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")
        fig.tight_layout()
        return self._save(fig, out_dir, "echo_chamber_analysis.png")

    def plot_all(self, results: dict[str, Any], out_dir: str | Path) -> list[Path]:
        """Save reusable echo chamber plots from pipeline outputs."""
        return [path for path in [
            self.plot_attitude_scores(results.get("tweets", pd.DataFrame()), out_dir),
            self.plot_user_attitudes(results.get("user_attitudes", pd.DataFrame()), out_dir),
            self.plot_exposure_diversity(results.get("user_attitudes", pd.DataFrame()), out_dir),
            self.plot_community_stats(results.get("community_stats", pd.DataFrame()), out_dir),
            self.plot_echo_chamber_scores(results.get("echo_chamber_metrics", {}), out_dir),
        ] if path is not None]
