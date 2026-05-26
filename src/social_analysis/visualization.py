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
