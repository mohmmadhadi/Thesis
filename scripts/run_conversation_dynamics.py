"""Run conversation dynamics analysis."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.conversation_dynamics import ConversationDynamicsPipeline
from social_analysis.data_loader import DataLoader
from social_analysis.visualization import ConversationDynamicsVisualizer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load sentiment results and save conversation dynamics outputs."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])
    outputs_dir = resolve_project_path(paths["outputs"]) / "03_conversation_dynamics"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    loader.ensure_directory(tables_dir)
    loader.ensure_directory(plots_dir)

    input_path = processed_dir / "sentiment_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected input not found: {input_path}. Run scripts/run_sentiment.py first."
        )

    data = loader.load_csv(input_path, low_memory=False)
    pipeline = ConversationDynamicsPipeline(
        config=config,
        sort_col=config["columns"].get("round", "round"),
    )
    tweet_level, summaries = pipeline.run(data)
    loader.save_dataframe(tweet_level, processed_dir / "tweet_level_dynamics.csv")
    loader.save_dataframe(summaries, processed_dir / "thread_summaries.csv")
    loader.save_dataframe(tweet_level, tables_dir / "tweet_level_dynamics.csv")
    loader.save_dataframe(summaries, tables_dir / "thread_summaries.csv")

    visualizer = ConversationDynamicsVisualizer()
    signal = config.get("conversation", {}).get("signal", "roberta_compound")
    visualizer.plot_sample_threads(tweet_level, summaries, signal=signal, out_dir=plots_dir)
    visualizer.plot_arc_distribution(summaries, out_dir=plots_dir)
    visualizer.plot_emotion_heatmap(summaries, out_dir=plots_dir)
    visualizer.plot_sentiment_over_relative_position(tweet_level, signal=signal, out_dir=plots_dir)
    visualizer.plot_valence_arousal_by_arc(summaries, out_dir=plots_dir)
    visualizer.plot_changepoint_timing(summaries, out_dir=plots_dir)


if __name__ == "__main__":
    main()
