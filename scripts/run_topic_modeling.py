"""Run topic modeling analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.topic_modeling import TopicModelingPipeline
from social_analysis.visualization import TopicModelingVisualizer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load tweets/sentiment results and save topic-modeling outputs."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])
    outputs_dir = resolve_project_path(paths["outputs"]) / "05_topic_modeling"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    html_dir = outputs_dir / "html"
    loader.ensure_directory(tables_dir)
    loader.ensure_directory(plots_dir)
    loader.ensure_directory(html_dir)

    tweets_path = processed_dir / "tweets.csv"
    sentiment_path = processed_dir / "sentiment_results.csv"
    if not tweets_path.exists():
        raise FileNotFoundError(f"Expected input not found: {tweets_path}. Run scripts/run_eda.py first.")
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"Expected input not found: {sentiment_path}. Run scripts/run_sentiment.py first."
        )

    tweets = loader.load_csv(tweets_path)
    sentiment = loader.load_csv(sentiment_path)
    pipeline = TopicModelingPipeline(config=config)
    results = pipeline.run(tweets, text_col=config["columns"]["text"])
    topic_df = results["data"]
    topic_drift = pipeline.build_topic_drift(topic_df, sentiment)

    loader.save_dataframe(topic_df, processed_dir / "topicmodel_tweets.csv")
    loader.save_dataframe(topic_drift, processed_dir / "topic_drift.csv")
    loader.save_dataframe(topic_df, tables_dir / "topicmodel_tweets.csv")
    loader.save_dataframe(topic_drift, tables_dir / "topic_drift.csv")
    loader.save_dataframe(results["bertopic_topic_info"], tables_dir / "bertopic_topic_info.csv")
    loader.save_dataframe(
        pd.DataFrame(results["lda_top_words"], columns=["topic_id", "top_words"]),
        tables_dir / "lda_top_words.csv",
    )
    pd.DataFrame(
        [
            {
                "lda_coherence": results["lda_coherence"],
                "lda_log_perplexity": results["lda_log_perplexity"],
            }
        ]
    ).to_csv(tables_dir / "topic_modeling_metrics.csv", index=False)

    visualizer = TopicModelingVisualizer()
    visualizer.plot_topic_counts(topic_df, plots_dir)
    visualizer.save_pyldavis(
        results["lda_model"],
        results["corpus"],
        results["dictionary"],
        html_dir,
    )


if __name__ == "__main__":
    main()
