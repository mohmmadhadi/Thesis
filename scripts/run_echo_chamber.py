"""Run echo chamber analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.echo_chamber import EchoChamberPipeline
from social_analysis.visualization import EchoChamberVisualizer, SentimentEmotionVisualizer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load sentiment results and save echo-chamber analysis outputs."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])
    outputs_dir = resolve_project_path(paths["outputs"]) / "07_echo_chamber"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    loader.ensure_directory(tables_dir)
    loader.ensure_directory(plots_dir)

    input_path = processed_dir / "sentiment_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected input not found: {input_path}. Run scripts/run_sentiment.py first."
        )

    data = loader.load_csv(input_path)
    pipeline = EchoChamberPipeline(config=config)
    results = pipeline.run(
        data,
        text_col="clean_text",
        user_col=config["columns"]["user_id"],
        day_col=config["columns"]["day"],
    )

    loader.save_dataframe(results["tweets"], processed_dir / "echo_chamber_tweets.csv")
    loader.save_dataframe(results["user_attitudes"], processed_dir / "echo_chamber_user_attitudes.csv")
    loader.save_dataframe(results["tweets"], tables_dir / "echo_chamber_tweets.csv")
    loader.save_dataframe(results["user_attitudes"], tables_dir / "echo_chamber_user_attitudes.csv")
    loader.save_dataframe(results["community_stats"].reset_index(), tables_dir / "echo_chamber_communities.csv")
    loader.save_dataframe(
        pd.DataFrame([results["graph_stats"]]),
        tables_dir / "echo_chamber_graph_stats.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["polarization"]]),
        tables_dir / "echo_chamber_polarization.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["homophily"]]),
        tables_dir / "echo_chamber_homophily.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["echo_chamber_metrics"]]),
        tables_dir / "echo_chamber_metrics.csv",
    )
    with (tables_dir / "echo_chamber_communities.json").open("w", encoding="utf-8") as handle:
        json.dump(results["communities"], handle, indent=2)

    SentimentEmotionVisualizer().plot_sentiment_distribution(results["tweets"], plots_dir)
    EchoChamberVisualizer().plot_all(results, plots_dir)
    # TODO: notebook topic-specific filtering and advanced sensitivity/landscape plots need confirmed inputs.


if __name__ == "__main__":
    main()
