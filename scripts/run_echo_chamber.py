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
    outputs_dir = resolve_project_path(paths["outputs"])
    loader.ensure_directory(outputs_dir)

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
    loader.save_dataframe(results["community_stats"].reset_index(), outputs_dir / "echo_chamber_communities.csv")
    loader.save_dataframe(
        pd.DataFrame([results["graph_stats"]]),
        outputs_dir / "echo_chamber_graph_stats.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["polarization"]]),
        outputs_dir / "echo_chamber_polarization.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["homophily"]]),
        outputs_dir / "echo_chamber_homophily.csv",
    )
    loader.save_dataframe(
        pd.DataFrame([results["echo_chamber_metrics"]]),
        outputs_dir / "echo_chamber_metrics.csv",
    )
    with (outputs_dir / "echo_chamber_communities.json").open("w", encoding="utf-8") as handle:
        json.dump(results["communities"], handle, indent=2)
    # TODO: notebook plotting and topic-specific filtering have not been extracted into this script.


if __name__ == "__main__":
    main()
