"""Run sentiment and emotion analysis."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.sentiment import SentimentEmotionPipeline


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load tweets.csv and save sentiment_results.csv."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])
    raw_dir = resolve_project_path(paths["raw_data"])

    input_path = processed_dir / "tweets.csv"
    if not input_path.exists():
        fallback = raw_dir / "tweets.csv"
        input_path = fallback if fallback.exists() else input_path
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected input not found: {input_path}. "
            "Run scripts/run_eda.py first or place tweets.csv in data/processed/."
        )

    tweets = loader.load_csv(input_path, low_memory=False)
    pipeline = SentimentEmotionPipeline(config=config)
    results = pipeline.transform_dataframe(tweets, text_col=config["columns"]["text"])
    loader.save_dataframe(results, processed_dir / "sentiment_results.csv")


if __name__ == "__main__":
    main()
