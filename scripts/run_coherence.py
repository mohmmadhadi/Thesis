"""Run response coherence scoring."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.coherence import ResponseCoherencePipeline
from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.visualization import CoherenceVisualizer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load tweets.csv and save coherence result tables."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])
    raw_dir = resolve_project_path(paths["raw_data"])
    outputs_dir = resolve_project_path(paths["outputs"]) / "04_response_coherence"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    loader.ensure_directory(tables_dir)
    loader.ensure_directory(plots_dir)

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
    pipeline = ResponseCoherencePipeline(config=config)
    results, summary = pipeline.run(tweets)
    loader.save_dataframe(results, processed_dir / "coherence_results.csv")
    loader.save_dataframe(summary, processed_dir / "coherence_summary.csv")
    loader.save_dataframe(results, tables_dir / "coherence_results.csv")
    loader.save_dataframe(summary, tables_dir / "coherence_summary.csv")
    CoherenceVisualizer().plot_all(results, plots_dir)


if __name__ == "__main__":
    main()
