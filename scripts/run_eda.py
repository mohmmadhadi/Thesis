"""Run the EDA data-preparation step from the simulation SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.visualization import EDAVisualizer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_sqlite_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    """Load all tables from a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';",
            conn,
        )["name"].tolist()
        return {table: pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in tables}


def build_notebook_csvs(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build the core CSV outputs used by downstream notebooks."""
    required = {
        "post",
        "rounds",
        "interests",
        "post_topics",
        "mentions",
        "reactions",
        "hashtags",
        "post_hashtags",
        "user_mgmt",
    }
    missing = required - set(tables)
    if missing:
        raise ValueError(f"Missing required SQLite tables: {sorted(missing)}")

    posts = tables["post"]
    time = tables["rounds"]
    topics_titles = tables["interests"]
    topics = tables["post_topics"]
    mentions = tables["mentions"]
    reactions = tables["reactions"]
    hashtags = tables["hashtags"]
    post_hashtags = tables["post_hashtags"]
    user_demo = tables["user_mgmt"]

    tweets = posts.merge(time, left_on="round", right_on="id", how="inner")
    tweets = tweets.drop(columns=["id_y"]).rename(columns={"id_x": "id"})

    reaction_counts = reactions.groupby(["post_id", "type"]).size().unstack(fill_value=0)
    for col in ["like", "dislike"]:
        if col not in reaction_counts.columns:
            reaction_counts[col] = 0
    post_reaction_summary = reaction_counts[["like", "dislike"]].reset_index()

    tweets = tweets.merge(post_reaction_summary, left_on="id", right_on="post_id", how="left")
    tweets = tweets.drop(columns=["post_id"]).fillna(0)

    topics = (
        topics.merge(topics_titles, left_on="topic_id", right_on="iid", how="inner")
        .rename(columns={"interest": "topic"})
        .drop(columns=["iid", "id"])
    )

    post_hashtags = (
        post_hashtags.merge(hashtags, left_on="hashtag_id", right_on="id", how="inner")
        .drop(columns=["id_x", "id_y"])
    )

    topic_tweets = tweets.merge(topics, left_on="id", right_on="post_id", how="outer")
    tweets_and_topics = (
        topic_tweets.groupby("post_id")["topic"].apply(list).reset_index(name="true_topics")
    )
    tweets = tweets.merge(tweets_and_topics, left_on="id", right_on="post_id", how="left")
    tweets = tweets.drop(columns=["post_id"])

    return {
        "tweets.csv": tweets,
        "topic_tweets.csv": topic_tweets,
        "mentions.csv": mentions,
        "reactions.csv": reactions,
        "post_hashtags.csv": post_hashtags,
        "user_demo.csv": user_demo,
    }


def main() -> None:
    """Load raw SQLite data and save downstream CSV inputs."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    raw_dir = resolve_project_path(paths["raw_data"])
    processed_dir = resolve_project_path(paths["processed_data"])
    outputs_dir = resolve_project_path(paths["outputs"]) / "01_eda"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    loader.ensure_directory(processed_dir)
    loader.ensure_directory(outputs_dir)
    loader.ensure_directory(tables_dir)
    loader.ensure_directory(plots_dir)

    db_path = raw_dir / "local-test.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Expected SQLite input not found: {db_path}. "
            "TODO: place the simulation database in data/raw/ or update configs/default.yaml."
        )

    tables = load_sqlite_tables(db_path)
    csv_outputs = build_notebook_csvs(tables)
    for filename, dataframe in csv_outputs.items():
        loader.save_dataframe(dataframe, processed_dir / filename)
        loader.save_dataframe(dataframe, tables_dir / filename)

    visualizer = EDAVisualizer()
    visualizer.plot_all(
        csv_outputs["tweets.csv"],
        csv_outputs["topic_tweets.csv"],
        plots_dir,
    )


if __name__ == "__main__":
    main()
