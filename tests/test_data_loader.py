from pathlib import Path
import sqlite3
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.data_loader import DataLoader


def test_save_dataframe_creates_parent_directory_and_loads_csv(tmp_path):
    loader = DataLoader()
    df = pd.DataFrame({"id": [1, 2], "tweet": ["hello", "world"]})
    output_path = tmp_path / "nested" / "tweets.csv"

    saved_path = loader.save_dataframe(df, output_path)
    loaded = loader.load_csv(saved_path)

    assert saved_path == output_path
    assert output_path.exists()
    pd.testing.assert_frame_equal(loaded, df)


def test_save_dataframe_can_include_index(tmp_path):
    loader = DataLoader()
    df = pd.DataFrame({"tweet": ["hello"]}, index=pd.Index([42], name="row_id"))
    output_path = tmp_path / "with_index.csv"

    loader.save_dataframe(df, output_path, index=True)
    loaded = loader.load_csv(output_path)

    assert loaded.to_dict(orient="records") == [{"row_id": 42, "tweet": "hello"}]


def test_ensure_directory_returns_created_path(tmp_path):
    loader = DataLoader()
    directory = tmp_path / "plots"

    result = loader.ensure_directory(directory)

    assert result == directory
    assert directory.is_dir()


def test_load_sqlite_table_reads_full_table(tmp_path):
    loader = DataLoader()
    db_path = tmp_path / "local-test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE tweets (id INTEGER, tweet TEXT)")
        conn.executemany(
            "INSERT INTO tweets VALUES (?, ?)",
            [(1, "hello"), (2, "world")],
        )

    loaded = loader.load_sqlite_table(db_path, "tweets")

    expected = pd.DataFrame({"id": [1, 2], "tweet": ["hello", "world"]})
    pd.testing.assert_frame_equal(loaded, expected)


def test_config_is_copied_on_init():
    config = {"paths": {"raw_data": "data/raw"}}
    loader = DataLoader(config)
    config["new_key"] = "changed"

    assert loader.config == {"paths": {"raw_data": "data/raw"}}
