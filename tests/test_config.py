from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.config import Config


def test_loads_default_config():
    config = Config()

    assert config["paths"]["raw_data"] == "data/raw/"
    assert config.get("models", {})["sentiment_model"] == (
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


def test_loads_custom_config_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("paths:\n  raw_data: custom/raw\n", encoding="utf-8")

    config = Config(path)

    assert config.path == path
    assert config["paths"]["raw_data"] == "custom/raw"


def test_get_returns_default_for_missing_key(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("paths: {}\n", encoding="utf-8")

    config = Config(path)

    assert config.get("missing", {"fallback": True}) == {"fallback": True}


def test_missing_config_raises_clear_file_not_found(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        Config(missing)


def test_empty_config_raises_value_error(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML config file is empty"):
        Config(path)


def test_invalid_yaml_raises_value_error(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("paths: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML config file"):
        Config(path)


def test_non_mapping_yaml_raises_value_error(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        Config(path)
