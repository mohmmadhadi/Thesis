"""YAML configuration loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Load and expose project settings from a YAML file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else self._default_path()
        self.settings = self._load_yaml(self.path)

    def __getitem__(self, key: str) -> Any:
        """Return a top-level config value."""
        return self.settings[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level config value or a default."""
        return self.settings.get(key, default)

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config file: {path}") from exc

        if data is None:
            raise ValueError(f"YAML config file is empty: {path}")
        if not isinstance(data, Mapping):
            raise ValueError(f"YAML config file must contain a mapping: {path}")

        return dict(data)
