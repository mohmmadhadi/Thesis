from pathlib import Path
import importlib
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize(
    "module_name",
    [
        "social_analysis.config",
        "social_analysis.data_loader",
        "social_analysis.preprocessing",
        "social_analysis.sentiment",
        "social_analysis.coherence",
        "social_analysis.conversation_dynamics",
        "social_analysis.topic_modeling",
        "social_analysis.user_clustering",
        "social_analysis.echo_chamber",
        "social_analysis.visualization",
        "social_analysis.utils",
    ],
)
def test_module_imports_without_side_effects(module_name: str) -> None:
    """Smoke-test module imports without constructing heavy models or pipelines."""
    module = importlib.import_module(module_name)

    assert module is not None
