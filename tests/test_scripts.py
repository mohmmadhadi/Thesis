from pathlib import Path
import importlib
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.run_eda",
        "scripts.run_sentiment",
        "scripts.run_conversation_dynamics",
        "scripts.run_coherence",
        "scripts.run_topic_modeling",
        "scripts.run_user_clustering",
        "scripts.run_echo_chamber",
    ],
)
def test_script_imports_and_exposes_main(module_name: str) -> None:
    """Smoke-test script imports without running pipeline entry points."""
    module = importlib.import_module(module_name)

    assert callable(module.main)
