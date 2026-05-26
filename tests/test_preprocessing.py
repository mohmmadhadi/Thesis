from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.preprocessing import TextPreprocessor


def test_clean_text_matches_shared_notebook_rules():
    preprocessor = TextPreprocessor()

    cleaned = preprocessor.clean_text(
        'Hello @user #Climate &amp; AI https://example.com  😊'
    )

    assert cleaned == "hello climate & ai 😊"


def test_clean_text_can_preserve_case_and_urls():
    preprocessor = TextPreprocessor(remove_urls=False, lowercase=False)

    cleaned = preprocessor.clean_text("Hello #AI www.example.com")

    assert cleaned == "Hello AI www.example.com"


def test_clean_text_can_remove_emojis():
    preprocessor = TextPreprocessor(keep_emojis=False)

    assert preprocessor.clean_text("Great work 😊 #AI") == "great work ai"


def test_clean_text_handles_empty_and_non_string_values():
    preprocessor = TextPreprocessor()

    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("   ") == ""
    assert preprocessor.clean_text(None) == ""


def test_transform_dataframe_returns_copy_with_output_column():
    preprocessor = TextPreprocessor()
    df = pd.DataFrame({"tweet": ["Hello @u #AI", "Visit http://example.com"]})

    result = preprocessor.transform_dataframe(df, "tweet")

    assert "clean_text" not in df.columns
    assert result["clean_text"].tolist() == ["hello ai", "visit"]


def test_transform_dataframe_supports_custom_output_column():
    preprocessor = TextPreprocessor()
    df = pd.DataFrame({"tweet": ["Hello #AI"]})

    result = preprocessor.transform_dataframe(df, "tweet", "processed_text")

    assert result["processed_text"].tolist() == ["hello ai"]
