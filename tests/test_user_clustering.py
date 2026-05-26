from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.user_clustering import FeatureExtractor


class FakeLemmatizer:
    def lemmatize(self, token):
        return {"cars": "car", "running": "run"}.get(token, token)


def simple_tokenizer(text):
    return text.split()


def make_extractor():
    return FeatureExtractor(
        stop_words={"and", "the"},
        lemmatizer=FakeLemmatizer(),
        tokenizer=simple_tokenizer,
        readability_func=lambda text: 42.0,
    )


def test_clean_tweet_matches_notebook_rules():
    text = "Hello @user! Visit https://example.com #AI, now."

    assert FeatureExtractor.clean_tweet(text) == "hello visit now"


def test_preprocess_for_lexical_features_matches_notebook_rules():
    extractor = make_extractor()

    result = extractor.preprocess_for_lexical_features("The cars, and running!")

    assert result == "car run"


def test_mtld_returns_zero_for_empty_or_single_token():
    assert FeatureExtractor.mtld([]) == 0.0
    assert FeatureExtractor.mtld(["one"]) == 0.0


def test_shannon_entropy_uses_token_frequencies():
    result = FeatureExtractor.shannon_entropy(["a", "a", "b", "b"])

    assert result == pytest.approx(1.0)


def test_lexical_features_short_text_returns_none_metrics():
    extractor = make_extractor()

    result = extractor.lexical_features("one two three four")

    assert result == {"mtld": None, "entropy": None, "token_count": 4}


def test_lexical_features_long_text_returns_metrics():
    extractor = make_extractor()

    result = extractor.lexical_features("one two three four five")

    assert result["token_count"] == 5
    assert result["mtld"] is not None
    assert result["entropy"] == pytest.approx(2.321928094887362)


def test_liwc_features_match_notebook_formulas():
    extractor = make_extractor()

    result = extractor.liwc_features("I think you agree! I do...")

    assert result["pronoun_ratio"] == pytest.approx(2 / (1 + 1e-5))
    assert result["avg_sent_length"] == pytest.approx(6 / 2)
    assert result["readability_score"] == 42.0
    assert result["exclamation_freq"] == pytest.approx(1 / 6)
    assert result["ellipsis_freq"] == pytest.approx(1 / 6)


def test_liwc_features_empty_text_returns_zeroes():
    extractor = make_extractor()

    result = extractor.liwc_features("")

    assert result.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_emoji_and_punctuation_rates_match_notebook_formulas():
    assert FeatureExtractor.emoji_rate("a😊!") == pytest.approx(1 / 3)
    assert FeatureExtractor.punctuation_rate("a😊!") == pytest.approx(1 / 3)


def test_transform_adds_feature_columns_without_clustering():
    extractor = make_extractor()
    df = pd.DataFrame(
        {
            "user_id": [1],
            "tweet": ["The cars, and running! I agree... 😊"],
        }
    )

    result = extractor.transform(df)

    expected_columns = {
        "clean_tweet",
        "cleaned_tweet",
        "tokens",
        "mtld",
        "entropy",
        "token_count",
        "pronoun_ratio",
        "avg_sent_length",
        "readability_score",
        "exclamation_freq",
        "ellipsis_freq",
        "emoji_rate",
        "punctuation_rate",
        "length",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "cleaned_tweet"] == "car run i agree 😊"


def test_aggregate_user_rates_matches_notebook_names():
    extractor = make_extractor()
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "emoji_rate": [0.0, 0.2, 0.5],
            "punctuation_rate": [0.1, 0.3, 0.4],
        }
    )

    result = extractor.aggregate_user_rates(df)

    assert result.columns.tolist() == ["user_id", "mean_emoji_rate", "mean_punct_rate"]
    assert result.loc[result["user_id"] == 1, "mean_emoji_rate"].iloc[0] == pytest.approx(0.1)
    assert result.loc[result["user_id"] == 1, "mean_punct_rate"].iloc[0] == pytest.approx(0.2)
