from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.conversation_dynamics import (
    ConversationDynamicsAnalyzer,
    ConversationDynamicsPipeline,
)


class NoChangepointAnalyzer(ConversationDynamicsAnalyzer):
    def detect_changepoints(self, values, penalty=None):
        return []


def test_validate_required_columns_raises_for_missing_sort_column():
    analyzer = ConversationDynamicsAnalyzer()
    df = pd.DataFrame({"thread_id": [1]})

    with pytest.raises(ValueError, match="Missing required columns"):
        analyzer.validate_required_columns(df, "round")


def test_load_and_validate_adds_turn_thread_len_and_relative_turn():
    analyzer = ConversationDynamicsAnalyzer()
    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 2],
            "round": [2, 1, 1],
            "roberta_compound": [0.2, 0.1, -0.1],
        }
    )

    result = analyzer.load_and_validate(df)

    assert result["round"].tolist() == [1, 2, 1]
    assert result["turn"].tolist() == [0, 1, 0]
    assert result["thread_len"].tolist() == [2, 2, 1]
    assert result["turn_rel"].tolist() == [0.0, 1.0, 0.0]


def test_compute_tweet_level_features_matches_notebook_columns():
    analyzer = ConversationDynamicsAnalyzer()
    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 1],
            "turn": [0, 1, 2],
            "roberta_compound": [0.0, 0.3, 0.6],
            "vader_compound": [0.0, -0.2, 0.5],
        }
    )

    result = analyzer.compute_tweet_level_features(df)

    assert result["roberta_compound_delta"].tolist()[1:] == [0.3, 0.3]
    assert result["roberta_compound_roll3_mean"].tolist() == [0.0, 0.15, 0.3]
    assert np.isnan(result.loc[0, "roberta_compound_roll3_std"])
    assert result["sentiment_agreement"].tolist() == pytest.approx([0.0, 0.5, 0.1])
    assert result["low_agreement"].tolist() == [False, True, False]


def test_arc_classification_matches_notebook_order():
    analyzer = ConversationDynamicsAnalyzer()

    assert analyzer.classify_arc_type(np.nan, 0.1, 0) == "unknown"
    assert analyzer.classify_arc_type(0.0, 0.3, 0) == "chaotic"
    assert analyzer.classify_arc_type(0.1, 0.1, 1) == "V-shape"
    assert analyzer.classify_arc_type(-0.1, 0.1, 1) == "inverted-V"
    assert analyzer.classify_arc_type(0.01, 0.1, 0) == "stable"
    assert analyzer.classify_arc_type(0.1, 0.1, 0) == "escalating"
    assert analyzer.classify_arc_type(-0.1, 0.1, 0) == "de-escalating"


def test_compute_thread_summaries_preserves_metrics_and_emotion_columns():
    analyzer = NoChangepointAnalyzer()
    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 1, 2, 2],
            "turn": [0, 1, 2, 0, 1],
            "roberta_compound": [0.0, 0.2, 0.4, -0.2, -0.1],
            "emotion_joy": [0.8, 0.7, 0.6, 0.1, 0.2],
            "emotion_anger": [0.1, 0.1, 0.1, 0.5, 0.4],
            "emotion_valence": [0.7, 0.6, 0.5, -0.4, -0.2],
            "emotion_arousal": [0.2, 0.3, 0.4, 0.6, 0.5],
            "low_agreement": [False, True, False, True, True],
        }
    )

    summary = analyzer.compute_thread_summaries(df)
    thread_one = summary[summary["thread_id"] == 1].iloc[0]
    thread_two = summary[summary["thread_id"] == 2].iloc[0]

    assert thread_one["thread_len"] == 3
    assert bool(thread_one["short_thread"]) is False
    assert thread_one["mean_sentiment"] == pytest.approx(0.2)
    assert thread_one["sentiment_delta"] == pytest.approx(0.4)
    assert thread_one["arc_type"] == "escalating"
    assert thread_one["n_changepoints"] == 0
    assert thread_one["changepoint_turns"] == "[]"
    assert thread_one["dominant_emotion"] == "joy"
    assert thread_one["emotion_joy_mean"] == pytest.approx(0.7)
    assert thread_one["emotion_valence_mean"] == pytest.approx(0.6)
    assert thread_one["mean_low_agreement"] == pytest.approx(1 / 3)
    assert bool(thread_two["short_thread"]) is True
    assert np.isnan(thread_two["n_changepoints"])


def test_pipeline_returns_tweet_level_and_summary():
    analyzer = NoChangepointAnalyzer()
    pipeline = ConversationDynamicsPipeline(analyzer=analyzer)
    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 1],
            "round": [0, 1, 2],
            "roberta_compound": [0.0, 0.1, 0.2],
        }
    )

    tweet_level, summary = pipeline.run(df)

    assert "roberta_compound_roll3_mean" in tweet_level.columns
    assert summary["thread_id"].tolist() == [1]
    assert summary.loc[0, "arc_type"] == "escalating"
