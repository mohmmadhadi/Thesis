from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.sentiment import (
    EmotionAnalyzer,
    SentimentEmotionPipeline,
    TransformerSentimentAnalyzer,
    VaderSentimentAnalyzer,
)


class FakeVader:
    def polarity_scores(self, text):
        if "bad" in text:
            return {"neg": 0.8, "neu": 0.2, "pos": 0.0, "compound": -0.6}
        return {"neg": 0.0, "neu": 0.2, "pos": 0.8, "compound": 0.7}


class FakeSentimentClassifier:
    def __call__(self, texts):
        return [
            [
                {"label": "LABEL_0", "score": 0.1},
                {"label": "LABEL_1", "score": 0.2},
                {"label": "LABEL_2", "score": 0.7},
            ]
            for _ in texts
        ]


class FakeEmotionClassifier:
    def __call__(self, texts):
        return [
            [
                {"label": "anger", "score": 0.10},
                {"label": "disgust", "score": 0.05},
                {"label": "fear", "score": 0.15},
                {"label": "joy", "score": 0.60},
                {"label": "neutral", "score": 0.20},
                {"label": "sadness", "score": 0.10},
                {"label": "surprise", "score": 0.30},
            ]
            for _ in texts
        ]


def make_vader_analyzer():
    analyzer = VaderSentimentAnalyzer.__new__(VaderSentimentAnalyzer)
    analyzer.analyzer = FakeVader()
    return analyzer


def test_vader_label_thresholds_match_notebook():
    assert VaderSentimentAnalyzer.label(0.05) == "positive"
    assert VaderSentimentAnalyzer.label(-0.05) == "negative"
    assert VaderSentimentAnalyzer.label(0.0) == "neutral"
    assert np.isnan(VaderSentimentAnalyzer.label(np.nan))


def test_vader_transform_dataframe_adds_notebook_columns():
    analyzer = make_vader_analyzer()
    df = pd.DataFrame({"clean_text": ["good", "", "bad"]})

    result = analyzer.transform_dataframe(df)

    assert result["vader_label"].tolist() == ["positive", np.nan, "negative"]
    assert result.loc[0, "vader_compound"] == 0.7
    assert np.isnan(result.loc[1, "vader_compound"])


def test_transformer_sentiment_uses_label_mapping_and_compound_score():
    analyzer = TransformerSentimentAnalyzer(classifier=FakeSentimentClassifier())

    result = analyzer.predict(["hello", ""])

    assert result.loc[0, "roberta_label"] == "positive"
    assert result.loc[0, "roberta_score"] == 0.7
    assert result.loc[0, "roberta_compound"] == 0.6
    assert np.isnan(result.loc[1, "roberta_label"])


def test_emotion_analyzer_adds_valence_and_arousal_formulas():
    analyzer = EmotionAnalyzer(classifier=FakeEmotionClassifier())

    result = analyzer.predict(["hello"])

    unpleasant = (0.10 + 0.15 + 0.05 + 0.10) / 4
    activated = (0.10 + 0.15 + 0.30 + 0.60) / 4
    deactivated = (0.10 + 0.05 + 0.20) / 3
    assert result.loc[0, "emotion_label"] == "joy"
    assert result.loc[0, "emotion_score"] == 0.60
    assert result.loc[0, "emotion_valence"] == 0.60 - unpleasant
    assert result.loc[0, "emotion_arousal"] == activated - deactivated


def test_pipeline_runs_with_injected_lightweight_analyzers():
    pipeline = SentimentEmotionPipeline(
        config={"models": {}},
        vader_analyzer=make_vader_analyzer(),
        sentiment_analyzer=TransformerSentimentAnalyzer(classifier=FakeSentimentClassifier()),
        emotion_analyzer=EmotionAnalyzer(classifier=FakeEmotionClassifier()),
    )
    df = pd.DataFrame({"tweet": ["Great #AI @user https://example.com"]})

    result = pipeline.transform_dataframe(df)

    expected_columns = {
        "clean_text",
        "is_empty",
        "vader_label",
        "roberta_label",
        "roberta_compound",
        "emotion_label",
        "emotion_valence",
        "emotion_arousal",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "clean_text"] == "great ai"
