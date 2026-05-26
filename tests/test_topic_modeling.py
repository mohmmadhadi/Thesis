from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.topic_modeling import (
    BERTopicModeler,
    LDATopicModeler,
    TopicModelingPipeline,
)


class FakeLemmatizer:
    def lemmatize(self, token):
        return {"cars": "car", "running": "run"}.get(token, token)


class FakeDictionary:
    def __init__(self, tokens):
        self.tokens = tokens
        self.filter_args = None

    def filter_extremes(self, no_below, no_above):
        self.filter_args = {"no_below": no_below, "no_above": no_above}

    def doc2bow(self, tokens):
        return list(enumerate(tokens))


class FakeLDAModel:
    def print_topics(self, num_topics):
        return [(0, '0.7*"ai" + 0.3*"health"'), (1, '0.5*"policy"')]

    def get_document_topics(self, doc):
        return [(0, 0.2), (1, 0.8)] if len(doc) > 1 else [(0, 1.0)]

    def log_perplexity(self, corpus):
        return -3.14


class FakeTopicModel:
    def __init__(self):
        self.docs = None

    def fit_transform(self, docs):
        self.docs = docs
        return [0, -1][: len(docs)], [[0.9], [0.1]][: len(docs)]

    def get_topic_info(self):
        return pd.DataFrame(
            {
                "Topic": [0, -1],
                "Name": ["0_ai_health", "-1_outliers"],
                "Count": [1, 1],
            }
        )


def simple_tokenizer(text):
    return text.split()


def make_lda_modeler(**kwargs):
    return LDATopicModeler(
        stop_words={"and", "the"},
        lemmatizer=FakeLemmatizer(),
        tokenizer=simple_tokenizer,
        **kwargs,
    )


def test_lda_preprocess_text_matches_notebook_steps():
    modeler = make_lda_modeler(min_tokens=2)

    result = modeler.preprocess_text("The CARS, and running!")

    assert result == "car run"


def test_lda_preprocess_text_drops_short_text_after_cleaning():
    modeler = make_lda_modeler(min_tokens=3)

    assert modeler.preprocess_text("AI and policy") == ""


def test_preprocess_dataframe_adds_input_tokens_and_filters_empty_rows():
    modeler = make_lda_modeler(min_tokens=2)
    df = pd.DataFrame({"tweet": ["The cars running", "AI"]})

    result = modeler.preprocess_dataframe(df)

    assert result["input"].tolist() == ["car run"]
    assert result["tokens"].tolist() == [["car", "run"]]


def test_build_dictionary_corpus_uses_notebook_filter_parameters():
    modeler = make_lda_modeler(
        dictionary_cls=FakeDictionary,
        no_below=10,
        no_above=0.5,
    )

    dictionary, corpus = modeler.build_dictionary_corpus([["ai", "health"]])

    assert dictionary.filter_args == {"no_below": 10, "no_above": 0.5}
    assert corpus == [[(0, "ai"), (1, "health")]]


def test_get_top_words_formats_topics_as_one_based_ids():
    modeler = make_lda_modeler()
    modeler.model = FakeLDAModel()

    assert modeler.get_top_words() == [
        (1, '0.7*"ai" + 0.3*"health"'),
        (2, '0.5*"policy"'),
    ]


def test_assign_topics_uses_dominant_document_topic():
    modeler = make_lda_modeler()
    df = pd.DataFrame({"tweet": ["a", "b"]})

    result = modeler.assign_topics(df, corpus=[["one"], ["one", "two"]], model=FakeLDAModel())

    assert result["lda_topic"].tolist() == [0, 1]


def test_log_perplexity_delegates_to_model():
    modeler = make_lda_modeler()

    assert modeler.compute_log_perplexity(corpus=[[]], model=FakeLDAModel()) == -3.14


def test_bertopic_assign_topics_and_names_with_fake_model():
    modeler = BERTopicModeler(topic_model=FakeTopicModel())
    df = pd.DataFrame({"input": ["ai health", ""]})

    result = modeler.assign_topics(df)

    assert result["bertopic_topic"].tolist() == [0, -1]
    assert result["bertopic_topic_name"].tolist() == ["0_ai_health", "-1_outliers"]


def test_build_topic_drift_preserves_notebook_columns_and_renames_sentiment():
    topic_df = pd.DataFrame(
        {
            "id": [1],
            "tweet": ["hello"],
            "user_id": [10],
            "comment_to": [-1],
            "thread_id": [100],
            "round": [0],
            "day": [1],
            "hour": [12],
            "bertopic_topic": [0],
            "bertopic_topic_name": ["0_ai"],
            "length": [3],
        }
    )
    sentiment_df = pd.DataFrame({"id": [1], "roberta_label": ["positive"]})

    result = TopicModelingPipeline.build_topic_drift(topic_df, sentiment_df)

    assert result.columns.tolist() == [
        "id",
        "tweet",
        "user_id",
        "comment_to",
        "thread_id",
        "round",
        "day",
        "hour",
        "bertopic_topic",
        "bertopic_topic_name",
        "length",
        "sentiment",
    ]
    assert result.loc[0, "sentiment"] == "positive"


def test_fit_requires_dictionary_and_corpus():
    modeler = make_lda_modeler()

    with pytest.raises(ValueError, match="Corpus and dictionary"):
        modeler.fit()
