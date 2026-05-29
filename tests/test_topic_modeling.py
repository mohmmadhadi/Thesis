from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.topic_modeling import (
    BERTopicModeler,
    LDATopicModeler,
    TopicModelingPipeline,
    token_length_report,
)
from social_analysis.visualization import TopicModelingVisualizer


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

    def __len__(self):
        return 42


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


class FakeFigure:
    def __init__(self):
        self.path = None

    def write_html(self, path):
        self.path = Path(path)
        self.path.write_text("<html></html>", encoding="utf-8")


class FakeVisualTopicModel(FakeTopicModel):
    def __init__(self):
        super().__init__()
        self.barchart_args = None

    def visualize_barchart(self, top_n_topics=10):
        self.barchart_args = {"top_n_topics": top_n_topics}
        return FakeFigure()

    def visualize_topics(self):
        return FakeFigure()


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


def test_lda_report_tables_preserve_notebook_columns():
    top_words = [(1, '0.7*"ai"'), (2, '0.5*"policy"')]
    top_words_table = LDATopicModeler.top_words_table(top_words)
    corpus_report = LDATopicModeler.corpus_report(FakeDictionary([]), [[], []])
    evaluation = LDATopicModeler.evaluation_report(0.51, -3.14)

    assert top_words_table.columns.tolist() == ["topic_id", "top_words"]
    assert top_words_table["topic_id"].tolist() == [1, 2]
    assert corpus_report.columns.tolist() == ["vocabulary_size", "corpus_size"]
    assert corpus_report.iloc[0].to_dict() == {"vocabulary_size": 42, "corpus_size": 2}
    assert evaluation.columns.tolist() == [
        "lda_coherence",
        "lda_log_perplexity",
        "coherence_note",
        "perplexity_note",
    ]
    assert evaluation.loc[0, "lda_coherence"] == 0.51


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


def test_bertopic_reports_topic_names_and_outliers():
    modeler = BERTopicModeler(topic_model=FakeTopicModel())
    topic_names = modeler.topic_name_mapping()
    mapping_table = BERTopicModeler.topic_name_mapping_table(topic_names)
    outliers = BERTopicModeler.outlier_report([0, 1, -1, -1])

    assert topic_names == {0: "0_ai_health", -1: "-1_outliers"}
    assert mapping_table.columns.tolist() == ["Topic", "Name"]
    assert mapping_table["Topic"].tolist() == [0, -1]
    assert outliers.columns.tolist() == [
        "num_topics_excluding_outliers",
        "outlier_tweets",
    ]
    assert outliers.iloc[0].to_dict() == {
        "num_topics_excluding_outliers": 2,
        "outlier_tweets": 2,
    }


def test_token_length_report_adds_length_and_describe_table():
    df = pd.DataFrame({"tokens": [["ai", "health"], ["policy"]]})

    result = token_length_report(df)

    assert result["data"]["length"].tolist() == [2, 1]
    assert result["describe"].columns.tolist() == ["value"]
    assert result["describe"].loc["count", "value"] == 2.0


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


def test_topic_visualizer_bertopic_html_exports_use_notebook_filenames(tmp_path):
    model = FakeVisualTopicModel()
    visualizer = TopicModelingVisualizer()

    barchart = visualizer.save_bertopic_barchart_html(model, tmp_path)
    topics = visualizer.save_bertopic_topics_html(model, tmp_path)

    assert barchart == tmp_path / "bertopic_barchart.html"
    assert topics == tmp_path / "bertopic_topics.html"
    assert barchart.exists()
    assert topics.exists()
    assert model.barchart_args == {"top_n_topics": 10}


def test_topic_visualizer_plot_methods_skip_missing_columns(tmp_path):
    visualizer = TopicModelingVisualizer()
    df = pd.DataFrame({"tweet": ["hello"]})

    assert visualizer.plot_lda_topic_distribution(df, tmp_path) is None
    assert visualizer.plot_token_length_distribution(df, tmp_path) is None
    assert visualizer.save_bertopic_barchart_html(object(), tmp_path) is None


def test_fit_requires_dictionary_and_corpus():
    modeler = make_lda_modeler()

    with pytest.raises(ValueError, match="Corpus and dictionary"):
        modeler.fit()
