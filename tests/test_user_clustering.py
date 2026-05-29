from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.user_clustering import FeatureExtractor, TweetClusterer, UserClusterer


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


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, "kwargs": kwargs})
        return [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]][: len(texts)]


class FakeReducer:
    def fit_transform(self, embeddings):
        return [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]][: len(embeddings)]


class FakeClusterer:
    def fit_predict(self, embeddings_2d):
        return [0, 0, -1][: len(embeddings_2d)]


class FakeTfidfVectorizer:
    def __init__(self, max_features=None, stop_words=None):
        self.max_features = max_features
        self.stop_words = stop_words
        self.idf_ = [2.0, 1.0, 3.0, 0.5]

    def fit(self, texts):
        self.texts = texts
        return self

    def get_feature_names_out(self):
        return ["alpha", "beta", "gamma", "delta"]


class FakeScaler:
    def fit_transform(self, data):
        return data.to_numpy()


class FakePCA:
    def __init__(self, output):
        self.output = output

    def fit_transform(self, data):
        return self.output[: len(data)]


class FakeGMM:
    def fit_predict(self, data):
        return [0, 1, 0][: len(data)]


class FakeHDBSCAN:
    def fit_predict(self, data):
        return [1, -1, 1][: len(data)]


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


def test_tweet_clusterer_prepare_tweets_cleans_and_filters_empty_rows():
    clusterer = TweetClusterer(
        embedding_model=FakeEmbeddingModel(),
        reducer=FakeReducer(),
        clusterer=FakeClusterer(),
        vectorizer_cls=FakeTfidfVectorizer,
    )
    df = pd.DataFrame({"tweet": ["Hello @u #AI!", "https://example.com @u #tag"]})

    result = clusterer.prepare_tweets(df)

    assert result["clean_tweet"].tolist() == ["hello"]


def test_tweet_clusterer_compute_embeddings_uses_notebook_encode_options():
    embedding_model = FakeEmbeddingModel()
    clusterer = TweetClusterer(embedding_model=embedding_model)

    embeddings = clusterer.compute_embeddings(["hello", "world"])

    assert embeddings.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert embedding_model.calls[0]["kwargs"] == {
        "batch_size": 64,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }


def test_tweet_clusterer_build_cluster_labels_uses_lowest_idf_words_and_noise():
    clusterer = TweetClusterer(vectorizer_cls=FakeTfidfVectorizer)
    df = pd.DataFrame(
        {
            "clean_tweet": ["alpha beta", "beta delta", "noise"],
            "cluster": [0, 0, -1],
        }
    )

    labels = clusterer.build_cluster_labels(df)

    assert labels == { -1: "Noise", 0: "Cluster 0: delta, beta, alpha, gamma" }


def test_tweet_clusterer_fit_transform_adds_coordinates_clusters_and_labels():
    clusterer = TweetClusterer(
        embedding_model=FakeEmbeddingModel(),
        reducer=FakeReducer(),
        clusterer=FakeClusterer(),
        vectorizer_cls=FakeTfidfVectorizer,
    )
    df = pd.DataFrame(
        {
            "tweet": [
                "First cluster text",
                "Second cluster text",
                "Noise text",
            ]
        }
    )

    result = clusterer.fit_transform(df)

    assert result["x"].tolist() == [0.0, 1.0, 2.0]
    assert result["y"].tolist() == [0.0, 1.0, 2.0]
    assert result["cluster"].tolist() == [0, 0, -1]
    assert result["cluster_label"].tolist() == [
        "Cluster 0: delta, beta, alpha, gamma",
        "Cluster 0: delta, beta, alpha, gamma",
        "Noise",
    ]
    assert clusterer.n_clusters_ == 1
    assert clusterer.n_noise_ == 1


def test_tweet_clusterer_cluster_count_summary_counts_noise_like_notebook():
    tweets = pd.DataFrame({"cluster": [0, 0, 1, -1, -1]})

    result = TweetClusterer.cluster_count_summary(tweets)

    assert result == {"n_clusters": 2, "n_noise": 2}


def test_tweet_clusterer_cluster_trait_crosstab_returns_counts_percent_annotations():
    tweets = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1, -1],
            "cluster_label": ["A", "A", "B", "B", "Noise"],
            "ex": ["high", "low", "high", "high", "low"],
        }
    )

    result = TweetClusterer.cluster_trait_crosstab(tweets, "ex")

    assert set(result) == {"counts", "row_percent", "annotation"}
    assert result["counts"].index.tolist() == ["A", "B"]
    assert result["counts"].columns.tolist() == ["high", "low"]
    assert result["counts"].loc["A", "high"] == 1
    assert result["row_percent"].loc["B", "high"] == pytest.approx(1.0)
    assert result["annotation"].loc["A", "high"] == "1<br>(50%)"


def test_tweet_clusterer_overlap_diagnostics_returns_ari_nmi_and_missing_trait():
    tweets = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1, -1],
            "cluster_label": ["A", "A", "B", "B", "Noise"],
            "ex": ["high", "low", "high", "low", "low"],
        }
    )

    result = TweetClusterer.cluster_trait_overlap_diagnostics(
        tweets,
        traits=["ex", "missing_trait"],
    )

    assert result.columns.tolist() == [
        "trait",
        "chi2",
        "p",
        "dof",
        "significance",
        "ari",
        "nmi",
        "ari_nmi_status",
        "n_values",
    ]
    ex_row = result[result["trait"] == "ex"].iloc[0]
    missing_row = result[result["trait"] == "missing_trait"].iloc[0]
    assert ex_row["ari_nmi_status"] == "computed"
    assert pd.notna(ex_row["ari"])
    assert pd.notna(ex_row["nmi"])
    assert missing_row["significance"] == "missing"
    assert missing_row["ari_nmi_status"] == "missing"
    assert missing_row["n_values"] == 0


def test_user_clusterer_add_length_features_matches_notebook_names():
    agents = pd.DataFrame({"id": [1, 2]})
    tweets = pd.DataFrame({"user_id": [1, 1, 2], "tweet": ["aa", "aaaa", "bbb"]})

    result = UserClusterer.add_length_features(agents, tweets)

    assert result.loc[result["id"] == 1, "mean_len"].iloc[0] == pytest.approx(3.0)
    assert result.loc[result["id"] == 2, "std_len"].iloc[0] == 0.0


def test_user_clusterer_add_sentiment_features_fills_missing_values():
    agents = pd.DataFrame({"id": [1, 2]})
    sentiments = pd.DataFrame(
        {
            "user_id": [1, 1],
            "roberta_score": [0.2, 0.8],
            "emotion_score": [0.1, 0.3],
        }
    )

    result = UserClusterer.add_sentiment_features(agents, sentiments)

    assert result.loc[result["id"] == 1, "mean_sent"].iloc[0] == pytest.approx(0.5)
    assert result.loc[result["id"] == 2, "mean_sent"].iloc[0] == 0.0
    assert result.loc[result["id"] == 2, "std_emo"].iloc[0] == 0.0


def test_user_clusterer_compute_user_tweet_cosine_matches_notebook_logic():
    tweets = pd.DataFrame({"user_id": [1, 1, 2]})
    embeddings = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).to_numpy()

    result = UserClusterer.compute_user_tweet_cosine(tweets, embeddings)

    assert result.loc[result["id"] == 1, "avg_tweet_cosine_similarity"].iloc[0] == 1.0
    assert result.loc[result["id"] == 2, "avg_tweet_cosine_similarity"].iloc[0] == 1.0


def test_user_clusterer_add_reply_coherence_uses_notebook_weights_and_missingness():
    agents = pd.DataFrame({"id": [1, 2]})
    reply_sim = pd.DataFrame(
        {
            "user_id": [1],
            "cosine_similarity": [0.5],
            "bs_f1": [0.8],
            "cross_encoder_score": [0.6],
        }
    )

    result = UserClusterer.add_reply_coherence(agents, reply_sim)

    expected = 0.20 * 0.5 + 0.35 * 0.8 + 0.45 * 0.6
    assert result.loc[result["id"] == 1, "avg_reply_coherence"].iloc[0] == pytest.approx(expected)
    assert result["has_parent"].tolist() == [1, 0]
    assert result.loc[result["id"] == 2, "avg_reply_coherence"].iloc[0] == pytest.approx(expected)


def test_user_clusterer_prepare_clustering_matrix_selects_features_and_drops_na():
    clusterer = UserClusterer()
    agents = pd.DataFrame(
        {
            "mean_sent": [0.1, 0.2],
            "std_emo": [0.0, 0.1],
            "mtld": [5.0, None],
            "pronoun_ratio": [1.0, 2.0],
            "mean_emoji_rate": [0.0, 0.1],
            "avg_reply_coherence": [0.5, 0.6],
            "has_parent": [1, 0],
        }
    )

    matrix = clusterer.prepare_clustering_matrix(agents)

    assert matrix.shape == (1, 7)
    assert clusterer.selected_features_ == [
        "mean_sent",
        "std_emo",
        "mtld",
        "pronoun_ratio",
        "mean_emoji_rate",
        "avg_reply_coherence",
        "has_parent",
    ]


def test_user_clusterer_clustering_matrix_summary_reports_shape_and_dropped_agents():
    clusterer = UserClusterer()
    agents = pd.DataFrame(
        {
            "mean_sent": [0.1, 0.2],
            "std_emo": [0.0, 0.1],
            "mtld": [5.0, None],
            "pronoun_ratio": [1.0, 2.0],
            "mean_emoji_rate": [0.0, 0.1],
            "avg_reply_coherence": [0.5, 0.6],
            "has_parent": [1, 0],
        }
    )

    result = clusterer.clustering_matrix_summary(agents)

    assert result["shape"] == (1, 7)
    assert result["dropped_agents"] == 1
    assert result["cluster_features"] == clusterer.selected_features_
    assert result["matrix"].shape == (1, 7)


def test_user_clusterer_fit_clusters_uses_injected_components():
    clusterer = UserClusterer(
        scaler=FakeScaler(),
        pca=FakePCA([[0.1, 0.2], [0.3, 0.4]]),
        pca_vis=FakePCA([[1.0, 2.0], [3.0, 4.0]]),
        gmm=FakeGMM(),
        hdbscan_clusterer=FakeHDBSCAN(),
    )
    matrix = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    labels_gmm, labels_hdb, x_pca, x_2d = clusterer.fit_clusters(matrix)

    assert labels_gmm.tolist() == [0, 1]
    assert labels_hdb.tolist() == [1, -1]
    assert x_pca.tolist() == [[0.1, 0.2], [0.3, 0.4]]
    assert x_2d.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_user_clusterer_assign_cluster_labels_uses_matrix_index():
    agents = pd.DataFrame({"id": [10, 11, 12], "name": ["a", "b", "c"]})
    matrix = pd.DataFrame({"feature": [1.0, 2.0]}, index=[0, 2])

    result = UserClusterer.assign_cluster_labels(
        agents,
        matrix,
        labels_gmm=pd.Series([0, 1]).to_numpy(),
        labels_hdbscan=pd.Series([1, -1]).to_numpy(),
    )

    assert result["id"].tolist() == [10, 12]
    assert result["gmm_cluster"].tolist() == [0, 1]
    assert result["hdbscan_cluster"].tolist() == [1, -1]


def test_user_clusterer_agent_cluster_count_summary_counts_noise():
    result = UserClusterer.agent_cluster_count_summary(
        labels_gmm=pd.Series([0, 1, 1]).to_numpy(),
        labels_hdbscan=pd.Series([2, -1, 2]).to_numpy(),
    )

    assert result == {
        "gmm_clusters": 2,
        "hdbscan_clusters": 1,
        "hdbscan_noise": 1,
    }


def test_user_clusterer_profile_clusters_returns_crosstab_and_binary_metrics():
    agents = pd.DataFrame(
        {
            "Agent_Cluster": [0, 0, 1, 1],
            "gender": ["f", "m", "f", "m"],
        }
    )

    profile = UserClusterer.profile_clusters(agents, "gender")

    assert "crosstab" in profile
    assert "raw_crosstab" in profile
    assert "chi2" in profile
    assert "ari" in profile
    assert "nmi" in profile


def test_user_clusterer_profile_clusters_describes_numeric_trait_with_many_values():
    agents = pd.DataFrame(
        {
            "Agent_Cluster": [0, 0, 1, 1] * 6,
            "age": list(range(24)),
        }
    )

    profile = UserClusterer.profile_clusters(agents, "age")

    assert "describe" in profile


def test_user_clusterer_profile_held_out_traits_skips_missing_and_preserves_profiles():
    agents = pd.DataFrame(
        {
            "Agent_Cluster": [0, 0, 1, 1],
            "gender": ["f", "m", "f", "m"],
            "age": [20, 21, 22, 23],
        }
    )

    result = UserClusterer.profile_held_out_traits(
        agents,
        traits=["gender", "missing_trait", "age"],
    )

    assert set(result) == {"gender", "age"}
    assert "crosstab" in result["gender"]
    assert "crosstab" in result["age"]


def test_user_clusterer_behavioral_correlation_matrix_uses_existing_behavior_cols():
    agents = pd.DataFrame(
        {
            "mean_len": [1.0, 2.0, 3.0],
            "mean_sent": [0.1, 0.2, 0.3],
            "notebook_ignored": [10, 11, 12],
        }
    )

    result = UserClusterer.behavioral_correlation_matrix(agents)

    assert result.index.tolist() == ["mean_len", "mean_sent"]
    assert result.columns.tolist() == ["mean_len", "mean_sent"]
    assert result.loc["mean_len", "mean_sent"] == pytest.approx(1.0)


def test_user_clusterer_cluster_profile_summary_returns_mean_features():
    agents = pd.DataFrame(
        {
            "Agent_Cluster": [0, 0, 1],
            "mean_sent": [0.2, 0.4, -0.5],
            "mtld": [10.0, 20.0, 30.0],
        }
    )

    result = UserClusterer.cluster_profile_summary(
        agents,
        feature_cols=["mean_sent", "mtld", "missing_feature"],
    )

    assert result.columns.tolist() == ["mean_sent", "mtld"]
    assert result.loc[0, "mean_sent"] == pytest.approx(0.3)
    assert result.loc[1, "mtld"] == pytest.approx(30.0)
