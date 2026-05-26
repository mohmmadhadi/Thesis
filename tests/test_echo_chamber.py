from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.echo_chamber import (
    ANTI_ANCHORS,
    PRO_ANCHORS,
    AttitudeScorer,
    EchoChamberAnalyzer,
    EchoChamberPipeline,
    InteractionNetworkBuilder,
    StanceEstimator,
)


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        values = {
            "pro": [1.0, 0.0],
            "anti": [0.0, 1.0],
            "neutral": [0.5, 0.5],
        }
        return [values[text] for text in texts]


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        scores = []
        for text, anchor in pairs:
            if "revolutionize healthcare" in anchor:
                scores.append(0.9 if text == "pro" else 0.1)
            elif "economic boom" in anchor:
                scores.append(0.8 if text == "pro" else 0.2)
            elif "powerful tool" in anchor:
                scores.append(0.7 if text == "pro" else 0.3)
            elif "mass global unemployment" in anchor:
                scores.append(0.2 if text == "pro" else 0.8)
            elif "existential threat" in anchor:
                scores.append(0.1 if text == "pro" else 0.9)
            else:
                scores.append(0.3 if text == "pro" else 0.7)
        return np.array(scores)


class FakeEchoChamberAnalyzer:
    def measure_polarization(self, attitudes):
        return {"variance": 0.1, "bimodality": 0.2, "inter_group_distance": 0.3}

    def measure_homophily(self, graph, attitudes):
        return {"assortativity": 0.4, "homophily_ratio": 0.5}

    def calculate_exposure_diversity(self, graph, attitudes):
        return {node: 0.25 for node in graph.nodes()}

    def detect_communities(self, graph):
        return {node: index for index, node in enumerate(graph.nodes())}

    def compute_modularity(self, communities, graph):
        return 0.42

    def add_community_labels(self, user_attitudes, communities, user_col="user_id"):
        return EchoChamberAnalyzer.add_community_labels(
            user_attitudes,
            communities,
            user_col=user_col,
        )

    def compute_community_stats(self, user_attitudes):
        return pd.DataFrame(
            {
                "size": [1, 1],
                "avg_attitude": [0.2, -0.2],
                "attitude_std": [0.1, 0.1],
                "avg_diversity": [0.25, 0.25],
            },
            index=[0, 1],
        )

    def compute_echo_chamber_score(
        self,
        polarization,
        homophily,
        diversity_stats,
        community_stats,
    ):
        return {
            "overall_score": 0.6,
            "polarization_score": 0.1,
            "homophily_score": 0.2,
            "diversity_score": 0.3,
            "separation_score": 0.4,
        }


class ExplodingStanceEstimator:
    def transform_dataframe(self, *args, **kwargs):
        raise AssertionError("Stance estimator should not be called")


class AddingStanceEstimator:
    def __init__(self):
        self.called = False

    def transform_dataframe(self, df, text_col="clean_text", initial_score_col="roberta_compound"):
        self.called = True
        result = df.copy()
        result["stance_ensemble"] = result[initial_score_col]
        return result


def test_build_stance_prototypes_uses_notebook_thresholds():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    scores = pd.Series([0.4, 0.5, -0.4, 0.0])

    proto_pro, proto_anti = StanceEstimator.build_stance_prototypes(embeddings, scores)

    assert proto_pro.tolist() == pytest.approx([0.9, 0.1])
    assert proto_anti.tolist() == pytest.approx([0.0, 1.0])


def test_build_stance_prototypes_uses_zero_vector_when_group_empty():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = pd.Series([0.0, 0.1])

    proto_pro, proto_anti = StanceEstimator.build_stance_prototypes(embeddings, scores)

    assert proto_pro.tolist() == [0.0, 0.0]
    assert proto_anti.tolist() == [0.0, 0.0]


def test_calculate_stance_similarity_returns_difference_and_components():
    embedding = np.array([1.0, 0.0])
    proto_pro = np.array([1.0, 0.0])
    proto_anti = np.array([0.0, 1.0])

    stance_score, sim_pro, sim_anti = StanceEstimator.calculate_stance_similarity(
        embedding,
        proto_pro,
        proto_anti,
    )

    assert stance_score == pytest.approx(1.0)
    assert sim_pro == pytest.approx(1.0)
    assert sim_anti == pytest.approx(0.0)


def test_compute_cross_encoder_score_uses_all_pro_and_anti_anchors():
    cross_encoder = FakeCrossEncoder()
    estimator = StanceEstimator(
        embedding_model=FakeEmbeddingModel(),
        cross_encoder=cross_encoder,
        load_embedding_model=False,
        load_cross_encoder=False,
    )

    score = estimator.compute_cross_encoder_score("pro")

    expected = np.mean([0.9, 0.8, 0.7]) - np.mean([0.2, 0.1, 0.3])
    assert score == pytest.approx(expected)
    assert len(cross_encoder.calls) == 2
    assert len(cross_encoder.calls[0]) == len(PRO_ANCHORS)
    assert len(cross_encoder.calls[1]) == len(ANTI_ANCHORS)


def test_fuse_stance_scores_matches_notebook_weights():
    df = pd.DataFrame(
        {
            "stance_similarity": [-1.0, 0.0, 1.0],
            "cross_encoder_score": [-2.0, 0.0, 2.0],
        }
    )

    result = StanceEstimator.fuse_stance_scores(df)

    assert result["stance_similarity_norm"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert result["cross_encoder_norm"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert result["stance_ensemble"].tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_transform_dataframe_returns_notebook_stance_columns_with_fakes():
    estimator = StanceEstimator(
        embedding_model=FakeEmbeddingModel(),
        cross_encoder=FakeCrossEncoder(),
        load_embedding_model=False,
        load_cross_encoder=False,
    )
    df = pd.DataFrame(
        {
            "clean_text": ["pro", "anti", "neutral"],
            "roberta_compound": [0.5, -0.5, 0.0],
        }
    )

    result = estimator.transform_dataframe(df)

    expected_columns = {
        "stance_score_initial",
        "stance_similarity",
        "sim_pro",
        "sim_anti",
        "cross_encoder_score",
        "stance_similarity_norm",
        "cross_encoder_norm",
        "stance_ensemble",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "stance_score_initial"] == 0.5
    assert result.loc[0, "stance_similarity"] > result.loc[1, "stance_similarity"]


def test_attitude_score_uses_notebook_formula():
    scorer = AttitudeScorer()

    result = scorer.calculate_attitude_score(stance=0.5, sentiment=-0.8)

    expected = np.tanh(0.5 * (1 + 0.5 * abs(-0.8)))
    assert result == pytest.approx(expected)


def test_add_sentiment_and_attitude_scores_use_notebook_column_names():
    scorer = AttitudeScorer()
    df = pd.DataFrame(
        {
            "stance_ensemble": [0.5, -0.5],
            "roberta_compound": [0.2, -0.4],
        }
    )

    result = scorer.transform_dataframe(df)

    assert result["sentiment_score"].tolist() == [0.2, -0.4]
    assert result["attitude_score"].tolist() == pytest.approx(
        [
            np.tanh(0.5 * (1 + 0.5 * 0.2)),
            np.tanh(-0.5 * (1 + 0.5 * 0.4)),
        ]
    )


def test_aggregate_user_attitudes_matches_notebook_formulas():
    user_df = pd.DataFrame(
        {
            "attitude_score": [0.2, 0.8],
            "day": [1, 3],
            "like": [0, 3],
            "reaction_count": [2, 5],
        }
    )

    result = AttitudeScorer.aggregate_user_attitudes(user_df)

    weights = np.exp(np.array([1, 3]) - 3)
    assert result["mean_attitude"] == pytest.approx(0.5)
    assert result["weighted_attitude"] == pytest.approx(
        np.average([0.2, 0.8], weights=weights)
    )
    assert result["engagement_weighted_attitude"] == pytest.approx(
        np.average([0.2, 0.8], weights=[1, 4])
    )
    assert result["attitude_std"] == pytest.approx(np.std([0.2, 0.8]))
    assert result["num_posts"] == 2
    assert result["total_likes"] == 3
    assert result["total_reactions"] == 7


def test_compute_user_attitudes_sets_single_post_std_to_nan():
    scorer = AttitudeScorer()
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "attitude_score": [0.2, 0.8, -0.5],
            "day": [1, 2, 1],
            "like": [0, 1, 2],
            "reaction_count": [1, 2, 3],
        }
    )

    result = scorer.compute_user_attitudes(df)

    user_one = result[result["user_id"] == 1].iloc[0]
    user_two = result[result["user_id"] == 2].iloc[0]
    assert user_one["attitude_std"] == pytest.approx(np.std([0.2, 0.8]))
    assert np.isnan(user_two["attitude_std"])


def test_interaction_network_builder_adds_user_nodes_with_attributes():
    builder = InteractionNetworkBuilder()
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1],
            "mean_attitude": [0.25],
            "num_posts": [3],
            "attitude_std": [0.1],
        }
    )

    graph = builder.build_graph(
        pd.DataFrame(columns=["id", "user_id", "comment_to", "shared_from"]),
        user_attitudes,
    )

    assert graph.nodes[1]["attitude"] == 0.25
    assert graph.nodes[1]["num_posts"] == 3
    assert graph.nodes[1]["attitude_std"] == 0.1


def test_interaction_network_builder_reply_and_retweet_edges_match_notebook_logic():
    builder = InteractionNetworkBuilder()
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "mean_attitude": [0.1, -0.2, 0.3],
            "num_posts": [1, 2, 3],
            "attitude_std": [0.0, 0.1, 0.2],
        }
    )
    df = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "user_id": [1, 2, 2, 3],
            "comment_to": [-1, 10, 10, 13],
            "shared_from": [-1, -1, -1, 11],
        }
    )

    graph = builder.build_graph(df, user_attitudes)

    assert graph[2][1]["weight"] == 2
    assert graph[2][1]["type"] == "reply"
    assert graph[3][2]["weight"] == 1
    assert graph[3][2]["type"] == "retweet"
    assert not graph.has_edge(3, 3)


def test_interaction_network_builder_ignores_targets_not_in_user_nodes():
    builder = InteractionNetworkBuilder()
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1],
            "mean_attitude": [0.1],
            "num_posts": [1],
            "attitude_std": [0.0],
        }
    )
    df = pd.DataFrame(
        {
            "id": [10, 11],
            "user_id": [1, 2],
            "comment_to": [-1, 10],
            "shared_from": [-1, -1],
        }
    )

    graph = builder.build_graph(df, user_attitudes)

    assert graph.number_of_edges() == 0


def test_interaction_network_builder_returns_undirected_graph_and_stats():
    builder = InteractionNetworkBuilder()
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2],
            "mean_attitude": [0.1, -0.1],
            "num_posts": [1, 1],
            "attitude_std": [0.0, 0.0],
        }
    )
    df = pd.DataFrame(
        {
            "id": [10, 11],
            "user_id": [1, 2],
            "comment_to": [-1, 10],
            "shared_from": [-1, -1],
        }
    )

    directed, undirected = builder.build(df, user_attitudes)
    stats = builder.graph_stats(directed)

    assert directed.is_directed()
    assert not undirected.is_directed()
    assert stats["number_of_nodes"] == 2
    assert stats["number_of_edges"] == 1
    assert stats["number_of_connected_components"] == 1
    assert stats["average_degree"] == pytest.approx(1.0)


def test_echo_chamber_analyzer_measure_polarization_matches_formula():
    attitudes = {1: -0.5, 2: 0.5, 3: 0.2, 4: -0.2}

    result = EchoChamberAnalyzer.measure_polarization(attitudes)

    assert result["variance"] == pytest.approx(np.var([-0.5, 0.5, 0.2, -0.2]))
    assert result["inter_group_distance"] == pytest.approx(np.mean([0.5, 0.2]) - np.mean([-0.5, -0.2]))
    assert "bimodality" in result


def test_echo_chamber_analyzer_measure_polarization_empty_returns_zeroes():
    assert EchoChamberAnalyzer.measure_polarization({}) == {
        "variance": 0,
        "bimodality": 0,
        "inter_group_distance": 0,
    }


def test_echo_chamber_analyzer_measure_homophily_counts_same_sign_edges():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    attitudes = {1: 0.5, 2: 0.2, 3: -0.1, 4: -0.4}

    result = EchoChamberAnalyzer.measure_homophily(graph, attitudes)

    assert result["homophily_ratio"] == pytest.approx(2 / 3)
    assert "assortativity" in result
    assert graph.nodes[1]["attitude"] == 0.5


def test_echo_chamber_analyzer_measure_homophily_no_edges_returns_zeroes():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])

    assert EchoChamberAnalyzer.measure_homophily(graph, {1: 0.1, 2: -0.1}) == {
        "assortativity": 0,
        "homophily_ratio": 0,
    }


def test_echo_chamber_analyzer_calculate_exposure_diversity_matches_formula():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (1, 3)])
    graph.add_node(4)
    attitudes = {2: -0.5, 3: 0.5}

    result = EchoChamberAnalyzer.calculate_exposure_diversity(graph, attitudes)

    neighbor_attitudes = np.array([-0.5, 0.5])
    hist, _ = np.histogram(neighbor_attitudes, bins=np.array([-1, -0.5, 0, 0.5, 1]))
    probs = hist / hist.sum()
    from scipy.stats import entropy

    expected = (
        np.std(neighbor_attitudes)
        + entropy(probs + 1e-10) / 2
        + (np.max(neighbor_attitudes) - np.min(neighbor_attitudes))
    ) / 3
    assert result[1] == pytest.approx(expected)
    assert result[4] == 0


def test_echo_chamber_analyzer_detect_communities_no_edges_assigns_zero():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])

    assert EchoChamberAnalyzer.detect_communities(graph) == {1: 0, 2: 0}


def test_echo_chamber_analyzer_add_labels_and_community_stats_match_notebook_columns():
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "propagated_attitude": [0.5, 0.7, -0.4],
            "exposure_diversity": [0.1, 0.3, 0.8],
        }
    )

    labeled = EchoChamberAnalyzer.add_community_labels(user_attitudes, {1: 0, 2: 0, 3: 1})
    stats = EchoChamberAnalyzer.compute_community_stats(labeled)

    assert labeled["community"].tolist() == [0, 0, 1]
    assert stats.columns.tolist() == ["size", "avg_attitude", "attitude_std", "avg_diversity"]
    assert stats.loc[0, "size"] == 2
    assert stats.loc[0, "avg_attitude"] == pytest.approx(0.6)
    assert stats.index.tolist()[0] == 0


def test_echo_chamber_analyzer_compute_echo_chamber_score_matches_notebook_weights():
    polarization = {"variance": 0.25, "bimodality": 0.35, "inter_group_distance": 0.75}
    homophily = {"assortativity": 0.4, "homophily_ratio": 0.6}
    diversity = pd.Series([0.2, 0.4])
    community_stats = pd.DataFrame(
        {
            "avg_attitude": [0.5, -0.5],
            "attitude_std": [0.2, 0.2],
        }
    )

    result = EchoChamberAnalyzer.compute_echo_chamber_score(
        polarization,
        homophily,
        diversity,
        community_stats,
    )

    polarization_score = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5
    homophily_score = 0.5 * 0.4 + 0.5 * 0.6
    diversity_score = 1 - diversity.mean()
    separation_score = min(1.0, community_stats["avg_attitude"].var() / (community_stats["attitude_std"].mean() + 0.1))
    expected = (
        0.3 * polarization_score
        + 0.3 * homophily_score
        + 0.2 * diversity_score
        + 0.2 * separation_score
    )
    assert result["overall_score"] == pytest.approx(expected)
    assert result["polarization_score"] == pytest.approx(polarization_score)
    assert result["homophily_score"] == pytest.approx(homophily_score)
    assert result["diversity_score"] == pytest.approx(diversity_score)
    assert result["separation_score"] == pytest.approx(separation_score)


def test_echo_chamber_pipeline_orchestrates_existing_stance_columns():
    df = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "user_id": [1, 2, 1],
            "day": [1, 1, 2],
            "clean_text": ["a", "b", "c"],
            "roberta_compound": [0.2, -0.4, 0.6],
            "stance_ensemble": [0.5, -0.5, 0.25],
            "like": [1, 2, 3],
            "reaction_count": [2, 3, 4],
            "comment_to": [-1, 10, -1],
            "shared_from": [-1, -1, 11],
        }
    )
    pipeline = EchoChamberPipeline(
        stance_estimator=ExplodingStanceEstimator(),
        analyzer=FakeEchoChamberAnalyzer(),
    )

    result = pipeline.run(df)

    assert result["tweets"]["sentiment_score"].tolist() == [0.2, -0.4, 0.6]
    assert "attitude_score" in result["tweets"].columns
    assert result["final_day"] == 2
    expected_user_one = result["tweets"][result["tweets"]["user_id"] == 1][
        "attitude_score"
    ].mean()
    assert result["propagated_attitudes"][1] == pytest.approx(expected_user_one)
    assert result["graph"].has_edge(2, 1)
    assert result["graph"].has_edge(1, 2)
    assert result["user_attitudes"]["propagated_attitude"].notna().all()
    assert result["user_attitudes"]["exposure_diversity"].tolist() == [0.25, 0.25]
    assert result["modularity"] == 0.42
    assert result["echo_chamber_metrics"]["overall_score"] == 0.6


def test_echo_chamber_pipeline_runs_stance_estimation_when_required():
    df = pd.DataFrame(
        {
            "id": [10, 11],
            "user_id": [1, 2],
            "day": [1, 1],
            "clean_text": ["a", "b"],
            "roberta_compound": [0.2, -0.4],
            "like": [1, 2],
            "reaction_count": [2, 3],
            "comment_to": [-1, 10],
            "shared_from": [-1, -1],
        }
    )
    stance_estimator = AddingStanceEstimator()
    pipeline = EchoChamberPipeline(
        stance_estimator=stance_estimator,
        analyzer=FakeEchoChamberAnalyzer(),
    )

    result = pipeline.run(df)

    assert stance_estimator.called
    assert result["tweets"]["stance_ensemble"].tolist() == [0.2, -0.4]
    assert "echo_chamber_metrics" in result
