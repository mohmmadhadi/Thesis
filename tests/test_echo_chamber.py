from pathlib import Path
import sys
import types

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


def install_fake_louvain(monkeypatch):
    """Install a tiny fake Louvain module for dependency-free tests."""
    fake_community = types.SimpleNamespace()

    def best_partition(graph):
        partition = {}
        for index, nodes in enumerate(nx.connected_components(graph)):
            for node in nodes:
                partition[node] = index
        return partition

    def modularity(partition, graph):
        return 0.0

    fake_community.best_partition = best_partition
    fake_community.modularity = modularity
    monkeypatch.setitem(sys.modules, "community", fake_community)


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

    def optimize_echo_chamber_weights(self, user_attitudes, community_stats, graph):
        return (
            {
                "polarization": 0.1,
                "homophily": 0.2,
                "diversity": 0.3,
                "separation": 0.4,
            },
            object(),
        )

    def compute_optimized_component_scores(
        self,
        daily_polarization,
        daily_homophily,
        user_attitudes,
        community_stats,
        final_day=None,
    ):
        return {
            "polarization_score": 0.2,
            "homophily_score": 0.3,
            "diversity_score": 0.4,
            "separation_score": 0.5,
        }

    def compute_optimized_echo_chamber_score(self, metrics, optimized_weights):
        result = dict(metrics)
        result["overall_score"] = 0.4
        return result

    def sensitivity_analysis(self, metrics, optimized_weights):
        return pd.DataFrame({"scheme": ["Optimized"], "score": [0.4]})

    def optimize_echo_chamber_weights_2d(self, user_attitudes, community_stats, graph):
        metrics_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "polarization": [0.2, 0.3],
                "homophily": [0.0, 1.0],
                "diversity": [0.5, 0.6],
                "separation": [0.7, 0.8],
            }
        )
        return (
            {
                "pc1": {
                    "polarization": 0.25,
                    "homophily": 0.25,
                    "diversity": 0.25,
                    "separation": 0.25,
                },
                "pc2": {
                    "polarization": 0.1,
                    "homophily": 0.2,
                    "diversity": 0.3,
                    "separation": 0.4,
                },
                "variances": np.array([0.6, 0.4]),
            },
            object(),
            metrics_df,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        )

    def compute_2d_optimized_score(self, metrics, weights_2d):
        return {
            "score_pc1": 0.35,
            "score_pc2": 0.45,
            "final_combined_score": 0.39,
        }

    def build_echo_chamber_landscape_dataframe(
        self,
        metrics_df,
        metrics_scaled,
        pca_model,
        user_attitudes,
    ):
        result = metrics_df.copy()
        result["PC1_Ideological"] = [0.0, 1.0]
        result["PC2_Structural"] = [0.0, 1.0]
        return result

    def build_echo_chamber_landscape_data(
        self,
        metrics_df,
        metrics_scaled,
        pca_model,
        user_attitudes,
    ):
        return {
            "landscape_df": self.build_echo_chamber_landscape_dataframe(
                metrics_df,
                metrics_scaled,
                pca_model,
                user_attitudes,
            ),
            "explained_variance_ratio": np.array([0.6, 0.4]),
            "x_col": "PC1_Ideological",
            "y_col": "PC2_Structural",
        }

    def analyze_sub_communities(
        self,
        graph,
        user_attitudes,
        top_n=5,
        user_col="user_id",
        community_col="community",
    ):
        labeled = user_attitudes.copy()
        labeled["parent_community"] = labeled["community"]
        labeled["sub_community"] = labeled["community"].map(lambda value: f"{value}_0")
        return {
            "sub_communities": labeled[
                ["user_id", "parent_community", "sub_community"]
            ],
            "user_attitudes": labeled,
            "sub_echo_scores": pd.DataFrame(
                {
                    "sub_community": ["0_0", "1_0"],
                    "size": [1, 1],
                    "echo_chamber_score": [0.5, 0.4],
                    "polarization": [0.1, 0.2],
                    "homophily": [0.3, 0.4],
                    "diversity": [0.5, 0.6],
                    "separation": [0.0, 0.0],
                }
            ),
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


def test_echo_chamber_analyzer_detect_sub_communities_preserves_notebook_columns(
    monkeypatch,
):
    install_fake_louvain(monkeypatch)
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (3, 4)])
    graph.add_nodes_from([5, 6])
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "community": [0, 0, 0, 0, 1, 1],
        }
    )

    result = EchoChamberAnalyzer.detect_sub_communities(
        graph,
        user_attitudes,
        top_n=2,
    )

    assert result.columns.tolist() == [
        "user_id",
        "parent_community",
        "sub_community",
    ]
    assert set(result["user_id"]) == {1, 2, 3, 4, 5, 6}
    assert set(result["parent_community"]) == {0, 1}
    assert all(
        str(sub).startswith(f"{parent}_")
        for parent, sub in zip(result["parent_community"], result["sub_community"])
    )
    small_parent = result[result["parent_community"] == 1]
    assert small_parent["sub_community"].tolist() == ["1_0", "1_0"]


def test_echo_chamber_analyzer_add_sub_community_labels_fills_non_top_communities():
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "community": [0, 0, 1],
            "parent_community": ["old", "old", "old"],
            "sub_community": ["old", "old", "old"],
        }
    )
    sub_communities = pd.DataFrame(
        {
            "user_id": [1, 2],
            "parent_community": [0, 0],
            "sub_community": ["0_0", "0_0"],
        }
    )

    result = EchoChamberAnalyzer.add_sub_community_labels(
        user_attitudes,
        sub_communities,
    )

    assert result["sub_community"].tolist() == ["0_0", "0_0", "None"]
    assert result["parent_community"].iloc[0] == 0
    assert pd.isna(result["parent_community"].iloc[2])


def test_echo_chamber_analyzer_compute_sub_community_echo_scores_columns_and_sort():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (3, 4)])
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "propagated_attitude": [0.5, 0.5, -0.4, -0.4, 0.1],
            "exposure_diversity": [0.2, 0.2, 0.7, 0.7, 0.5],
            "sub_community": ["0_0", "0_0", "0_1", "0_1", "None"],
        }
    )

    result = EchoChamberAnalyzer.compute_sub_community_echo_scores(
        graph,
        user_attitudes,
    )

    assert result.columns.tolist() == [
        "sub_community",
        "size",
        "echo_chamber_score",
        "polarization",
        "homophily",
        "diversity",
        "separation",
    ]
    assert set(result["sub_community"]) == {"0_0", "0_1"}
    assert result["size"].tolist() == [2, 2]
    assert result["echo_chamber_score"].is_monotonic_decreasing
    assert np.issubdtype(result["echo_chamber_score"].dtype, np.number)


def test_echo_chamber_analyzer_analyze_sub_communities_returns_merged_scores(
    monkeypatch,
):
    install_fake_louvain(monkeypatch)
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (3, 4)])
    user_attitudes = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "community": [0, 0, 0, 0, 1],
            "propagated_attitude": [0.5, 0.5, -0.4, -0.4, 0.2],
            "exposure_diversity": [0.2, 0.2, 0.7, 0.7, 0.5],
        }
    )

    result = EchoChamberAnalyzer.analyze_sub_communities(
        graph,
        user_attitudes,
        top_n=1,
    )

    assert set(result) == {"sub_communities", "user_attitudes", "sub_echo_scores"}
    assert "parent_community" in result["user_attitudes"].columns
    assert "sub_community" in result["user_attitudes"].columns
    assert result["user_attitudes"].loc[
        result["user_attitudes"]["user_id"] == 5,
        "sub_community",
    ].iloc[0] == "None"
    assert result["sub_echo_scores"].columns.tolist() == [
        "sub_community",
        "size",
        "echo_chamber_score",
        "polarization",
        "homophily",
        "diversity",
        "separation",
    ]


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


def test_echo_chamber_analyzer_improved_bimodality_insufficient_sample_keys():
    values = np.array([-0.5, 0.0, 0.5])

    result = EchoChamberAnalyzer.improved_bimodality_test(values, min_sample_size=50)

    assert result == {
        "sample_size": 3,
        "bimodal": False,
        "reason": "insufficient_data",
        "dip_statistic": None,
        "separation_index": None,
        "variance": pytest.approx(np.var(values)),
    }


def test_echo_chamber_analyzer_improved_bimodality_detects_synthetic_bimodal_distribution():
    left = np.linspace(-0.85, -0.55, 60)
    right = np.linspace(0.55, 0.85, 60)
    values = np.concatenate([left, right])

    result = EchoChamberAnalyzer.improved_bimodality_test(values, min_sample_size=50)

    assert result["sample_size"] == 120
    assert result["bimodal"] is True
    assert result["separation_index"] > 0.3
    assert result["bimodality_coefficient"] > 0.555
    assert result["num_peaks"] >= 2
    assert result["variance"] == pytest.approx(np.var(values))
    assert "dip_pvalue" in result


def test_echo_chamber_analyzer_improved_bimodality_unimodal_distribution_not_majority_vote():
    values = np.linspace(-0.2, 0.2, 120)

    result = EchoChamberAnalyzer.improved_bimodality_test(values, min_sample_size=50)

    assert result["sample_size"] == 120
    assert result["bimodal"] is False
    assert result["separation_index"] <= 0.3
    assert result["variance"] == pytest.approx(np.var(values))


def test_echo_chamber_analyzer_calculate_bimodality_and_trend_match_notebook_formula():
    history = {
        1: {1: -0.5, 2: 0.5, 3: 0.2},
        2: {1: -0.8, 2: -0.6, 3: 0.6, 4: 0.8, 5: 0.0},
    }

    trend = EchoChamberAnalyzer.calculate_bimodality_trend(history)

    assert trend.columns.tolist() == ["day", "bimodality"]
    assert trend["day"].tolist() == [1, 2]
    assert trend.loc[0, "bimodality"] == 0
    assert trend.loc[1, "bimodality"] == pytest.approx(
        EchoChamberAnalyzer.calculate_bimodality(history[2])
    )


def test_echo_chamber_analyzer_build_bimodality_plot_data_returns_notebook_arrays():
    values = np.concatenate([np.linspace(-0.85, -0.55, 60), np.linspace(0.55, 0.85, 60)])

    result = EchoChamberAnalyzer.build_bimodality_plot_data(values, min_sample_size=50)

    assert set(result) == {
        "bimodality_result",
        "hist",
        "bin_edges",
        "peaks",
        "peak_positions",
        "x_range",
        "kde_density",
    }
    assert len(result["hist"]) == 30
    assert len(result["bin_edges"]) == 31
    assert len(result["peak_positions"]) == len(result["peaks"])
    assert result["x_range"].shape == (200, 1)
    assert result["kde_density"].shape == (200,)


def _pca_user_attitudes():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "propagated_attitude": [0.8, 0.4, -0.6, -0.2],
            "exposure_diversity": [0.1, 0.3, 0.6, 0.8],
            "community": [0, 0, 1, 1],
        }
    )


def _pca_community_stats():
    return pd.DataFrame(
        {
            "avg_attitude": [0.6, -0.4],
            "attitude_std": [0.2, 0.3],
            "avg_diversity": [0.2, 0.7],
            "size": [2, 2],
        },
        index=[0, 1],
    )


def _pca_graph():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    return graph


def test_echo_chamber_analyzer_optimized_weights_are_pca_derived_and_normalized():
    weights, pca = EchoChamberAnalyzer.optimize_echo_chamber_weights(
        _pca_user_attitudes(),
        _pca_community_stats(),
        _pca_graph(),
    )

    assert set(weights) == {"polarization", "homophily", "diversity", "separation"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(isinstance(value, float) for value in weights.values())
    assert all(value >= 0 for value in weights.values())
    assert hasattr(pca, "components_")
    assert pca.n_components == 1
    assert list(weights.values()) != pytest.approx([0.3, 0.3, 0.2, 0.2])


def test_echo_chamber_analyzer_optimized_component_scores_match_notebook_columns():
    daily_pol = {
        1: {"variance": 0.1, "bimodality": 0.2, "inter_group_distance": 0.3},
        2: {"variance": 0.25, "bimodality": 0.4, "inter_group_distance": 0.7},
    }
    daily_homo = {
        1: {"assortativity": 0.1, "homophily_ratio": 0.2},
        2: {"assortativity": 0.3, "homophily_ratio": 0.6},
    }

    result = EchoChamberAnalyzer.compute_optimized_component_scores(
        daily_pol,
        daily_homo,
        _pca_user_attitudes(),
        _pca_community_stats(),
    )

    assert result == {
        "polarization_score": pytest.approx(0.5),
        "homophily_score": pytest.approx(0.6),
        "diversity_score": pytest.approx(1 - np.mean([0.1, 0.3, 0.6, 0.8])),
        "separation_score": pytest.approx(
            min(1.0, _pca_community_stats()["avg_attitude"].var() / 0.3)
        ),
    }


def test_echo_chamber_analyzer_optimized_score_and_sensitivity_are_not_fixed_only():
    metrics = {
        "polarization_score": 0.5,
        "homophily_score": 0.6,
        "diversity_score": 0.7,
        "separation_score": 0.8,
    }
    weights = {
        "polarization": 0.1,
        "homophily": 0.2,
        "diversity": 0.3,
        "separation": 0.4,
    }

    optimized = EchoChamberAnalyzer.compute_optimized_echo_chamber_score(metrics, weights)
    sensitivity = EchoChamberAnalyzer.sensitivity_analysis(optimized, weights)

    assert optimized["overall_score"] == pytest.approx(
        0.1 * 0.5 + 0.2 * 0.6 + 0.3 * 0.7 + 0.4 * 0.8
    )
    assert sensitivity.columns.tolist() == ["scheme", "score"]
    assert sensitivity["scheme"].tolist() == [
        "Equal",
        "Polarization-heavy",
        "Homophily-heavy",
        "Diversity-heavy",
        "Separation-heavy",
        "Optimized",
    ]
    assert sensitivity.loc[sensitivity["scheme"] == "Optimized", "score"].iloc[0] == pytest.approx(
        optimized["overall_score"]
    )


def test_echo_chamber_analyzer_2d_pca_outputs_weights_metrics_and_landscape_shape():
    weights_2d, pca, metrics_df, metrics_scaled = (
        EchoChamberAnalyzer.optimize_echo_chamber_weights_2d(
            _pca_user_attitudes(),
            _pca_community_stats(),
            _pca_graph(),
        )
    )
    component_scores = {
        "polarization_score": 0.5,
        "homophily_score": 0.6,
        "diversity_score": 0.7,
        "separation_score": 0.8,
    }
    score = EchoChamberAnalyzer.compute_2d_optimized_score(component_scores, weights_2d)
    landscape = EchoChamberAnalyzer.build_echo_chamber_landscape_dataframe(
        metrics_df,
        metrics_scaled,
        pca,
        _pca_user_attitudes(),
    )

    assert set(weights_2d) == {"pc1", "pc2", "variances"}
    assert sum(weights_2d["pc1"].values()) == pytest.approx(1.0)
    assert sum(weights_2d["pc2"].values()) == pytest.approx(1.0)
    assert metrics_df.columns.tolist() == [
        "user_id",
        "polarization",
        "homophily",
        "diversity",
        "separation",
    ]
    assert metrics_scaled.shape == (4, 4)
    assert pca.n_components == 2
    assert set(score) == {"score_pc1", "score_pc2", "final_combined_score"}
    assert {"PC1_Ideological", "PC2_Structural"}.issubset(landscape.columns)
    assert len(landscape) == 4


def test_echo_chamber_analyzer_landscape_data_preserves_notebook_plot_fields():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    metrics_df = pd.DataFrame(
        {
            "user_id": [10, 11, 12, 13],
            "polarization": [0.9, 0.7, 0.2, 0.1],
            "homophily": [1.0, 0.8, 0.1, 0.2],
            "diversity": [0.8, 0.6, 0.3, 0.4],
            "separation": [0.7, 0.5, 0.2, 0.1],
        }
    )
    original_columns = metrics_df.columns.tolist()
    user_df = pd.DataFrame(
        {
            "user_id": [10, 11, 12, 13],
            "community": [0, 0, 1, 1],
            "propagated_attitude": [0.8, 0.6, -0.4, -0.2],
        }
    )
    scaler = StandardScaler()
    feature_cols = ["polarization", "homophily", "diversity", "separation"]
    metrics_scaled = scaler.fit_transform(metrics_df[feature_cols])
    pca = PCA(n_components=2)
    pca.fit(metrics_scaled)

    result = EchoChamberAnalyzer.build_echo_chamber_landscape_data(
        metrics_df,
        metrics_scaled,
        pca,
        user_df,
    )
    landscape = result["landscape_df"]

    assert metrics_df.columns.tolist() == original_columns
    assert {"PC1_Ideological", "PC2_Structural"}.issubset(landscape.columns)
    assert landscape["user_id"].tolist() == [10, 11, 12, 13]
    assert landscape["community"].tolist() == [0, 0, 1, 1]
    assert landscape["propagated_attitude"].tolist() == [0.8, 0.6, -0.4, -0.2]
    assert feature_cols == ["polarization", "homophily", "diversity", "separation"]
    assert all(column in landscape.columns for column in feature_cols)
    assert result["explained_variance_ratio"].shape == (2,)
    assert result["pca_components"].shape == (2, 4)
    assert result["x_col"] == "PC1_Ideological"
    assert result["y_col"] == "PC2_Structural"
    assert result["hue_col"] == "community"
    assert result["size_col"] == "polarization"
    assert result["title"] == "Echo Chamber Landscape: Ideological vs. Structural Bias"
    assert result["size_range"] == (30, 300)
    assert result["palette"] == "Set1"
    assert result["edgecolor"] == "black"
    assert result["origin_lines"] == {"x": 0, "y": 0}
    assert [item["label"] for item in result["quadrant_annotations"]] == [
        "Double Trap\n(Radical & Isolated)",
        "Bridge Agents\n(Moderate & Diverse)",
    ]


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
