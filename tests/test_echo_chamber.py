from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.echo_chamber import ANTI_ANCHORS, PRO_ANCHORS, StanceEstimator


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
