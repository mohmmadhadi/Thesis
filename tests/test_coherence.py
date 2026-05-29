from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from social_analysis.coherence import CoherenceScorer, ResponseCoherencePipeline
from social_analysis.visualization import CoherenceVisualizer


def make_lightweight_scorer() -> CoherenceScorer:
    return CoherenceScorer(
        load_embedding_model=False,
        load_cross_encoder_model=False,
    )


def test_flag_low_coherence_uses_notebook_thresholds_and_consensus():
    scorer = make_lightweight_scorer()
    df = pd.DataFrame(
        {
            "cosine_similarity": [0.20, 0.30, 0.20],
            "bs_f1": [0.80, 0.90, 0.90],
            "cross_encoder_score": [0.80, 0.40, 0.40],
        }
    )

    result = scorer.flag_low_coherence(df)

    assert result["flag_low_cosine"].tolist() == [True, False, True]
    assert result["flag_low_bertscore"].tolist() == [True, False, False]
    assert result["flag_low_crossenc"].tolist() == [False, True, True]
    assert result["flag_consensus"].tolist() == [True, False, True]


def test_thread_summary_composite_formula_matches_notebook():
    scorer = make_lightweight_scorer()
    df = pd.DataFrame(
        {
            "thread_id": [1, 1, 2],
            "id": [10, 11, 20],
            "cosine_similarity": [0.2, 0.4, 0.9],
            "bs_f1": [0.8, 0.9, 1.0],
            "cross_encoder_score": [0.3, 0.6, 0.9],
            "flag_consensus": [True, False, False],
        }
    )

    summary = scorer.summarise_by_thread(df)
    thread_one = summary[summary["thread_id"] == 1].iloc[0]

    expected = (0.3 + ((0.85 - 0.8) / 0.2) + 0.45) / 3
    assert thread_one["n_replies"] == 2
    assert thread_one["pct_flagged_consensus"] == 0.5
    assert thread_one["composite_coherence"] == pytest.approx(expected)
    assert summary["composite_coherence"].tolist() == sorted(
        summary["composite_coherence"].tolist()
    )


def test_row_composite_formula_matches_heatmap_logic():
    df = pd.DataFrame(
        {
            "cosine_similarity": [0.6],
            "bs_f1": [0.85],
            "cross_encoder_score": [0.9],
        }
    )

    result = CoherenceScorer.compute_row_composite(df)

    assert result.iloc[0] == pytest.approx((0.6 + 0.5 + 0.9) / 3)


def test_coherence_diagnostic_reports_match_notebook_tables():
    results = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "thread_id": [10, 10, 11],
            "round": [1, 2, 1],
            "root_tweet": ["root a", "root a", "root b"],
            "tweet": ["bad reply", "ok reply", "good reply"],
            "cosine_similarity": [0.2, 0.4, 0.9],
            "bs_f1": [0.8, 0.9, 0.95],
            "cross_encoder_score": [0.4, 0.6, 0.9],
            "flag_low_cosine": [True, False, False],
            "flag_low_bertscore": [True, False, False],
            "flag_low_crossenc": [True, False, False],
            "flag_consensus": [True, False, False],
        }
    )
    summary = make_lightweight_scorer().summarise_by_thread(results)

    stats = CoherenceScorer.score_descriptive_stats(results)
    failures = CoherenceScorer.threshold_failure_counts(results)
    examples = CoherenceScorer.thread_summary_examples(summary, n=1)
    flagged = CoherenceScorer.flagged_reply_examples(results)
    report = CoherenceScorer.diagnostics_report(results, summary)

    assert stats["metric"].tolist() == [
        "cosine_similarity",
        "bs_f1",
        "cross_encoder_score",
    ]
    assert "mean" in stats.columns
    assert failures.columns.tolist() == [
        "metric",
        "label",
        "threshold",
        "below_threshold",
        "total",
        "pct_below_threshold",
    ]
    assert failures.loc[failures["metric"] == "cosine_similarity", "below_threshold"].iloc[0] == 1
    assert examples["least_coherent_threads"].shape[0] == 1
    assert examples["most_coherent_threads"].shape[0] == 1
    assert flagged["id"].tolist() == [1]
    assert set(report) == {
        "score_descriptive_stats",
        "threshold_failure_counts",
        "least_coherent_threads",
        "most_coherent_threads",
        "flagged_reply_examples",
    }


def test_coherence_plot_ready_tables_match_notebook_shapes():
    results = pd.DataFrame(
        {
            "thread_id": [1, 1, 2, 2],
            "round": [1, 2, 1, 2],
            "cosine_similarity": [0.2, 0.4, 0.8, 0.9],
            "bs_f1": [0.8, 0.85, 0.9, 0.95],
            "cross_encoder_score": [0.3, 0.5, 0.8, 0.9],
        }
    )

    decay = CoherenceScorer.coherence_decay_table(results)
    pivot = CoherenceScorer.thread_heatmap_pivot(results)

    assert decay.columns.tolist() == [
        "round",
        "cosine_similarity_mean",
        "cosine_similarity_std",
        "cosine_similarity_count",
        "bs_f1_mean",
        "bs_f1_std",
        "bs_f1_count",
        "cross_encoder_score_mean",
        "cross_encoder_score_std",
        "cross_encoder_score_count",
    ]
    assert pivot.index.tolist() == [1, 2]
    assert pivot.columns.tolist() == [1, 2]
    assert pivot.loc[1, 1] == pytest.approx((0.2 + ((0.8 - 0.7) / 0.3) + 0.3) / 3)


def test_coherence_visualizer_uses_notebook_filenames_and_skips_missing(tmp_path):
    visualizer = CoherenceVisualizer()
    results = pd.DataFrame(
        {
            "thread_id": [1, 1, 2, 2, 3, 3],
            "round": [1, 2, 1, 2, 1, 2],
            "cosine_similarity": [0.2, 0.4, 0.8, 0.9, 0.3, 0.5],
            "bs_f1": [0.8, 0.85, 0.9, 0.95, 0.86, 0.88],
            "cross_encoder_score": [0.3, 0.5, 0.8, 0.9, 0.4, 0.6],
            "flag_consensus": [True, False, False, False, True, False],
        }
    )

    outputs = visualizer.plot_all(results, tmp_path)

    assert [path.name for path in outputs] == [
        "fig1_score_distributions.png",
        "fig2_score_correlations.png",
        "fig3_coherence_decay.png",
        "fig4_thread_heatmap.png",
    ]
    assert all(path.exists() for path in outputs)
    assert visualizer.plot_score_distributions(pd.DataFrame({"tweet": ["x"]}), tmp_path) is None


def test_prepare_thread_root_pairs_matches_notebook_logic():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "thread_id": [100, 100, 100],
            "comment_to": [-1, 1, 2],
            "round": [0, 1, 2],
            "user_id": [10, 11, 12],
            "tweet": ["root", "reply one", "reply two"],
        }
    )

    roots, replies = ResponseCoherencePipeline.prepare_thread_root_pairs(df)

    assert roots.loc[100, "root_id"] == 1
    assert replies["id"].tolist() == [2, 3]
    assert replies["root_id"].tolist() == [1, 1]
    assert replies["root_tweet"].tolist() == ["root", "root"]


def test_prepare_direct_parent_pairs_matches_notebook_logic():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "thread_id": [100, 100, 100],
            "comment_to": [-1, 1, 2],
            "round": [0, 1, 2],
            "user_id": [10, 11, 12],
            "tweet": ["root", "reply one", "reply two"],
        }
    )

    roots, replies = ResponseCoherencePipeline.prepare_direct_parent_pairs(df)

    assert roots["root_id"].tolist() == [1, 2]
    assert replies["id"].tolist() == [2, 3]
    assert replies["root_id"].tolist() == [1, 2]
    assert replies["root_tweet"].tolist() == ["root", "reply one"]


def test_prepare_pairs_falls_back_to_min_round_when_no_comment_root():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "thread_id": [100, 100],
            "comment_to": [0, 1],
            "round": [1, 2],
            "user_id": [10, 11],
            "tweet": ["first", "reply"],
        }
    )

    roots, replies = ResponseCoherencePipeline.prepare_thread_root_pairs(df)

    assert roots.loc[100, "root_id"] == 1
    assert replies["id"].tolist() == [2]
