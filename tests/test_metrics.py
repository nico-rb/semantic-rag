"""
Unit tests for the pure retrieval-quality logic: the NDCG@10 and Recall@k
metrics and the Reciprocal Rank Fusion. These are the pieces every reported
number depends on, and they run without network access or on-disk indexes.
"""

import math

from evaluate import ndcg_at_k, recall_at_k
from src.retrieval.fusion import reciprocal_rank_fusion

# --------------------------------------------------------------
# ndcg_at_k
# --------------------------------------------------------------


def test_perfect_ranking_scores_one():
    relevant = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], relevant) == 1.0


def test_inverted_ranking_scores_below_one():
    relevant = {"a": 1}
    # The only relevant doc is last, so DCG < IDCG.
    assert ndcg_at_k(["x", "y", "a"], relevant) < 1.0


def test_no_relevant_retrieved_scores_zero():
    relevant = {"a": 1}
    assert ndcg_at_k(["x", "y", "z"], relevant) == 0.0


def test_no_judgments_scores_zero_without_dividing_by_zero():
    # idcg == 0; must return 0.0 rather than raising.
    assert ndcg_at_k(["a", "b"], {}) == 0.0


def test_known_value_single_relevant_at_rank_two():
    # One relevant doc placed second. Ideal puts it first.
    relevant = {"a": 1}
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert ndcg_at_k(["x", "a"], relevant) == expected


def test_graded_relevance_rewards_higher_grade_first():
    relevant = {"a": 2, "b": 1}
    # Best order puts the grade-2 doc first; this is the ideal, so score is 1.0.
    assert ndcg_at_k(["a", "b"], relevant) == 1.0
    # Swapping them is worse.
    assert ndcg_at_k(["b", "a"], relevant) < 1.0


def test_only_top_k_counts():
    # A relevant doc beyond position k must not contribute.
    relevant = {"a": 1}
    predicted = ["x"] * 10 + ["a"]
    assert ndcg_at_k(predicted, relevant, k=10) == 0.0


# --------------------------------------------------------------
# recall_at_k
# --------------------------------------------------------------


def test_all_relevant_retrieved_scores_one():
    relevant = {"a": 1, "b": 1}
    assert recall_at_k(["a", "b", "c"], relevant, k=3) == 1.0


def test_recall_no_relevant_retrieved_scores_zero():
    assert recall_at_k(["x", "y"], {"a": 1}, k=2) == 0.0


def test_partial_recall():
    # One of two relevant docs retrieved.
    relevant = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x"], relevant, k=2) == 0.5


def test_no_judgments_scores_zero():
    assert recall_at_k(["a", "b"], {}, k=2) == 0.0


def test_relevant_beyond_k_does_not_count():
    # The relevant doc sits at position 3 but k=2, so it is not "retrieved".
    relevant = {"a": 1}
    assert recall_at_k(["x", "y", "a"], relevant, k=2) == 0.0


def test_zero_grade_docs_are_not_relevant():
    # A doc graded 0 is not a relevant target; recall has nothing to find.
    relevant = {"a": 0}
    assert recall_at_k(["a"], relevant, k=1) == 0.0


# --------------------------------------------------------------
# reciprocal_rank_fusion
# --------------------------------------------------------------


def test_doc_ranked_high_in_both_lists_wins():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    # "a" is rank 1 in both lists, so it must come out on top.
    assert fused[0][0] == "a"


def test_rrf_score_matches_formula():
    # Single list, single doc at rank 1: score == 1 / (k + 1).
    fused = reciprocal_rank_fusion([["a"]], k=60)
    assert fused == [("a", 1 / 61)]


def test_doc_in_both_lists_beats_doc_in_one():
    # "shared" appears in both lists (lower per-list rank) but accumulates;
    # "solo" is rank 1 in one list only.
    fused = dict(reciprocal_rank_fusion([["solo", "shared"], ["x", "shared"]]))
    assert fused["shared"] > fused["solo"]


def test_empty_rankings_return_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
