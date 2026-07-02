import math

from guessbench.scoring import (
    bootstrap_ci,
    effective_interpretations,
    entropy_nats,
    score_assignments,
)


def test_entropy_single_cluster_is_zero():
    assert entropy_nats([0] * 20) == 0.0
    assert effective_interpretations([0] * 20) == 1.0


def test_entropy_all_distinct():
    n = 20
    assignments = list(range(n))
    assert math.isclose(entropy_nats(assignments), math.log(n))
    assert math.isclose(effective_interpretations(assignments), n)


def test_entropy_hand_computed_two_clusters():
    # 15/20 and 5/20: H = -(0.75 ln 0.75 + 0.25 ln 0.25)
    assignments = [0] * 15 + [1] * 5
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert math.isclose(entropy_nats(assignments), expected)
    assert math.isclose(effective_interpretations(assignments), math.exp(expected))


def test_entropy_uniform_four_clusters():
    assignments = [0, 1, 2, 3] * 5
    assert math.isclose(entropy_nats(assignments), math.log(4))
    assert math.isclose(effective_interpretations(assignments), 4.0)


def test_bootstrap_ci_degenerate_is_tight():
    low, high = bootstrap_ci([0] * 20, iterations=200, seed=1)
    assert low == 1.0
    assert high == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    assignments = [0] * 10 + [1] * 6 + [2] * 4
    ei = effective_interpretations(assignments)
    low, high = bootstrap_ci(assignments, iterations=1000, seed=1)
    assert low <= ei <= high
    assert low >= 1.0
    assert high <= len(assignments)


def test_bootstrap_deterministic_for_seed():
    assignments = [0] * 10 + [1] * 10
    assert bootstrap_ci(assignments, iterations=300, seed=7) == bootstrap_ci(
        assignments, iterations=300, seed=7
    )


def test_score_assignments_fields():
    assignments = [0] * 10 + [1] * 6 + [2] * 4
    score = score_assignments(assignments, bootstrap_iterations=200, seed=0)
    assert score.n == 20
    assert score.k == 3
    assert math.isclose(score.ei, math.exp(score.entropy))
    assert score.ei_ci_low <= score.ei <= score.ei_ci_high
