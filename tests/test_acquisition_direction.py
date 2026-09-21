"""Public AskTell acquisition direction and disabled retrieval regressions."""
import numpy as np
import pytest

from boicl.asktell import AskTellFewShotTopk
from boicl.aqfxns import greedy, log_expected_improvement, upper_confidence_bound
from boicl.llm_model import GaussDist, make_dd


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("base", [-20.0, 20.0])
@pytest.mark.parametrize("spread", [0.0, 1.5])
@pytest.mark.parametrize(
    "acquisition",
    [
        "expected_improvement",
        "probability_of_improvement",
        "log_expected_improvement",
        "upper_confidence_bound",
        "greedy",
    ],
)
def test_public_ask_ranks_raw_objectives_in_requested_direction(
    maximize, base, spread, acquisition
):
    model = AskTellFewShotTopk(selector_k=None, maximize=maximize)
    model.tell("seed1", base - 1)
    model.tell("seed2", base + 1)
    distributions = [
        make_dd([base - 3 - spread, base - 3 + spread], [0.5] * 2),
        make_dd([base + 3 - spread, base + 3 + spread], [0.5] * 2),
    ]
    model.predict = lambda *args, **kwargs: distributions
    selected, scores, means = model.ask(
        ["low", "high"], aq_fxn=acquisition, inv_filter=0, k=2
    )
    assert selected[0] == ("high" if maximize else "low")
    assert scores[0] > scores[1]
    assert means == ([base + 3, base - 3] if maximize else [base - 3, base + 3])


@pytest.mark.parametrize("aug_random_filter", [0, 1])
def test_zero_inverse_filter_scores_full_eligible_pool_and_preserves_failed_ids(
    monkeypatch, aug_random_filter
):
    model = AskTellFewShotTopk(selector_k=None, maximize=False)
    model.tell("seed1", -5)
    model.tell("seed2", -6)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Disabled inverse filtering must not generate or retrieve a proposal"
        )

    monkeypatch.setattr(model, "inv_predict", forbidden)
    from boicl.pool import Pool

    monkeypatch.setattr(Pool, "approx_sample", forbidden)
    queried = []

    def predict(candidates, **kwargs):
        queried.extend(candidates)
        return [
            make_dd([-7, -7], [0.5, 0.5]),
            make_dd([], []),
            make_dd([-9, -9], [0.5, 0.5]),
        ]

    model.predict = predict
    selected, scores, means = model.ask(
        ["seed1", "a", "failed", "c", "seed2"],
        inv_filter=0,
        aug_random_filter=aug_random_filter,
        k=3,
    )
    assert queried == ["a", "failed", "c"]
    assert selected == ["c", "a"] and means == [-9, -7] and scores == [3, 1]
    assert [
        (row["candidate"], row["status"]) for row in model.last_prediction_records
    ] == [
        ("a", "scored"),
        ("failed", "insufficient_or_invalid_samples"),
        ("c", "scored"),
    ]


@pytest.mark.parametrize("gaussian", [True, False])
def test_minimizing_ucb_rewards_spread_without_flipping_it(gaussian):
    stable = GaussDist(10, 0) if gaussian else make_dd([10, 10], [0.5, 0.5])
    spread = GaussDist(12, 4) if gaussian else make_dd([8, 16], [0.5, 0.5])
    assert upper_confidence_bound(stable, 10, 1, maximize=False) == -10
    assert upper_confidence_bound(spread, 10, 1, maximize=False) == -8
    assert upper_confidence_bound(spread, 10, 1) == 16


def test_log_ei_direction_and_margin_match_raw_improvement():
    dist = make_dd([-12, -8], [0.5, 0.5])
    assert log_expected_improvement(dist, -9, xi=1, maximize=False) == pytest.approx(
        np.log(1)
    )
    assert log_expected_improvement(dist, -9, maximize=True) == pytest.approx(
        np.log(0.5)
    )
    assert greedy(make_dd([-12, -12, -8], [1 / 3] * 3), -9, maximize=False) == 12


@pytest.mark.parametrize("k", [1, 10])
def test_enabled_inverse_filter_adds_distinct_random_candidate_and_respects_limits(
    monkeypatch, k
):
    from boicl.pool import Pool

    model = AskTellFewShotTopk(selector_k=None, objective_bounds=(0, 100))
    model.tell("seed1", 10)
    model.tell("seed2", 20)
    original_pool = Pool(["seed1", "a", "b", "b", "c", "d", "e", "seed2"])
    original_pool.choose("e")
    calls, scored = [], []

    def inverse(target, **kwargs):
        calls.append(("inverse", target))
        assert 20 <= target <= 100
        return "query recipe"

    def retrieve(pool, query, count, **kwargs):
        assert list(pool) == ["a", "b", "c", "d"]
        assert query == "query recipe" and count == 2
        assert kwargs == {"lambda_mult": 0.25}
        calls.append(("retrieve", count))
        return ["a", "b"]

    def sample(pool, count):
        available = list(pool)
        assert available == [
            "c",
            "d",
        ], "Random additions must exclude retrieved and observed candidates"
        assert count == 1
        calls.append(("sample", count))
        return available[:count]

    def predict(candidates, **kwargs):
        scored.extend(candidates)
        return [make_dd([value, value], [0.5, 0.5]) for value in [40, 50, 60]]

    monkeypatch.setattr(model, "inv_predict", inverse)
    monkeypatch.setattr(Pool, "approx_sample", retrieve)
    monkeypatch.setattr(Pool, "sample", sample)
    monkeypatch.setattr(model, "predict", predict)
    selected, scores, means = model.ask(
        original_pool, inv_filter=2, aug_random_filter=1, lambda_mult=0.25, k=k
    )
    assert [call[0] for call in calls] == ["inverse", "retrieve", "sample"]
    assert scored == ["a", "b", "c"]
    assert selected == ["c", "b", "a"][:k]
    assert scores == [40, 30, 20][:k] and means == [60, 50, 40][:k]
    assert len(selected) == min(k, 3)
    assert list(original_pool) == ["seed1", "a", "b", "c", "d", "seed2"]


def test_filter_budget_covering_remaining_pool_scores_each_eligible_candidate_once(
    monkeypatch,
):
    from boicl.pool import Pool

    model = AskTellFewShotTopk(selector_k=None)
    model.tell("seed1", 1)
    model.tell("seed2", 2)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Retrieval and random sampling are unnecessary when the filter covers the whole pool"
        )

    monkeypatch.setattr(model, "inv_predict", forbidden)
    monkeypatch.setattr(Pool, "approx_sample", forbidden)
    monkeypatch.setattr(Pool, "sample", forbidden)
    scored = []

    def predict(candidates, **kwargs):
        scored.extend(candidates)
        return [make_dd([value, value], [0.5, 0.5]) for value in [3, 4]]

    monkeypatch.setattr(model, "predict", predict)
    selected, scores, means = model.ask(
        ["seed1", "a", "b", "seed2", "a"], inv_filter=2, aug_random_filter=1, k=5
    )
    assert scored == ["a", "b"]
    assert selected == ["b", "a"] and scores == [2, 1] and means == [4, 3]


@pytest.mark.parametrize("candidates", [[], ["seed1", "seed2", "seed1"]])
def test_public_ask_exhausted_pool_returns_empty_without_prediction_or_retrieval(
    monkeypatch, candidates
):
    from boicl.pool import Pool

    model = AskTellFewShotTopk(selector_k=None)
    model.tell("seed1", 1)
    model.tell("seed2", 2)

    def forbidden(*args, **kwargs):
        pytest.fail("An exhausted pool must make no model, retrieval or random request")

    monkeypatch.setattr(model, "inv_predict", forbidden)
    monkeypatch.setattr(model, "predict", forbidden)
    monkeypatch.setattr(Pool, "approx_sample", forbidden)
    monkeypatch.setattr(Pool, "sample", forbidden)
    assert model.ask(candidates, inv_filter=2, aug_random_filter=1, k=10) == (
        [],
        [],
        [],
    )
