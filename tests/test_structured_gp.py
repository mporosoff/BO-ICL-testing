"""Offline numerical checks for the source GP and corrected bounded posterior."""

import itertools
import socket

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from boicl.structured_gp import (
    DEFAULTS,
    FEATURE_ORDER,
    PaddedLogitTransform,
    StructuredGP,
    bounded_expected_improvement,
    bounded_mixture_summary,
    effective_noise,
    log_marginal_likelihood,
    log_prior,
    matern52,
    sample_posterior,
    transform_features,
)


def design(t=550, ramp=5, flow=30, hold=0.5, ratio=2, gas="H2", cid=None):
    return {
        "candidate_id": cid or f"{t}-{ramp}-{flow}-{hold}-{ratio}-{gas}",
        "temperature_C": t,
        "ramp_C_per_min": ramp,
        "flow_sccm": flow,
        "hold_h": hold,
        "sucrose_to_AMT_mass_ratio": ratio,
        "AMT_to_sucrose_mass_ratio": 1 / ratio,
        "gas": gas,
    }


def observation(candidate, outcome=50, oid=None, **extra):
    return {
        "candidate_id": candidate["candidate_id"],
        "moc_wt_pct": outcome,
        "moc_wt_pct_sigma": 1.5,
        "gof": 1.0,
        "closure_gap_wt_pct": 0.0,
        "observation_id": oid or candidate["candidate_id"] + "-obs",
        **extra,
    }


def source_seeds():
    candidates = [
        design(800, 15, 30, 0.5, 1, "N2"),
        design(600, 15, 30, 2, 1, "H2"),
        design(900, 15, 30, 2, 1, "H2"),
    ]
    observations = [
        observation(candidates[0], 72.1, "M7", gof=0.517),
        observation(candidates[1], 83.8, "M12", gof=0.548, moc_wt_pct_sigma=5.67),
        observation(candidates[2], 23.4, "M13", gof=0.932),
    ]
    return candidates, observations


def fast_settings(**extra):
    # Explicitly shorter numerical-test chain; production defaults remain pinned.
    return {"burn_in": 25, "retained_draws": 80, "predict_thin": 8, **extra}


def test_source_feature_order_full_bounds_and_constant_seed_dimensions():
    low = design(gas="N2")
    high = design(900, 15, 100, 10, 0.5, "H2")
    np.testing.assert_allclose(transform_features([low, high]), [[0] * 6, [1] * 6])
    seeds, _ = source_seeds()
    x = transform_features(seeds)
    assert FEATURE_ORDER == (
        "temperature_C",
        "ramp_C_per_min",
        "flow_sccm",
        "hold_h",
        "AMT_to_sucrose_mass_ratio",
        "gas",
    )
    np.testing.assert_allclose(x[:, 1], 1)
    np.testing.assert_allclose(x[:, 2], 0)
    np.testing.assert_allclose(x[:, 4], 0.5)
    np.testing.assert_allclose(x[1], [1 / 7, 1, 0, np.log(4) / np.log(20), 0.5, 1])
    changed_labels = [
        {**row, "moc_wt_pct": 100, "gof": 200, "closure_gap_wt_pct": 99}
        for row in seeds
    ]
    np.testing.assert_array_equal(transform_features(changed_labels), x)


@pytest.mark.parametrize(
    "patch",
    [
        {"temperature_C": 555},
        {"ramp_C_per_min": 0},
        {"flow_sccm": float("nan")},
        {"hold_h": 0},
        {"AMT_to_sucrose_mass_ratio": -1},
        {"gas": "Ar"},
        {"AMT_to_sucrose_mass_ratio": 2, "sucrose_to_AMT_mass_ratio": 2},
    ],
)
def test_feature_validation_rejects_bad_units_grid_and_ratio(patch):
    with pytest.raises(ValueError):
        transform_features([{**design(), **patch}])


def test_reciprocal_ratio_inferred_and_custom_constant_space_safe():
    row = design(ratio=0.5)
    row.pop("AMT_to_sucrose_mass_ratio")
    assert transform_features([row])[0, 4] == 1
    full = [design(550), design(900)]
    x = transform_features(full, full_space=full)
    np.testing.assert_array_equal(x[0], np.zeros(6))
    np.testing.assert_array_equal(x[1], [1, 0, 0, 0, 0, 0])


def test_noise_formula_seed_override_missing_policy_and_negative_gap():
    _, observations = source_seeds()
    sigmas, flags = effective_noise(observations)
    np.testing.assert_array_equal(sigmas, [1.5, 5.67, 1.5])
    assert flags == []
    candidate = design()
    cases = [
        observation(candidate, moc_wt_pct_sigma=0, gof=0.1),
        observation(candidate, moc_wt_pct_sigma=2, gof=3, closure_gap_wt_pct=-8),
    ]
    np.testing.assert_allclose(effective_noise(cases)[0], [0.5, 10])
    missing = observation(candidate)
    missing.pop("closure_gap_wt_pct")
    with pytest.raises(ValueError, match="closure_gap_wt_pct"):
        effective_noise([missing])
    assumed, flags = effective_noise(
        [missing], missing_policy="fallback", fallbacks={"closure_gap_wt_pct": 2}
    )
    assert assumed[0] == 2.5
    assert flags[0]["origin"] == "explicit fallback"
    assert flags[0]["assumed_value"] == 2
    with pytest.raises(ValueError):
        effective_noise([observation(candidate, moc_wt_pct_sigma=-1)])


def test_padded_logit_variance_derivative_and_standardization_once():
    values = np.asarray([0, 23.4, 72.1, 83.8, 100])
    sigmas = np.asarray([0.5, 1.5, 1.5, 5.67, 0.5])
    transform = PaddedLogitTransform().fit(values)
    expected_z = np.log((values + 0.65) / (100.65 - values))
    np.testing.assert_allclose(
        transform.forward(values),
        (expected_z - expected_z.mean()) / expected_z.std(ddof=1),
    )
    derivative = 1 / (values + 0.65) + 1 / (100.65 - values)
    expected_var = (sigmas * derivative) ** 2 / expected_z.var(ddof=1)
    np.testing.assert_allclose(transform.variance(values, sigmas), expected_var)
    np.testing.assert_allclose(
        transform.inverse(transform.forward(values)), values, atol=1e-13
    )
    np.testing.assert_allclose(transform.inverse([-1e6, 1e6]), [0, 100])


@pytest.mark.parametrize("values", [[], [0], [100], [50, 50, 50], [0, 0]])
def test_small_or_constant_outcomes_use_declared_unit_scale(values):
    transform = PaddedLogitTransform().fit(values)
    assert transform.scale == 1
    assert np.all(np.isfinite(transform.bounds))
    assert np.all(np.isfinite(transform.variance(values, np.full(len(values), 0.5))))
    assert "unit logit" in transform.scaling_policy


def test_matern_covariance_and_log_prior_are_source_ard_values():
    a = np.asarray([[0] * 6, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], float)
    ls = np.asarray([1, 2, 3, 4, 5, 6], float)
    covariance = matern52(a, a, ls, 2.0)
    expected = 2 * (1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5))
    assert covariance[0, 1] == pytest.approx(expected)
    assert covariance[0, 2] > covariance[0, 1]
    np.testing.assert_allclose(np.diag(covariance), 2)
    theta = np.r_[np.full(6, np.log(np.sqrt(6))), 0]
    assert log_prior(theta) == 0
    theta[0] += 0.75
    theta[-1] = 1
    assert log_prior(theta) == pytest.approx(-1)


def test_sampler_matches_pinned_source_one_shot_tuning_and_rng():
    candidates, observations = source_seeds()
    x = transform_features(candidates)
    values = np.asarray([row["moc_wt_pct"] for row in observations])
    transform = PaddedLogitTransform().fit(values)
    y = transform.forward(values)
    noise = transform.variance(values, effective_noise(observations)[0])
    # Independent transcription of the notebook's sampling loop, with a tiny chain.
    rng = np.random.default_rng(616)
    th = np.r_[np.full(6, np.log(np.sqrt(6))), 0.0]
    lp = log_marginal_likelihood(th, x, y, noise) + log_prior(th)
    expected, step, accepted = [], 0.3, 0
    n, burn = 25, 15
    for i in range(n + burn):
        prop = th + step * rng.standard_normal(7)
        proposed_lp = log_marginal_likelihood(prop, x, y, noise) + log_prior(prop)
        if np.log(rng.random()) < proposed_lp - lp:
            th, lp, accepted = prop, proposed_lp, accepted + 1
        if i == burn - 1:
            step *= float(np.clip((accepted / burn) / 0.3, 0.3, 3.0))
        if i >= burn:
            expected.append(th.copy())
    actual, diagnostics = sample_posterior(x, y, noise, retained_draws=n, burn_in=burn)
    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["final_proposal_step"] == step
    assert diagnostics["random_state"] == rng.bit_generator.state
    assert diagnostics["acceptance_rate"] == accepted / (n + burn)
    assert len(diagnostics["effective_sample_size"]) == 7
    assert "do not establish convergence" in diagnostics["warnings"][0]
    assert DEFAULTS["burn_in"] == 1000 and DEFAULTS["retained_draws"] == 4000
    assert DEFAULTS["predict_thin"] == 20 and DEFAULTS["proposal_step"] == 0.3


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_capped_ei_matches_independent_quadrature_and_has_no_impossible_improvement(
    direction,
):
    lo, hi = -2.0, 3.0
    best, xi = (1.0 if direction == "maximize" else -0.7), 0.01
    means, variances = np.asarray([-8, -1, 2, 8], float), np.asarray(
        [0.7, 1.3, 2.1, 0.6]
    )
    actual = bounded_expected_improvement(
        means, variances, best, [lo, hi], xi=xi, direction=direction
    )
    sign = 1 if direction == "maximize" else -1
    for mu, variance, value in zip(means, variances, actual):
        sd = np.sqrt(variance)
        improvement = lambda z: max(sign * (z - best) - xi, 0)
        integral = quad(
            lambda z: improvement(z) * norm.pdf(z, mu, sd),
            lo,
            hi,
            epsabs=1e-11,
            points=[best],
        )[0]
        integral += improvement(lo) * norm.cdf(lo, mu, sd)
        integral += improvement(hi) * norm.sf(hi, mu, sd)
        assert value == pytest.approx(integral, abs=2e-8)
    physical_limit = hi if direction == "maximize" else lo
    np.testing.assert_array_equal(
        bounded_expected_improvement(
            means, variances, physical_limit, [lo, hi], xi=0, direction=direction
        ),
        0,
    )


def test_degenerate_component_ei_retains_deterministic_improvement():
    result = bounded_expected_improvement([0, 2, 20], [0, 0, 0], 1, [-2, 3], xi=0)
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_bounded_mixture_moments_quantiles_and_endpoint_masses_against_quadrature():
    transform = PaddedLogitTransform(mean=0.8, scale=1.7)
    means = np.asarray([-3.0, 0.6, 3.4])[:, None]
    variances = np.asarray([0.3, 1.4, 2.8])[:, None]
    actual = bounded_mixture_summary(means, variances, transform, quadrature_nodes=160)
    lo, hi = transform.bounds
    moments = []
    conditional_variances = []
    for mu, variance in zip(means[:, 0], variances[:, 0]):
        sd = np.sqrt(variance)
        f = lambda z: 101.3 * expit(0.8 + 1.7 * z) - 0.65
        moment1 = quad(lambda z: f(z) * norm.pdf(z, mu, sd), lo, hi, epsabs=1e-9)[
            0
        ] + 100 * norm.sf(hi, mu, sd)
        moment2 = quad(lambda z: f(z) ** 2 * norm.pdf(z, mu, sd), lo, hi, epsabs=1e-9)[
            0
        ] + 10000 * norm.sf(hi, mu, sd)
        moments.append(moment1)
        conditional_variances.append(moment2 - moment1**2)
    expected_mean = np.mean(moments)
    expected_variance = np.mean(conditional_variances) + np.var(moments)
    assert actual["mean"][0] == pytest.approx(expected_mean, abs=0.002)
    assert actual["variance"][0] == pytest.approx(expected_variance, abs=0.03)
    assert actual["within_component_variance"][0] == pytest.approx(
        np.mean(conditional_variances), abs=0.03
    )
    assert actual["lower_boundary_mass"][0] == pytest.approx(
        np.mean(norm.cdf((lo - means) / np.sqrt(variances)))
    )
    assert actual["upper_boundary_mass"][0] == pytest.approx(
        np.mean(norm.sf((hi - means) / np.sqrt(variances)))
    )
    # This mixture has endpoint atoms, so its outer 95% interval reaches both bounds.
    assert actual["lower95"][0] == pytest.approx(0, abs=1e-10)
    assert actual["upper95"][0] == pytest.approx(100, abs=1e-10)
    assert abs(expected_mean - transform.inverse(means.mean())) > 0.5


def test_mixture_nonlinear_quantiles_match_seeded_monte_carlo():
    transform = PaddedLogitTransform(mean=0.4, scale=1.3)
    means = np.asarray([-0.7, 0.3, 1.2])
    variances = np.asarray([0.2, 0.5, 0.3])
    summary = bounded_mixture_summary(means, variances, transform)
    rng = np.random.default_rng(1942)
    component = rng.integers(0, 3, size=500_000)
    draws = transform.inverse(
        rng.normal(means[component], np.sqrt(variances[component]))
    )
    assert summary["mean"][0] == pytest.approx(draws.mean(), abs=0.12)
    assert summary["sd"][0] == pytest.approx(draws.std(), abs=0.1)
    np.testing.assert_allclose(
        [summary["lower95"][0], summary["upper95"][0]],
        np.quantile(draws, [0.025, 0.975]),
        atol=0.2,
    )


def test_zero_variance_mixture_remains_a_mixture_with_between_component_uncertainty():
    transform = PaddedLogitTransform()
    bounds = transform.bounds
    summary = bounded_mixture_summary(
        [bounds[0] - 1, 0, bounds[1] + 1], [0, 0, 0], transform
    )
    assert summary["mean"][0] == pytest.approx(50)
    assert summary["variance"][0] == pytest.approx(np.var([0, 50, 100]))
    assert summary["within_component_variance"][0] == 0
    assert summary["lower_boundary_mass"][0] == pytest.approx(1 / 3)


def test_source_fixture_maximin_recommendation_no_network_or_credentials(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Structured GP attempted a network call")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidates = [
        design(t, ramp, flow, hold, ratio, gas)
        for ratio, t, ramp, hold, gas, flow in itertools.product(
            [2, 1, 0.5],
            range(550, 901, 10),
            [5, 10, 15],
            [0.5, 2, 5, 10],
            ["H2", "N2"],
            [30, 60, 100],
        )
    ]
    _, observations = source_seeds()
    assert len(candidates) == 7776
    engine = StructuredGP(fast_settings()).fit(candidates, observations)
    result = engine.recommend(candidates)
    assert result["candidate_id"] == design(550, 5, 100, 10, 2, "N2")["candidate_id"]
    assert result["stage"] == "maximin"
    assert "3 of 10" in result["selection_reason"]
    assert len(result["scores"]) == 7773
    assert (
        0
        <= result["prediction"]["lower95"]
        <= result["prediction"]["mean"]
        <= result["prediction"]["upper95"]
        <= 100
    )
    assert result["diagnostics"]["effective_sigmas_pp"] == [1.5, 5.67, 1.5]


def test_nine_vs_ten_distinct_measured_not_pending_excluded_or_repeat_counts():
    candidates = [design(t) for t in range(550, 721, 10)]
    first_nine = [observation(row, 10 + 3 * i) for i, row in enumerate(candidates[:9])]
    first_nine += [observation(candidates[0], 0, "physical-repeat-1")]
    first_nine += [observation(candidates[9], 99, "pending", record_status="pending")]
    first_nine += [observation(candidates[10], 99, "excluded", training_included=False)]
    engine = StructuredGP(fast_settings()).fit(candidates, first_nine)
    assert engine.recommend(candidates)["stage"] == "maximin"
    assert len(engine.observations) == 10 and len(engine.measured_ids) == 9
    engine.fit(
        candidates, first_nine + [observation(candidates[9], 100, "tenth-measured")]
    )
    result = engine.recommend(candidates, excluded_ids=[candidates[10]["candidate_id"]])
    assert result["stage"] == "expected_improvement"
    assert result["unique_design_count"] == 10
    assert result["measurement_count"] == 11
    assert result["score"] == 0  # An observed 100 wt% cannot be improved.
    assert all(
        row["candidate_id"] != candidates[10]["candidate_id"]
        for row in result["scores"]
    )


@pytest.mark.parametrize("value", [0, 50, 100])
def test_single_constant_and_boundary_gp_fits_are_finite_with_uncertainty(value):
    candidates = [design(550), design(600), design(900)]
    engine = StructuredGP(fast_settings()).fit(
        candidates, [observation(candidates[0], value)]
    )
    result = engine.predict(candidates[1:])
    assert np.all(np.isfinite(result["mean"]))
    assert np.all(result["sd"] > 0)
    assert np.all(result["lower95"] >= 0) and np.all(result["upper95"] <= 100)
    future = engine.predict(candidates[1:], observation_sigma_pp=8)
    assert future["uncertainty_type"].startswith("future-measurement")
    engine.fit(
        candidates,
        [observation(candidates[0], value), observation(candidates[1], value)],
    )
    assert engine.transform.scale == 1.0
    assert np.isfinite(engine.recommend(candidates)["prediction"]["sd"])


def test_empty_campaign_initial_design_exhaustion_and_stable_ties():
    candidates = [design(550), design(900)]
    engine = StructuredGP(fast_settings()).fit(candidates, [])
    result = engine.recommend(candidates)
    assert result["status"] == "initial_design" and result["prediction"] is None
    assert result["candidate_id"] == candidates[0]["candidate_id"]
    assert (
        engine.recommend(
            candidates, excluded_ids=[row["candidate_id"] for row in candidates]
        )["status"]
        == "exhausted"
    )
    with pytest.raises(ValueError, match="at least one"):
        engine.predict(candidates)


def test_minimization_uses_lower_incumbent_and_lower_physical_bound():
    candidates = [design(t) for t in range(550, 681, 10)]
    observations = [observation(row, i * 5) for i, row in enumerate(candidates[:10])]
    engine = StructuredGP(fast_settings(direction="minimize")).fit(
        candidates, observations
    )
    result = engine.recommend(candidates)
    assert result["stage"] == "expected_improvement" and result["score"] == 0


def test_fit_and_prediction_cancellation_preserve_previous_complete_fit():
    candidates, observations = source_seeds()
    engine = StructuredGP(fast_settings()).fit(candidates, observations)
    samples = engine.samples.copy()
    with pytest.raises(InterruptedError):
        engine.fit(candidates, [], cancel=lambda: True)
    np.testing.assert_array_equal(engine.samples, samples)
    with pytest.raises(InterruptedError):
        engine.predict(candidates, cancel=lambda: True)


def test_refinements_not_counted_as_independent_physical_measurements():
    candidate = design()
    rows = [
        observation(candidate, 50, "v1", measurement_id="measurement-1"),
        observation(candidate, 55, "v2", measurement_id="measurement-1"),
    ]
    with pytest.raises(ValueError, match="one active refinement"):
        StructuredGP(fast_settings()).fit([candidate], rows)
