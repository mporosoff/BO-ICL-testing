"""Offline six-parameter GP for the crystal 810c3f7 MoC continuation.

The training transform, covariance, priors and random-walk sampler follow the
pinned notebook. The operational posterior is an explicitly corrected mixture:
each Gaussian component is clamped in standardized padded-logit coordinates
before acquisition and inverse transformation. This places probability mass at
0 and 100 wt%; it is not a truncated Gaussian or a moment-matched Gaussian.

No model-provider or embedding client is imported or created by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import expit, ndtr, ndtri


METHOD_VERSION = "moc-structured-gp-bounded-mixture-v1"
SAMPLER_VERSION = "crystal-810c3f7-rwm-one-shot-burn-adjustment-v1"
FEATURE_ORDER = (
    "temperature_C",
    "ramp_C_per_min",
    "flow_sccm",
    "hold_h",
    "AMT_to_sucrose_mass_ratio",
    "gas",
)
FEATURE_SPEC = {
    "order": list(FEATURE_ORDER),
    "transforms": [
        "(T-550)/350",
        "(r-5)/10",
        "(f-30)/70",
        "(ln(h)-ln(0.5))/ln(20)",
        "(log2(q)+1)/2",
        "H2=1;N2=0",
    ],
    "physical_bounds": [[550, 900], [5, 15], [30, 100], [0.5, 10], [0.5, 2], [0, 1]],
    "scaling_corpus": "fixed full design space; no measured outcomes",
}
DEFAULTS = {
    "seed": 616,
    "gof_power": 1.0,
    "sigma_floor_pp": 0.5,
    "logit_delta_pp": 0.65,
    "burn_in": 1000,
    "retained_draws": 4000,
    "predict_thin": 20,
    "proposal_step": 0.3,
    "ei_after_unique_measured_designs": 10,
    "ei_xi_standardized_logit": 0.01,
    "direction": "maximize",
    "missing_metadata_policy": "error",
    "metadata_fallbacks": {},
    "chunk_size": 512,
    "quadrature_nodes": 96,
    "feature_spec": None,
    "objective_field": "moc_wt_pct",
    "objective_bounds": [0.0, 100.0],
    "noise_policy": "moc_quality",
    "default_observation_sigma": 1.0,
    "units": "wt%",
}


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def transform_features(
    rows: Sequence[Mapping],
    *,
    strict_grid=True,
    full_space: Sequence[Mapping] | None = None,
) -> np.ndarray:
    """Return the six features in their pinned order, never fitted on outcomes.

    The MoC default uses fixed source bounds even if every observed seed has the
    same ramp, flow or ratio. Explicit custom spaces may supply ``full_space``;
    transformed constant columns then map to zero without division by zero.
    """
    values = []
    grids = (
        [*range(550, 901, 10)],
        [5, 10, 15],
        [30, 60, 100],
        [0.5, 2, 5, 10],
        [0.5, 1, 2],
    )
    for row in rows:
        numeric = [_number(row.get(name), name) for name in FEATURE_ORDER[:4]]
        q = row.get("AMT_to_sucrose_mass_ratio")
        displayed = row.get("sucrose_to_AMT_mass_ratio")
        if q in (None, ""):
            displayed = _number(displayed, "sucrose_to_AMT_mass_ratio")
            if displayed <= 0:
                raise ValueError("Sucrose:AMT ratio must be positive")
            q = 1.0 / displayed
        q = _number(q, "AMT_to_sucrose_mass_ratio")
        if q <= 0 or numeric[3] <= 0:
            raise ValueError("Hold time and AMT:sucrose ratio must be positive")
        if displayed not in (None, "") and not np.isclose(
            q * _number(displayed, "sucrose_to_AMT_mass_ratio"), 1
        ):
            raise ValueError("AMT:sucrose and sucrose:AMT ratios must be reciprocals")
        numeric.append(q)
        gas = row.get("gas")
        if gas not in ("H2", "N2"):
            raise ValueError("Gas must be H2 or N2")
        if strict_grid:
            for name, value, allowed in zip(FEATURE_ORDER[:5], numeric, grids):
                if not np.any(np.isclose(value, allowed, rtol=0, atol=1e-8)):
                    raise ValueError(
                        f"{name}={value} is outside the MoC synthesis grid"
                    )
        t, r, f, h, q = numeric
        values.append(
            [
                (t - 550) / 350,
                (r - 5) / 10,
                (f - 30) / 70,
                np.log(h / 0.5) / np.log(20),
                (np.log2(q) + 1) / 2,
                1.0 if gas == "H2" else 0.0,
            ]
        )
    result = np.asarray(values, dtype=float).reshape(-1, 6)
    if full_space is not None:
        full = transform_features(full_space, strict_grid=strict_grid)
        if len(full) == 0:
            raise ValueError(
                "Custom full-space scaling requires at least one candidate"
            )
        lo, hi = full.min(axis=0), full.max(axis=0)
        width = hi - lo
        if len(result) and (np.any(result < lo - 1e-10) or np.any(result > hi + 1e-10)):
            raise ValueError("A design is outside the configured full-space bounds")
        result = np.divide(
            result - lo, width, out=np.zeros_like(result), where=width > 0
        )
    return result


def effective_noise(
    observations: Sequence[Mapping],
    *,
    sigma_floor_pp=0.5,
    gof_power=1.0,
    missing_policy="error",
    fallbacks=None,
):
    """Return effective sigma (percentage points) and explicit fallback flags.

    An imported zero gap is retained as an override. Missing metadata is rejected
    unless the caller explicitly selects ``fallback`` and supplies each missing
    field's assumed value; it never becomes a measured zero implicitly.
    """
    floor = _number(sigma_floor_pp, "sigma_floor_pp")
    power = _number(gof_power, "gof_power")
    if floor <= 0 or power < 0:
        raise ValueError("sigma_floor_pp must be positive and gof_power nonnegative")
    if missing_policy not in ("error", "fallback"):
        raise ValueError("missing_metadata_policy must be error or fallback")
    fallbacks = fallbacks or {}
    fields = ("moc_wt_pct_sigma", "gof", "closure_gap_wt_pct")
    sigmas, flags = [], []
    for index, row in enumerate(observations):
        vals = []
        for field in fields:
            value = row.get(field)
            if value is None or value == "":
                if missing_policy != "fallback" or field not in fallbacks:
                    raise ValueError(
                        f"Observation {row.get('observation_id', index)} lacks {field}; configure an explicit fallback"
                    )
                value = fallbacks[field]
                flags.append(
                    {
                        "observation_id": row.get("observation_id", str(index)),
                        "field": field,
                        "assumed_value": float(value),
                        "origin": "explicit fallback",
                    }
                )
            vals.append(_number(value, field))
        esd, gof, gap = vals
        if esd < 0 or gof < 0:
            raise ValueError("Reported esd and GOF cannot be negative")
        sigma = np.hypot(max(esd * max(gof, 1.0) ** power, floor), gap)
        if not np.isfinite(sigma):
            raise ValueError("Effective observation sigma is not finite")
        sigmas.append(sigma)
    return np.asarray(sigmas, dtype=float), flags


@dataclass
class PaddedLogitTransform:
    delta: float = 0.65
    mean: float = 0.0
    scale: float = 1.0
    scaling_policy: str = "unfitted"
    lower: float = 0.0
    upper: float = 100.0

    def __post_init__(self):
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("Padded-logit delta must be positive")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Outcome standardization scale must be positive")
        if (
            not np.isfinite(self.lower)
            or not np.isfinite(self.upper)
            or self.lower >= self.upper
        ):
            raise ValueError("Outcome bounds must be finite and increasing")

    def raw(self, outcomes):
        outcomes = np.asarray(outcomes, dtype=float)
        if np.any(~np.isfinite(outcomes)) or np.any(
            (outcomes < self.lower) | (outcomes > self.upper)
        ):
            raise ValueError(
                f"Outcomes must be finite values in [{self.lower},{self.upper}]"
            )
        return np.log(
            (outcomes - self.lower + self.delta) / (self.upper + self.delta - outcomes)
        )

    def fit(self, outcomes):
        z = self.raw(outcomes)
        self.mean = float(z.mean()) if z.size else 0.0
        sample_sd = float(z.std(ddof=1)) if z.size > 1 else 0.0
        if np.isfinite(sample_sd) and sample_sd > 1e-6:
            self.scale = sample_sd
            self.scaling_policy = "observed sample SD, ddof=1"
        else:
            self.scale = 1.0
            self.scaling_policy = "unit logit scale for zero/single/constant outcomes"
        return self

    def forward(self, outcomes):
        return (self.raw(outcomes) - self.mean) / self.scale

    def variance(self, outcomes, sigma):
        outcomes, sigma = np.broadcast_arrays(
            np.asarray(outcomes, float), np.asarray(sigma, float)
        )
        self.raw(outcomes)
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Observation sigmas must be finite and nonnegative")
        derivative = (self.upper - self.lower + 2 * self.delta) / (
            (outcomes - self.lower + self.delta) * (self.upper + self.delta - outcomes)
        )
        return (derivative * sigma / self.scale) ** 2

    @property
    def bounds(self):
        return self.forward([self.lower, self.upper])

    def inverse(self, latent, *, clamp=True):
        latent = np.asarray(latent, dtype=float)
        if clamp:
            latent = np.clip(latent, *self.bounds)
        raw = (
            (self.upper - self.lower + 2 * self.delta)
            * expit(self.mean + self.scale * latent)
            + self.lower
            - self.delta
        )
        return np.clip(raw, self.lower, self.upper) if clamp else raw

    def as_dict(self):
        return {
            "delta_pp": self.delta,
            "mean_logit": self.mean,
            "scale_logit": self.scale,
            "scaling_policy": self.scaling_policy,
            "standardized_bounds": self.bounds.tolist(),
            "physical_bounds": [self.lower, self.upper],
            "bounded_posterior": "clamp latent draws before EI and inverse; endpoint atoms allowed",
        }


class StandardOutcomeTransform:
    """Generic unbounded objective: fold-independent ordinary standardization."""

    lower, upper = -np.inf, np.inf

    def fit(self, outcomes):
        outcomes = np.asarray(outcomes, float)
        if np.any(~np.isfinite(outcomes)):
            raise ValueError("Outcomes must be finite")
        self.mean = float(outcomes.mean()) if len(outcomes) else 0.0
        sd = float(outcomes.std(ddof=1)) if len(outcomes) > 1 else 0.0
        self.scale = sd if np.isfinite(sd) and sd > 1e-6 else 1.0
        return self

    def forward(self, outcomes):
        return (np.asarray(outcomes, float) - self.mean) / self.scale

    def inverse(self, latent, *, clamp=True):
        return np.asarray(latent, float) * self.scale + self.mean

    def variance(self, outcomes, sigma):
        sigma = np.asarray(sigma, float)
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Observation sigma must be finite and nonnegative")
        return (sigma / self.scale) ** 2

    @property
    def bounds(self):
        return np.asarray([-np.inf, np.inf])

    def as_dict(self):
        return {
            "transform": "standardize",
            "mean": self.mean,
            "scale": self.scale,
            "physical_bounds": None,
            "bounded_posterior": False,
        }


def matern52(a, b, lengthscales, signal_variance):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ls = np.asarray(lengthscales, float)
    r = np.sqrt(
        5.0
        * np.maximum(
            np.sum(((a[:, None, :] - b[None, :, :]) / ls) ** 2, axis=-1), 1e-12
        )
    )
    return float(signal_variance) * (1 + r + r * r / 3) * np.exp(-r)


def log_prior(theta):
    theta = np.asarray(theta, float)
    if theta.ndim != 1 or len(theta) < 2 or not np.all(np.isfinite(theta)):
        return -np.inf
    dimension = len(theta) - 1
    return float(
        -0.5 * np.sum(((theta[:-1] - np.log(np.sqrt(dimension))) / 0.75) ** 2)
        - 0.5 * theta[-1] ** 2
    )


def log_marginal_likelihood(theta, x, y, noise):
    if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 100):
        return -np.inf
    ls, sf2 = np.exp(theta[:-1]), np.exp(theta[-1])
    k = matern52(x, x, ls, sf2) + np.diag(noise) + 1e-8 * np.eye(len(y))
    try:
        factor = np.linalg.cholesky(k)
        alpha = cho_solve((factor, True), y, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf
    return float(
        -0.5 * y @ alpha
        - np.log(np.diag(factor)).sum()
        - 0.5 * len(y) * np.log(2 * np.pi)
    )


def _check_cancel(cancel):
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise InterruptedError("Structured GP calculation canceled")


def _effective_sample_size(samples):
    """Single-chain initial-positive-pair autocorrelation ESS; diagnostic only."""
    n = len(samples)
    result = []
    for column in np.asarray(samples).T:
        centered = column - column.mean()
        if n < 4 or np.sum(centered**2) <= 1e-30:
            result.append(1.0)
            continue
        fft = np.fft.rfft(centered, n=2 * n)
        acov = np.fft.irfft(fft * fft.conj())[:n]
        acf = acov / acov[0]
        positive_sum = 0.0
        for i in range(1, n - 1, 2):
            pair = acf[i] + acf[i + 1]
            if pair <= 0:
                break
            positive_sum += pair
        result.append(float(np.clip(n / (1 + 2 * positive_sum), 1, n)))
    return result


def sample_posterior(
    x,
    y,
    noise,
    *,
    retained_draws=4000,
    burn_in=1000,
    seed=616,
    proposal_step=0.3,
    random_state=None,
    cancel=None,
    progress=None,
):
    """Source Metropolis sampler; one adjustment at the final burn-in iteration.

    Start at the prior mean, not at a fitted MAP. The source multiplies the step
    by ``clip((burn_acceptance / .3), .3, 3)`` exactly once; no further adaptation
    occurs among retained draws. All draws and final RNG state are returned.
    """
    for name, value, minimum in (
        ("retained_draws", retained_draws, 1),
        ("burn_in", burn_in, 0),
    ):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    retained_draws, burn_in = int(retained_draws), int(burn_in)
    if not np.isfinite(proposal_step) or proposal_step <= 0:
        raise ValueError("proposal_step must be positive")
    rng = np.random.default_rng(seed)
    if random_state is not None:
        rng.bit_generator.state = random_state
    dimension = x.shape[1]
    theta = np.r_[np.full(dimension, np.log(np.sqrt(dimension))), 0.0]
    lp = log_marginal_likelihood(theta, x, y, noise) + log_prior(theta)
    out = np.empty((retained_draws, dimension + 1))
    log_density = np.empty(retained_draws)
    step, accepted, burn_accepted, kept_accepted = float(proposal_step), 0, 0, 0
    total = retained_draws + burn_in
    for i in range(total):
        _check_cancel(cancel)
        proposal = theta + step * rng.standard_normal(dimension + 1)
        proposed_lp = log_marginal_likelihood(proposal, x, y, noise) + log_prior(
            proposal
        )
        if np.log(rng.random()) < proposed_lp - lp:
            theta, lp = proposal, proposed_lp
            accepted += 1
            if i < burn_in:
                burn_accepted += 1
            else:
                kept_accepted += 1
        if i == burn_in - 1:
            step *= float(np.clip((accepted / burn_in) / 0.3, 0.3, 3.0))
        if i >= burn_in:
            out[i - burn_in] = theta
            log_density[i - burn_in] = lp
        if progress is not None and (i % 100 == 0 or i == total - 1):
            progress({"stage": "sampling", "completed": i + 1, "total": total})
    ess = _effective_sample_size(out)
    trace_indices = np.unique(
        np.linspace(0, retained_draws - 1, min(200, retained_draws), dtype=int)
    )
    warnings = [
        "One random-walk chain; acceptance and ESS do not establish convergence."
    ]
    if min(ess) < 50:
        warnings.append(
            "Low effective sample size (<50) in at least one hyperparameter; prediction may be unstable."
        )
    if kept_accepted == 0:
        warnings.append(
            "Sampler retained no accepted moves; posterior sampling failed to explore."
        )
    diagnostics = {
        "sampler_version": SAMPLER_VERSION,
        "burn_in": burn_in,
        "retained_draws": retained_draws,
        "acceptance_rate": accepted / total,
        "burn_acceptance_rate": burn_accepted / burn_in if burn_in else None,
        "retained_acceptance_rate": kept_accepted / retained_draws,
        "initial_proposal_step": float(proposal_step),
        "final_proposal_step": step,
        "effective_sample_size": ess,
        "effective_sample_size_method": "single-chain initial positive autocorrelation pairs",
        "trace_indices": trace_indices.tolist(),
        "log_hyperparameter_trace": out[trace_indices].tolist(),
        "log_posterior_trace": log_density[trace_indices].tolist(),
        "warnings": warnings,
        "random_state": rng.bit_generator.state,
    }
    return out, diagnostics


def _positive_normal_part(mu, sd, threshold):
    difference = mu - threshold
    safe_sd = np.where(sd > 0, sd, 1.0)
    z = difference / safe_sd
    result = difference * ndtr(z) + sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return np.where(sd > 0, result, np.maximum(difference, 0))


def bounded_expected_improvement(
    means, variances, best, bounds, *, xi=0.01, direction="maximize"
):
    """Exact component EI for a clamped Gaussian in standardized logit units."""
    means, variances = np.broadcast_arrays(
        np.asarray(means, float), np.asarray(variances, float)
    )
    if (
        np.any(~np.isfinite(means))
        or np.any(~np.isfinite(variances))
        or np.any(variances < 0)
    ):
        raise ValueError(
            "Posterior means/variances must be finite and variances nonnegative"
        )
    if xi < 0 or not np.isfinite(xi):
        raise ValueError("EI xi must be finite and nonnegative")
    lo, hi = map(float, bounds)
    if direction == "minimize":
        means, best, lo, hi = -means, -best, -hi, -lo
    elif direction != "maximize":
        raise ValueError("direction must be maximize or minimize")
    threshold = float(best) + xi
    if threshold >= hi:
        return np.zeros_like(means)
    sd = np.sqrt(variances)
    if np.isposinf(hi):
        return np.maximum(_positive_normal_part(means, sd, threshold), 0)
    if threshold < lo:
        bounded_mean = (
            means
            + _positive_normal_part(-means, sd, -lo)
            - _positive_normal_part(means, sd, hi)
        )
        return np.maximum(bounded_mean - threshold, 0)
    ei = _positive_normal_part(means, sd, threshold) - _positive_normal_part(
        means, sd, hi
    )
    return np.clip(ei, 0, hi - threshold)


def bounded_mixture_summary(
    means, variances, transform, *, quadrature_nodes=96, quantiles=(0.025, 0.975)
):
    """Moments and exact mixture-CDF quantiles after clamp and nonlinear inverse.

    Inputs have shape (hyperparameter components, candidate points). Numerical
    Gauss-Legendre integration is over each component's interior probability;
    endpoint masses are included separately. Variance is E[var]+var[E], in wt%.
    """
    means, variances = np.broadcast_arrays(
        np.asarray(means, float), np.asarray(variances, float)
    )
    if means.ndim == 1:
        means, variances = means[:, None], variances[:, None]
    if means.ndim != 2 or means.shape[0] == 0:
        raise ValueError("Posterior mixture must contain components x candidates")
    if (
        np.any(~np.isfinite(means))
        or np.any(~np.isfinite(variances))
        or np.any(variances < 0)
    ):
        raise ValueError(
            "Posterior mixture values must be finite with nonnegative variances"
        )
    if quadrature_nodes < 16 or int(quadrature_nodes) != quadrature_nodes:
        raise ValueError("Use at least 16 quadrature nodes")
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be strictly between zero and one")
    lo, hi = transform.bounds
    if not np.isfinite(lo) and not np.isfinite(hi):
        mean = means.mean(axis=0) * transform.scale + transform.mean
        within = variances.mean(axis=0) * transform.scale**2
        between = means.var(axis=0) * transform.scale**2
        sd = np.sqrt(variances)
        safe_sd = np.where(sd > 0, sd, 1.0)
        interval = []
        for q in quantiles:
            left = np.min(means - 12 * sd, axis=0)
            right = np.max(means + 12 * sd, axis=0)
            for _ in range(52):
                mid = (left + right) / 2
                cdf = np.where(
                    sd > 0,
                    ndtr((mid[None, :] - means) / safe_sd),
                    means <= mid[None, :],
                ).mean(axis=0)
                left = np.where(cdf < q, mid, left)
                right = np.where(cdf >= q, mid, right)
            interval.append(transform.inverse((left + right) / 2))
        return {
            "mean": mean,
            "sd": np.sqrt(within + between),
            "variance": within + between,
            "within_component_variance": within,
            "between_component_variance": between,
            "lower95": interval[0],
            "upper95": interval[-1],
            "lower_boundary_mass": np.zeros_like(mean),
            "upper_boundary_mass": np.zeros_like(mean),
            "uncertainty_type": "latent-function unbounded posterior mixture",
            "quadrature_nodes": 0,
        }
    sd = np.sqrt(variances)
    positive = sd > 0
    safe_sd = np.where(positive, sd, 1.0)
    lower_mass = np.where(positive, ndtr((lo - means) / safe_sd), means <= lo)
    upper_mass = np.where(positive, ndtr((means - hi) / safe_sd), means >= hi)
    interior = np.maximum(1 - lower_mass - upper_mass, 0)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_nodes))
    component_mean = transform.lower * lower_mass + transform.upper * upper_mass
    component_second = (
        transform.lower**2 * lower_mass + transform.upper**2 * upper_mass
    )
    # Iterate over quadrature nodes to keep memory O(components*candidates).
    for node, weight in zip(nodes, weights):
        probabilities = np.clip(
            lower_mass + interior * ((node + 1) / 2),
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        z = means + safe_sd * ndtri(probabilities)
        outcome = transform.inverse(z)
        factor = weight * 0.5 * interior
        component_mean += factor * outcome
        component_second += factor * outcome * outcome
    deterministic = transform.inverse(means)
    component_mean = np.where(positive, component_mean, deterministic)
    component_second = np.where(positive, component_second, deterministic**2)
    component_variance = np.maximum(component_second - component_mean**2, 0)
    mixture_mean = component_mean.mean(axis=0)
    within = component_variance.mean(axis=0)
    between = component_mean.var(axis=0)
    lower_prob, upper_prob = lower_mass.mean(axis=0), upper_mass.mean(axis=0)
    interval = []
    for q in quantiles:
        left, right = np.full(means.shape[1], lo), np.full(means.shape[1], hi)
        for _ in range(52):
            middle = (left + right) * 0.5
            cdf = np.where(
                positive,
                ndtr((middle[None, :] - means) / safe_sd),
                means <= middle[None, :],
            ).mean(axis=0)
            left = np.where(cdf < q, middle, left)
            right = np.where(cdf >= q, middle, right)
        latent_quantile = np.where(
            q <= lower_prob, lo, np.where(q > 1 - upper_prob, hi, (left + right) / 2)
        )
        interval.append(transform.inverse(latent_quantile))
    return {
        "mean": mixture_mean,
        "sd": np.sqrt(within + between),
        "variance": within + between,
        "within_component_variance": within,
        "between_component_variance": between,
        "lower95": interval[0],
        "upper95": interval[-1],
        "lower_boundary_mass": lower_prob,
        "upper_boundary_mass": upper_prob,
        "uncertainty_type": "latent-function bounded posterior mixture",
        "quadrature_nodes": int(quadrature_nodes),
    }


def _included(row):
    inclusion = row.get("training_included", True)
    if (
        inclusion is False
        or inclusion == 0
        or (isinstance(inclusion, str) and inclusion.lower() in ("false", "0", "no"))
    ):
        return False
    return row.get("record_status", row.get("status", "measured")) in (
        "measured",
        "completed",
        "active",
    )


class StructuredGP:
    """Reusable offline GP, accepting canonical candidate and observation dicts.

    ``fit(candidates, observations)`` uses only included measured records.
    ``recommend(eligible_candidates)`` scores the supplied finite set, excludes
    measured IDs, and preserves input ordering to break ties. The campaign layer
    supplies exclusions and reservations; they never count toward coverage.
    """

    def __init__(self, settings=None, **overrides):
        self.settings = {**DEFAULTS, **(settings or {}), **overrides}
        for key in (
            "retained_draws",
            "predict_thin",
            "ei_after_unique_measured_designs",
            "chunk_size",
            "quadrature_nodes",
        ):
            value = self.settings[key]
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            self.settings[key] = int(value)
        if self.settings["direction"] not in ("maximize", "minimize"):
            raise ValueError("direction must be maximize or minimize")
        xi = _number(
            self.settings["ei_xi_standardized_logit"], "ei_xi_standardized_logit"
        )
        if xi < 0:
            raise ValueError("EI xi must be nonnegative")
        self.observations = []
        self.measured_ids = set()
        self.feature_transform = None
        if self.settings["feature_spec"] is not None:
            from .generic_import import FeatureTransform

            self.feature_transform = FeatureTransform(self.settings["feature_spec"])
        self.dimension = (
            self.feature_transform.dimension if self.feature_transform else 6
        )
        self.method_version = (
            "structured-gp-generic-v1" if self.feature_transform else METHOD_VERSION
        )
        if (
            self.feature_transform
            and "units" not in (settings or {})
            and "units" not in overrides
        ):
            self.settings["units"] = ""
        self.transform = self._outcome_transform()
        self.samples = np.empty((0, self.dimension + 1))
        self.diagnostics = {}
        self._factors = []
        self._fitted = False

    def _features(self, rows):
        return (
            self.feature_transform.transform(rows)
            if self.feature_transform
            else transform_features(rows)
        )

    def _outcome_transform(self):
        bounds = self.settings["objective_bounds"]
        if bounds is None:
            return StandardOutcomeTransform().fit([])
        return PaddedLogitTransform(
            self.settings["logit_delta_pp"], lower=bounds[0], upper=bounds[1]
        )

    def fit(self, candidates, observations, *, cancel=None, progress=None):
        _check_cancel(cancel)
        candidate_map = {}
        for row in candidates:
            candidate_id = row.get("candidate_id")
            if not candidate_id or candidate_id in candidate_map:
                raise ValueError("Candidate IDs must be present and unique")
            candidate_map[candidate_id] = row
        # Validate all inputs; a data-dependent feature scaler is never fitted.
        self._features(list(candidate_map.values()))
        included = [dict(row) for row in observations if _included(row)]
        active_measurement_ids = [
            row.get(
                "physical_measurement_id",
                row.get("measurement_id", row.get("observation_id")),
            )
            for row in included
        ]
        nonempty_ids = [key for key in active_measurement_ids if key is not None]
        if len(nonempty_ids) != len(set(nonempty_ids)):
            raise ValueError(
                "Only one active refinement per physical measurement may enter fitting"
            )
        for row in included:
            if row.get("candidate_id") not in candidate_map:
                raise ValueError(
                    "An observation refers to a candidate outside the fixed design space"
                )
        x = self._features([candidate_map[row["candidate_id"]] for row in included])
        objective = self.settings["objective_field"]
        outcomes = np.asarray(
            [_number(row.get(objective), objective) for row in included]
        )
        transform = self._outcome_transform().fit(outcomes)
        if self.settings["noise_policy"] == "reported_or_fixed":
            fallback = _number(
                self.settings["default_observation_sigma"], "default_observation_sigma"
            )
            if fallback < 0:
                raise ValueError("Default observation sigma cannot be negative")
            sigmas = np.asarray(
                [
                    _number(
                        row["objective_sigma"]
                        if row.get("objective_sigma") is not None
                        else fallback,
                        "objective_sigma",
                    )
                    for row in included
                ]
            )
            if np.any(sigmas < 0):
                raise ValueError("Observation sigma cannot be negative")
            flags = [
                {
                    "observation_id": row.get("observation_id"),
                    "field": "objective_sigma",
                    "assumed_value": fallback,
                    "origin": "configured generic fixed sigma; GOF/closure retained as metadata, not used",
                }
                for row in included
                if row.get("objective_sigma") is None
            ]
        elif self.settings["noise_policy"] == "moc_quality":
            sigmas, flags = effective_noise(
                included,
                sigma_floor_pp=self.settings["sigma_floor_pp"],
                gof_power=self.settings["gof_power"],
                missing_policy=self.settings["missing_metadata_policy"],
                fallbacks=self.settings["metadata_fallbacks"],
            )
        else:
            raise ValueError("Unknown structured GP observation-noise policy")
        y, noise = transform.forward(outcomes), transform.variance(outcomes, sigmas)
        if len(included):
            samples, diagnostics = sample_posterior(
                x,
                y,
                noise,
                retained_draws=self.settings["retained_draws"],
                burn_in=self.settings["burn_in"],
                seed=self.settings["seed"],
                proposal_step=self.settings["proposal_step"],
                random_state=self.settings.get("random_state"),
                cancel=cancel,
                progress=progress,
            )
        else:
            samples, diagnostics = np.empty((0, self.dimension + 1)), {
                "warnings": [
                    "No observations: deterministic first-eligible initial design; no fitted prediction."
                ]
            }
        factors = []
        for theta in samples[:: self.settings["predict_thin"]]:
            _check_cancel(cancel)
            ls, sf2 = np.exp(theta[:-1]), np.exp(theta[-1])
            factor = np.linalg.cholesky(
                matern52(x, x, ls, sf2) + np.diag(noise) + 1e-8 * np.eye(len(y))
            )
            factors.append(
                (ls, sf2, factor, cho_solve((factor, True), y, check_finite=False))
            )
        # Publish a complete fit only after sampling and factors finish.
        self.observations, self.measured_ids = included, {
            row["candidate_id"] for row in included
        }
        self.x, self.y, self.noise, self.outcomes = x, y, noise, outcomes
        self.transform, self.samples, self._factors = transform, samples, factors
        self.diagnostics = {
            **diagnostics,
            "method_version": self.method_version,
            "feature_spec": self.settings["feature_spec"]
            if self.feature_transform
            else FEATURE_SPEC,
            "feature_order": self.feature_transform.order
            if self.feature_transform
            else list(FEATURE_ORDER),
            "outcome_transform": transform.as_dict(),
            "noise_policy": self.settings["noise_policy"],
            "effective_sigmas": sigmas.tolist(),
            "objective_units": self.settings["units"],
            "metadata_fallback_flags": flags,
            "measurement_count": len(included),
            "unique_design_count": len(self.measured_ids),
            "prediction_components": len(factors),
            "predict_thin": self.settings["predict_thin"],
            "tie_policy": "first candidate in supplied source order",
        }
        if not self.feature_transform:
            self.diagnostics["effective_sigmas_pp"] = sigmas.tolist()
        self._fitted = True
        return self

    def posterior_components(self, rows, *, cancel=None):
        if not self._fitted or not self._factors:
            raise ValueError(
                "Posterior prediction requires at least one fitted observation"
            )
        xs = self._features(rows)
        means, variances = [], []
        for ls, sf2, factor, alpha in self._factors:
            _check_cancel(cancel)
            cross = matern52(xs, self.x, ls, sf2)
            projected = solve_triangular(
                factor, cross.T, lower=True, check_finite=False
            )
            means.append(cross @ alpha)
            variances.append(np.maximum(sf2 - np.sum(projected**2, axis=0), 1e-12))
        return np.asarray(means), np.asarray(variances)

    def predict(self, rows, *, cancel=None, observation_sigma_pp=None):
        """Report bounded mixture moments/quantiles in wt%, with explicit type.

        Future-measurement intervals require explicit assumed future sigmas;
        delta-method noise is transformed at each component's bounded mean.
        Acquisition always uses the latent-function posterior instead.
        """
        means, variances = self.posterior_components(rows, cancel=cancel)
        if observation_sigma_pp is not None:
            sigmas = np.asarray(observation_sigma_pp, float)
            variances = variances + self.transform.variance(
                self.transform.inverse(means), sigmas
            )
        result = bounded_mixture_summary(
            means,
            variances,
            self.transform,
            quadrature_nodes=self.settings["quadrature_nodes"],
        )
        if observation_sigma_pp is not None:
            result[
                "uncertainty_type"
            ] = "future-measurement bounded posterior mixture; delta-method assumed noise"
        return result

    def recommend(
        self, eligible_candidates, *, excluded_ids=(), cancel=None, progress=None
    ):
        if not self._fitted:
            raise ValueError(
                "Call fit with the full candidate space before requesting a recommendation"
            )
        _check_cancel(cancel)
        excluded = self.measured_ids | set(excluded_ids)
        candidates = [
            row for row in eligible_candidates if row["candidate_id"] not in excluded
        ]
        ids = [row["candidate_id"] for row in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Eligible candidate IDs must be unique")
        common = {
            "measurement_count": len(self.observations),
            "unique_design_count": len(self.measured_ids),
            "diagnostics": self.diagnostics,
            "method_version": self.method_version,
        }
        if not candidates:
            return {
                **common,
                "status": "exhausted",
                "candidate_id": None,
                "prediction": None,
            }
        if not self.observations:
            return {
                **common,
                "status": "initial_design",
                "candidate_id": ids[0],
                "stage": "initial_design",
                "selection_reason": "No measured designs: first eligible candidate in source order",
                "score": None,
                "acquisition_units": None,
                "prediction": None,
                "scores": [],
            }
        unique_count = len(self.measured_ids)
        threshold = self.settings["ei_after_unique_measured_designs"]
        exploring = unique_count < threshold
        scores = np.empty(len(candidates))
        chunk_size = self.settings["chunk_size"]
        incumbent = float(
            np.max(self.y)
            if self.settings["direction"] == "maximize"
            else np.min(self.y)
        )
        for start in range(0, len(candidates), chunk_size):
            _check_cancel(cancel)
            chunk = candidates[start : start + chunk_size]
            if exploring:
                xs = self._features(chunk)
                scores[start : start + len(chunk)] = np.linalg.norm(
                    xs[:, None, :] - self.x[None, :, :], axis=2
                ).min(axis=1)
            else:
                means, variances = self.posterior_components(chunk, cancel=cancel)
                scores[start : start + len(chunk)] = bounded_expected_improvement(
                    means,
                    variances,
                    incumbent,
                    self.transform.bounds,
                    xi=self.settings["ei_xi_standardized_logit"],
                    direction=self.settings["direction"],
                ).mean(axis=0)
            if progress is not None:
                progress(
                    {
                        "stage": "maximin" if exploring else "expected_improvement",
                        "completed": min(start + chunk_size, len(candidates)),
                        "total": len(candidates),
                    }
                )
        index = int(np.argmax(scores))
        prediction = self.predict([candidates[index]], cancel=cancel)
        prediction = {
            key: float(value[0]) if isinstance(value, np.ndarray) else value
            for key, value in prediction.items()
        }
        stage = "maximin" if exploring else "expected_improvement"
        reason = (
            f"Space-filling: {unique_count} of {threshold} measured designs; greatest distance to active observed designs"
            if exploring
            else f"Expected improvement over {unique_count} distinct measured designs"
        )
        return {
            **common,
            "status": "recommended",
            "candidate_id": ids[index],
            "stage": stage,
            "selection_reason": reason,
            "score": float(scores[index]),
            "acquisition_units": f"scaled {self.dimension}-feature Euclidean distance"
            if exploring
            else "standardized objective units"
            if self.settings["objective_bounds"] is None
            else "standardized padded-logit units",
            "prediction": prediction,
            "scores": [
                {"candidate_id": cid, "score": float(score)}
                for cid, score in zip(ids, scores)
            ],
        }
