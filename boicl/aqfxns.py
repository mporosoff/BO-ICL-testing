import numpy as np
from scipy.stats import norm
from .llm_model import DiscreteDist, GaussDist


def expected_improvement(dist, best, xi=0.0, maximize=True):
    """Expected improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return expected_improvement_d(dist.probs, dist.values, best, xi, maximize)
    elif isinstance(dist, GaussDist):
        direction = 1 if maximize else -1
        return expected_improvement_g(
            direction * dist.mean(), dist.std(), direction * best + xi
        )


def log_expected_improvement(dist, best, xi=0.0, maximize=True):
    """Log Expected improvement for the given discrete distribution"""
    return np.log(expected_improvement(dist, best, xi, maximize) + 1e-15)


# I think it's just taking the log of the final EI computation. Will test this later
# def log_expected_improvement(dist, best):
#     """Log Expected improvement for the given discrete distribution"""
#     if isinstance(dist, DiscreteDist):
#         return np.log(expected_improvement_d(dist.probs, dist.values, best))
#     elif isinstance(dist, GaussDist):
#         return np.log(expected_improvement_g(dist.mean(), dist.std(), best))


def probability_of_improvement(dist, best, xi=0.0, maximize=True):
    """Probability of improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return probability_of_improvement_d(dist.probs, dist.values, best, xi, maximize)
    elif isinstance(dist, GaussDist):
        direction = 1 if maximize else -1
        return probability_of_improvement_g(
            direction * dist.mean(), dist.std(), direction * best + xi
        )


def upper_confidence_bound(dist, best, _lambda, maximize=True):
    """Optimistic utility; larger scores are preferred in either direction."""
    if isinstance(dist, DiscreteDist):
        return upper_confidence_bound_d(
            dist.probs, dist.values, best, _lambda, maximize
        )
    elif isinstance(dist, GaussDist):
        return upper_confidence_bound_g(
            dist.mean(), dist.std(), best, _lambda, maximize
        )


def greedy(dist, best, maximize=True):
    """Greedy selection (most likely point) for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return greedy_d(dist.probs, dist.values, best, maximize)
    elif isinstance(dist, GaussDist):
        return greedy_g(dist.mean(), dist.std(), best, maximize)


def expected_improvement_d(probs, values, best, xi=0.0, maximize=True):
    """Expected improvement for the given discrete distribution"""
    direction = 1 if maximize else -1
    ei = np.sum(np.maximum(direction * (np.asarray(values) - best) - xi, 0) * probs)
    return ei


def log_expected_improvement_d(probs, values, best, xi=0.0, maximize=True):
    """Log Expected improvement for the given discrete distribution"""
    return np.log(expected_improvement_d(probs, values, best, xi, maximize) + 1e-15)


def probability_of_improvement_d(probs, values, best, xi=0.0, maximize=True):
    """Probability of improvement for the given discrete distribution"""
    direction = 1 if maximize else -1
    pi = np.sum((direction * (np.asarray(values) - best) > xi).astype(float) * probs)
    return pi


def upper_confidence_bound_d(probs, values, best, _lambda, maximize=True):
    """Upper confidence bound for the given discrete distribution"""
    values = np.asarray(values)
    mu = np.sum(values * probs)
    sigma = np.sqrt(np.sum((values - mu) ** 2 * probs))
    return (1 if maximize else -1) * mu + _lambda * sigma


def greedy_d(probs, values, best, maximize=True):
    """Greedy selection (most likely point) for the given discrete distribution"""
    return (1 if maximize else -1) * values[np.argmax(probs)]


def expected_improvement_g(mean, std, best):
    """Expected improvement for the given Gaussian distribution"""
    eps = 1e-15
    z = (mean - best) / (std + eps)
    ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def log_expected_improvement_g(mean, std, best, xi=0.0, maximize=True):
    """Log Expected improvement for the given Gaussian distribution"""
    direction = 1 if maximize else -1
    return np.log(
        expected_improvement_g(direction * mean, std, direction * best + xi) + 1e-15
    )


def probability_of_improvement_g(mean, std, best):
    """Probability of improvement for the given Gaussian distribution"""
    if std == 0:
        return float(mean > best)
    eps = 1e-15
    z = (mean - best) / (std + eps)
    pi = norm.cdf(z)
    return pi


def upper_confidence_bound_g(mean, std, best, _lambda, maximize=True):
    """Upper confidence bound for the given Gaussian distribution"""
    return (1 if maximize else -1) * mean + _lambda * std


def greedy_g(mean, std, best, maximize=True):
    """Greedy selection (most likely point) for the given Gaussian distribution"""
    return (1 if maximize else -1) * mean
