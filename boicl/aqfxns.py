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


def log_expected_improvement(dist, best):
    """Log Expected improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return log_expected_improvement_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return log_expected_improvement_g(dist.mean(), dist.std(), best)


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


def upper_confidence_bound(dist, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return upper_confidence_bound_d(dist.probs, dist.values, best, _lambda)
    elif isinstance(dist, GaussDist):
        return upper_confidence_bound_g(dist.mean(), dist.std(), best, _lambda)


def greedy(dist, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return greedy_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return greedy_g(dist.mean(), dist.std(), best)


def expected_improvement_d(probs, values, best, xi=0.0, maximize=True):
    """Expected improvement for the given discrete distribution"""
    direction = 1 if maximize else -1
    ei = np.sum(np.maximum(direction * (np.asarray(values) - best) - xi, 0) * probs)
    return ei


def log_expected_improvement_d(probs, values, best):
    """Log Expected improvement for the given discrete distribution"""
    # ei = np.sum(np.maximum(values - best, 0) * probs)
    log_ei = np.log(np.sum(np.maximum(values - best, 0) * probs) + 1e-15)
    return log_ei


def probability_of_improvement_d(probs, values, best, xi=0.0, maximize=True):
    """Probability of improvement for the given discrete distribution"""
    direction = 1 if maximize else -1
    pi = np.sum((direction * (np.asarray(values) - best) > xi).astype(float) * probs)
    return pi


def upper_confidence_bound_d(probs, values, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    mu = np.sum(values * probs)
    sigma = np.sqrt(np.sum((values - mu) ** 2 * probs))
    return mu + _lambda * sigma


def greedy_d(probs, values, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    return values[np.argmax(probs)]


def expected_improvement_g(mean, std, best):
    """Expected improvement for the given Gaussian distribution"""
    eps = 1e-15
    z = (mean - best) / (std + eps)
    ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def log_expected_improvement_g(mean, std, best):
    """Log Expected improvement for the given Gaussian distribution"""
    return np.log(expected_improvement_g(mean, std, best) + 1e-15)


def probability_of_improvement_g(mean, std, best):
    """Probability of improvement for the given Gaussian distribution"""
    if std == 0:
        return float(mean > best)
    eps = 1e-15
    z = (mean - best) / (std + eps)
    pi = norm.cdf(z)
    return pi


def upper_confidence_bound_g(mean, std, best, _lambda):
    """Upper confidence bound for the given Gaussian distribution"""
    eps = 1e-15
    return mean + _lambda * (std + eps)


def greedy_g(mean, std, best):
    """Greedy selection (most likely point) for the given Gaussian distribution"""
    return mean
