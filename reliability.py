"""Satellite reliability: failure-time sampling and analytical reliability R(t).

Two models are supported:
- "exponential": constant failure rate, R(t) = exp(-t / mttf)
- "weibull":     R(t) = exp(-(t / theta)^beta)

All times are in seconds (SI).
"""

import numpy as np
from scipy.special import gamma


def weibull_mean(beta, theta_s):
    """Mean time to failure of a Weibull distribution.

    Inputs:
        beta    : shape parameter [-]
        theta_s : scale parameter [s]
    Output:
        mean time to failure [s], theta * Gamma(1 + 1/beta)
    """
    return theta_s * gamma(1.0 + 1.0 / beta)


def sample_failure_times(n, model, rng, beta=None, theta_s=None, mttf_s=None):
    """Draw n random failure times from the chosen reliability model.

    Inputs:
        n       : number of samples [-]
        model   : "exponential" or "weibull"
        rng     : numpy Generator, e.g. np.random.default_rng(seed)
        beta    : Weibull shape parameter [-]      (weibull only)
        theta_s : Weibull scale parameter [s]      (weibull only)
        mttf_s  : mean time to failure [s]         (exponential only)
    Output:
        array of shape (n,) with failure times [s]
    """
    if model == "exponential":
        return rng.exponential(scale=mttf_s, size=n)
    if model == "weibull":
        # numpy draws the standard Weibull (theta = 1); scale it by theta
        return theta_s * rng.weibull(beta, size=n)
    raise ValueError(f"Unknown reliability model: {model}")


def reliability_analytic(t_s, model, beta=None, theta_s=None, mttf_s=None):
    """Analytical reliability R(t) = probability of surviving beyond time t.

    Inputs:
        t_s     : time(s) since launch [s], scalar or array
        model   : "exponential" or "weibull"
        beta, theta_s, mttf_s : as in sample_failure_times
    Output:
        R(t) [-], same shape as t_s
    """
    t_s = np.asarray(t_s, dtype=float)
    if model == "exponential":
        return np.exp(-t_s / mttf_s)
    if model == "weibull":
        return np.exp(-(t_s / theta_s) ** beta)
    raise ValueError(f"Unknown reliability model: {model}")
