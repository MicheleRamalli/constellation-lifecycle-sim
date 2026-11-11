"""Validation of analytic.binomial_interval (Clopper-Pearson) against closed forms."""

import numpy as np
from scipy.stats import binom

from analytic import binomial_interval


def test_all_successes_and_no_successes_closed_form():
    """n_ok = n: low = (alpha/2)^(1/n), high = 1. n_ok = 0: mirror image."""
    n, alpha = 1000, 0.05
    low, high = binomial_interval(n, n, 1.0 - alpha)
    assert high == 1.0
    np.testing.assert_allclose(low, (alpha / 2) ** (1.0 / n), rtol=1e-12)
    low0, high0 = binomial_interval(0, n, 1.0 - alpha)
    assert low0 == 0.0
    np.testing.assert_allclose(high0, 1.0 - (alpha / 2) ** (1.0 / n), rtol=1e-12)


def test_bounds_satisfy_the_definition():
    """Definition: P(X >= k | p = low) = alpha/2 and P(X <= k | p = high) = alpha/2."""
    k, n, alpha = 940, 1000, 0.05
    low, high = binomial_interval(k, n, 1.0 - alpha)
    np.testing.assert_allclose(binom.sf(k - 1, n, low), alpha / 2, rtol=1e-8)
    np.testing.assert_allclose(binom.cdf(k, n, high), alpha / 2, rtol=1e-8)
    assert low < k / n < high
