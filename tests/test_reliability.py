"""Validation of reliability.py: Monte Carlo survival fraction vs analytical R(t).

Statistical tolerance: the number of survivors at time t out of N satellites
is Binomial(N, R(t)), so the survival fraction has standard deviation
sigma = sqrt(R (1 - R) / N). We accept |fraction - R| < 4 sigma
(false-failure probability ~6e-5 per point, normal approximation).
The seed is fixed, so the test is deterministic.
"""

import numpy as np
import pytest

import params
from reliability import sample_failure_times, reliability_analytic

N = 1_000_000
SEED = 42
N_SIGMA = 4.0
CHECK_TIMES_S = np.array([1.0, 5.0, 10.0, 15.0]) * params.SECONDS_PER_YEAR

THETA_S = params.WEIBULL_THETA_YEARS * params.SECONDS_PER_YEAR
BETA = params.WEIBULL_BETA
# Exponential comparison case. Matching the Weibull mean would give
# MTTF ~6400 years and R(15 yr) ~0.998, an almost trivial test. Instead we
# choose MTTF so that the exponential has the same R(10 yr) as the Weibull:
# exp(-t_ref / MTTF) = R_weibull(t_ref)  =>  MTTF = t_ref / (-ln R_weibull(t_ref)).
# This gives a realistic survival level over a typical mission duration.
T_REF_S = 10.0 * params.SECONDS_PER_YEAR
R_WEIBULL_REF = reliability_analytic(T_REF_S, "weibull", beta=BETA, theta_s=THETA_S)
MTTF_S = T_REF_S / (-np.log(R_WEIBULL_REF))

MODELS = {
    "weibull": dict(beta=BETA, theta_s=THETA_S),
    "exponential": dict(mttf_s=MTTF_S),
}


@pytest.mark.parametrize("model", ["weibull", "exponential"])
def test_survival_fraction_matches_analytic(model):
    rng = np.random.default_rng(SEED)
    t_fail_s = sample_failure_times(N, model, rng, **MODELS[model])

    for t_s in CHECK_TIMES_S:
        fraction = np.mean(t_fail_s > t_s)
        r = reliability_analytic(t_s, model, **MODELS[model])
        sigma = np.sqrt(r * (1.0 - r) / N)
        assert abs(fraction - r) < N_SIGMA * sigma, (
            f"{model}, t = {t_s / params.SECONDS_PER_YEAR:.0f} yr: "
            f"MC {fraction:.5f} vs R(t) {r:.5f} (4 sigma = {N_SIGMA * sigma:.5f})"
        )


def test_exponential_sample_mean():
    # For an exponential distribution std = mean, so the sample mean has
    # standard deviation mttf / sqrt(N) (central limit theorem).
    rng = np.random.default_rng(SEED)
    t_fail_s = sample_failure_times(N, "exponential", rng, mttf_s=MTTF_S)
    sigma = MTTF_S / np.sqrt(N)
    assert abs(np.mean(t_fail_s) - MTTF_S) < N_SIGMA * sigma


def test_small_category_six_months():
    """Dubos, Castet & Saleh (IAC-09-D1.3.6, p. 5): R(6 months) ~ 96.8 % for the Small
    category; with the Table 3 parameters R(0.5 yr) = 0.9684 (tolerance: the text gives
    one decimal in percent, +-0.05 %)."""
    r = reliability_analytic(0.5 * params.SECONDS_PER_YEAR, "weibull", beta=params.WEIBULL_BETA_SMALL,
                             theta_s=params.WEIBULL_THETA_SMALL_YEARS * params.SECONDS_PER_YEAR)
    assert r == pytest.approx(0.968, abs=0.0005)


def test_small_category_survival_matches_analytic():
    """Monte Carlo sampling with the Small-category parameters reproduces R(t) (4 sigma)."""
    model = dict(beta=params.WEIBULL_BETA_SMALL,
                 theta_s=params.WEIBULL_THETA_SMALL_YEARS * params.SECONDS_PER_YEAR)
    t_fail_s = sample_failure_times(N, "weibull", np.random.default_rng(SEED), **model)
    for t_s in CHECK_TIMES_S:
        r = reliability_analytic(t_s, "weibull", **model)
        sigma = np.sqrt(r * (1.0 - r) / N)
        assert abs(np.mean(t_fail_s > t_s) - r) < N_SIGMA * sigma
