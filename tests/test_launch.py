"""Launch success: Jeffreys estimator, GCAT counting, one draw per launch."""

import os

import numpy as np
import pytest
from scipy.stats import beta

import params
from launch_log import count_launches, jeffreys, read_launches
from lifecycle import LAUNCH_FAILURE, launch_satellites

YEAR_S = params.SECONDS_PER_YEAR
WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}


def test_jeffreys_no_data_is_the_prior():
    """n = 0: posterior = prior Beta(1/2, 1/2), mean 1/2, interval from its quantiles."""
    mean, lo, hi = jeffreys(0, 0)
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(beta.ppf(0.025, 0.5, 0.5))
    assert hi == pytest.approx(beta.ppf(0.975, 0.5, 0.5))


def test_jeffreys_mean_formula():
    """Posterior mean (k + 1/2) / (n + 1), interval contains the mean."""
    mean, lo, hi = jeffreys(643, 644)
    assert mean == pytest.approx(643.5 / 645)
    assert lo < mean < hi <= 1.0


def write_tsv(path, rows):
    """Minimal GCAT-like file: 33 tab-separated columns, '#' header."""
    lines = ["#header", "# Updated"]
    for date, lv, var, mission, code in rows:
        col = ["-"] * 33
        col[2], col[3], col[4], col[8], col[26] = date, lv, var, mission, code
        lines.append("\t".join(col))
    path.write_text("\n".join(lines) + "\n")


def test_count_launches_synthetic(tmp_path):
    """Orbital/deep-space only, partial failure counted as failure, prefix filter."""
    f = tmp_path / "launch.tsv"
    write_tsv(f, [("2020 Jan 1", "Falcon 9", "FT5", "Starlink", "OS  "),
                  ("2020 Jan 2", "Falcon 9", "FT5", "Transporter-1", "OS"),
                  ("2020 Jan 3", "Falcon 9", "FT5", "Starlink", "OF40"),
                  ("2020 Jan 4", "Falcon 9", "FT5", "Probe", "DS"),
                  ("2020 Jan 5", "Falcon 9", "FT5", "Dragon IFA", "AS"),   # suborbital: out
                  ("2020 Jan 6", "Falcon 9", "FT4", "Old", "OF"),          # other variant: out
                  ("2020 Jan 7", "Atlas V", "551", "X", "OF")])            # other vehicle: out
    rows = read_launches(f)
    assert count_launches(rows, "Falcon 9", "FT5") == (4, 1)
    assert count_launches(rows, "Falcon 9", "FT5", "Transporter") == (1, 0)


def test_unknown_outcome_raises(tmp_path):
    f = tmp_path / "launch.tsv"
    write_tsv(f, [("2020 Jan 1", "Falcon 9", "FT5", "X", "OU")])
    with pytest.raises(ValueError):
        count_launches(read_launches(f), "Falcon 9", "FT5")


@pytest.mark.skipif(not os.path.exists(params.LAUNCH_LOG_RAW), reason="raw GCAT file not present")
def test_params_match_raw_file():
    """The counts in params.py are those of the downloaded file."""
    rows = read_launches(params.LAUNCH_LOG_RAW)
    assert count_launches(rows, params.LAUNCH_LV_TYPE, params.LAUNCH_VARIANT) == (
        params.LAUNCH_N, params.LAUNCH_N_FAIL)
    assert count_launches(rows, params.LAUNCH_LV_TYPE, params.LAUNCH_VARIANT, "Transporter") == (
        params.LAUNCH_N_TRANSPORTER, params.LAUNCH_N_FAIL_TRANSPORTER)


def test_launch_is_all_or_nothing_with_frequency_1_minus_p():
    """Each launch loses all or none of its satellites; failed launches ~ Binomial(n, 1 - p)."""
    rng = np.random.default_rng(5)
    n_launches, n_sats, p = 20000, 4, 0.9
    lost = np.array([np.sum(launch_satellites(n_sats, 0.0, rng, 5 * YEAR_S, p, WEIBULL)[1]
                            == LAUNCH_FAILURE) for _ in range(n_launches)])
    assert set(np.unique(lost)) <= {0, n_sats}
    freq = np.mean(lost == n_sats)
    sigma = np.sqrt(p * (1 - p) / n_launches)
    assert abs(freq - (1 - p)) < 4 * sigma
