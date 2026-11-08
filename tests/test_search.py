"""Validation of search.py on criteria with a known answer."""

import numpy as np

from search import bisect_interval, search_max_interval


def test_monotone_step_function():
    """Criterion passes for T <= 237 days: the search must return exactly 237."""
    calls = []

    def evaluate(t_days):
        calls.append(t_days)
        return 1.0 if t_days <= 237 else 0.0

    result = search_max_interval(evaluate, np.arange(90, 366, 15), 0.95)
    assert result["t_star_days"] == 237
    assert result["bracketed"] and not result["non_monotone"]
    assert len(calls) <= 19 + 4  # 19 grid points + log2(15) bisection steps


def test_non_monotone_is_flagged_and_first_bracket_used():
    """A passing grid point after the first failure is flagged and ignored."""
    def evaluate(t_days):
        return 1.0 if t_days <= 200 or 300 <= t_days <= 310 else 0.0

    result = search_max_interval(evaluate, np.arange(90, 366, 15), 0.95)
    assert result["t_star_days"] == 200
    assert result["non_monotone"]


def test_edge_cases():
    always = search_max_interval(lambda t: 1.0, np.arange(90, 366, 15), 0.95)
    never = search_max_interval(lambda t: 0.0, np.arange(90, 366, 15), 0.95)
    assert always["t_star_days"] == 360 and not always["bracketed"]
    assert np.isnan(never["t_star_days"])
    assert bisect_interval(lambda t: float(t <= 100), 90, 105, 0.95) == 100
