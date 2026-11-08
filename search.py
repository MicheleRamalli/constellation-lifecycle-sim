"""Search of the largest launch interval T that meets the mission criterion.

Criterion: the fraction of Monte Carlo runs with availability >= target is at
least `confidence`. The function `evaluate(T_days)` returns that fraction.

Method: coarse sweep, then bisection inside the bracket.
- A sweep alone gives T only to the grid resolution.
- Bisection alone needs a monotone criterion, which is not guaranteed here
  (the cadence can resonate with the design life, and the criterion is a Monte
  Carlo estimate). The sweep finds the FIRST grid value that fails; the
  bisection (on integer days) then works between the last passing grid value
  and that one. Grid values that pass again after the first failure are
  reported as "non monotone" and are not used (conservative choice).

Common random numbers: every evaluation uses the same master seed, so two
nearby T share most random draws and their difference is mostly due to T, not
to Monte Carlo noise. The correlation is only partial: a different T changes
the number of satellites launched and therefore which random numbers each
satellite receives.
"""

import numpy as np


def bisect_interval(evaluate, lo_days, hi_days, confidence):
    """Bisection on integer days between a passing and a failing interval.

    Inputs:
        evaluate   : function T_days -> fraction of runs meeting the target [-]
        lo_days    : interval that meets the criterion [days]
        hi_days    : interval that does not meet it [days], hi > lo
        confidence : required fraction of runs [-]
    Output:
        largest passing interval found [days] (int)
    """
    lo, hi = int(lo_days), int(hi_days)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if evaluate(mid) >= confidence:
            lo = mid
        else:
            hi = mid
    return lo


def search_max_interval(evaluate, grid_days, confidence):
    """Coarse sweep on grid_days (ascending), then bisection in the first bracket.

    Inputs:
        evaluate   : function T_days -> fraction of runs meeting the target [-]
        grid_days  : candidate intervals [days], ascending integers
        confidence : required fraction of runs [-]
    Output: dict with
        "t_star_days"  : largest passing interval [days] (NaN if the first grid value fails,
                         last grid value if all pass)
        "fractions"    : criterion value at each grid point [-]
        "bracketed"    : True if a failing grid value was found
        "non_monotone" : True if some grid value passes after the first failure
    """
    fractions = np.array([evaluate(int(t)) for t in grid_days])
    passing = fractions >= confidence
    result = {"fractions": fractions, "bracketed": not passing.all(), "non_monotone": False}
    if not passing[0]:
        result["t_star_days"] = np.nan
        return result
    if passing.all():
        result["t_star_days"] = int(grid_days[-1])
        return result
    i_fail = int(np.argmin(passing))  # first failing grid value
    result["non_monotone"] = bool(passing[i_fail:].any())
    result["t_star_days"] = bisect_interval(evaluate, grid_days[i_fail - 1], grid_days[i_fail],
                                            confidence)
    return result
