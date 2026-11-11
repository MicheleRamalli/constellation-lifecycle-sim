"""Launch success probability from J. McDowell's GCAT launch log.

Source: J. McDowell, General Catalog of Artificial Space Objects (GCAT), launch list
https://planet4589.org/space/gcat/tsv/launch/launch.tsv ; column definitions in
https://planet4589.org/space/gcat/web/launch/lcols.html
LaunchCode: first letter = category (O orbital, D deep space, A atmospheric, ...),
second letter = outcome (S success, F failure, U unknown, E pad explosion), optionally
followed by a partial-success percentage (e.g. OF40).

Run once:  python launch_log.py  -> prints the counts copied into params.py.
The raw file stays out of git (like the raw TLEs).
"""

import numpy as np
from scipy.stats import beta

import params


def read_launches(path):
    """Read the GCAT launch list (tab separated, header lines start with '#').

    Input: path of launch.tsv
    Output: list of dicts with keys date, lv_type, variant, mission, launch_code (all str)
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            col = [c.strip() for c in line.rstrip("\n").split("\t")]
            rows.append({"date": col[2], "lv_type": col[3], "variant": col[4],
                         "mission": col[8], "launch_code": col[26]})
    return rows


def count_launches(rows, lv_type, variant, mission_prefix=""):
    """Count orbital/deep-space launches of one vehicle and their failures.

    Only categories O (orbital) and D (deep space) are counted: suborbital tests
    (e.g. category A, the Dragon in-flight abort) are not orbital launch attempts.
    Outcome F (also with a partial-success percentage, e.g. OF40) is a failure:
    in the lifecycle model a failed launch loses all its satellites.
    Outcomes other than S and F (U unknown, E pad explosion) raise an error, so
    they are noticed and decided explicitly.

    Inputs: rows from read_launches, LV_Type, Variant, optional mission name prefix
    Outputs: (n launches, n failures)
    """
    n, n_fail = 0, 0
    for r in rows:
        if r["lv_type"] != lv_type or r["variant"] != variant:
            continue
        if not r["mission"].startswith(mission_prefix):
            continue
        code = r["launch_code"]
        if code[0] not in "OD":
            continue
        if code[1] not in "SF":
            raise ValueError(f"Outcome code {code} on {r['date']}: decide how to count it")
        n += 1
        n_fail += code[1] == "F"
    return n, n_fail


def jeffreys(n_success, n, confidence=0.95):
    """Jeffreys estimate of a success probability.

    Posterior Beta(k + 1/2, n - k + 1/2) from the Jeffreys prior Beta(1/2, 1/2).
    Inputs: successes k, trials n, confidence level of the equal-tailed interval
    Outputs: (posterior mean (k + 1/2)/(n + 1), lower bound, upper bound)
    """
    a, b = n_success + 0.5, n - n_success + 0.5
    tail = (1.0 - confidence) / 2.0
    return a / (a + b), beta.ppf(tail, a, b), beta.ppf(1.0 - tail, a, b)


def main():
    rows = read_launches(params.LAUNCH_LOG_RAW)
    for label, prefix in [("Falcon 9 Block 5, all", ""), ("Falcon 9 Block 5, Transporter", "Transporter")]:
        n, n_fail = count_launches(rows, params.LAUNCH_LV_TYPE, params.LAUNCH_VARIANT, prefix)
        mean, lo, hi = jeffreys(n - n_fail, n)
        print(f"{label}: n = {n}, failures = {n_fail}; Jeffreys mean {mean:.5f}, "
              f"95 % interval {lo:.5f}-{hi:.5f}")


if __name__ == "__main__":
    main()
