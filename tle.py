"""Two-line element sets (TLE): parser, Brouwer mean motion, derived CSV of elements.

TLE format (fixed columns, 1-based, as in the CelesTrak / NORAD documentation):
  line 1: catalog number 3-7, international designator 10-17, epoch year 19-20,
          epoch day of year 21-32, first derivative of the mean motion divided by 2
          34-43 [rev/day^2], checksum 69
  line 2: inclination 9-16 [deg], RAAN 18-25 [deg], eccentricity 27-33 (leading
          decimal point implied), argument of perigee 35-42 [deg], mean anomaly
          44-51 [deg], mean motion 53-63 [rev/day], checksum 69
  checksum = sum of the digits of columns 1-68, with '-' counting 1, modulo 10.

The TLE mean motion is the SGP4 (Kozai) mean motion. The Brouwer mean motion and
semi-major axis are recovered with Hoots F.R., Roehrich R.L. (1980), "Spacetrack
Report No. 3: Models for propagation of NORAD element sets". Here with the WGS84
constants of params.py (SGP4 itself uses WGS72; the difference on a is far below a
metre-level concern for this use).
No SGP4 propagation: the elements are only used at their epoch, propagated with
the J2 secular rates of coverage.py where needed.
"""

import csv
import datetime

import numpy as np

import params


def checksum(line):
    """TLE checksum of one line: digits of columns 1-68, '-' counts 1, modulo 10."""
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


def epoch_datetime(year_2d, day_of_year):
    """TLE epoch -> UTC datetime. Years 57-99 are 1957-1999, 00-56 are 2000-2056."""
    year = 1900 + year_2d if year_2d >= 57 else 2000 + year_2d
    return (datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(days=day_of_year - 1.0))


def parse_tle(name, line1, line2):
    """Parse one TLE (name + two lines), checking line numbers, catalog numbers and checksums.

    Output: dict with name, norad, intdes, epoch (datetime UTC), inc_deg, raan_deg,
            ecc, argp_deg, mean_anomaly_deg, n_rev_day, ndot_rev_day2 (first
            derivative of the mean motion; the TLE field holds ndot/2)
    """
    line1, line2 = line1.rstrip(), line2.rstrip()
    if len(line1) != 69 or len(line2) != 69 or line1[0] != "1" or line2[0] != "2":
        raise ValueError(f"Malformed TLE for {name!r}")
    if line1[2:7] != line2[2:7]:
        raise ValueError(f"Catalog numbers differ in the two lines of {name!r}")
    for line in (line1, line2):
        if checksum(line) != int(line[68]):
            raise ValueError(f"Checksum error in TLE of {name!r}")
    return {
        "name": name.strip(),
        "norad": int(line1[2:7]),
        "intdes": line1[9:17].strip(),
        "epoch": epoch_datetime(int(line1[18:20]), float(line1[20:32])),
        "inc_deg": float(line2[8:16]),
        "raan_deg": float(line2[17:25]),
        "ecc": float("0." + line2[26:33]),
        "argp_deg": float(line2[34:42]),
        "mean_anomaly_deg": float(line2[43:51]),
        "n_rev_day": float(line2[52:63]),
        "ndot_rev_day2": 2.0 * float(line1[33:43]),
    }


def read_tle_text(text):
    """Parse a three-line TLE file (name line + two element lines per object)."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("A three-line TLE file must have 3 lines per object")
    return [parse_tle(*lines[i:i + 3]) for i in range(0, len(lines), 3)]


def brouwer_mean_motion(n_kozai_rad_s, ecc, inc_rad):
    """Brouwer mean motion and semi-major axis from the TLE (Kozai) mean motion.

    Hoots & Roehrich (1980), SGP4 initialisation, with k2 = J2 R^2 / 2:
        a1 = (mu / n0^2)^(1/3)
        d1 = (3/2) k2 / a1^2 (3 cos^2 i - 1) / (1 - e^2)^(3/2)
        a0 = a1 (1 - d1/3 - d1^2 - 134/81 d1^3)
        d0 = (3/2) k2 / a0^2 (3 cos^2 i - 1) / (1 - e^2)^(3/2)
        n'' = n0 / (1 + d0),  a'' = a0 / (1 - d0)
    Inputs: Kozai mean motion [rad/s], eccentricity [-], inclination [rad]
    Outputs: Brouwer mean motion [rad/s], Brouwer semi-major axis [m]
    """
    k2 = 0.5 * params.J2_EARTH * params.R_EARTH**2
    factor = 1.5 * k2 * (3.0 * np.cos(inc_rad)**2 - 1.0) / (1.0 - ecc**2)**1.5
    a1 = (params.MU_EARTH / n_kozai_rad_s**2)**(1.0 / 3.0)
    d1 = factor / a1**2
    a0 = a1 * (1.0 - d1 / 3.0 - d1**2 - 134.0 / 81.0 * d1**3)
    d0 = factor / a0**2
    return n_kozai_rad_s / (1.0 + d0), a0 / (1.0 - d0)


def element_rows(records):
    """Rows of the derived table: name, norad, intdes, epoch, a [km], e, i, RAAN, u [deg],
    ndot [rev/day^2].

    a is the Brouwer semi-major axis; u = argument of perigee + mean anomaly
    (mean argument of latitude, adequate for near-circular orbits).
    """
    rows = []
    for r in records:
        n_rad_s = r["n_rev_day"] * 2.0 * np.pi / params.SECONDS_PER_DAY
        _, a_m = brouwer_mean_motion(n_rad_s, r["ecc"], np.radians(r["inc_deg"]))
        rows.append({
            "name": r["name"], "norad": r["norad"], "intdes": r["intdes"],
            "epoch_utc": r["epoch"].strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "a_km": f"{a_m / 1e3:.4f}", "ecc": f"{r['ecc']:.7f}",
            "inc_deg": f"{r['inc_deg']:.4f}", "raan_deg": f"{r['raan_deg']:.4f}",
            "u_deg": f"{(r['argp_deg'] + r['mean_anomaly_deg']) % 360.0:.4f}",
            "n_rev_day": f"{r['n_rev_day']:.8f}", "ndot_rev_day2": f"{r['ndot_rev_day2']:.4e}",
        })
    return rows


def write_elements_csv(path, rows, header_lines):
    """Write the derived table, with '#' comment lines (source, download date) on top."""
    with open(path, "w", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_elements_csv(path):
    """Read the derived table. Output: dict of numpy arrays (angles in rad, a in m, epoch in s).

    Keys: name, norad, intdes, epoch_s (seconds since the earliest epoch of the
    table), epoch0 (that earliest epoch, datetime), a_m, ecc, inc_rad, raan_rad, u_rad,
    n_rad_s (TLE mean motion), ndot_rad_s2 (TLE first derivative of the mean motion).
    """
    rev_day = 2.0 * np.pi / params.SECONDS_PER_DAY  # 1 rev/day in rad/s
    with open(path) as f:
        reader = csv.DictReader(l for l in f if not l.startswith("#"))
        rows = list(reader)
    epochs = [datetime.datetime.fromisoformat(r["epoch_utc"]).replace(tzinfo=datetime.timezone.utc)
              for r in rows]
    epoch0 = min(epochs)
    return {
        "name": np.array([r["name"] for r in rows]),
        "norad": np.array([int(r["norad"]) for r in rows]),
        "intdes": np.array([r["intdes"] for r in rows]),
        "epoch0": epoch0,
        "epoch_s": np.array([(e - epoch0).total_seconds() for e in epochs]),
        "a_m": np.array([float(r["a_km"]) * 1e3 for r in rows]),
        "ecc": np.array([float(r["ecc"]) for r in rows]),
        "inc_rad": np.radians([float(r["inc_deg"]) for r in rows]),
        "raan_rad": np.radians([float(r["raan_deg"]) for r in rows]),
        "u_rad": np.radians([float(r["u_deg"]) for r in rows]),
        "n_rad_s": np.array([float(r["n_rev_day"]) for r in rows]) * rev_day,
        "ndot_rad_s2": np.array([float(r["ndot_rev_day2"]) for r in rows])
                       * rev_day / params.SECONDS_PER_DAY,
    }
