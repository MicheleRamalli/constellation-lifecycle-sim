"""Validation of tle.py on a published TLE and on analytical special cases."""

import datetime

import numpy as np
import pytest

import params
import tle

# Example TLE of the ISS from Wikipedia, "Two-line element set" (decoded fields
# listed there; both checksums are 7). Transcription verified with the checksums.
ISS = ("ISS (ZARYA)",
       "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
       "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")


def test_published_tle_fields():
    r = tle.parse_tle(*ISS)
    assert r["norad"] == 25544 and r["intdes"] == "98067A"
    assert r["inc_deg"] == 51.6416 and r["raan_deg"] == 247.4627
    assert r["ecc"] == 0.0006703
    assert r["argp_deg"] == 130.5360 and r["mean_anomaly_deg"] == 325.0288
    assert r["n_rev_day"] == 15.72125391
    assert r["ndot_rev_day2"] == 2 * -0.00002182   # the TLE field holds ndot / 2
    assert tle.checksum(ISS[1]) == 7 and tle.checksum(ISS[2]) == 7


def test_epoch_conversion():
    """Year 08 -> 2008; day 264.51782528 of a leap year -> 20 September, 12:25:40.104 UTC."""
    epoch = tle.parse_tle(*ISS)["epoch"]
    expected = datetime.datetime(2008, 9, 20, tzinfo=datetime.timezone.utc) + \
        datetime.timedelta(days=0.51782528)
    assert abs((epoch - expected).total_seconds()) < 1e-3
    assert tle.epoch_datetime(57, 1.0).year == 1957 and tle.epoch_datetime(56, 1.0).year == 2056


def test_checksum_error_is_detected():
    bad = ISS[2][:20] + "8" + ISS[2][21:]   # change one digit, keep the old checksum
    with pytest.raises(ValueError):
        tle.parse_tle(ISS[0], ISS[1], bad)


def test_brouwer_recovery_special_cases():
    """At the 'magic' inclination 3 cos^2 i = 1 the J2 term vanishes: a'' = (mu/n^2)^(1/3).

    For a near-polar 500 km orbit the correction is negative (3 cos^2 i - 1 < 0)
    and of the order of J2 (R/a)^2 ~ 1e-3 relative, i.e. a few km.
    """
    n = 15.2 * 2 * np.pi / params.SECONDS_PER_DAY
    a_kepler = (params.MU_EARTH / n**2)**(1 / 3)
    inc_magic = np.arccos(np.sqrt(1 / 3))
    n_b, a_b = tle.brouwer_mean_motion(n, 0.0, inc_magic)
    assert np.isclose(a_b, a_kepler, rtol=1e-14) and np.isclose(n_b, n, rtol=1e-14)
    _, a_b = tle.brouwer_mean_motion(n, 0.0, np.radians(97.4))
    assert 0.5e3 < abs(a_b - a_kepler) < 10e3


def test_csv_roundtrip(tmp_path):
    rows = tle.element_rows([tle.parse_tle(*ISS)])
    path = tmp_path / "elements.csv"
    tle.write_elements_csv(path, rows, ["source: test"])
    el = tle.read_elements_csv(path)
    assert el["norad"][0] == 25544
    assert np.isclose(np.degrees(el["raan_rad"][0]), 247.4627)
    assert np.isclose(np.degrees(el["u_rad"][0]), (130.5360 + 325.0288) % 360)
    rev_day = 2 * np.pi / 86400.0
    assert np.isclose(el["n_rad_s"][0], 15.72125391 * rev_day)
    assert np.isclose(el["ndot_rad_s2"][0], -2 * 0.00002182 * rev_day / 86400.0)


def test_iride_csv_content():
    """The derived IRIDE table holds exactly the NORAD numbers of params.IRIDE_NORAD."""
    el = tle.read_elements_csv(params.IRIDE_ELEMENTS_CSV)
    assert sorted(el["norad"].tolist()) == sorted(params.IRIDE_NORAD)
