"""Download the IRIDE Eaglet II TLEs from CelesTrak ONCE and write the derived CSV.

Run by hand: python fetch_tle.py  (never called by main.py or by the tests).
- CelesTrak usage: a single download. If today's raw file already exists it is
  reused and nothing is downloaded.
- The raw TLE file is saved in inputs/tle/ (excluded from git: the CelesTrak
  usage policy does not mention redistribution and the repository will be public).
- The derived table params.IRIDE_ELEMENTS_CSV (name, NORAD, epoch, a, e, i, RAAN, u)
  is written with the source and the download date on top; it is tracked in git
  and is what the simulation reads.
"""

import datetime
import os
import urllib.request

import params
import tle


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(os.path.dirname(params.IRIDE_TLE_RAW_PATTERN), exist_ok=True)
    raw_path = params.IRIDE_TLE_RAW_PATTERN.format(date=now.strftime("%Y-%m-%d"))

    if os.path.exists(raw_path):
        print(f"Using the file already downloaded today: {raw_path}")
        with open(raw_path) as f:
            text = f.read()
        # download time = time the raw file was written
        now = datetime.datetime.fromtimestamp(os.path.getmtime(raw_path), datetime.timezone.utc)
    else:
        request = urllib.request.Request(params.IRIDE_TLE_URL,
                                         headers={"User-Agent": "constellation-lifecycle student project"})
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("ascii")
        with open(raw_path, "w") as f:
            f.write(text)
        print(f"Downloaded {params.IRIDE_TLE_URL} -> {raw_path}")

    records = tle.read_tle_text(text)  # checks format and checksums
    found = {r["norad"] for r in records}
    expected = set(params.IRIDE_NORAD)
    missing = sorted(expected - found)
    extra = sorted((r["norad"], r["name"]) for r in records if r["norad"] not in expected)
    print(f"Objects in the file: {len(records)}; expected NORAD found: {len(expected & found)}")
    if extra:
        print("Not in the NORAD list (ignored):", extra)
    if missing:
        raise SystemExit(f"Expected NORAD numbers missing from the download: {missing}")

    used = sorted((r for r in records if r["norad"] in expected), key=lambda r: r["norad"])
    header = [f"Derived from CelesTrak GP data: {params.IRIDE_TLE_URL}",
              f"Downloaded {now.strftime('%Y-%m-%d %H:%M')} UTC (raw TLE file not redistributed)",
              "a = Brouwer semi-major axis (Hoots & Roehrich 1980) [km]; i, RAAN in deg at epoch;",
              "u = argument of perigee + mean anomaly at epoch [deg];",
              "n = TLE (Kozai) mean motion [rev/day]; ndot = TLE first derivative of n [rev/day^2]"]
    tle.write_elements_csv(params.IRIDE_ELEMENTS_CSV, tle.element_rows(used), header)
    print(f"Wrote {len(used)} rows to {params.IRIDE_ELEMENTS_CSV}")


if __name__ == "__main__":
    main()
