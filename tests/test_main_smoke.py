"""End-to-end run of main.py in the default fast mode, without the external DRAMA files.

All the paths outside the repository (DRAMA 4.1.4 installation, DRAMA_projects runs) are
redirected to a folder that does not exist, as on a machine where they are not installed:
the run must finish, skipping those cases with a message. Figures go to a temporary folder.
Slow (~3 min): excluded from the default pytest run, run with  pytest -m slow
"""

import os
import sys

import pytest

import main
import params

FIGURES = ["revisit_vs_latitude.png", "revisit_table.png", "active_vs_time.png",
           "availability_vs_order_time.png", "cost_vs_availability.png", "iride_scenario.png",
           "decay.png", "end_of_life.png", "solar_cycle.png"]


@pytest.mark.slow
def test_main_fast_mode_without_drama(tmp_path, monkeypatch, capsys):
    missing = str(tmp_path / "not_installed")
    monkeypatch.setattr(params, "OSCAR_FORECAST_PATH", os.path.join(missing, "fap_mon.dat"))
    monkeypatch.setattr(params, "SARA_POPULATION_1D_PATH", os.path.join(missing, "pop.csv"))
    monkeypatch.setattr(params, "OSCAR_S3B_RUNS",
                        {k: {**v, "folder": os.path.join(missing, k)} for k, v in params.OSCAR_S3B_RUNS.items()})
    monkeypatch.setattr(params, "SARA_RUNS",
                        {k: {**v, "folder": os.path.join(missing, k)} for k, v in params.SARA_RUNS.items()})
    # main() lowers the run counts in fast mode: restore them after the test
    monkeypatch.setattr(params, "N_RUNS", params.N_RUNS)
    monkeypatch.setattr(params, "N_RUNS_SWEEP", params.N_RUNS_SWEEP)
    monkeypatch.setattr(main, "CADENCE_CACHE", {})
    # fast mode must write to FIG_DIR_FAST and leave FIG_DIR (tracked --full figures) untouched;
    # FIG_DIR is also set here so that monkeypatch restores it after main() switches it
    monkeypatch.setattr(main, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(main, "FIG_DIR_FAST", str(tmp_path / "figures_fast"))
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    out = capsys.readouterr().out
    assert "Mode: FAST" in out and "APPROXIMATE" in out
    assert "not found" in out                        # the missing DRAMA cases are reported
    assert not os.path.exists(tmp_path / "figures")  # nothing written where the --full figures live
    for name in FIGURES:
        assert os.path.exists(tmp_path / "figures_fast" / name), name
