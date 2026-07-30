import os
import pathlib

import pandas as pd

from run_years_worked_analysis import SUCCESS_TIERS, main


def test_main_produces_grid_and_safe_wr_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo_root = pathlib.Path(__file__).parent.parent

    grid_df, safe_tables = main(
        scenarios_path=str(repo_root / "config" / "scenarios.yaml"),
        historical_path=str(repo_root / "data" / "historical_returns.csv"),
        output_dir=str(tmp_path / "output"),
    )

    assert isinstance(grid_df, pd.DataFrame)
    assert set(grid_df["scenario"].unique()) == {
        "single_taipei", "single_taichung", "couple_taipei", "couple_taichung",
    }
    assert set(grid_df["years_worked"].unique()) == {0, 1, 2, 3, 4}
    assert grid_df["success_rate"].between(0, 1).all()

    assert os.path.exists(tmp_path / "output" / "years_worked_wr_grid.csv")
    for success_column, basename, _label, _title in SUCCESS_TIERS:
        assert success_column in safe_tables
        assert os.path.exists(tmp_path / "output" / f"{basename}.csv")
        assert os.path.exists(tmp_path / "output" / f"{basename}.png")
