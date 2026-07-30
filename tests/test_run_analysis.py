import os

import pandas as pd

from run_analysis import main


def test_main_produces_full_summary_and_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # symlink-free copy isn't needed: run against the real project files by
    # pointing cwd-relative paths at the actual repo config/data via absolute paths.
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent

    summary_df = main(
        scenarios_path=str(repo_root / "config" / "scenarios.yaml"),
        historical_path=str(repo_root / "data" / "historical_returns.csv"),
        output_dir=str(tmp_path / "output"),
    )

    assert isinstance(summary_df, pd.DataFrame)
    # 4 scenarios x 6 years_worked values (0-5)
    assert len(summary_df) == 24
    assert set(summary_df["scenario"].unique()) == {
        "single_taipei", "single_taichung", "couple_taipei", "couple_taichung",
    }
    assert set(summary_df["years_worked"].unique()) == {0, 1, 2, 3, 4, 5}
    assert summary_df["success_rate"].between(0, 1).all()

    assert os.path.exists(tmp_path / "output" / "summary.csv")
    assert os.path.exists(tmp_path / "output" / "success_rate_chart.png")
    assert os.path.exists(tmp_path / "output" / "ending_balance_chart.png")
