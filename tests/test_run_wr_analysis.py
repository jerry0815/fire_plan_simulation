import os
import pathlib

import pandas as pd

from run_wr_analysis import main


def test_main_produces_wr_summary_and_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo_root = pathlib.Path(__file__).parent.parent

    summary_df = main(
        scenarios_path=str(repo_root / "config" / "scenarios.yaml"),
        historical_path=str(repo_root / "data" / "historical_returns.csv"),
        output_dir=str(tmp_path / "output"),
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert set(summary_df["scenario"].unique()) == {
        "single_taipei", "single_taichung", "couple_taipei", "couple_taichung",
    }
    assert summary_df["success_rate"].between(0, 1).all()
    assert summary_df["capped_work_success_rate"].between(0, 1).all()
    assert summary_df["comfortable_success_rate"].between(0, 1).all()
    assert summary_df["no_cut_success_rate"].between(0, 1).all()
    # stricter tiers can never exceed the looser ones they're nested inside
    assert (summary_df["capped_work_success_rate"] <= summary_df["success_rate"]).all()
    assert (summary_df["comfortable_success_rate"] <= summary_df["capped_work_success_rate"]).all()
    assert (summary_df["no_cut_success_rate"] <= summary_df["comfortable_success_rate"]).all()
    assert (summary_df["implied_initial_capital"] > 0).all()

    assert os.path.exists(tmp_path / "output" / "wr_summary.csv")
    assert os.path.exists(tmp_path / "output" / "wr_success_rate_chart.png")
    assert os.path.exists(tmp_path / "output" / "wr_capped_work_success_rate_chart.png")
    assert os.path.exists(tmp_path / "output" / "wr_comfortable_success_rate_chart.png")
    assert os.path.exists(tmp_path / "output" / "wr_no_cut_success_rate_chart.png")
