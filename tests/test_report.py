import os

import pandas as pd

from src.report import (
    build_summary_table,
    plot_ending_balance,
    plot_success_rate,
    write_summary_csv,
)


def _sample_all_results():
    return {
        ("single_taipei", 0): {
            "success_rate": 0.8, "median_ending_balance": 1_000_000,
            "p10_ending_balance": 100_000, "avg_years_tier1_cut": 2.0, "avg_years_tier2_worked": 0.5,
        },
        ("single_taipei", 1): {
            "success_rate": 0.9, "median_ending_balance": 1_500_000,
            "p10_ending_balance": 300_000, "avg_years_tier1_cut": 1.0, "avg_years_tier2_worked": 0.1,
        },
        ("couple_taichung", 0): {
            "success_rate": 0.85, "median_ending_balance": 1_200_000,
            "p10_ending_balance": 200_000, "avg_years_tier1_cut": 1.5, "avg_years_tier2_worked": 0.3,
        },
    }


def test_build_summary_table_shape_and_sort_order():
    df = build_summary_table(_sample_all_results())
    assert list(df.columns) == [
        "scenario", "years_worked", "success_rate", "median_ending_balance",
        "p10_ending_balance", "avg_years_tier1_cut", "avg_years_tier2_worked",
    ]
    assert len(df) == 3
    assert df.iloc[0]["scenario"] == "couple_taichung"
    assert df.iloc[1]["scenario"] == "single_taipei"
    assert df.iloc[1]["years_worked"] == 0
    assert df.iloc[2]["years_worked"] == 1


def test_write_summary_csv_round_trips(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "summary.csv"
    write_summary_csv(df, str(out_path))

    read_back = pd.read_csv(out_path)
    assert len(read_back) == len(df)
    assert list(read_back.columns) == list(df.columns)


def test_plot_success_rate_creates_nonempty_file(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "success_rate.png"
    plot_success_rate(df, str(out_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_ending_balance_creates_nonempty_file(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "ending_balance.png"
    plot_ending_balance(df, str(out_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
