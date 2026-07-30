import textwrap

import pandas as pd
import pytest

from src.backtest import (
    aggregate_results,
    load_historical_data,
    run_rolling_backtest,
    run_withdrawal_rate_sweep,
    run_wr_years_worked_grid,
    safe_withdrawal_rate_table,
    wrapped_window,
)
from src.scenarios import Scenario


def test_load_historical_data_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Year,Global_Market_Return,Inflation_Rate\n2000,0.1,0.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_historical_data(str(csv_path))


def test_load_historical_data_too_short_raises(tmp_path):
    csv_path = tmp_path / "short.csv"
    csv_path.write_text(
        "Year,Global_Market_Return,Inflation_Rate,Cash_Yield\n2000,0.1,0.02,0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 5 years"):
        load_historical_data(str(csv_path))


def test_load_historical_data_sorts_by_year(tmp_path):
    csv_path = tmp_path / "unsorted.csv"
    csv_path.write_text(
        textwrap.dedent("""\
            Year,Global_Market_Return,Inflation_Rate,Cash_Yield
            2002,0.1,0.02,0.01
            2000,0.1,0.02,0.01
            2001,0.1,0.02,0.01
            2003,0.1,0.02,0.01
            2004,0.1,0.02,0.01
        """),
        encoding="utf-8",
    )
    df = load_historical_data(str(csv_path))
    assert df["Year"].tolist() == [2000, 2001, 2002, 2003, 2004]


def test_wrapped_window_cycles_past_end_of_data():
    df = pd.DataFrame({"Year": [2000, 2001, 2002, 2003, 2004]})
    window = wrapped_window(df, start_index=3, length=7)
    assert window["Year"].tolist() == [2003, 2004, 2000, 2001, 2002, 2003, 2004]


def _make_flat_scenario(annual_cost, current_capital, monthly_work_income):
    return Scenario(
        name="test",
        annual_cost=annual_cost,
        current_capital=current_capital,
        current_age=35,
        monthly_work_income=monthly_work_income,
    )


def test_run_rolling_backtest_trial_count_matches_dataset_size():
    historical_df = pd.DataFrame({
        "Year": [2000, 2001, 2002, 2003, 2004],
        "Global_Market_Return": [0.05, 0.05, 0.05, 0.05, 0.05],
        "Inflation_Rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cash_Yield": [0.01, 0.01, 0.01, 0.01, 0.01],
    })
    scenario = _make_flat_scenario(annual_cost=300_000, current_capital=10_000_000, monthly_work_income=40_000)
    engine_params = {
        "cash_tent_size_years": 3,
        "tier_1_wr_threshold": 0.048,
        "tier_2_wr_threshold": 0.070,
        "budget_cut_percentage": 0.10,
        "barista_annual_income": 240_000,
    }

    trial_df = run_rolling_backtest(scenario, years_worked=1, historical_df=historical_df, horizon_years=2, engine_params=engine_params)

    assert len(trial_df) == 5
    assert set(trial_df.columns) == {"start_year", "survived", "ending_balance", "years_tier1_cut", "years_tier2_worked"}
    assert trial_df["survived"].all()  # flat positive returns, huge capital, lean budget -> always survives


def test_run_rolling_backtest_marks_accumulation_depletion_as_failed():
    historical_df = pd.DataFrame({
        "Year": [2000, 2001, 2002, 2003, 2004],
        "Global_Market_Return": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Inflation_Rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cash_Yield": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    scenario = _make_flat_scenario(annual_cost=360_000, current_capital=50_000, monthly_work_income=0)
    engine_params = {
        "cash_tent_size_years": 3,
        "tier_1_wr_threshold": 0.048,
        "tier_2_wr_threshold": 0.070,
        "budget_cut_percentage": 0.10,
        "barista_annual_income": 240_000,
    }

    trial_df = run_rolling_backtest(scenario, years_worked=1, historical_df=historical_df, horizon_years=2, engine_params=engine_params)

    assert not trial_df["survived"].any()
    assert (trial_df["years_tier1_cut"] == 0).all()
    assert (trial_df["years_tier2_worked"] == 0).all()


def test_run_withdrawal_rate_sweep_computes_implied_capital_and_success_rate():
    historical_df = pd.DataFrame({
        "Year": [2000, 2001, 2002, 2003, 2004],
        "Global_Market_Return": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Inflation_Rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cash_Yield": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    scenario = _make_flat_scenario(annual_cost=300_000, current_capital=0, monthly_work_income=0)
    engine_params = {
        "cash_tent_size_years": 3,
        "tier_1_wr_threshold": 0.048,
        "tier_2_wr_threshold": 0.070,
        "budget_cut_percentage": 0.10,
        "barista_annual_income": 240_000,
    }

    wr_df = run_withdrawal_rate_sweep(
        scenario, [0.03, 0.10], historical_df, horizon_years=2, engine_params=engine_params
    )

    assert list(wr_df["withdrawal_rate"]) == [0.03, 0.10]
    assert round(wr_df.iloc[0]["implied_initial_capital"]) == 10_000_000
    assert round(wr_df.iloc[1]["implied_initial_capital"]) == 3_000_000
    # 3% withdrawal rate on a flat 0%-return world always stays in the safe guardrail state
    assert wr_df.iloc[0]["success_rate"] == 1.0
    assert wr_df.iloc[0]["capped_work_success_rate"] == 1.0
    assert wr_df.iloc[0]["comfortable_success_rate"] == 1.0
    assert wr_df.iloc[0]["no_cut_success_rate"] == 1.0
    # 10% withdrawal rate always triggers tier2 (cut + work) every year -> never a "no_cut" success,
    # and with horizon_years=2 both years hit tier2, so it also clears the <5-years-worked cap easily
    assert wr_df.iloc[1]["no_cut_success_rate"] == 0.0
    assert wr_df.iloc[1]["capped_work_success_rate"] == 1.0
    assert set(wr_df.columns) == {
        "withdrawal_rate", "implied_initial_capital", "success_rate",
        "capped_work_success_rate", "comfortable_success_rate", "no_cut_success_rate",
        "median_ending_balance", "p10_ending_balance",
        "avg_years_tier1_cut", "avg_years_tier2_worked",
    }


def test_aggregate_results_computes_expected_stats():
    trial_df = pd.DataFrame({
        "start_year": [2000, 2001, 2002, 2003],
        "survived": [True, True, True, False],
        "ending_balance": [100, 200, 300, 0],
        "years_tier1_cut": [0, 1, 2, 0],
        "years_tier2_worked": [0, 0, 1, 0],
    })
    agg = aggregate_results(trial_df)
    assert agg["success_rate"] == 0.75
    assert agg["capped_work_success_rate"] == 0.75  # none of the survived trials hit the 5-year cap
    assert agg["comfortable_success_rate"] == 0.5
    assert agg["no_cut_success_rate"] == 0.25
    assert agg["median_ending_balance"] == 150.0
    assert agg["avg_years_tier1_cut"] == 0.75
    assert agg["avg_years_tier2_worked"] == 0.25


def test_aggregate_results_capped_work_excludes_trials_at_or_over_the_limit():
    trial_df = pd.DataFrame({
        "start_year": [2000, 2001, 2002],
        "survived": [True, True, True],
        "ending_balance": [100, 200, 300],
        "years_tier1_cut": [4, 5, 10],
        "years_tier2_worked": [4, 5, 10],  # under, at, and over the 5-year cap
    })
    agg = aggregate_results(trial_df)
    assert agg["capped_work_success_rate"] == pytest.approx(1 / 3)


def test_run_wr_years_worked_grid_adds_years_worked_dimension():
    historical_df = pd.DataFrame({
        "Year": [2000, 2001, 2002, 2003, 2004],
        "Global_Market_Return": [0.05, 0.05, 0.05, 0.05, 0.05],
        "Inflation_Rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cash_Yield": [0.01, 0.01, 0.01, 0.01, 0.01],
    })
    scenario = _make_flat_scenario(annual_cost=300_000, current_capital=0, monthly_work_income=40_000)
    engine_params = {
        "cash_tent_size_years": 3,
        "tier_1_wr_threshold": 0.048,
        "tier_2_wr_threshold": 0.070,
        "budget_cut_percentage": 0.10,
        "barista_annual_income": 240_000,
    }

    grid_df = run_wr_years_worked_grid(
        scenario, [0.05], [0, 1], historical_df, horizon_years=2, engine_params=engine_params
    )

    assert list(grid_df["years_worked"]) == [0, 1]
    assert list(grid_df["withdrawal_rate"]) == [0.05, 0.05]
    assert round(grid_df.iloc[0]["implied_initial_capital"]) == 6_000_000  # 300,000 / 0.05
    # working one more year (positive savings, positive returns) can only ever help or match
    assert grid_df.iloc[1]["success_rate"] >= grid_df.iloc[0]["success_rate"]


def test_safe_withdrawal_rate_table_returns_max_safe_rate_or_nan():
    grid_df = pd.DataFrame({
        "scenario": ["a", "a", "a", "b", "b"],
        "years_worked": [0, 0, 1, 0, 0],
        "withdrawal_rate": [0.03, 0.05, 0.03, 0.03, 0.05],
        "success_rate": [1.00, 0.50, 1.00, 0.80, 0.95],
    })

    safe_df = safe_withdrawal_rate_table(grid_df, threshold=0.95, success_column="success_rate")

    row_a0 = safe_df[(safe_df["scenario"] == "a") & (safe_df["years_worked"] == 0)].iloc[0]
    assert row_a0["safe_withdrawal_rate"] == 0.03
    row_a1 = safe_df[(safe_df["scenario"] == "a") & (safe_df["years_worked"] == 1)].iloc[0]
    assert row_a1["safe_withdrawal_rate"] == 0.03
    row_b0 = safe_df[(safe_df["scenario"] == "b") & (safe_df["years_worked"] == 0)].iloc[0]
    assert row_b0["safe_withdrawal_rate"] == 0.05
