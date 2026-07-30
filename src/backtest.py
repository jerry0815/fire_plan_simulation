import pandas as pd

from src.accumulation import run_accumulation
from src.decumulation import run_decumulation, summarize_decumulation


def load_historical_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"Year", "Global_Market_Return", "Inflation_Rate", "Cash_Yield"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"historical data missing columns: {missing}")
    if len(df) < 5:
        raise ValueError("historical data must contain at least 5 years")
    return df.sort_values("Year").reset_index(drop=True)


def wrapped_window(df: pd.DataFrame, start_index: int, length: int) -> pd.DataFrame:
    n = len(df)
    indices = [(start_index + i) % n for i in range(length)]
    return df.iloc[indices].reset_index(drop=True)


def run_rolling_backtest(scenario, years_worked, historical_df, horizon_years, engine_params) -> pd.DataFrame:
    n = len(historical_df)
    total_len = years_worked + horizon_years
    trials = []

    for start_idx in range(n):
        window = wrapped_window(historical_df, start_idx, total_len)
        accum_window = window.iloc[:years_worked]
        decum_window = window.iloc[years_worked:].reset_index(drop=True)
        start_year = historical_df.iloc[start_idx]["Year"]

        ending_capital = run_accumulation(
            starting_capital=scenario.current_capital,
            monthly_income=scenario.monthly_work_income,
            annual_cost=scenario.annual_cost,
            years_worked=years_worked,
            returns=accum_window["Global_Market_Return"].tolist(),
        )

        if ending_capital <= 0:
            trials.append({
                "start_year": start_year,
                "survived": False,
                "ending_balance": ending_capital,
                "years_tier1_cut": 0,
                "years_tier2_worked": 0,
            })
            continue

        records = run_decumulation(
            initial_capital=ending_capital,
            initial_annual_budget=scenario.annual_cost,
            cash_tent_years=engine_params["cash_tent_size_years"],
            tier1_wr=engine_params["tier_1_wr_threshold"],
            tier2_wr=engine_params["tier_2_wr_threshold"],
            budget_cut_pct=engine_params["budget_cut_percentage"],
            barista_annual_income=engine_params["barista_annual_income"],
            returns=decum_window["Global_Market_Return"].tolist(),
            inflation=decum_window["Inflation_Rate"].tolist(),
            cash_yields=decum_window["Cash_Yield"].tolist(),
        )
        summary = summarize_decumulation(records)

        trials.append({
            "start_year": start_year,
            "survived": summary["survived"],
            "ending_balance": summary["ending_balance"],
            "years_tier1_cut": summary["years_tier1_cut"],
            "years_tier2_worked": summary["years_tier2_worked"],
        })

    return pd.DataFrame(trials)


def aggregate_results(trial_df: pd.DataFrame) -> dict:
    return {
        "success_rate": trial_df["survived"].mean(),
        "median_ending_balance": trial_df["ending_balance"].median(),
        "p10_ending_balance": trial_df["ending_balance"].quantile(0.10),
        "avg_years_tier1_cut": trial_df["years_tier1_cut"].mean(),
        "avg_years_tier2_worked": trial_df["years_tier2_worked"].mean(),
    }
