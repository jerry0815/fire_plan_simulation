import os

import pandas as pd

from src.backtest import load_historical_data, run_withdrawal_rate_sweep
from src.report import plot_wr_success_rate, write_summary_csv
from src.scenarios import load_scenarios

ENGINE_PARAMS = {
    "cash_tent_size_years": 3,
    "tier_1_wr_threshold": 0.048,
    "tier_2_wr_threshold": 0.070,
    "budget_cut_percentage": 0.10,
    "barista_annual_income": 240_000,
}
HORIZON_YEARS = 40
WITHDRAWAL_RATES = [round(0.025 + 0.0025 * i, 4) for i in range(23)]  # 2.50% .. 8.00% step 0.25%
SUCCESS_THRESHOLD = 0.95


def main(
    scenarios_path: str = "config/scenarios.yaml",
    historical_path: str = "data/historical_returns.csv",
    output_dir: str = "output",
):
    scenarios = load_scenarios(scenarios_path)
    historical_df = load_historical_data(historical_path)

    frames = []
    for scenario_name, scenario in scenarios.items():
        wr_df = run_withdrawal_rate_sweep(
            scenario, WITHDRAWAL_RATES, historical_df, HORIZON_YEARS, ENGINE_PARAMS
        )
        wr_df.insert(0, "scenario", scenario_name)
        frames.append(wr_df)
    summary_df = pd.concat(frames, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    write_summary_csv(summary_df, os.path.join(output_dir, "wr_summary.csv"))
    plot_wr_success_rate(summary_df, os.path.join(output_dir, "wr_success_rate_chart.png"))

    return summary_df


def print_safe_withdrawal_rates(summary_df: pd.DataFrame, threshold: float = SUCCESS_THRESHOLD) -> None:
    for scenario_name, group in summary_df.groupby("scenario"):
        safe = group[group["success_rate"] >= threshold]
        if not safe.empty:
            safe_wr = safe["withdrawal_rate"].max()
            print(f"{scenario_name}: safe withdrawal rate (>= {threshold:.0%} success) = {safe_wr:.2%}")
        else:
            print(f"{scenario_name}: no withdrawal rate in sweep achieves >= {threshold:.0%} success")


if __name__ == "__main__":
    result = main()
    print(result.to_string(index=False))
    print()
    print_safe_withdrawal_rates(result)
