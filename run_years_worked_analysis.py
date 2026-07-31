import os

import pandas as pd

from src.backtest import load_historical_data, run_wr_years_worked_grid, safe_withdrawal_rate_table
from src.report import plot_safe_wr_vs_years_worked, write_summary_csv
from src.scenarios import load_scenarios

ENGINE_PARAMS = {
    "cash_tent_size_years": 3,
    "tier_1_wr_threshold": 0.048,
    "tier_2_wr_threshold": 0.070,
    "budget_cut_percentage": 0.30,
    "barista_annual_income": 240_000,
}
HORIZON_YEARS = 40
WITHDRAWAL_RATES = [round(0.025 + 0.0025 * i, 4) for i in range(23)]  # 2.50% .. 8.00% step 0.25%
YEARS_WORKED_RANGE = range(0, 5)  # 0-4: less than 5 additional years of work
SUCCESS_THRESHOLD = 0.95

# (success column, output basename, console label, chart title)
SUCCESS_TIERS = [
    ("success_rate", "safe_wr_vs_years_worked_bare_survival",
     "bare survival", "Safe Withdrawal Rate vs. Years Worked (balance > 0 at year 40)"),
    ("capped_work_success_rate", "safe_wr_vs_years_worked_capped_work",
     "< 5 years of work", "Safe Withdrawal Rate vs. Years Worked (fewer than 5 years of forced work)"),
    ("comfortable_success_rate", "safe_wr_vs_years_worked_comfortable",
     "no forced work", "Safe Withdrawal Rate vs. Years Worked (never forced back to work)"),
    ("no_cut_success_rate", "safe_wr_vs_years_worked_no_cut",
     "no cuts at all", "Safe Withdrawal Rate vs. Years Worked (never took a lifestyle cut)"),
]


def main(
    scenarios_path: str = "config/scenarios.yaml",
    historical_path: str = "data/historical_returns.csv",
    output_dir: str = "output",
):
    scenarios = load_scenarios(scenarios_path)
    historical_df = load_historical_data(historical_path)

    frames = []
    for scenario_name, scenario in scenarios.items():
        grid_df = run_wr_years_worked_grid(
            scenario, WITHDRAWAL_RATES, YEARS_WORKED_RANGE, historical_df, HORIZON_YEARS, ENGINE_PARAMS
        )
        grid_df.insert(0, "scenario", scenario_name)
        frames.append(grid_df)
    grid_df = pd.concat(frames, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    write_summary_csv(grid_df, os.path.join(output_dir, "years_worked_wr_grid.csv"))

    safe_tables = {}
    for success_column, basename, _label, title in SUCCESS_TIERS:
        safe_df = safe_withdrawal_rate_table(grid_df, SUCCESS_THRESHOLD, success_column)
        safe_tables[success_column] = safe_df
        write_summary_csv(safe_df, os.path.join(output_dir, f"{basename}.csv"))
        plot_safe_wr_vs_years_worked(safe_df, os.path.join(output_dir, f"{basename}.png"), title=title)

    return grid_df, safe_tables


if __name__ == "__main__":
    grid_df, safe_tables = main()
    for success_column, _basename, label, _title in SUCCESS_TIERS:
        print(f"-- safe withdrawal rate by {label} (>= {SUCCESS_THRESHOLD:.0%}), by years worked --")
        pivot = safe_tables[success_column].pivot(
            index="scenario", columns="years_worked", values="safe_withdrawal_rate"
        )
        print((pivot * 100).round(2).to_string())
        print()
