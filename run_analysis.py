import os

from src.backtest import aggregate_results, load_historical_data, run_rolling_backtest
from src.report import build_summary_table, plot_ending_balance, plot_success_rate, write_summary_csv
from src.scenarios import load_scenarios

ENGINE_PARAMS = {
    "cash_tent_size_years": 3,
    "tier_1_wr_threshold": 0.048,
    "tier_2_wr_threshold": 0.070,
    "budget_cut_percentage": 0.30,
    "barista_annual_income": 240_000,
}
HORIZON_YEARS = 40
YEARS_WORKED_RANGE = range(0, 6)


def main(
    scenarios_path: str = "config/scenarios.yaml",
    historical_path: str = "data/historical_returns.csv",
    output_dir: str = "output",
):
    scenarios = load_scenarios(scenarios_path)
    historical_df = load_historical_data(historical_path)

    all_results = {}
    for scenario_name, scenario in scenarios.items():
        for years_worked in YEARS_WORKED_RANGE:
            trial_df = run_rolling_backtest(
                scenario, years_worked, historical_df, HORIZON_YEARS, ENGINE_PARAMS
            )
            all_results[(scenario_name, years_worked)] = aggregate_results(trial_df)

    summary_df = build_summary_table(all_results)

    os.makedirs(output_dir, exist_ok=True)
    write_summary_csv(summary_df, os.path.join(output_dir, "summary.csv"))
    plot_success_rate(summary_df, os.path.join(output_dir, "success_rate_chart.png"))
    plot_ending_balance(summary_df, os.path.join(output_dir, "ending_balance_chart.png"))

    return summary_df


if __name__ == "__main__":
    result = main()
    print(result.to_string(index=False))
