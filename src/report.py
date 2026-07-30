import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_summary_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for (scenario_name, years_worked), agg in all_results.items():
        row = {"scenario": scenario_name, "years_worked": years_worked}
        row.update(agg)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["scenario", "years_worked"])
        .reset_index(drop=True)
    )


def write_summary_csv(summary_df: pd.DataFrame, path: str) -> None:
    summary_df.to_csv(path, index=False)


def plot_success_rate(summary_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario_name, group in summary_df.groupby("scenario"):
        group = group.sort_values("years_worked")
        ax.plot(group["years_worked"], group["success_rate"] * 100, marker="o", label=scenario_name)
    ax.set_xlabel("Years Worked Before FIRE")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Retirement Success Rate vs. Years Worked")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_ending_balance(summary_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario_name, group in summary_df.groupby("scenario"):
        group = group.sort_values("years_worked")
        ax.plot(group["years_worked"], group["median_ending_balance"], marker="o", label=f"{scenario_name} (median)")
        ax.plot(group["years_worked"], group["p10_ending_balance"], marker="x", linestyle="--", label=f"{scenario_name} (p10)")
    ax.set_xlabel("Years Worked Before FIRE")
    ax.set_ylabel("Ending Portfolio Balance (NTD)")
    ax.set_title("Ending Balance vs. Years Worked")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)
