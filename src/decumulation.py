def run_decumulation(
    initial_capital: float,
    initial_annual_budget: float,
    cash_tent_years: float,
    tier1_wr: float,
    tier2_wr: float,
    budget_cut_pct: float,
    barista_annual_income: float,
    returns: list[float],
    inflation: list[float],
    cash_yields: list[float],
) -> list[dict]:
    horizon = len(returns)
    cash_tent = initial_annual_budget * cash_tent_years
    equity = initial_capital - cash_tent
    cumulative_inflation = 1.0
    depleted = False
    records: list[dict] = []

    for t in range(horizon):
        if t > 0:
            cumulative_inflation *= (1 + inflation[t - 1])
        target_budget = initial_annual_budget * cumulative_inflation
        total_portfolio = equity + cash_tent

        if depleted or total_portfolio <= 0:
            records.append({
                "year_index": t,
                "total_portfolio_jan1": max(total_portfolio, 0.0),
                "target_budget": target_budget,
                "wr": None,
                "state": "Depleted",
                "net_drain": 0.0,
                "end_portfolio": 0.0,
            })
            equity = 0.0
            cash_tent = 0.0
            depleted = True
            continue

        wr = target_budget / total_portfolio

        if wr < tier1_wr:
            state = "A"
            actual_spend = target_budget
            barista_income = 0.0
        elif wr < tier2_wr:
            state = "B"
            actual_spend = target_budget * (1 - budget_cut_pct)
            barista_income = 0.0
        else:
            state = "C"
            actual_spend = target_budget * (1 - budget_cut_pct)
            barista_income = barista_annual_income

        required_withdrawal = max(actual_spend - barista_income, 0.0)

        drain_from_cash = min(required_withdrawal, cash_tent)
        cash_tent -= drain_from_cash
        remaining = required_withdrawal - drain_from_cash
        equity -= remaining

        equity = equity * (1 + returns[t])
        cash_tent = cash_tent * (1 + cash_yields[t])

        end_portfolio = equity + cash_tent
        if end_portfolio <= 0:
            equity = 0.0
            cash_tent = 0.0
            end_portfolio = 0.0
            depleted = True

        records.append({
            "year_index": t,
            "total_portfolio_jan1": total_portfolio,
            "target_budget": target_budget,
            "wr": wr,
            "state": state,
            "net_drain": required_withdrawal,
            "end_portfolio": end_portfolio,
        })

    return records


def summarize_decumulation(records: list[dict]) -> dict:
    ending_balance = records[-1]["end_portfolio"]
    years_tier1_cut = sum(1 for r in records if r["state"] in ("B", "C"))
    years_tier2_worked = sum(1 for r in records if r["state"] == "C")
    return {
        "ending_balance": ending_balance,
        "years_tier1_cut": years_tier1_cut,
        "years_tier2_worked": years_tier2_worked,
        "survived": ending_balance > 0,
    }
