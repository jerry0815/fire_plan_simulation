def run_accumulation(
    starting_capital: float,
    monthly_income: float,
    annual_cost: float,
    years_worked: int,
    returns: list[float],
) -> float:
    capital = starting_capital
    annual_savings = monthly_income * 12 - annual_cost

    for r in returns[:years_worked]:
        capital = capital * (1 + r) + annual_savings
        if capital <= 0:
            return capital

    return capital
