from src.accumulation import run_accumulation


def test_zero_years_worked_returns_starting_capital_unchanged():
    result = run_accumulation(
        starting_capital=1_000_000,
        monthly_income=40_000,
        annual_cost=360_000,
        years_worked=0,
        returns=[],
    )
    assert result == 1_000_000


def test_positive_savings_compounds_with_market_returns():
    # annual_savings = 40,000*12 - 360,000 = 120,000
    result = run_accumulation(
        starting_capital=0,
        monthly_income=40_000,
        annual_cost=360_000,
        years_worked=2,
        returns=[0.05, 0.03],
    )
    # year1: 0*1.05 + 120,000 = 120,000
    # year2: 120,000*1.03 + 120,000 = 243,600
    assert round(result) == 243_600


def test_negative_savings_still_draws_down_capital_without_erroring():
    # annual_savings = 20,000*12 - 360,000 = -120,000
    result = run_accumulation(
        starting_capital=1_000_000,
        monthly_income=20_000,
        annual_cost=360_000,
        years_worked=1,
        returns=[0.05],
    )
    # 1,000,000*1.05 - 120,000 = 930,000
    assert round(result) == 930_000


def test_depletion_during_accumulation_returns_nonpositive_value():
    result = run_accumulation(
        starting_capital=50_000,
        monthly_income=0,
        annual_cost=360_000,
        years_worked=1,
        returns=[0.0],
    )
    assert result <= 0
    assert round(result) == -310_000
