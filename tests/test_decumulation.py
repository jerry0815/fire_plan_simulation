from src.decumulation import run_decumulation, summarize_decumulation


def test_state_a_safe_no_cut():
    records = run_decumulation(
        initial_capital=10_000_000,
        initial_annual_budget=300_000,
        cash_tent_years=3,
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=240_000,
        returns=[0.05],
        inflation=[0.0],
        cash_yields=[0.01],
    )
    r = records[0]
    assert r["total_portfolio_jan1"] == 10_000_000
    assert r["target_budget"] == 300_000
    assert round(r["wr"], 4) == 0.03
    assert r["state"] == "A"
    assert r["net_drain"] == 300_000
    # cash_tent = 900,000 -> 600,000 after drain -> *1.01 = 606,000
    # equity = 9,100,000 unchanged by drain -> *1.05 = 9,555,000
    assert round(r["end_portfolio"]) == 10_161_000


def test_state_b_tier1_cut_no_work():
    records = run_decumulation(
        initial_capital=5_000_000,
        initial_annual_budget=300_000,
        cash_tent_years=3,
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=240_000,
        returns=[0.0],
        inflation=[0.0],
        cash_yields=[0.0],
    )
    r = records[0]
    assert round(r["wr"], 4) == 0.06
    assert r["state"] == "B"
    assert r["net_drain"] == 270_000
    # cash_tent = 900,000 - 270,000 = 630,000; equity unchanged at 4,100,000
    assert round(r["end_portfolio"]) == 4_730_000


def test_state_c_tier2_cut_and_work():
    records = run_decumulation(
        initial_capital=3_000_000,
        initial_annual_budget=300_000,
        cash_tent_years=3,
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=240_000,
        returns=[0.0],
        inflation=[0.0],
        cash_yields=[0.0],
    )
    r = records[0]
    assert round(r["wr"], 4) == 0.10
    assert r["state"] == "C"
    # actual_spend = 270,000; barista = 240,000 -> required withdrawal = 30,000
    assert r["net_drain"] == 30_000
    # cash_tent = 900,000 - 30,000 = 870,000; equity unchanged at 2,100,000
    assert round(r["end_portfolio"]) == 2_970_000


def test_drain_order_cash_tent_then_equity():
    records = run_decumulation(
        initial_capital=3_000_000,
        initial_annual_budget=300_000,
        cash_tent_years=0.05,  # cash_tent = 15,000, smaller than required withdrawal
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=240_000,
        returns=[0.0],
        inflation=[0.0],
        cash_yields=[0.0],
    )
    r = records[0]
    # required withdrawal = 30,000; cash_tent only has 15,000 -> drains to 0,
    # remaining 15,000 comes from equity (2,985,000 - 15,000 = 2,970,000)
    assert r["net_drain"] == 30_000
    assert round(r["end_portfolio"]) == 2_970_000


def test_inflation_compounds_target_budget_across_years():
    records = run_decumulation(
        initial_capital=100_000_000,  # huge, so it always stays in State A
        initial_annual_budget=300_000,
        cash_tent_years=3,
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=240_000,
        returns=[0.0, 0.0, 0.0],
        inflation=[0.05, 0.03, 0.02],
        cash_yields=[0.0, 0.0, 0.0],
    )
    assert round(records[0]["target_budget"]) == 300_000
    assert round(records[1]["target_budget"]) == 315_000
    assert round(records[2]["target_budget"]) == 324_450


def test_depletion_stops_at_zero_and_survived_is_false():
    records = run_decumulation(
        initial_capital=100_000,
        initial_annual_budget=300_000,
        cash_tent_years=0.1,
        tier1_wr=0.042,
        tier2_wr=0.070,
        budget_cut_pct=0.10,
        barista_annual_income=0,
        returns=[-0.5, 0.0],
        inflation=[0.0, 0.0],
        cash_yields=[0.0, 0.0],
    )
    summary = summarize_decumulation(records)
    assert summary["survived"] is False
    assert summary["ending_balance"] == 0.0


def test_summarize_counts_tier1_and_tier2_years():
    records = [
        {"state": "A", "end_portfolio": 100},
        {"state": "B", "end_portfolio": 90},
        {"state": "C", "end_portfolio": 80},
        {"state": "C", "end_portfolio": 70},
    ]
    summary = summarize_decumulation(records)
    assert summary["ending_balance"] == 70
    assert summary["years_tier1_cut"] == 3  # B + C + C
    assert summary["years_tier2_worked"] == 2  # C + C
    assert summary["survived"] is True
