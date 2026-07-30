# Taiwan Lean-FIRE Rolling-Window Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rolling-window (cFIREsim-style) backtesting tool that evaluates lean-FIRE safety across 4 scenarios (single/couple × Taipei/Taichung), for 0–5 years of pre-FIRE "back to work" accumulation at 40,000 NTD/month.

**Architecture:** A pre-FIRE accumulation phase (savings invested at historical market returns) feeds into an existing dual-trigger Guyton-Klinger decumulation engine (per `spec.md`), both walked across every historical starting year with wraparound, aggregated into a success-rate table and charts.

**Tech Stack:** Python 3.10+, pandas, numpy, matplotlib, PyYAML, pytest.

## Global Constraints

- Language: Python 3.10+.
- Dependencies (pin in `requirements.txt`): `pandas`, `numpy`, `matplotlib`, `pyyaml`, `pytest`.
- All monetary amounts are NTD (Taiwan dollars); no currency conversion anywhere.
- Historical market data (`data/historical_returns.csv`) is a researched approximation for backtesting only — must ship with a disclaimer, not audited financial data (per spec section 2, non-goals).
- Guardrail engine parameters (`cash_tent_size_years`, `tier_1_wr_threshold`, `tier_2_wr_threshold`, `budget_cut_percentage`, `barista_annual_income`) are global across all 4 scenarios in v1, not per-scenario.
- Retirement horizon defaults to 40 years; `years_worked` is evaluated over `range(0, 6)` (0 through 5 inclusive).
- Scenario placeholder fields (`current_capital_ntd`, `current_age`) must stay clearly marked as user-editable in `config/scenarios.yaml`.
- No web server, no network calls at runtime — all inputs are static local files.

---

### Task 1: Project scaffolding + historical dataset

**Files:**
- Create: `requirements.txt`
- Create: `data/historical_returns.csv`
- Create: `data/README.md`
- Create: `tests/test_backtest_data.py`

**Interfaces:**
- Produces: `data/historical_returns.csv` with columns `Year, Global_Market_Return, Inflation_Rate, Cash_Yield`, 55 rows (1970–2024), which Task 5's `load_historical_data()` will consume.

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
pytest>=7.4
```

- [ ] **Step 2: Create `data/historical_returns.csv` with the researched approximation dataset**

```csv
Year,Global_Market_Return,Inflation_Rate,Cash_Yield
1970,0.040,0.057,0.065
1971,0.143,0.044,0.043
1972,0.190,0.032,0.041
1973,-0.147,0.062,0.070
1974,-0.265,0.110,0.079
1975,0.372,0.091,0.058
1976,0.238,0.058,0.050
1977,-0.070,0.065,0.053
1978,0.065,0.076,0.072
1979,0.185,0.113,0.100
1980,0.317,0.135,0.115
1981,-0.047,0.103,0.140
1982,0.204,0.062,0.107
1983,0.223,0.032,0.086
1984,0.061,0.043,0.096
1985,0.316,0.036,0.075
1986,0.186,0.019,0.060
1987,0.052,0.036,0.058
1988,0.166,0.041,0.067
1989,0.317,0.048,0.081
1990,-0.031,0.054,0.075
1991,0.305,0.042,0.054
1992,0.076,0.030,0.035
1993,0.101,0.030,0.030
1994,0.013,0.026,0.043
1995,0.376,0.028,0.055
1996,0.230,0.030,0.050
1997,0.334,0.023,0.051
1998,0.286,0.016,0.048
1999,0.210,0.022,0.046
2000,-0.091,0.034,0.059
2001,-0.119,0.028,0.034
2002,-0.221,0.016,0.016
2003,0.287,0.023,0.010
2004,0.109,0.027,0.014
2005,0.049,0.034,0.031
2006,0.158,0.032,0.047
2007,0.055,0.029,0.044
2008,-0.370,0.038,0.014
2009,0.265,-0.004,0.002
2010,0.151,0.016,0.001
2011,0.021,0.032,0.001
2012,0.160,0.021,0.001
2013,0.324,0.015,0.001
2014,0.137,0.016,0.000
2015,0.014,0.001,0.001
2016,0.120,0.013,0.003
2017,0.218,0.021,0.009
2018,-0.044,0.024,0.019
2019,0.315,0.018,0.021
2020,0.184,0.012,0.004
2021,0.287,0.047,0.000
2022,-0.181,0.080,0.020
2023,0.263,0.041,0.050
2024,0.250,0.029,0.052
```

- [ ] **Step 3: Create `data/README.md` documenting the dataset's approximation methodology**

```markdown
# Historical Returns Dataset

**This is a researched approximation for backtesting, not audited financial data.**

- `Global_Market_Return`: approximated from US large-cap total returns (a common
  simplification in toy FIRE backtesters, since a true MSCI ACWI series pre-1988
  doesn't exist and full global diversification data requires paid sources).
  Real global-diversified returns may differ, especially in decades where
  international equities notably over- or under-performed the US.
- `Inflation_Rate`: approximated from US CPI-U annual change. Taiwan's actual
  inflation history differs from this in most years; treat as a rough proxy.
- `Cash_Yield`: approximated from US 3-month T-bill annual yield, as a proxy for
  a cash-tent's holding yield.

Edit this file's data directly if you have a more precise dataset you trust more
(e.g. an actual MSCI ACWI total return series or Taiwan CPI series).
```

- [ ] **Step 4: Write a test verifying the CSV loads and has the expected shape**

```python
# tests/test_backtest_data.py
import csv
from pathlib import Path

def test_historical_csv_has_required_columns_and_row_count():
    path = Path(__file__).parent.parent / "data" / "historical_returns.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 55
    required_cols = {"Year", "Global_Market_Return", "Inflation_Rate", "Cash_Yield"}
    assert required_cols.issubset(rows[0].keys())
    assert rows[0]["Year"] == "1970"
    assert rows[-1]["Year"] == "2024"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_backtest_data.py -v`
Expected: PASS (2 checks: row count and column/year bounds)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt data/historical_returns.csv data/README.md tests/test_backtest_data.py
git commit -m "Add project scaffolding and historical returns dataset"
```

---

### Task 2: Scenario config + loader

**Files:**
- Create: `config/scenarios.yaml`
- Create: `src/__init__.py`
- Create: `src/scenarios.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces: `Scenario` dataclass with fields `name: str, annual_cost: float, current_capital: float, current_age: int, monthly_work_income: float`, and `load_scenarios(path: str) -> dict[str, Scenario]`. Consumed by Task 5's `run_rolling_backtest`.

- [ ] **Step 1: Create `config/scenarios.yaml` with researched lean-FIRE cost breakdowns**

```yaml
single_taipei:
  monthly_costs_ntd:
    rent: 13000
    food: 9000
    utilities: 1800
    transport: 1200
    health_insurance: 1000
    phone_internet: 700
    misc_buffer: 3000
  placeholders:
    current_capital_ntd: 0        # EDIT ME
    current_age: 35               # EDIT ME
    monthly_work_income_ntd: 40000

single_taichung:
  monthly_costs_ntd:
    rent: 9000
    food: 8000
    utilities: 1700
    transport: 1000
    health_insurance: 1000
    phone_internet: 700
    misc_buffer: 2500
  placeholders:
    current_capital_ntd: 0        # EDIT ME
    current_age: 35               # EDIT ME
    monthly_work_income_ntd: 40000

couple_taipei:
  monthly_costs_ntd:
    rent: 18000
    food: 16000
    utilities: 2500
    transport: 2200
    health_insurance: 2000
    phone_internet: 1200
    misc_buffer: 4500
  placeholders:
    current_capital_ntd: 0        # EDIT ME
    current_age: 35               # EDIT ME
    monthly_work_income_ntd: 40000

couple_taichung:
  monthly_costs_ntd:
    rent: 12000
    food: 14000
    utilities: 2300
    transport: 1800
    health_insurance: 2000
    phone_internet: 1200
    misc_buffer: 3800
  placeholders:
    current_capital_ntd: 0        # EDIT ME
    current_age: 35               # EDIT ME
    monthly_work_income_ntd: 40000
```

- [ ] **Step 2: Create empty `src/__init__.py`**

```python
```

- [ ] **Step 3: Write the failing test for `load_scenarios`**

```python
# tests/test_scenarios.py
import textwrap
import pytest
from src.scenarios import load_scenarios, Scenario

def test_load_scenarios_computes_annual_cost_and_placeholders(tmp_path):
    yaml_content = textwrap.dedent("""
        single_taipei:
          monthly_costs_ntd:
            rent: 13000
            food: 9000
          placeholders:
            current_capital_ntd: 500000
            current_age: 40
            monthly_work_income_ntd: 40000
    """)
    config_path = tmp_path / "scenarios.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    scenarios = load_scenarios(str(config_path))

    assert set(scenarios.keys()) == {"single_taipei"}
    s = scenarios["single_taipei"]
    assert isinstance(s, Scenario)
    assert s.name == "single_taipei"
    assert s.annual_cost == (13000 + 9000) * 12
    assert s.current_capital == 500000
    assert s.current_age == 40
    assert s.monthly_work_income == 40000


def test_load_scenarios_missing_field_raises(tmp_path):
    yaml_content = textwrap.dedent("""
        single_taipei:
          monthly_costs_ntd:
            rent: 13000
          placeholders:
            current_age: 40
            monthly_work_income_ntd: 40000
    """)
    config_path = tmp_path / "scenarios.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="current_capital_ntd"):
        load_scenarios(str(config_path))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scenarios'`

- [ ] **Step 5: Implement `src/scenarios.py`**

```python
from dataclasses import dataclass

import yaml


@dataclass
class Scenario:
    name: str
    annual_cost: float
    current_capital: float
    current_age: int
    monthly_work_income: float


def load_scenarios(path: str) -> dict[str, Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    scenarios: dict[str, Scenario] = {}
    for key, cfg in raw.items():
        monthly_costs = cfg.get("monthly_costs_ntd")
        if monthly_costs is None:
            raise ValueError(f"scenario '{key}' missing 'monthly_costs_ntd'")

        placeholders = cfg.get("placeholders")
        if placeholders is None:
            raise ValueError(f"scenario '{key}' missing 'placeholders'")

        for field in ("current_capital_ntd", "current_age", "monthly_work_income_ntd"):
            if field not in placeholders:
                raise ValueError(f"scenario '{key}' missing placeholders.{field}")

        annual_cost = sum(monthly_costs.values()) * 12
        scenarios[key] = Scenario(
            name=key,
            annual_cost=annual_cost,
            current_capital=placeholders["current_capital_ntd"],
            current_age=placeholders["current_age"],
            monthly_work_income=placeholders["monthly_work_income_ntd"],
        )
    return scenarios
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_scenarios.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add config/scenarios.yaml src/__init__.py src/scenarios.py tests/test_scenarios.py
git commit -m "Add scenario config and loader"
```

---

### Task 3: Decumulation engine (dual-trigger guardrail)

**Files:**
- Create: `src/decumulation.py`
- Test: `tests/test_decumulation.py`

**Interfaces:**
- Produces:
  - `run_decumulation(initial_capital: float, initial_annual_budget: float, cash_tent_years: float, tier1_wr: float, tier2_wr: float, budget_cut_pct: float, barista_annual_income: float, returns: list[float], inflation: list[float], cash_yields: list[float]) -> list[dict]` — each record dict has keys `year_index, total_portfolio_jan1, target_budget, wr, state, net_drain, end_portfolio` (`state` is one of `"A"`, `"B"`, `"C"`, `"Depleted"`).
  - `summarize_decumulation(records: list[dict]) -> dict` with keys `ending_balance, years_tier1_cut, years_tier2_worked, survived`.
- Consumed by Task 5's `run_rolling_backtest`.

- [ ] **Step 1: Write failing tests covering each guardrail state, drain order, and inflation compounding**

```python
# tests/test_decumulation.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decumulation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.decumulation'`

- [ ] **Step 3: Implement `src/decumulation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decumulation.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/decumulation.py tests/test_decumulation.py
git commit -m "Add dual-trigger guardrail decumulation engine"
```

---

### Task 4: Accumulation phase

**Files:**
- Create: `src/accumulation.py`
- Test: `tests/test_accumulation.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure function).
- Produces: `run_accumulation(starting_capital: float, monthly_income: float, annual_cost: float, years_worked: int, returns: list[float]) -> float`. Consumed by Task 5's `run_rolling_backtest`. A return value `<= 0` signals the trial failed before decumulation began.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_accumulation.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accumulation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.accumulation'`

- [ ] **Step 3: Implement `src/accumulation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accumulation.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/accumulation.py tests/test_accumulation.py
git commit -m "Add pre-FIRE accumulation phase"
```

---

### Task 5: Rolling-window backtest orchestration

**Files:**
- Create: `src/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `Scenario` from `src/scenarios.py` (Task 2, fields `annual_cost, current_capital, monthly_work_income`); `run_accumulation` from `src/accumulation.py` (Task 4); `run_decumulation`/`summarize_decumulation` from `src/decumulation.py` (Task 3).
- Produces:
  - `load_historical_data(path: str) -> pandas.DataFrame` (columns `Year, Global_Market_Return, Inflation_Rate, Cash_Yield`, sorted by Year).
  - `wrapped_window(df: pandas.DataFrame, start_index: int, length: int) -> pandas.DataFrame`.
  - `run_rolling_backtest(scenario: Scenario, years_worked: int, historical_df: pandas.DataFrame, horizon_years: int, engine_params: dict) -> pandas.DataFrame` (columns `start_year, survived, ending_balance, years_tier1_cut, years_tier2_worked`). `engine_params` keys: `cash_tent_size_years, tier_1_wr_threshold, tier_2_wr_threshold, budget_cut_percentage, barista_annual_income`.
  - `aggregate_results(trial_df: pandas.DataFrame) -> dict` with keys `success_rate, median_ending_balance, p10_ending_balance, avg_years_tier1_cut, avg_years_tier2_worked`.
- Consumed by Task 7's `run_analysis.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest.py
import textwrap

import pandas as pd
import pytest

from src.backtest import (
    aggregate_results,
    load_historical_data,
    run_rolling_backtest,
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
    assert agg["median_ending_balance"] == 150.0
    assert agg["avg_years_tier1_cut"] == 0.75
    assert agg["avg_years_tier2_worked"] == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtest'`

- [ ] **Step 3: Implement `src/backtest.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtest.py tests/test_backtest.py
git commit -m "Add rolling-window backtest orchestration"
```

---

### Task 6: Report generation

**Files:**
- Create: `src/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `aggregate_results()`'s output dict shape from Task 5 (keys `success_rate, median_ending_balance, p10_ending_balance, avg_years_tier1_cut, avg_years_tier2_worked`).
- Produces:
  - `build_summary_table(all_results: dict[tuple[str, int], dict]) -> pandas.DataFrame` (columns `scenario, years_worked` + the aggregate keys, sorted by scenario then years_worked).
  - `write_summary_csv(summary_df: pandas.DataFrame, path: str) -> None`.
  - `plot_success_rate(summary_df: pandas.DataFrame, path: str) -> None`.
  - `plot_ending_balance(summary_df: pandas.DataFrame, path: str) -> None`.
- Consumed by Task 7's `run_analysis.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_report.py
import os

import pandas as pd

from src.report import (
    build_summary_table,
    plot_ending_balance,
    plot_success_rate,
    write_summary_csv,
)


def _sample_all_results():
    return {
        ("single_taipei", 0): {
            "success_rate": 0.8, "median_ending_balance": 1_000_000,
            "p10_ending_balance": 100_000, "avg_years_tier1_cut": 2.0, "avg_years_tier2_worked": 0.5,
        },
        ("single_taipei", 1): {
            "success_rate": 0.9, "median_ending_balance": 1_500_000,
            "p10_ending_balance": 300_000, "avg_years_tier1_cut": 1.0, "avg_years_tier2_worked": 0.1,
        },
        ("couple_taichung", 0): {
            "success_rate": 0.85, "median_ending_balance": 1_200_000,
            "p10_ending_balance": 200_000, "avg_years_tier1_cut": 1.5, "avg_years_tier2_worked": 0.3,
        },
    }


def test_build_summary_table_shape_and_sort_order():
    df = build_summary_table(_sample_all_results())
    assert list(df.columns) == [
        "scenario", "years_worked", "success_rate", "median_ending_balance",
        "p10_ending_balance", "avg_years_tier1_cut", "avg_years_tier2_worked",
    ]
    assert len(df) == 3
    assert df.iloc[0]["scenario"] == "couple_taichung"
    assert df.iloc[1]["scenario"] == "single_taipei"
    assert df.iloc[1]["years_worked"] == 0
    assert df.iloc[2]["years_worked"] == 1


def test_write_summary_csv_round_trips(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "summary.csv"
    write_summary_csv(df, str(out_path))

    read_back = pd.read_csv(out_path)
    assert len(read_back) == len(df)
    assert list(read_back.columns) == list(df.columns)


def test_plot_success_rate_creates_nonempty_file(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "success_rate.png"
    plot_success_rate(df, str(out_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_ending_balance_creates_nonempty_file(tmp_path):
    df = build_summary_table(_sample_all_results())
    out_path = tmp_path / "ending_balance.png"
    plot_ending_balance(df, str(out_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`

- [ ] **Step 3: Implement `src/report.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "Add summary table and chart generation"
```

---

### Task 7: Entrypoint wiring + end-to-end smoke test

**Files:**
- Create: `run_analysis.py`
- Test: `tests/test_run_analysis.py`

**Interfaces:**
- Consumes: `load_scenarios` (Task 2), `load_historical_data`/`run_rolling_backtest`/`aggregate_results` (Task 5), `build_summary_table`/`write_summary_csv`/`plot_success_rate`/`plot_ending_balance` (Task 6).
- Produces: `main() -> pandas.DataFrame` (the summary table), and as a side effect writes `output/summary.csv`, `output/success_rate_chart.png`, `output/ending_balance_chart.png`.

- [ ] **Step 1: Write a failing end-to-end smoke test against the real config and data files**

```python
# tests/test_run_analysis.py
import os

import pandas as pd

from run_analysis import main


def test_main_produces_full_summary_and_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # symlink-free copy isn't needed: run against the real project files by
    # pointing cwd-relative paths at the actual repo config/data via absolute paths.
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent

    summary_df = main(
        scenarios_path=str(repo_root / "config" / "scenarios.yaml"),
        historical_path=str(repo_root / "data" / "historical_returns.csv"),
        output_dir=str(tmp_path / "output"),
    )

    assert isinstance(summary_df, pd.DataFrame)
    # 4 scenarios x 6 years_worked values (0-5)
    assert len(summary_df) == 24
    assert set(summary_df["scenario"].unique()) == {
        "single_taipei", "single_taichung", "couple_taipei", "couple_taichung",
    }
    assert set(summary_df["years_worked"].unique()) == {0, 1, 2, 3, 4, 5}
    assert summary_df["success_rate"].between(0, 1).all()

    assert os.path.exists(tmp_path / "output" / "summary.csv")
    assert os.path.exists(tmp_path / "output" / "success_rate_chart.png")
    assert os.path.exists(tmp_path / "output" / "ending_balance_chart.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_analysis'`

- [ ] **Step 3: Implement `run_analysis.py`**

```python
import os

from src.backtest import aggregate_results, load_historical_data, run_rolling_backtest
from src.report import build_summary_table, plot_ending_balance, plot_success_rate, write_summary_csv
from src.scenarios import load_scenarios

ENGINE_PARAMS = {
    "cash_tent_size_years": 3,
    "tier_1_wr_threshold": 0.048,
    "tier_2_wr_threshold": 0.070,
    "budget_cut_percentage": 0.10,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_analysis.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

Run: `pytest -v`
Expected: All tests across all modules PASS

- [ ] **Step 6: Run the tool for real and eyeball the output**

Run: `python run_analysis.py`
Expected: A 24-row table prints to the console, and `output/summary.csv`, `output/success_rate_chart.png`, `output/ending_balance_chart.png` are created.

- [ ] **Step 7: Commit**

```bash
git add run_analysis.py tests/test_run_analysis.py
git commit -m "Wire up end-to-end analysis entrypoint"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers design section 4.2 (historical data). Task 2 covers section 4.1 (scenarios). Task 3 covers section 5.3 (decumulation, i.e. spec.md verbatim). Task 4 covers section 5.2 (accumulation). Task 5 covers sections 5.1 and 5.4 (rolling windows + aggregation). Task 6 covers section 6 (output/reporting). Task 7 covers wiring + section 7 error handling (via Task 5's `load_historical_data`/`load_scenarios` validation, already tested in Tasks 2 and 5). Section 9's "open parameters" are placeholders in `config/scenarios.yaml` (Task 2) and constants in `run_analysis.py` (Task 7), both editable post-generation.
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and runnable as written.
- **Type consistency:** `Scenario` dataclass fields (`annual_cost`, `current_capital`, `monthly_work_income`) match usage in `run_rolling_backtest` (Task 5). Decumulation record keys (`state`, `end_portfolio`, etc.) match `summarize_decumulation`'s usage. `aggregate_results`' output dict keys match `build_summary_table`'s consumption in Task 6, and both match the assertions in Task 7's smoke test.
