# Taiwan Lean-FIRE Rolling-Window Simulator — Design

## 1. Purpose

Extend the existing dual-trigger Guyton-Klinger decumulation engine (`spec.md`) into a full FIRE planning
tool that answers: **"How much does working a few more years buy me in retirement safety, across the
FIRE plans I'm actually considering (single vs. couple, Taipei vs. Taichung)?"**

Two phases are modeled per trial:

- **Accumulation** — 0 to 5 additional years of work at a placeholder 40,000 NTD/month income, saving the
  difference between that income and the scenario's living cost, invested and grown using real historical
  market sequences.
- **Decumulation** — the existing dual-trigger guardrail engine from `spec.md`, unchanged in its core
  logic, run for a fixed retirement horizon.

Both phases are evaluated across **every possible historical starting year** (rolling-window backtesting,
cFIREsim/FIRECalc-style), not just a single historical pass, so the output is a success *rate*, not a
single pass/fail.

## 2. Non-goals

- Not a general portfolio optimizer or tax planner. No Taiwan tax-specific modeling (capital gains,
  NHI premium brackets, etc.) beyond a lump "misc/insurance" cost line.
- Not sourcing real-time or audited financial data. All historical return/inflation/cash-yield figures
  are researched, good-faith approximations for backtesting — explicitly labeled as such in the dataset
  and not to be relied on for actual investment decisions.
- No web UI. Python scripts + CSV/PNG output only.
- No per-scenario custom guardrail thresholds in v1 — all scenarios share the same tier thresholds and
  budget-cut percentage from `spec.md`'s configurable parameters; only the *cost of living* and *capital*
  differ per scenario.

## 3. Project structure

```
fire_plan_simulation/
├── spec.md                        # existing — core decumulation engine spec, unchanged
├── config/
│   └── scenarios.yaml             # single/couple × Taipei/Taichung cost breakdowns + placeholders
├── data/
│   └── historical_returns.csv     # Year, Global_Market_Return, Inflation_Rate, Cash_Yield
├── src/
│   ├── decumulation.py            # Guyton-Klinger dual-trigger engine (implements spec.md section 5)
│   ├── accumulation.py            # pre-FIRE savings/growth phase
│   ├── backtest.py                # rolling-window orchestration, wraparound, success-rate aggregation
│   ├── scenarios.py               # loads scenario config, computes annual lean-FIRE budget per scenario
│   └── report.py                  # CSV tables + matplotlib charts
├── run_analysis.py                # entrypoint: loops years_worked 0-5 × 4 scenarios
├── output/                        # generated CSVs + PNG charts (gitignored contents, dir kept)
└── tests/
    ├── test_decumulation.py
    ├── test_accumulation.py
    └── test_backtest.py
```

## 4. Data & scenarios

### 4.1 `config/scenarios.yaml`

One entry per scenario: `single_taipei`, `single_taichung`, `couple_taipei`, `couple_taichung`. Each has:

```yaml
single_taipei:
  monthly_costs_ntd:          # itemized, editable — researched lean-FIRE estimates
    rent: 15000                # small studio/1BR, outer districts
    food: 9000
    utilities: 2000
    transport: 1500
    health_insurance: 1500     # NHI premium + buffer
    phone_internet: 1000
    misc_buffer: 3000
  placeholders:
    current_capital_ntd: 0     # EDIT ME
    current_age: 35            # EDIT ME
    monthly_work_income_ntd: 40000
```

`annual_cost = sum(monthly_costs_ntd) * 12`. This feeds `initial_annual_budget` in the decumulation engine
and the expense side of the accumulation phase. All four scenarios share the same `monthly_work_income_ntd`
default (40,000 NTD) per the user's stated back-to-work assumption, but it's overridable per scenario.

Couple scenarios are NOT simply single×2 — shared costs (rent, utilities, some misc) scale sub-linearly;
food/health insurance scale ~linearly. The researched breakdown reflects that directly rather than via a
multiplier.

### 4.2 `data/historical_returns.csv`

Columns: `Year, Global_Market_Return, Inflation_Rate, Cash_Yield`, covering a researched approximation of
~1970–2024 (MSCI ACWI/VT-style global equity total return proxy, blended US/Taiwan-relevant CPI, and a
cash-yield proxy). A header comment in the file (or a companion `data/README.md`) documents that this is
an approximation assembled for backtesting purposes, not verified financial data, and names the general
sources/methodology used to approximate it.

## 5. Simulation logic

### 5.1 Rolling window construction

Let `N` = number of years in the historical dataset. For a trial with `years_worked = w` and retirement
horizon `H` (default 40 years), each candidate **retirement start year** `s` in the dataset requires a
sequence of `w + H` consecutive years. Since `w + H` will usually exceed `N`, sequences wrap around
(cyclically reuse the dataset from its start) once they run past the last available year — the standard
cFIREsim approach to extending short historical datasets. Every year in the dataset is tried as a
retirement start year, giving `N` trials per (scenario, years_worked) combination.

### 5.2 Accumulation phase (`accumulation.py`)

For trial starting at retirement year `s` with `w` years worked: walk the `w` years immediately preceding
`s` in the (wrapped) historical sequence. Each year:

```
annual_savings = monthly_work_income_ntd * 12 - scenario.annual_cost
capital = capital * (1 + Global_Market_Return[year]) + annual_savings
```

Starting `capital` = scenario's `placeholders.current_capital_ntd`. If `annual_savings` is negative
(income doesn't cover costs), it still applies (draws down capital) — this surfaces scenarios where the
placeholder income doesn't even cover living costs, which is itself useful signal. If accumulation-phase
capital reaches ≤ 0 before decumulation begins, the trial is recorded as failed immediately (same
depletion rule as section 5.3) and decumulation is not run for that trial.

### 5.3 Decumulation phase (`decumulation.py`)

Implements `spec.md` section 5 exactly (inflation adjustment, Jan-1 WR evaluation, three-state guardrail,
drain order cash-tent-then-equity, market return application), parameterized by:
- `initial_capital` = accumulation phase's ending capital
- `initial_annual_budget` = scenario's `annual_cost`
- `start_year`/`end_year` = the wrapped `H`-year window starting at `s`
- shared tier thresholds / cut percentage / barista income from a global config (not per-scenario in v1)

Runs for the full `H` years regardless of whether the portfolio depletes; if `Equity_Balance +
Cash_Tent_Balance` hits ≤0, the trial is marked failed from that year forward (no negative-balance
"recovery").

### 5.4 Backtest aggregation (`backtest.py`)

For each (scenario, `years_worked` in 0..5): run all `N` rolling trials, then compute:
- `success_rate` = fraction of trials where ending balance > 0 at year `H`
- `median_ending_balance`, `p10_ending_balance` (10th percentile, for downside view)
- `avg_years_tier1_cut`, `avg_years_tier2_worked` (averaged across trials)

Output: one row per (scenario, years_worked) — 4 scenarios × 6 years_worked values = 24 rows.

## 6. Output & reporting (`report.py`)

- `output/summary.csv` — the 24-row aggregation table above.
- `output/success_rate_chart.png` — line chart, x-axis = years worked (0–5), y-axis = success rate (0–100%),
  one line per scenario (4 lines), so the earlier-retirement-vs-safety tradeoff is visible at a glance.
- `output/ending_balance_chart.png` — median + p10 ending balance per scenario × years_worked (bar or line),
  showing not just pass/fail but margin of safety.
- Console: pretty-printed summary table via pandas on `run_analysis.py` completion.

## 7. Error handling & edge cases

- Missing/malformed `scenarios.yaml` fields → raise a clear `ValueError` naming the missing field and
  scenario key (config errors should fail loudly, not silently default).
- `historical_returns.csv` with fewer than `H` years total → still valid (wraparound handles it), but if
  fewer than ~5 years total, raise an error (wraparound on a near-empty dataset isn't meaningful).
- Negative `annual_savings` during accumulation is allowed (see 5.2) — not an error condition.
- Portfolio depletion mid-year in decumulation is clamped at 0, not allowed to go negative.

## 8. Testing

- `test_decumulation.py`: hand-computed fixtures for each guardrail state (A/B/C) transition and the
  drain order (cash-tent-first, then equity), verifying against manually calculated expected values.
- `test_accumulation.py`: verifies compounding + savings math over a few years with known inputs, and the
  negative-savings (income < costs) path.
- `test_backtest.py`: verifies wraparound indexing (e.g., a window that runs off the end of a small
  synthetic dataset correctly cycles back to the start) and success-rate aggregation arithmetic on a
  synthetic dataset with known pass/fail outcomes.

## 9. Open parameters left as placeholders (edit before relying on results)

- `current_capital_ntd`, `current_age` per scenario (all default 0 / 35 — **must edit**)
- `monthly_work_income_ntd` (defaults to the stated 40,000 NTD for all scenarios)
- Retirement horizon `H` (default 40 years) and tier thresholds / cut % / barista income (from `spec.md`'s
  existing configurable parameters, applied globally across scenarios in v1)
