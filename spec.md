# AI Agent Project Specification: Lean FIRE "Dual-Trigger" Simulator
## 1. Project Overview
Build a backtesting simulation engine to test sequence-of-returns risk for an early retirement (FIRE) portfolio. The simulator must evaluate how a globally diversified portfolio survives historical market data using a "Dual-Trigger Guyton-Klinger Guardrail" system combined with a "Cash Tent" and "Barista FIRE" (part-time work) fallback.
## 2. Core Tech Stack Recommendation
​  **Language:** Python (using pandas for data handling) or TypeScript/React (if building a web UI).
​  **Data Structure:** Year-by-year iteration loop.
## 3. Configurable Input Parameters
The user must be able to adjust these variables via variables or a UI config object:
​  initial_capital (e.g., 9000000, 10285000, or 12000000)
​  initial_annual_budget (e.g., 360000)
​  cash_tent_size_years (e.g., 3) -> Calculated as initial_annual_budget * 3. This amount is subtracted from initial_capital and held in cash.
​  equity_portfolio -> Calculated as initial_capital - cash tent amount.
​  tier_1_wr_threshold (e.g., 0.042 or 0.048) -> Triggers a budget cut.
​  tier_2_wr_threshold (e.g., 0.060 or 0.070) -> Triggers a budget cut + part-time work.
​  budget_cut_percentage (e.g., 0.10)
​  barista_annual_income (e.g., 240000)
​  start_year (e.g., 1998 or 2000)
​  end_year (e.g., 2024)
## 4. Required Datasets
The agent must structure the program to accept a CSV or JSON file containing historical yearly data with the following columns:
​  Year (YYYY)
​  Global_Market_Return (e.g., -0.142 for -14.2%. Represents MSCI ACWI / VT total return).
​  Inflation_Rate (e.g., 0.03 for 3%. Represents YoY CPI increase).
​  Cash_Yield (e.g., 0.015 for 1.5%. Represents return on the Cash Tent).
## 5. The Simulation Loop Logic (Yearly Evaluation)
For each year from start_year to end_year, execute the following logic in exact order:
**Step 1: Inflation Adjustment**
​  If Year > start_year, multiply the base initial_annual_budget by the cumulative inflation up to that year to get the target_budget. (Do not compound the cut budget, always track the true inflated target).
**Step 2: January 1st Evaluation (Withdrawal Rate)**
​  Calculate Total_Portfolio = Equity_Balance + Cash_Tent_Balance.
​  Calculate Jan_1_WR = target_budget / Total_Portfolio.
**Step 3: Determine Action State (Guardrails)**
​  **State A (Safe):** If Jan_1_WR < tier_1_wr_threshold: 
    *   Actual_Spend = target_budget.
    *   Barista_Income = 0.
​  **State B (Tier 1 - Mild Threat):** If Jan_1_WR >= tier_1_wr_threshold AND < tier_2_wr_threshold:
    *   Actual_Spend = target_budget * (1 - budget_cut_percentage).
    *   Barista_Income = 0.
​  **State C (Tier 2 - Severe Threat):** If Jan_1_WR >= tier_2_wr_threshold:
    *   Actual_Spend = target_budget * (1 - budget_cut_percentage).
    *   Barista_Income = barista_annual_income.
**Step 4: Execute Withdrawals (Net Drain)**
​  Calculate Required_Withdrawal = Actual_Spend - Barista_Income.
​  If Required_Withdrawal < 0, set to 0 (income covers all expenses).
​  **Drain Order:**
    1. Subtract Required_Withdrawal from Cash_Tent_Balance first.
    2. If Cash_Tent_Balance hits 0, subtract the remainder of Required_Withdrawal from Equity_Balance.
**Step 5: Apply Market Returns (End of Year)**
​  Equity_Balance = Equity_Balance * (1 + Global_Market_Return for that year).
​  Cash_Tent_Balance = Cash_Tent_Balance * (1 + Cash_Yield for that year).
**Step 6: Log Data and Loop**
​  Save the year's metrics (Start Portfolio, Target Budget, Jan 1 WR, Action State, Net Drain, End Portfolio) to an array/dataframe.
​  Move to the next year.
## 6. Expected Outputs & Display
The program must output a structured data table (or terminal readout) with the following columns:
1.  Year
2.  Jan 1st Total Portfolio
3.  Target Budget (Inflated)
4.  Jan 1st Withdrawal Rate (Percentage)
5.  Action Taken (Safe, 10% Cut, or 10% Cut + Work)
6.  End of Year Portfolio Balance
**Summary Metrics required at the end of the simulation:**
​  Ending Portfolio Balance.
​  Total Years Worked (Count of Tier 2 triggers).
​  Total Years with Lifestyle Cut (Count of Tier 1 & Tier 2 triggers).

