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
