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
