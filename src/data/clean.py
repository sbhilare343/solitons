from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/sp500.csv")
OUTPUT_PATH = Path("data/processed/sp500_returns.csv")


def create_returns(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Create daily log returns from S&P 500 closing prices."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run src/data/download.py first."
        )

    df = pd.read_csv(input_path, parse_dates=["Date"])

    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in the input data.")

    df = df.sort_values("Date").copy()

    # Log transformation
    df["log_price"] = np.log(df["Close"])
    # First order differencing
    df["log_return"] = df["log_price"].diff()

    # Remove the first row because it has no previous observation.
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    # Express returns in percentage points.
    df["return_pct"] = 100.0 * df["log_return"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "Date",
        "Close",
        "log_price",
        "log_return",
        "return_pct",
    ]

    df[columns].to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    print(df[columns].head())
    print("\nSummary:")
    print(df["return_pct"].describe())

    return df[columns]


if __name__ == "__main__":
    create_returns()
