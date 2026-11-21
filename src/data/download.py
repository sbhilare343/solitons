from pathlib import Path

import pandas as pd
import yfinance as yf


TICKER = "^GSPC"
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_PATH = Path("data/raw/sp500.csv")


def download_sp500(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Download S&P 500 historical daily data and save it as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No data was downloaded.")

    # Handle possible MultiIndex columns returned by yfinance.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Keep only the fields needed for the initial project.
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[col for col in required_columns if col in df.columns]]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    print(df.head())
    print(df.tail())

    return df


if __name__ == "__main__":
    download_sp500()
