from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


DATA_PATH = Path("data/processed/sp500_returns.csv")


CANDIDATE_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 0, 1),
    (2, 0, 0),
    (2, 0, 1),
]


# def load_returns(path: Path = DATA_PATH) -> pd.Series:
#     """Load daily S&P 500 log returns."""
#     if not path.exists():
#         raise FileNotFoundError(f"Missing data file: {path}")

#     df = pd.read_csv(path, parse_dates=["Date"])
#     return df["return_pct"].dropna()

def load_returns(path: Path = DATA_PATH) -> pd.Series:
    """Load daily S&P 500 log returns while preserving dates."""

    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    required_columns = {"Date", "return_pct"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns: {required_columns}"
        )

    if df["Date"].isna().any():
        raise ValueError("Missing dates detected.")

    df["return_pct"] = pd.to_numeric(
        df["return_pct"],
        errors="raise",
    )

    if df["return_pct"].isna().any():
        raise ValueError("Missing returns detected.")

    if not np.isfinite(df["return_pct"].to_numpy()).all():
        raise ValueError("Non-finite returns detected.")

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("Date")

    returns = df.set_index("Date")["return_pct"].astype(float)

    returns.index.name = "Date"
    returns.name = "return_pct"

    return returns


def fit_arima(series: pd.Series, order: tuple[int, int, int]):
    """Fit an ARIMA model with a constant."""
    model = ARIMA(
        series,
        order=order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    return model.fit()


def compare_models(series: pd.Series) -> pd.DataFrame:
    """Fit candidate ARIMA models and compare information criteria."""

    results = []

    for order in CANDIDATE_ORDERS:
        try:
            fitted = fit_arima(series, order)

            results.append(
                {
                    "order": order,
                    "AIC": fitted.aic,
                    "BIC": fitted.bic,
                    "HQIC": fitted.hqic,
                }
            )

            print(
                f"ARIMA{order}: "
                f"AIC={fitted.aic:.3f}, "
                f"BIC={fitted.bic:.3f}, "
                f"HQIC={fitted.hqic:.3f}"
            )

        except Exception as exc:
            print(f"ARIMA{order} failed: {exc}")

    return (
        pd.DataFrame(results)
        .sort_values("AIC")
        .reset_index(drop=True)
    )


def main():
    returns = load_returns()

    print(f"Number of observations: {len(returns)}")

    results = compare_models(returns)

    print("\n=== Model Ranking ===")
    print(results.to_string(index=False))

    best_order = results.iloc[0]["order"]

    print(f"\nBest model according to AIC: ARIMA{best_order}")

    fitted = fit_arima(returns, best_order)

    print("\n=== Best ARIMA Summary ===")
    print(fitted.summary())

    residuals = fitted.resid

    print("\nResidual summary:")
    print(residuals.describe())


if __name__ == "__main__":
    main()