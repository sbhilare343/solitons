from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


INPUT_PATH = Path("data/processed/sp500_returns.csv")
FIGURES_PATH = Path("results/figures")


def load_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load processed S&P 500 return data."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run the data pipeline first."
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def adf_test(series: pd.Series) -> None:
    """Perform Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna(), autolag="AIC")

    print("\n=== Augmented Dickey-Fuller Test ===")
    print(f"ADF statistic : {result[0]:.6f}")
    print(f"p-value       : {result[1]:.6g}")
    print(f"Lags used     : {result[2]}")
    print(f"Observations  : {result[3]}")

    print("\nCritical values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.6f}")

    if result[1] < 0.05:
        print("\nConclusion: Reject H0 -> series appears stationary.")
    else:
        print("\nConclusion: Fail to reject H0 -> stationarity is not established.")


def ljung_box_test(series: pd.Series, name: str, lags: list[int]) -> None:
    """Perform Ljung-Box test."""
    result = acorr_ljungbox(
        series.dropna(),
        lags=lags,
        return_df=True,
    )

    print(f"\n=== Ljung-Box Test: {name} ===")
    print(result.to_string())

    for lag in lags:
        p_value = result.loc[lag, "lb_pvalue"]

        if p_value < 0.05:
            print(f"Lag {lag}: evidence of autocorrelation.")
        else:
            print(f"Lag {lag}: no significant autocorrelation detected.")


def arch_test(series: pd.Series, nlags: int = 10) -> None:
    """Perform Engle's ARCH LM test."""
    statistic, p_value, _, _ = het_arch(
        series.dropna(),
        nlags=nlags,
    )

    print("\n=== ARCH LM Test ===")
    print(f"LM statistic : {statistic:.6f}")
    print(f"p-value      : {p_value:.6g}")

    if p_value < 0.05:
        print("Conclusion: ARCH effects detected.")
    else:
        print("Conclusion: No significant ARCH effects detected.")


def create_plots(df: pd.DataFrame) -> None:
    """Create diagnostic plots."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    returns = df["return_pct"].dropna()

    # Return series
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["return_pct"])
    plt.title("S&P 500 Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "sp500_returns.png", dpi=150)
    plt.close()

    # Squared returns
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["return_pct"] ** 2)
    plt.title("S&P 500 Squared Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Squared Return")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "sp500_squared_returns.png", dpi=150)
    plt.close()

    # ACF of returns
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(returns, lags=40, ax=ax)
    ax.set_title("ACF of S&P 500 Log Returns")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "acf_returns.png", dpi=150)
    plt.close(fig)

    # PACF of returns
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(returns, lags=40, ax=ax, method="ywm")
    ax.set_title("PACF of S&P 500 Log Returns")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "pacf_returns.png", dpi=150)
    plt.close(fig)

    # ACF of squared returns
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(returns ** 2, lags=40, ax=ax)
    ax.set_title("ACF of Squared S&P 500 Log Returns")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "acf_squared_returns.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {FIGURES_PATH}")


def main() -> None:
    df = load_data()

    returns = df["return_pct"]

    print("Dataset shape:", df.shape)
    print("Date range:", df["Date"].min(), "to", df["Date"].max())

    print("\n=== Return Summary ===")
    print(returns.describe())

    adf_test(returns)

    ljung_box_test(
        returns,
        name="Returns",
        lags=[5, 10, 20],
    )

    ljung_box_test(
        returns ** 2,
        name="Squared Returns",
        lags=[5, 10, 20],
    )

    arch_test(returns, nlags=10)

    create_plots(df)


if __name__ == "__main__":
    main()