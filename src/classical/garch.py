from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# from sarima import fit_arima, load_returns
from src.classical.sarima import fit_arima, load_returns

OUTPUT_PATH = Path("results/forecasts/garch_fitted_volatility.csv")


def fit_garch(residuals: pd.Series):
    """
    Fit GARCH(1,1) to residuals from the ARIMA mean model.

    We use mean='Zero' because the residuals are already centered
    by the ARIMA model.
    """
    model = arch_model(
        residuals,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    return model.fit(
        disp="off",
        update_freq=0,
    )


# def residual_diagnostics(
#     residuals: pd.Series,
#     standardized_residuals: pd.Series,
# ):
#     """Check whether GARCH has removed volatility dependence."""

#     print("\n=== ARCH LM Test: ARIMA Residuals ===")

#     arch_stat, arch_pvalue, _, _ = het_arch(
#         residuals.dropna(),
#         nlags=10,
#     )

#     print(f"LM statistic : {arch_stat:.6f}")
#     print(f"p-value      : {arch_pvalue:.6g}")

#     print("\n=== Ljung-Box: Standardized Residuals ===")

#     lb_resid = acorr_ljungbox(
#         standardized_residuals.dropna(),
#         lags=[5, 10, 20],
#         return_df=True,
#     )

#     print(lb_resid)

#     print("\n=== Ljung-Box: Squared Standardized Residuals ===")

#     lb_squared = acorr_ljungbox(
#         standardized_residuals.dropna() ** 2,
#         lags=[5, 10, 20],
#         return_df=True,
#     )

#     print(lb_squared)

#     return {
#         "arch_stat": arch_stat,
#         "arch_pvalue": arch_pvalue,
#         "ljung_box_residuals": lb_resid,
#         "ljung_box_squared": lb_squared,
#     }

def residual_diagnostics(
    residuals: pd.Series,
    standardized_residuals: pd.Series,
):
    """Check whether GARCH has removed remaining dependence."""

    # ARCH test BEFORE GARCH
    print("\n=== ARCH LM Test: ARIMA Residuals (Before GARCH) ===")

    arch_stat_before, arch_pvalue_before, _, _ = het_arch(
        residuals.dropna(),
        nlags=10,
    )

    print(f"LM statistic : {arch_stat_before:.6f}")
    print(f"p-value      : {arch_pvalue_before:.6g}")

    # ARCH test AFTER GARCH
    print("\n=== ARCH LM Test: Standardized Residuals (After GARCH) ===")

    arch_stat_after, arch_pvalue_after, _, _ = het_arch(
        standardized_residuals.dropna(),
        nlags=10,
    )

    print(f"LM statistic : {arch_stat_after:.6f}")
    print(f"p-value      : {arch_pvalue_after:.6g}")

    # Ljung-Box on standardized residuals
    print("\n=== Ljung-Box: Standardized Residuals ===")

    lb_resid = acorr_ljungbox(
        standardized_residuals.dropna(),
        lags=[5, 10, 20],
        return_df=True,
    )

    print(lb_resid)

    # Ljung-Box on squared standardized residuals
    print("\n=== Ljung-Box: Squared Standardized Residuals ===")

    lb_squared = acorr_ljungbox(
        standardized_residuals.dropna() ** 2,
        lags=[5, 10, 20],
        return_df=True,
    )

    print(lb_squared)

    return {
        "arch_before": {
            "stat": arch_stat_before,
            "pvalue": arch_pvalue_before,
        },
        "arch_after": {
            "stat": arch_stat_after,
            "pvalue": arch_pvalue_after,
        },
        "ljung_box_residuals": lb_resid,
        "ljung_box_squared": lb_squared,
    }

def main():
    # 1. Load returns
    returns = load_returns()

    print(f"Observations: {len(returns)}")

    # 2. Fit selected ARIMA model
    arima_order = (2, 0, 0)

    print(f"\nFitting ARIMA{arima_order}...")

    arima_result = fit_arima(
        returns,
        arima_order,
    )

    residuals = arima_result.resid.dropna()

    print("\nARIMA residuals:")
    print(residuals.describe())

    # 3. Fit GARCH
    print("\nFitting GARCH(1,1) with Student-t innovations...")

    garch_result = fit_garch(residuals)

    print("\n=== GARCH RESULTS ===")
    print(garch_result.summary())

    # 4. Standardized residuals
    standardized_residuals = pd.Series(
        garch_result.std_resid,
        index=residuals.index,
        name="standardized_residual",
    )

    print("\nStandardized residual summary:")
    print(standardized_residuals.describe())

    # 5. Diagnostics
    residual_diagnostics(
        residuals,
        standardized_residuals,
    )

    # 6. Conditional volatility
    conditional_volatility = pd.Series(
        garch_result.conditional_volatility,
        index=residuals.index,
        name="conditional_volatility",
    )

    output = pd.DataFrame(
        {
            "residual": residuals,
            "conditional_volatility": conditional_volatility,
            "standardized_residual": standardized_residuals,
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(OUTPUT_PATH)

    print(
        f"\nSaved fitted volatility data to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()