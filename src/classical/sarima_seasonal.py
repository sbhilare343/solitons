"""Supplementary seasonal-ARIMA check for the Solitons S&P 500 project.

Purpose:
- Compare a small, predeclared non-seasonal ARIMA / seasonal SARIMA candidate set.
- Fit on the frozen training partition only.
- Produce rolling one-step validation forecasts with fixed fitted parameters.
- Never read the sealed test partition.

Run from repository root:
    uv run python -m src.classical.sarima_seasonal
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

DATA = Path("data/processed/sp500_returns.csv")
OUT_DIR = Path("results/sarima")
TRAIN_N, VALIDATION_N = 2815, 603
SEASONAL_PERIOD = 5  # five trading sessions; an empirical hypothesis, not a calendar identity
CANDIDATES = [
    {"name":"ARIMA(0,0,0)", "order":[0,0,0], "seasonal_order":[0,0,0,0]},
    {"name":"ARIMA(1,0,0)", "order":[1,0,0], "seasonal_order":[0,0,0,0]},
    {"name":"ARIMA(1,0,1)", "order":[1,0,1], "seasonal_order":[0,0,0,0]},
    {"name":"SARIMA(0,0,0)(1,0,0)[5]", "order":[0,0,0], "seasonal_order":[1,0,0,5]},
    {"name":"SARIMA(1,0,0)(1,0,0)[5]", "order":[1,0,0], "seasonal_order":[1,0,0,5]},
    {"name":"SARIMA(0,0,1)(0,0,1)[5]", "order":[0,0,1], "seasonal_order":[0,0,1,5]},
    {"name":"SARIMA(1,0,1)(1,0,1)[5]", "order":[1,0,1], "seasonal_order":[1,0,1,5]},
]

def main():
    df = pd.read_csv(DATA, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if len(df) != 4022 or df["return_pct"].isna().any():
        raise ValueError("Data do not match the registered 4,022-observation series.")
    y = df["return_pct"].astype(float).to_numpy()
    train, val = y[:TRAIN_N], y[TRAIN_N:TRAIN_N+VALIDATION_N]
    dates = df["Date"].iloc[TRAIN_N:TRAIN_N+VALIDATION_N].dt.strftime("%Y-%m-%d")
    rows, forecasts, fitted_objects = [], {}, {}
    for spec in CANDIDATES:
        order, seasonal = tuple(spec["order"]), tuple(spec["seasonal_order"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(pd.Series(train), order=order, seasonal_order=seasonal,
                            trend="c", enforce_stationarity=True,
                            enforce_invertibility=True)
            fit = model.fit(disp=False, maxiter=200)
        k, n = len(fit.params), len(train)
        aicc = float(fit.aic + 2*k*(k+1)/(n-k-1))
        pred = np.asarray(fit.get_prediction(start=len(train),
                          end=len(train)+len(val)-1, dynamic=False).predicted_mean)
        resid = np.asarray(fit.resid)
        model_df = order[0] + order[2] + seasonal[0] + seasonal[2]  # AR + MA + seasonal AR + seasonal MA
        lb = acorr_ljungbox(resid, lags=[5,10,20], model_df=model_df, return_df=True)
        arch = het_arch(resid, nlags=10)
        row = {
            "model":spec["name"], "order":list(order), "seasonal_order":list(seasonal),
            "aic":float(fit.aic), "aicc":aicc, "bic":float(fit.bic),
            "converged":bool(fit.mle_retvals.get("converged", False)),
            "validation_rmse":float(np.sqrt(mean_squared_error(val,pred))),
            "validation_mae":float(mean_absolute_error(val,pred)),
            "ljung_box_residual_p_lag5":float(lb.loc[5,"lb_pvalue"]),
            "ljung_box_residual_p_lag10":float(lb.loc[10,"lb_pvalue"]),
            "ljung_box_residual_p_lag20":float(lb.loc[20,"lb_pvalue"]),
            "arch_lm_p_lag10":float(arch[1]),
            "parameter_pvalues":{str(k):float(v) for k,v in fit.pvalues.items()},
            "warnings":[str(w.message) for w in caught],
        }
        rows.append(row); forecasts[spec["name"]] = pred.tolist()
        fitted_objects[spec["name"]] = row
    metrics = pd.DataFrame(rows).sort_values("aicc")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR/"sarima_candidate_metrics.csv", index=False)
    fc = pd.DataFrame({"Date":dates.to_numpy(), "actual_return_pct":val})
    for name, pred in forecasts.items():
        fc[name] = pred
    fc.to_csv(OUT_DIR/"sarima_validation_forecasts.csv", index=False)
    seasonal_acf = acf(train, nlags=30, fft=True)
    summary = {
        "scope":"supplementary SARIMA diagnostic; training fit and validation evaluation only",
        "train_n":len(train), "validation_n":len(val),
        "train_start":str(df.Date.iloc[0].date()), "train_end":str(df.Date.iloc[TRAIN_N-1].date()),
        "validation_start":str(df.Date.iloc[TRAIN_N].date()),
        "validation_end":str(df.Date.iloc[TRAIN_N+VALIDATION_N-1].date()),
        "test_partition_read":False, "seasonal_period":5,
        "training_return_acf_lag5":float(seasonal_acf[5]),
        "candidate_results":rows,
        "interpretation":"Treat the seasonal specification as exploratory. AICc improvement alone is not evidence of a useful seasonal mean. Check residual whiteness and ARCH effects, and require a meaningful validation gain. Do not use the sealed test for selection."
    }
    (OUT_DIR/"sarima_diagnostics.json").write_text(json.dumps(summary,indent=2))
    print(metrics[["model","aicc","validation_rmse","validation_mae",
                   "ljung_box_residual_p_lag10","arch_lm_p_lag10"]].to_string(index=False))
    print(f"Training ACF at lag 5: {seasonal_acf[5]:.6f}")
    print("Sealed test read: False")

if __name__ == "__main__":
    main()
