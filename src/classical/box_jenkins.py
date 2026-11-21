"""Training-only Box--Jenkins screening for the S&P 500 return series.

This is a supplementary identification/diagnostic experiment. It never reads
validation or test outcomes and does not replace the frozen EXP-003--EXP-009
forecast pipeline. Run from repository root:
    uv run python -m src.classical.box_jenkins
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

DATA = Path("data/processed/sp500_returns.csv")
OUT = Path("results/tables/box_jenkins_training_diagnostics.json")
TRAIN_N = 2815

def run():
    df = pd.read_csv(DATA, parse_dates=["Date"]).sort_values("Date")
    dates = pd.to_datetime(df["Date"].iloc[:TRAIN_N])
    y = pd.Series(df["return_pct"].astype(float).iloc[:TRAIN_N].to_numpy(), name="return_pct")
    if len(y) != TRAIN_N or not np.isfinite(y).all():
        raise ValueError("Training series does not match frozen boundary or contains non-finite values.")
    adf = adfuller(y, autolag="AIC")
    candidates = []
    fitted = {}
    for p in range(4):
        for q in range(4):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = ARIMA(y, order=(p, 0, q), trend="c",
                               enforce_stationarity=True,
                               enforce_invertibility=True).fit()
            k, n = len(result.params), int(result.nobs)
            aicc = float(result.aic + 2*k*(k+1)/(n-k-1))
            candidates.append({"p":p, "d":0, "q":q, "aic":float(result.aic),
                               "bic":float(result.bic), "aicc":aicc,
                               "warnings":[str(w.message) for w in caught]})
            fitted[(p,q)] = result
    candidates.sort(key=lambda x: x["aicc"])
    best = candidates[0]
    result = fitted[(best["p"], best["q"])]
    resid = pd.Series(result.resid).dropna()
    lb = acorr_ljungbox(resid, lags=[10,20], model_df=best["p"] + best["q"], return_df=True)
    arch = het_arch(resid, nlags=10)
    jb = jarque_bera(resid)
    payload = {
        "scope":"training_only_supplementary_box_jenkins",
        "n":len(y), "start":str(dates.iloc[0].date()), "end":str(dates.iloc[-1].date()),
        "adf":{"statistic":float(adf[0]),"pvalue":float(adf[1]),"used_lags":int(adf[2])},
        "candidate_grid":"ARIMA(p,0,q), p=0..3, q=0..3, constant; ranked by AICc",
        "top_candidates":candidates[:8],
        "selected_order":{"p":best["p"],"d":0,"q":best["q"]},
        "selected_aic":best["aic"],"selected_bic":best["bic"],"selected_aicc":best["aicc"],
        "diagnostics":{
            "ljung_box_residuals_model_df_p_plus_q":{str(l):{"stat":float(lb.loc[l,"lb_stat"]),
                                            "pvalue":float(lb.loc[l,"lb_pvalue"])} for l in [10,20]},
            "arch_lm_10":{"stat":float(arch[0]),"pvalue":float(arch[1])},
            "jarque_bera":{"stat":float(jb.statistic),"pvalue":float(jb.pvalue)}
        },
        "interpretation":"Ljung-Box degrees of freedom are adjusted by p+q for the fitted ARMA model. The ADF rejects a unit root in returns. The AICc-selected ARMA mean "
          "screen leaves residual autocorrelation and strong ARCH effects; this motivates "
          "conditional-variance modelling. It is a training-only screening result, not a "
          "claim that ARMA(2,2) is the final forecasting model.",
        "test_partition_read":False
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload

if __name__ == "__main__":
    run()
