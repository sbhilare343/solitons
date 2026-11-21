"""Rebuild the descriptive figures embedded in paper/report.tex.

All forecast panels read the immutable experiment artifacts named below. The
return-history panel is descriptive and is not used for model selection.
Run from the repository root with `uv run python paper/make_figures.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
DATA = ROOT / "data" / "processed" / "sp500_returns.csv"
GARCH_CSV = ROOT / "results" / "forecasts" / "exp003_validation_forecasts_20261116T113919Z_6cb9.csv"
PATCH_CSV = ROOT / "results" / "forecasts" / "exp005_patchtst_validation_20261116T164839Z_17b6.csv"
TEST_CSV = ROOT / "results" / "forecasts" / "exp009_test_forecasts_20261116T182517Z_13fd.csv"
EXP003_JSON = ROOT / "results" / "tables" / "exp003_garch_mean_spec_20261116T113919Z_6cb9.json"
EXP004_CSV = ROOT / "results" / "tables" / "exp004_qlike_by_proxy_20261116T124356Z_d85f.csv"
EXP005_JSON = ROOT / "results" / "tables" / "exp005_patchtst_20261116T164839Z_17b6.json"
EXP006_JSON = ROOT / "results" / "tables" / "exp006_fair_comparison_20261116T165032Z_b8f8.json"
EXP007_JSON = ROOT / "results" / "tables" / "exp007_combinations_20261116T165831Z_7e52.json"
EXP009_JSON = ROOT / "results" / "tables" / "exp009_test_results_20261116T182517Z_13fd.json"

NAVY = "#18324b"
BLUE = "#2878a8"
CORAL = "#d66b4d"
GREEN = "#34856b"
GOLD = "#c3922d"
GREY = "#687684"
PURPLE = "#7566a8"
COLORS = {"zero": NAVY, "training_mean": BLUE, "constant": GREEN,
          "ar1": CORAL, "ar2": GOLD, "patchtst": PURPLE,
          "frozen_combination": "#263e55"}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9aa5ad", "axes.labelcolor": NAVY,
        "text.color": NAVY, "xtick.color": GREY, "ytick.color": GREY,
        "grid.color": "#d7dde1", "grid.linewidth": 0.55,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig1_data_and_volatility() -> None:
    df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
    train_end = pd.Timestamp("2021-03-11")
    validation_end = pd.Timestamp("2023-08-03")
    boundaries = [df.index.min(), train_end, validation_end, df.index.max()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1]})
    sections = [(df.index.min(), train_end, NAVY),
                (train_end, validation_end, BLUE),
                (validation_end, df.index.max(), CORAL)]
    for start, end, col in sections:
        ax1.axvspan(start, end, color=col, alpha=0.035, lw=0)
        ax2.axvspan(start, end, color=col, alpha=0.035, lw=0)
    ax1.plot(df.index, df.return_pct, lw=0.48, color=NAVY, alpha=0.82)
    ax1.axhline(0, color=GREY, lw=0.7)
    ax1.set_ylabel("Daily log return (pp)")
    ax1.set_title("S&P 500 daily log returns and 20-observation realized volatility")
    ax1.grid(axis="y", alpha=0.6)
    rolling_sd = df.return_pct.rolling(20, min_periods=20).std(ddof=1)
    ax2.plot(df.index, rolling_sd, lw=0.9, color=CORAL)
    ax2.set_ylabel("Rolling SD (pp)")
    ax2.set_xlabel("Trading date")
    ax2.grid(axis="y", alpha=0.6)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (ax1, ax2):
        ax.axvline(train_end, color=BLUE, ls="--", lw=0.9)
        ax.axvline(validation_end, color=CORAL, ls="--", lw=0.9)
    handles = [plt.Line2D([0],[0], color=NAVY, lw=2, label="Training (2,815)"),
               plt.Line2D([0],[0], color=BLUE, lw=2, label="Validation (603)"),
               plt.Line2D([0],[0], color=CORAL, lw=2, label="Test (604; shown descriptively)"),
               plt.Line2D([0],[0], color=CORAL, lw=1, label="20-observation SD")]
    ax1.legend(handles=handles, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig1_returns_volatility.png")


def fig2_train_distribution() -> None:
    df = pd.read_csv(DATA, parse_dates=["Date"])
    train = df.loc[df.Date <= "2021-03-11", "return_pct"].to_numpy(float)
    mu, sd = float(train.mean()), float(train.std(ddof=0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.linspace(np.quantile(train, 0.002), np.quantile(train, 0.998), 500)
    ax1.hist(train, bins=70, density=True, color=BLUE, alpha=0.72,
             edgecolor="white", linewidth=0.25)
    ax1.plot(x, norm.pdf(x, loc=mu, scale=sd), color=CORAL, lw=2,
             label=f"Normal fit: mean={mu:.3f}, SD={sd:.3f}")
    ax1.set_xlim(np.quantile(train, .002), np.quantile(train, .998))
    ax1.set_xlabel("Daily return (percentage points)")
    ax1.set_ylabel("Density")
    ax1.set_title("Training return distribution")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=.45)

    probabilities = (np.arange(1, len(train)+1)-0.5)/len(train)
    empirical = np.sort(train)
    theoretical = norm.ppf(probabilities, loc=mu, scale=sd)
    ax2.scatter(theoretical, empirical, s=6, alpha=.28, color=NAVY, rasterized=True)
    lo, hi = np.quantile(np.r_[theoretical, empirical], [.005, .995])
    ax2.plot([lo, hi], [lo, hi], color=CORAL, lw=1.5, label="Gaussian reference")
    ax2.set_xlabel("Fitted Normal quantiles (pp)")
    ax2.set_ylabel("Empirical return quantiles (pp)")
    ax2.set_title("Normal Q–Q diagnostic (training only)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=.4)
    fig.tight_layout()
    save(fig, "fig2_train_distribution.png")


def _validation_forecasts() -> pd.DataFrame:
    classical = pd.read_csv(GARCH_CSV, parse_dates=["Date"])
    patch = pd.read_csv(PATCH_CSV, parse_dates=["Date"])
    if not classical.Date.equals(patch.Date):
        raise ValueError("EXP-003 and EXP-005 dates differ; refusing to plot misaligned forecasts.")
    combo = json.loads(EXP007_JSON.read_text())
    frame = classical.copy()
    frame["patchtst"] = patch.patchtst_forecast
    frame["frozen_combination"] = combo["best_validation_forecast"]
    return frame


def _metrics(frame: pd.DataFrame, names: list[str]) -> dict[str, float]:
    actual = frame.actual_return_pct.to_numpy(float)
    out = {}
    mapping = {"zero": "baseline_mean_zero", "training_mean": "baseline_mean_training",
               "constant": "mean_forecast_constant", "ar1": "mean_forecast_ar1",
               "ar2": "mean_forecast_ar2", "patchtst": "patchtst",
               "frozen_combination": "frozen_combination"}
    for name in names:
        forecast = frame[mapping[name]].to_numpy(float)
        out[name] = float(np.mean((actual-forecast)**2)**0.5)
    return out


def fig3_rmse_by_partition() -> None:
    val = _validation_forecasts()
    test = pd.read_csv(TEST_CSV, parse_dates=["Date"])
    names = ["zero", "training_mean", "constant", "ar1", "ar2", "patchtst", "frozen_combination"]
    vm = _metrics(val, names)
    actual = test.actual_return_pct.to_numpy(float)
    tm = {name: float(np.mean((actual-test[name].to_numpy(float))**2)**0.5) for name in names}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    labels = ["Zero", "Train mean", "GARCH const.", "GARCH AR(1)", "GARCH AR(2)", "PatchTST", "Frozen combo"]
    for ax, metrics, title in zip(axes, [vm, tm], ["Validation · 603 origins", "Sealed test · 604 origins"]):
        values = [metrics[n] for n in names]
        colors = [COLORS[n] for n in names]
        bars = ax.barh(labels, values, color=colors, height=.68, alpha=.9)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("RMSE (percentage points)")
        ax.grid(axis="x", alpha=.55)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values)*1.17)
        for bar, val_ in zip(bars, values):
            ax.text(val_+max(values)*.012, bar.get_y()+bar.get_height()/2, f"{val_:.3f}",
                    va="center", fontsize=7.7, color=NAVY)
    axes[0].set_ylabel("Forecast")
    fig.suptitle("Point forecast errors on matched one-step windows", y=1.02,
                 color=NAVY, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_rmse_validation_test.png")


def fig4_qlike_proxy_panels() -> None:
    data = pd.read_csv(EXP004_CSV)
    proxy_order = ["p1_squared_return", "p2_parkinson", "p3_ewma_094"]
    titles = ["P1 · squared return", "P2 · Parkinson range", "P3 · EWMA proxy"]
    model_order = ["b2_ewma_forecast", "ar2", "ar1", "constant", "zero", "b1_constant_variance"]
    labels = {"b2_ewma_forecast": "EWMA forecast", "ar2": "GARCH AR(2)",
              "ar1": "GARCH AR(1)", "constant": "GARCH constant mean",
              "zero": "GARCH zero mean", "b1_constant_variance": "Constant variance"}
    colors = {"b2_ewma_forecast": BLUE, "b1_constant_variance": GREY,
              "ar1": CORAL, "ar2": GOLD, "constant": GREEN, "zero": NAVY}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.7))
    for ax, proxy, title in zip(axes, proxy_order, titles):
        sub = data[data.proxy == proxy].set_index("model")
        ordered = sorted(model_order, key=lambda m: float(sub.loc[m, "qlike"]))
        vals = [float(sub.loc[m, "qlike"]) for m in ordered]
        bars = ax.barh([labels[m] for m in ordered], vals,
                       color=[colors[m] for m in ordered], height=.7)
        ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlabel("QLIKE (within-proxy scale)")
        ax.grid(axis="x", alpha=.45); ax.set_axisbelow(True)
        for bar, value in zip(bars, vals):
            ax.text(value + (max(vals)-min(vals))*.035, bar.get_y()+bar.get_height()/2,
                    f"{value:.3f}", va="center", fontsize=7.2)
        ax.set_xlim(min(vals)*.94, max(vals)*1.22)
    fig.suptitle("Variance forecast rankings depend on the realized-volatility proxy",
                 y=1.02, color=NAVY, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_qlike_by_proxy.png")


def fig5_patchtst_learning_curves() -> None:
    info = json.loads(EXP005_JSON.read_text())
    p1 = info["phase1"]["history"]
    p2 = info["phase2"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    epochs = np.arange(1, len(p1["train"])+1)
    axes[0].plot(epochs, p1["train"], color=BLUE, marker="o", ms=3, label="Train MSE")
    axes[0].plot(epochs, p1["val"], color=CORAL, marker="o", ms=3, label="Internal-val MSE")
    best = int(info["phase1"]["best_epoch"])
    axes[0].axvline(best, color=GREEN, ls="--", lw=1.2, label=f"Selected epoch {best}")
    axes[0].set_title("Phase 1 · epoch selection inside training")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Scaled-target MSE")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=.45)
    e2 = np.arange(1, len(p2["train"])+1)
    axes[1].plot(e2, p2["train"], color=NAVY, marker="o", ms=3)
    axes[1].set_title("Phase 2 · fresh fit on all training windows")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training MSE")
    axes[1].grid(alpha=.45)
    fig.suptitle("PatchTST optimization history (validation partition untouched)",
                 y=1.02, color=NAVY, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_patchtst_learning_curves.png")


def fig6_rolling_rmse() -> None:
    val = _validation_forecasts()
    test = pd.read_csv(TEST_CSV, parse_dates=["Date"])
    window = 60
    focus = ["zero", "ar1", "patchtst", "frozen_combination"]
    vals = {"zero": val.baseline_mean_zero, "ar1": val.mean_forecast_ar1,
            "patchtst": val.patchtst, "frozen_combination": val.frozen_combination}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharey=True)
    for ax, frame, forecasts, title in [
        (axes[0], val, vals, "Validation (rolling 60-observation RMSE)"),
        (axes[1], test, {k: test[k] for k in focus}, "Test (descriptive; no reselection)")]:
        y = frame.actual_return_pct.to_numpy(float)
        for name in focus:
            if name not in forecasts: continue
            p = np.asarray(forecasts[name], dtype=float)
            squared = pd.Series((y-p)**2)
            rolling = np.sqrt(squared.rolling(window, min_periods=window).mean())
            labels = {"zero": "Zero", "ar1": "GARCH AR(1)",
                      "patchtst": "PatchTST", "frozen_combination": "Frozen combination"}
            ax.plot(frame.Date, rolling, color=COLORS[name], lw=1.15,
                    label=labels[name])
        ax.set_title(title, loc="left")
        ax.set_ylabel("Trailing RMSE (pp)")
        ax.grid(axis="y", alpha=.45)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xlabel("Forecast date")
    fig.suptitle("Forecast performance changes with the market period",
                 y=1.02, color=NAVY, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_rolling_rmse.png")


def fig7_test_dm_forest() -> None:
    info = json.loads(EXP009_JSON.read_text())
    items = [(k, v) for k, v in info["primary_dm_vs_each"].items()]
    labels = {"zero": "Zero", "training_mean": "Training mean", "constant": "GARCH constant",
              "ar1": "GARCH AR(1)", "ar2": "GARCH AR(2)", "patchtst": "PatchTST"}
    items.sort(key=lambda kv: kv[1]["mean_loss_difference_a_minus_b"])
    y = np.arange(len(items))
    means = np.array([d["mean_loss_difference_a_minus_b"] for _,d in items])
    lo = np.array([d["ci95_loss_difference"][0] for _,d in items])
    hi = np.array([d["ci95_loss_difference"][1] for _,d in items])
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for yi, (name, data), mean, low, high in zip(y, items, means, lo, hi):
        color = COLORS[name]
        ax.errorbar(mean, yi, xerr=[[mean-low],[high-mean]], fmt="o", color=color,
                    ecolor=color, capsize=3.2, markersize=5, lw=1.4)
        ax.text(high + .012, yi, f"p={data['pvalue_two_sided_t']:.3g}",
                va="center", fontsize=8, color=GREY)
    ax.axvline(0, color=NAVY, ls="--", lw=1)
    ax.set_yticks(y, [labels[k] for k,_ in items])
    ax.set_xlabel("Mean squared-loss difference: frozen combination − comparator (pp²)")
    ax.set_title("Sealed-test DM contrasts with 95% HAC intervals", loc="left")
    ax.grid(axis="x", alpha=.45); ax.set_axisbelow(True)
    ax.text(.01, -.22, "Negative values favor the frozen combination. Bartlett HAC, lag 5; pairwise tests unadjusted.",
            transform=ax.transAxes, fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, "fig7_test_dm_intervals.png")


def main() -> None:
    setup()
    functions = [fig1_data_and_volatility, fig2_train_distribution,
                 fig3_rmse_by_partition, fig4_qlike_proxy_panels,
                 fig5_patchtst_learning_curves, fig6_rolling_rmse,
                 fig7_test_dm_forest]
    for fn in functions:
        fn()
    for path in sorted(OUT.glob("fig*.png")):
        print(f"created {path}")
    print(f"Wrote {len(list(OUT.glob('fig*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
