"""
GARCH benchmark comparison against the UKF.

Fits GARCH(1,1), EGARCH(1,1) and GJR-GARCH(1,1) on the training split,
generates recursive 1-step-ahead OOS forecasts, then compares all models
plus EWMA on MAE, QLIKE and Corr(sigma, |r|).

Saves plots/benchmark.png.
"""

import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from arch import arch_model

from btc_modal import run_base_filter

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ── Style ─────────────────────────────────────────────────────────────────────

COLOURS = {
    "UKF":          "#2271B2",
    "GARCH(1,1)":   "#C0392B",
    "EGARCH(1,1)":  "#2E8B57",
    "GJR-GARCH":    "#D4853A",
    "EWMA":         "#888888",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
})

MODELS = ["UKF", "GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH", "EWMA"]


# ── Metric helpers ────────────────────────────────────────────────────────────

def mae(vol_oos, r_oos):
    return float(np.mean(np.abs(vol_oos - np.abs(r_oos))))


def qlike(vol_oos, r_oos):
    v = vol_oos ** 2
    return float(np.mean(np.log(v) + r_oos ** 2 / v))


def corr(vol_oos, r_oos):
    return float(np.corrcoef(vol_oos, np.abs(r_oos))[0, 1])


# ── Load data ─────────────────────────────────────────────────────────────────

print("Downloading BTC-USD data...")
btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
prices = btc["Close"].squeeze().dropna()
dates = prices.index
r = np.diff(np.log(prices.values.astype(float)))
dates_r = dates[1:]
n = len(r)
split = int(0.7 * n)
r_oos = r[split:]
n_oos = len(r_oos)
oos_dates = dates_r[split:]

r_pct = r * 100   # arch model uses percentage-scale returns
r_train_pct = r_pct[:split]
print(f"  {n} observations  |  train: {split}  |  OOS: {n_oos}")


# ── UKF ───────────────────────────────────────────────────────────────────────

print("\nRunning UKF...")
states_ukf, vol_ukf, vol_ewma, _ = run_base_filter(r)
vol_ukf_oos  = vol_ukf[split:]
vol_ewma_oos = vol_ewma[split:]


# ── GARCH helper: recursive 1-step-ahead OOS forecast ─────────────────────────

def _garch_oos(res, kind):
    """
    Produce OOS conditional-vol array by recursing from the last in-sample
    state using the actual OOS returns.  Returns vol in decimal (not pct).
    """
    cv = res.conditional_volatility
    last_cv = cv.values[-1] if hasattr(cv, "values") else cv[-1]
    sig2_prev = float(last_cv ** 2)
    params = res.params
    om = float(params["omega"])
    sig2 = np.zeros(n_oos)

    if kind == "garch":
        al = float(params["alpha[1]"])
        be = float(params["beta[1]"])
        r_prev = r_train_pct[-1]
        for t in range(n_oos):
            sig2[t] = om + al * r_prev ** 2 + be * sig2_prev
            r_prev = r_pct[split + t]
            sig2_prev = sig2[t]

    elif kind == "egarch":
        al = float(params["alpha[1]"])
        gm = float(params["gamma[1]"])
        be = float(params["beta[1]"])
        sqrt_2_pi = np.sqrt(2 / np.pi)
        log_var_prev = np.log(sig2_prev)
        sig_prev = np.sqrt(sig2_prev)
        r_prev = r_train_pct[-1]
        for t in range(n_oos):
            z = r_prev / (sig_prev + 1e-8)
            lv = om + be * log_var_prev + al * (abs(z) - sqrt_2_pi) + gm * z
            sig2[t] = np.exp(lv)
            r_prev = r_pct[split + t]
            log_var_prev = lv
            sig_prev = np.sqrt(sig2[t])

    elif kind == "gjr":
        al = float(params["alpha[1]"])
        gm = float(params["gamma[1]"])
        be = float(params["beta[1]"])
        r_prev = r_train_pct[-1]
        for t in range(n_oos):
            ind = 1.0 if r_prev < 0 else 0.0
            sig2[t] = om + al * r_prev ** 2 + gm * r_prev ** 2 * ind + be * sig2_prev
            r_prev = r_pct[split + t]
            sig2_prev = sig2[t]

    return np.sqrt(sig2) / 100   # back to decimal


# ── Fit and forecast GARCH models ─────────────────────────────────────────────

print("\nFitting GARCH models on training data...")

print("  GARCH(1,1) ...", end=" ")
res_garch = arch_model(
    r_train_pct, vol="Garch", p=1, q=1, dist="normal", mean="Constant"
).fit(disp="off", show_warning=False)
vol_garch_oos = _garch_oos(res_garch, "garch")
print(f"done  (omega={res_garch.params['omega']:.4f}, "
      f"alpha={res_garch.params['alpha[1]']:.4f}, "
      f"beta={res_garch.params['beta[1]']:.4f})")

print("  EGARCH(1,1) ...", end=" ")
res_egarch = arch_model(
    r_train_pct, vol="EGARCH", p=1, o=1, q=1, dist="normal", mean="Constant"
).fit(disp="off", show_warning=False)
vol_egarch_oos = _garch_oos(res_egarch, "egarch")
print(f"done  (omega={res_egarch.params['omega']:.4f}, "
      f"alpha={res_egarch.params['alpha[1]']:.4f}, "
      f"gamma={res_egarch.params['gamma[1]']:.4f}, "
      f"beta={res_egarch.params['beta[1]']:.4f})")

print("  GJR-GARCH(1,1) ...", end=" ")
res_gjr = arch_model(
    r_train_pct, vol="Garch", p=1, o=1, q=1, dist="normal", mean="Constant"
).fit(disp="off", show_warning=False)
vol_gjr_oos = _garch_oos(res_gjr, "gjr")
print(f"done  (omega={res_gjr.params['omega']:.4f}, "
      f"alpha={res_gjr.params['alpha[1]']:.4f}, "
      f"gamma={res_gjr.params['gamma[1]']:.4f}, "
      f"beta={res_gjr.params['beta[1]']:.4f})")


# ── Metrics ───────────────────────────────────────────────────────────────────

vols_oos = {
    "UKF":         vol_ukf_oos,
    "GARCH(1,1)":  vol_garch_oos,
    "EGARCH(1,1)": vol_egarch_oos,
    "GJR-GARCH":   vol_gjr_oos,
    "EWMA":        vol_ewma_oos,
}

metrics = {
    name: {
        "MAE":  mae(v, r_oos),
        "QLIKE": qlike(v, r_oos),
        "Corr": corr(v, r_oos),
    }
    for name, v in vols_oos.items()
}

W = 68
print(f"\n{'=' * W}")
print(f"{'Model':<18} {'MAE':>10} {'QLIKE':>12} {'Corr(σ,|r|)':>14}")
print(f"{'-' * W}")
for name in MODELS:
    m = metrics[name]
    best_mae   = min(metrics[n]["MAE"]   for n in MODELS)
    best_qlike = min(metrics[n]["QLIKE"] for n in MODELS)
    best_corr  = max(metrics[n]["Corr"]  for n in MODELS)
    flag = (
        " ★" if (m["MAE"] == best_mae and
                  m["QLIKE"] == best_qlike and
                  m["Corr"] == best_corr)
        else ("  MAE★"   if m["MAE"] == best_mae   else
              ("  QL★"   if m["QLIKE"] == best_qlike else
               ("  Corr★" if m["Corr"] == best_corr   else "")))
    )
    print(f"  {name:<16} {m['MAE']:>10.4f} {m['QLIKE']:>12.4f} "
          f"{m['Corr']:>14.4f}{flag}")
print(f"{'=' * W}")


# ── Plot ─────────────────────────────────────────────────────────────────────

print("\nBuilding benchmark plot...")

DATE_FMT = mdates.DateFormatter("%b '%y")
DATE_LOC = mdates.MonthLocator(interval=4)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                      height_ratios=[1.8, 1.0])
fig.suptitle(
    "UKF vs GARCH Family — BTC-USD OOS Volatility Benchmark (5-Year History)",
    fontsize=12, fontweight="bold",
)

# ── Top row: OOS vol time series (span all 3 columns) ────────────────────────

ax_vol = fig.add_subplot(gs[0, :])
ann_vol = np.sqrt(252) * 100

for name in MODELS:
    v = vols_oos[name]
    lw = 1.5 if name == "UKF" else 0.9
    ls = "-" if name in ("UKF", "EWMA") else "--"
    ax_vol.plot(oos_dates, v * ann_vol,
                color=COLOURS[name], lw=lw, ls=ls,
                alpha=1.0 if name == "UKF" else 0.8,
                label=name)

ax_vol.set_ylabel("Annualised volatility (%)")
ax_vol.set_title("Out-of-Sample Conditional Volatility Estimates — All Models")
ax_vol.legend(fontsize=8, ncol=5, loc="upper right")
ax_vol.xaxis.set_major_formatter(DATE_FMT)
ax_vol.xaxis.set_major_locator(DATE_LOC)
fig.autofmt_xdate(rotation=30, ha="right")

# ── Bottom row: one bar chart per metric ─────────────────────────────────────

colours = [COLOURS[n] for n in MODELS]
x = np.arange(len(MODELS))
tick_labels = ["UKF", "GARCH\n(1,1)", "EGARCH\n(1,1)", "GJR-\nGARCH", "EWMA"]

# MAE (lower = better)
ax_mae = fig.add_subplot(gs[1, 0])
mae_vals = [metrics[n]["MAE"] for n in MODELS]
bars = ax_mae.bar(x, mae_vals, color=colours, alpha=0.75, width=0.6)
best_idx = int(np.argmin(mae_vals))
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(1.5)
ax_mae.set_xticks(x)
ax_mae.set_xticklabels(tick_labels, fontsize=8)
ax_mae.set_ylabel("MAE")
ax_mae.set_title("MAE  (lower = better)\n★ = best")
ax_mae.set_ylim(0, max(mae_vals) * 1.15)
ax_mae.annotate("★", xy=(best_idx, mae_vals[best_idx]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=11)

# QLIKE (lower = better, more negative)
ax_ql = fig.add_subplot(gs[1, 1])
ql_vals = [metrics[n]["QLIKE"] for n in MODELS]
bars = ax_ql.bar(x, ql_vals, color=colours, alpha=0.75, width=0.6)
best_idx = int(np.argmin(ql_vals))
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(1.5)
ax_ql.set_xticks(x)
ax_ql.set_xticklabels(tick_labels, fontsize=8)
ax_ql.set_ylabel("QLIKE")
ax_ql.set_title("QLIKE  (more negative = better)\n★ = best")
bottom = min(ql_vals) * 1.02
top = max(ql_vals) * 0.98
ax_ql.set_ylim(bottom, top)
ax_ql.annotate("★", xy=(best_idx, ql_vals[best_idx]),
               xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=11)

# Corr (higher = better)
ax_co = fig.add_subplot(gs[1, 2])
corr_vals = [metrics[n]["Corr"] for n in MODELS]
bars = ax_co.bar(x, corr_vals, color=colours, alpha=0.75, width=0.6)
best_idx = int(np.argmax(corr_vals))
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(1.5)
ax_co.set_xticks(x)
ax_co.set_xticklabels(tick_labels, fontsize=8)
ax_co.set_ylabel("Pearson ρ")
ax_co.set_title("Corr(σ̂, |r|)  (higher = better)\n★ = best")
ax_co.set_ylim(0, max(corr_vals) * 1.15)
ax_co.annotate("★", xy=(best_idx, corr_vals[best_idx]),
               xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=11)

plt.savefig("plots/benchmark.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/benchmark.png")

# ── Print ranking summary ─────────────────────────────────────────────────────

print("\nRanking (1 = best):")
for metric_key, reverse in [("MAE", False), ("QLIKE", False), ("Corr", True)]:
    ranked = sorted(MODELS, key=lambda n: metrics[n][metric_key],
                    reverse=reverse)
    print(f"  {metric_key}: " + "  >  ".join(ranked))
