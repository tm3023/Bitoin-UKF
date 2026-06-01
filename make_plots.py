"""
Generate publication-quality plots for the BTC UKF analysis.

Saves to plots/:
  vol_comparison.png  -- price history + volatility + innovations  (main)
  return_fit.png      -- return series with UKF confidence bands + OOS scatter
  diagnostics.png     -- QQ plot, ACF of innovations, ACF of squared innovations
"""

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from btc_modal import run_base_filter

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ── Colours & style ───────────────────────────────────────────────────────────

BLUE   = "#2271B2"
GREEN  = "#2E8B57"
RED    = "#C0392B"
ORANGE = "#D4853A"
GREY   = "#666666"
LGREY  = "#CCCCCC"

plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.6,
    "figure.dpi":          150,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
})

DATE_FMT  = mdates.DateFormatter("%b '%y")
DATE_LOC  = mdates.MonthLocator(interval=6)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Downloading BTC-USD data...")
btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
prices = btc["Close"].squeeze().dropna()
dates  = prices.index                          # DatetimeIndex
price_vals = prices.values.astype(float)
r      = np.diff(np.log(price_vals))
dates_r = dates[1:]                            # dates aligned with returns
n      = len(r)
split  = int(0.7 * n)
split_date = dates_r[split]


# ── Run filter ────────────────────────────────────────────────────────────────

print("Running UKF...")
states, vol_ukf, vol_ewma, mu_h = run_base_filter(r)

vol_ukf_ann  = vol_ukf  * np.sqrt(252) * 100   # annualised %
vol_ewma_ann = vol_ewma * np.sqrt(252) * 100
slow_mode    = states[:, 0]
fast_mode    = states[:, 2]
fitted_mean  = slow_mode + fast_mode
innovations  = r - fitted_mean
std_innov    = innovations / (vol_ukf + 1e-8)

conf_95 = 1.96 / np.sqrt(n)


# ── Key BTC market events ─────────────────────────────────────────────────────

def _ts(s, tz):
    t = pd.Timestamp(s)
    return t.tz_localize(tz) if (tz is not None and t.tzinfo is None) else t


tz = dates_r.tz
EVENTS = [
    ("2020-03-13", "COVID\ncrash",     "above"),
    ("2021-11-10", "ATH\n$69k",        "above"),
    ("2022-11-11", "FTX\ncollapse",    "below"),
    ("2024-03-14", "New\nATH\n$73k",   "above"),
]


def annotate_events(ax, y_series, events, fontsize=7):
    for date_str, label, side in events:
        d = _ts(date_str, tz)
        if d < dates_r[0] or d > dates_r[-1]:
            continue
        idx = dates_r.searchsorted(d)
        yval = y_series[idx]
        dy = 18 if side == "above" else -24
        ax.annotate(
            label,
            xy=(dates_r[idx], yval),
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=fontsize,
            ha="center",
            color=GREY,
            arrowprops=dict(arrowstyle="-", color=LGREY, lw=0.7),
        )


# ── ACF helper ────────────────────────────────────────────────────────────────

def acf_manual(x, nlags=40):
    x = x - x.mean()
    var = np.var(x)
    return np.array([
        np.mean(x[:n - k] * x[k:]) / var if k < n else 0.0
        for k in range(nlags + 1)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Price + volatility + innovations
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 1: vol_comparison.png ...")

fig = plt.figure(figsize=(14, 11))
gs  = fig.add_gridspec(3, 1, hspace=0.38, height_ratios=[2.2, 2.2, 1.2])
fig.suptitle(
    "Unscented Kalman Filter — BTC-USD Daily Returns (5-Year History)",
    fontsize=12, fontweight="bold", y=0.99,
)

# Panel 1: price (log) + UKF vol shading on twin axis
ax1   = fig.add_subplot(gs[0])
ax1r  = ax1.twinx()
ax1r.spines["right"].set_visible(True)
ax1r.spines["top"].set_visible(False)

ax1r.plot(dates_r, price_vals[1:], color=BLUE, lw=1.1, label="BTC price (USD)", zorder=3)
ax1r.set_yscale("log")
ax1r.set_ylabel("Price (USD, log scale)", color=BLUE, fontsize=9)
ax1r.tick_params(axis="y", labelcolor=BLUE)

ax1.fill_between(dates_r, vol_ukf_ann, alpha=0.18, color=GREEN)
ax1.plot(dates_r, vol_ukf_ann, color=GREEN, lw=1.0, label="UKF vol (ann. %)", zorder=2)
ax1.set_ylabel("Annualised volatility (%)", color=GREEN, fontsize=9)
ax1.tick_params(axis="y", labelcolor=GREEN)
ax1.axvline(split_date, color=GREY, lw=1.0, ls="--", alpha=0.6)
ax1.set_title("BTC Price (log) with UKF Volatility Estimate")

annotate_events(ax1, vol_ukf_ann, EVENTS)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# Panel 2: UKF vs EWMA annualised vol
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax2.fill_between(
    dates_r, vol_ukf_ann,
    where=np.arange(n) >= split,
    alpha=0.07, color=GREEN, interpolate=True,
)
ax2.plot(dates_r, vol_ukf_ann,  color=GREEN, lw=1.3, label="UKF (this model)")
ax2.plot(dates_r, vol_ewma_ann, color=RED, lw=0.9, ls="--",
         alpha=0.85, label="EWMA (λ=0.94, benchmark)")
ax2.axvline(split_date, color=GREY, lw=1.0, ls="--", alpha=0.6, label="Train / test split")
ax2.set_ylabel("Annualised volatility (%)", fontsize=9)
ax2.set_title("Volatility Estimate: UKF vs EWMA — out-of-sample region shaded")
ax2.legend(fontsize=8, loc="upper right")

# Panel 3: standardised innovations
ax3 = fig.add_subplot(gs[2], sharex=ax1)

pos_mask = std_innov >= 0
ax3.bar(dates_r[pos_mask],  std_innov[pos_mask],  width=1.5, color=BLUE,   alpha=0.5)
ax3.bar(dates_r[~pos_mask], std_innov[~pos_mask], width=1.5, color=ORANGE, alpha=0.5)
ax3.axhline( 2, color=RED,   lw=0.9, ls="--", alpha=0.7, label="±2σ")
ax3.axhline(-2, color=RED,   lw=0.9, ls="--", alpha=0.7)
ax3.axhline( 0, color="black", lw=0.7)
ax3.set_ylabel("z-score", fontsize=9)
ax3.set_ylim(-6, 6)
ax3.set_title("Standardised Innovations (i.i.d. N(0,1) under model)")
ax3.legend(fontsize=8, loc="upper right")

for ax in (ax1, ax2, ax3):
    ax.xaxis.set_major_formatter(DATE_FMT)
    ax.xaxis.set_major_locator(DATE_LOC)

fig.autofmt_xdate(rotation=30, ha="right")
plt.savefig("plots/vol_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/vol_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Return fit + OOS evaluation scatter
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 2: return_fit.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Return Fit and Out-of-Sample Forecast Evaluation", fontsize=12,
             fontweight="bold")

# Left: return series with ±2σ UKF band
ax = axes[0]
upper = (fitted_mean + 2 * vol_ukf) * 100
lower = (fitted_mean - 2 * vol_ukf) * 100

ax.fill_between(dates_r, lower, upper, alpha=0.18, color=GREEN, label="±2σ UKF band")
ax.plot(dates_r, r * 100, color=BLUE, lw=0.4, alpha=0.6, label="Return")
ax.plot(dates_r, fitted_mean * 100, color=ORANGE, lw=0.8, label="UKF fitted mean")
ax.axvline(split_date, color=GREY, lw=1.0, ls="--", alpha=0.7, label="Train / test")
ax.set_ylabel("Daily log-return (%)")
ax.set_title("Returns vs UKF ±2σ Confidence Band")
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(DATE_FMT)
ax.xaxis.set_major_locator(DATE_LOC)

# Right: OOS forecast vol vs realised |return|
ax = axes[1]
vol_oos   = vol_ukf[split:]  * 100
ewma_oos  = vol_ewma[split:] * 100
abs_r_oos = np.abs(r[split:]) * 100

corr_ukf  = np.corrcoef(vol_oos,  abs_r_oos)[0, 1]
corr_ewma = np.corrcoef(ewma_oos, abs_r_oos)[0, 1]

ax.scatter(ewma_oos, abs_r_oos, s=5, alpha=0.25, color=RED,
           label=f"EWMA  (ρ = {corr_ewma:.2f})", zorder=1)
ax.scatter(vol_oos,  abs_r_oos, s=5, alpha=0.30, color=GREEN,
           label=f"UKF   (ρ = {corr_ukf:.2f})", zorder=2)

lim = max(vol_oos.max(), ewma_oos.max(), abs_r_oos.max()) * 1.05
ax.plot([0, lim], [0, lim], color="black", lw=0.8, ls="--", alpha=0.35, label="y = x")
ax.set_xlabel("Forecast volatility (%)")
ax.set_ylabel("Realised |return| (%)")
ax.set_title("OOS: Forecast σ vs Realised |r|  (last 30% of sample)")
ax.legend(fontsize=9)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots/return_fit.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/return_fit.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Innovation diagnostics
# ─────────────────────────────────────────────────────────────────────────────

print("Building Figure 3: diagnostics.png ...")

lags     = np.arange(41)
acf_z    = acf_manual(std_innov)
acf_z2   = acf_manual(std_innov**2)
kurt     = float(pd.Series(std_innov).kurtosis()) + 3   # excess → total

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Innovation Diagnostics", fontsize=12, fontweight="bold")

# Left: QQ plot
ax = axes[0]
(osm, osr), (slope, intercept, _) = stats.probplot(std_innov, dist="norm")
ax.scatter(osm, osr, s=4, alpha=0.4, color=BLUE, label=f"Innovations (kurt={kurt:.1f})")
ref_x = np.array([osm[0], osm[-1]])
ax.plot(ref_x, slope * ref_x + intercept, color=RED, lw=1.5, label="Normal reference")
ax.set_xlabel("Theoretical quantiles (Normal)")
ax.set_ylabel("Sample quantiles")
ax.set_title("QQ Plot: Innovations vs Normal(0,1)")
ax.legend(fontsize=8)

# Middle: ACF of innovations
ax = axes[1]
ax.bar(lags[1:], acf_z[1:],  color=BLUE,  alpha=0.65, width=0.8, label="ACF")
ax.axhline( conf_95, color=RED, lw=1.0, ls="--", label="95% CI")
ax.axhline(-conf_95, color=RED, lw=1.0, ls="--")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("ACF of Standardised Innovations\n(should be near zero if filter is calibrated)")
ax.legend(fontsize=8)
ax.set_ylim(-0.18, 0.25)

# Right: ACF of squared innovations (ARCH test)
ax = axes[2]
ax.bar(lags[1:], acf_z2[1:], color=ORANGE, alpha=0.65, width=0.8, label="ACF of z²")
ax.axhline( conf_95, color=RED, lw=1.0, ls="--", label="95% CI")
ax.axhline(-conf_95, color=RED, lw=1.0, ls="--")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("ACF of Squared Innovations\n(residual ARCH effects if significant lags remain)")
ax.legend(fontsize=8)
ax.set_ylim(-0.18, 0.25)

plt.tight_layout()
plt.savefig("plots/diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/diagnostics.png")

print("\nDone — all 3 figures saved to plots/")
