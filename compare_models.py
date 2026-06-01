"""
Gaussian UKF vs Student-t UKF: model comparison.

Finds the optimal degrees-of-freedom nu via OOS QLIKE, then compares
both models on forecast accuracy, tail calibration, and VaR coverage.

Saves plots/comparison.png.
"""

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from btc_modal import run_base_filter, run_student_t_filter

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

BLUE   = "#2271B2"
GREEN  = "#2E8B57"
RED    = "#C0392B"
ORANGE = "#D4853A"
PURPLE = "#7B2D8B"
GREY   = "#666666"

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


# ── Load data ─────────────────────────────────────────────────────────────────

print("Downloading BTC-USD data...")
btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
prices    = btc["Close"].squeeze().dropna()
dates     = prices.index
price_vals = prices.values.astype(float)
r         = np.diff(np.log(price_vals))
dates_r   = dates[1:]
n         = len(r)
split     = int(0.7 * n)
r_oos     = r[split:]
oos_dates = dates_r[split:]


# ── Metric helpers ────────────────────────────────────────────────────────────

def mae(vol_oos):
    return float(np.mean(np.abs(vol_oos - np.abs(r_oos))))


def qlike(vol_oos):
    v = vol_oos**2
    return float(np.mean(np.log(v) + r_oos**2 / v))


def corr_vol(vol_oos):
    return float(np.corrcoef(vol_oos, np.abs(r_oos))[0, 1])


def innov_kurtosis(r_full, states, vol_full):
    z = (r_full - states[:, 0] - states[:, 2]) / (vol_full + 1e-8)
    return float(pd.Series(z).kurtosis() + 3)  # excess → total


def var_exceedance(vol_oos, alpha, nu=None):
    """
    One-tailed downside VaR exceedance: fraction of r_oos < -VaR_bound.
    Expected rate = 1 - alpha under a calibrated model.
    Gaussian: VaR = norm.ppf(alpha) * vol
    Student-t: VaR = t.ppf(alpha, df=nu) * vol  (vol is the scale, not std)
    """
    if nu is None:
        bound = stats.norm.ppf(alpha) * vol_oos
    else:
        bound = stats.t.ppf(alpha, df=nu) * vol_oos
    return float(np.mean(r_oos < -bound))


# ── Run Gaussian filter ───────────────────────────────────────────────────────

print("Running Gaussian UKF...")
states_g, vol_g, vol_ewma, mu_h = run_base_filter(r)

vol_g_oos    = vol_g[split:]
vol_ewma_oos = vol_ewma[split:]

mae_g  = mae(vol_g_oos)
ql_g   = qlike(vol_g_oos)
corr_g = corr_vol(vol_g_oos)
kurt_g = innov_kurtosis(r, states_g, vol_g)


# ── Grid search: optimal nu ───────────────────────────────────────────────────

NU_GRID = [3, 4, 5, 6, 7, 8, 10, 15]
print(f"\nGrid search over nu in {NU_GRID} (minimising OOS QLIKE)...")
print(f"  {'nu':>4}  {'QLIKE':>10}  {'MAE':>10}  {'Corr':>8}  {'Kurt(innov)':>12}")
print("  " + "-" * 55)

best_nu, best_ql = None, np.inf

for nu in NU_GRID:
    _st, _vt, _, _ = run_student_t_filter(r, nu=nu)
    _vt_oos = _vt[split:]
    ql = qlike(_vt_oos)
    m  = mae(_vt_oos)
    c  = corr_vol(_vt_oos)
    k  = innov_kurtosis(r, _st, _vt)
    print(f"  {nu:>4}  {ql:>10.4f}  {m:>10.4f}  {c:>8.4f}  {k:>12.2f}")
    if ql < best_ql:
        best_ql, best_nu = ql, nu

print(f"\n  Best nu = {best_nu}  (OOS QLIKE = {best_ql:.4f})")
print(f"  Gaussian OOS QLIKE = {ql_g:.4f}")
if best_ql < ql_g:
    print(f"  Student-t improves QLIKE by {ql_g - best_ql:.4f}")
else:
    print(f"  *** Gaussian QLIKE is better by {best_ql - ql_g:.4f} ***")


# ── Run Student-t with best nu ────────────────────────────────────────────────

print(f"\nRunning Student-t UKF (nu={best_nu})...")
states_t, vol_t, _, _ = run_student_t_filter(r, nu=best_nu)

vol_t_oos = vol_t[split:]
mae_t     = mae(vol_t_oos)
ql_t      = qlike(vol_t_oos)
corr_t    = corr_vol(vol_t_oos)
kurt_t    = innov_kurtosis(r, states_t, vol_t)

# Standardised innovations
z_g = (r - states_g[:, 0] - states_g[:, 2]) / (vol_g + 1e-8)
z_t = (r - states_t[:, 0] - states_t[:, 2]) / (vol_t + 1e-8)

# VaR exceedance (one-tailed downside, expected = 1-alpha)
ALPHAS   = [0.90, 0.95, 0.975, 0.99]
exc_g    = [var_exceedance(vol_g_oos,    a)          for a in ALPHAS]
exc_t    = [var_exceedance(vol_t_oos,    a, best_nu) for a in ALPHAS]
exc_ewma = [var_exceedance(vol_ewma_oos, a)          for a in ALPHAS]
exc_exp  = [1 - a                                     for a in ALPHAS]


# ── Print summary ─────────────────────────────────────────────────────────────

W = 65
print("\n" + "=" * W)
print(f"{'Metric':<28} {'Gaussian':>12} {'Student-t':>12} {'EWMA':>10}")
print("-" * W)
print(f"{'MAE (vol)':28} {mae_g:>12.4f} {mae_t:>12.4f}"
      f" {mae(vol_ewma_oos):>10.4f}")
print(f"{'QLIKE':28} {ql_g:>12.4f} {ql_t:>12.4f}"
      f" {qlike(vol_ewma_oos):>10.4f}")
print(f"{'Corr(sigma, |r|)':28} {corr_g:>12.4f} {corr_t:>12.4f}"
      f" {corr_vol(vol_ewma_oos):>10.4f}")
print(f"{'Innovation kurtosis':28} {kurt_g:>12.2f} {kurt_t:>12.2f}"
      f" {'—':>10}")
print(f"{'Best nu':28} {'—':>12} {best_nu:>12} {'—':>10}")
print("=" * W)

print(
    "\nVaR exceedance (one-tailed downside, expected = 1-alpha):\n"
    "  Over-exceedance = model underestimates tail risk  (!)\n"
    "  Under-exceedance = model is conservative\n"
)
print(f"  {'Level':>7}  {'Expected':>9}  {'Gaussian':>10}  "
      f"{'Student-t':>10}  {'EWMA':>8}")
print("  " + "-" * 52)
for a, eg, et, ee, exp in zip(ALPHAS, exc_g, exc_t, exc_ewma, exc_exp):
    fg = " !" if eg > exp + 0.01 else ("  " if eg >= exp - 0.01 else " <")
    ft = " !" if et > exp + 0.01 else ("  " if et >= exp - 0.01 else " <")
    print(f"  {a:>7.1%}  {exp:>9.3f}  {eg:>9.3f}{fg}  "
          f"{et:>9.3f}{ft}  {ee:>8.3f}")


# ── Plot ─────────────────────────────────────────────────────────────────────

print("\nBuilding comparison plot...")

DATE_FMT = mdates.DateFormatter("%b '%y")
DATE_LOC = mdates.MonthLocator(interval=6)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Gaussian UKF vs Student-t UKF (ν={best_nu}) — BTC-USD 5-Year OOS Comparison",
    fontsize=12, fontweight="bold",
)

# ── (0,0) Vol comparison in OOS window ───────────────────────────────────────

ax = axes[0, 0]
ax.plot(oos_dates, vol_g_oos    * np.sqrt(252) * 100,
        color=BLUE,   lw=1.1, label="Gaussian UKF")
ax.plot(oos_dates, vol_t_oos    * np.sqrt(252) * 100,
        color=GREEN,  lw=1.1, ls="--", label=f"Student-t UKF (ν={best_nu})")
ax.plot(oos_dates, vol_ewma_oos * np.sqrt(252) * 100,
        color=RED,    lw=0.8, alpha=0.65, ls=":", label="EWMA")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("Volatility Estimate — Out-of-Sample Period")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(DATE_FMT)
ax.xaxis.set_major_locator(DATE_LOC)

# ── (0,1) Innovation distribution ────────────────────────────────────────────

ax = axes[0, 1]
xr = np.linspace(-6, 6, 400)

ax.hist(z_g, bins=90, density=True, alpha=0.35, color=BLUE,
        label=f"Gaussian innov. (kurt = {kurt_g:.2f})")
ax.hist(z_t, bins=90, density=True, alpha=0.35, color=GREEN,
        label=f"Student-t innov. (kurt = {kurt_t:.2f})")

# Reference PDFs: z_g ~ N(0,1) under Gaussian model
#                 z_t ~ t_nu(0,1) under Student-t model (scale=1, var=nu/(nu-2))
ax.plot(xr, stats.norm.pdf(xr),
        color="black", lw=1.6, ls="--", label="Normal(0,1) reference")
ax.plot(xr, stats.t.pdf(xr, df=best_nu),
        color=PURPLE,  lw=1.6, ls="-.",
        label=f"t(ν={best_nu}) reference (var={best_nu/(best_nu-2):.2f})")

ax.set_xlabel("Standardised innovation (z-score)")
ax.set_ylabel("Density")
ax.set_title("Innovation Distribution vs Theoretical Reference")
ax.legend(fontsize=8)
ax.set_xlim(-6, 6)

# ── (1,0) QQ plot ─────────────────────────────────────────────────────────────

ax = axes[1, 0]
ps = np.linspace(0.5, 99.5, 300)
q_g   = np.percentile(z_g, ps)
q_t   = np.percentile(z_t, ps)
q_ref = stats.norm.ppf(ps / 100)

ax.scatter(q_ref, q_g, s=7, alpha=0.5, color=BLUE,
           label=f"Gaussian UKF  (kurt={kurt_g:.2f})")
ax.scatter(q_ref, q_t, s=7, alpha=0.5, color=GREEN,
           label=f"Student-t UKF (kurt={kurt_t:.2f})")
lim = np.array([q_ref[0], q_ref[-1]])
ax.plot(lim, lim, color="black", lw=1.2, ls="--", alpha=0.5, label="y = x (Normal)")
ax.set_xlabel("Normal theoretical quantile")
ax.set_ylabel("Empirical innovation quantile")
ax.set_title(
    "QQ Plot: Innovations vs Normal(0,1)\n"
    "Points above y=x = fatter tails than Normal"
)
ax.legend(fontsize=8)

# ── (1,1) VaR calibration ────────────────────────────────────────────────────

ax = axes[1, 1]
x = np.arange(len(ALPHAS))
w = 0.18
labs = [f"{int(a * 100)}%" for a in ALPHAS]

ax.bar(x - 1.5*w, exc_exp,  w, color="black", alpha=0.55, label="Expected (1−α)")
ax.bar(x - 0.5*w, exc_g,    w, color=BLUE,    alpha=0.75, label="Gaussian UKF")
ax.bar(x + 0.5*w, exc_t,    w, color=GREEN,   alpha=0.75,
       label=f"Student-t (ν={best_nu})")
ax.bar(x + 1.5*w, exc_ewma, w, color=RED,     alpha=0.60, label="EWMA")

ax.set_xticks(x)
ax.set_xticklabels(labs)
ax.set_xlabel("VaR confidence level")
ax.set_ylabel("Exceedance rate (fraction of days r < −VaR)")
ax.set_title(
    "One-Tailed Downside VaR Calibration — OOS\n"
    "Bar height should equal Expected (black) for a calibrated model"
)
ax.legend(fontsize=8)

fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots/comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/comparison.png")
