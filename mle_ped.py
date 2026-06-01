"""
MLE parameter fitting via Prediction Error Decomposition (PED).

For any state-space model, the Kalman filter produces at each step:
  - innovation:            v_t = z_t - E[z_t | z_{1:t-1}]
  - innovation covariance: S_t = Var[z_t | z_{1:t-1}]

Under the model, v_t ~ N(0, S_t), so the exact log-likelihood is:

  log L(theta) = -0.5 * sum_t { k*log(2pi) + log|S_t| + v_t' S_t^{-1} v_t }

This is Harvey's (1989) prediction error decomposition -- the standard MLE
method for state-space models.  We maximise over the structural parameters:

  theta = [T_slow, zeta_slow, T_fast, zeta_fast, phi, log_q_h]

  T_slow, T_fast  : oscillator periods (days) -- the dominant BTC cycles
  zeta_slow/fast  : damping ratios
  phi             : log-variance AR(1) persistence
  log_q_h         : log of process noise on h (controls vol-of-vol)
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.linalg import inv
from scipy.optimize import minimize

from btc_modal import run_base_filter
from data_loader import load_returns

warnings.filterwarnings("ignore")

# Load returns and run the base filter so vol_ukf / vol_ewma are available
# for the three-way OOS comparison at the end of the script.
r, h_true = load_returns()
n = len(r)
mu_h = 2 * np.log(r.std())
_, vol_ukf, vol_ewma, _ = run_base_filter(r)


def make_fx(T_slow, zeta_slow, T_fast, zeta_fast, phi, mu_h_val):
    """Return an fx closure for the given modal parameters."""
    omega_slow = 2 * np.pi / T_slow
    omega_fast = 2 * np.pi / T_fast

    def _osc(omega, zeta, dt=1.0):
        wd = omega * np.sqrt(max(1 - zeta**2, 1e-8))
        e = np.exp(-zeta * omega * dt)
        c, s = np.cos(wd), np.sin(wd)
        return e * np.array([
            [c + zeta * omega / wd * s, s / wd],
            [-(omega**2) / wd * s, c - zeta * omega / wd * s],
        ])

    A1 = _osc(omega_slow, zeta_slow)
    A2 = _osc(omega_fast, zeta_fast)

    def fx(state, dt):
        p1, v1, p2, v2, h = state
        p1n, v1n = A1 @ [p1, v1]
        p2n, v2n = A2 @ [p2, v2]
        hn = mu_h_val + phi * (h - mu_h_val)
        return np.array([p1n, v1n, p2n, v2n, hn])

    return fx


def hx(state):
    p1, _, p2, _, h = state
    return np.array([p1 + p2, h - 1.27])


def neg_loglik(params, r_obs, mu_h_val, burn_in=50):
    """
    Return -log L(theta | r_{1:T}) via PED.
    burn_in: first steps excluded from likelihood (filter init -- standard).
    """
    T_slow, zeta_slow, T_fast, zeta_fast, phi, log_q_h = params

    if not (
        5 < T_slow < 90
        and 0.02 < zeta_slow < 0.95
        and 2 < T_fast < 20
        and 0.05 < zeta_fast < 0.95
        and 0.5 < phi < 0.9999
        and -8 < log_q_h < 1
    ):
        return 1e10

    q_h = np.exp(log_q_h)
    _fx = make_fx(T_slow, zeta_slow, T_fast, zeta_fast, phi, mu_h_val)

    sp = MerweScaledSigmaPoints(n=5, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=5, dim_z=2, dt=1.0, fx=_fx, hx=hx, points=sp
    )
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h_val])
    ukf.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    ukf.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, q_h])

    n_obs = len(r_obs)
    ll = 0.0
    k = 2

    for t in range(n_obs):
        h_est = ukf.x[4]
        ukf.R = np.array([
            [np.exp(h_est), 0.0],
            [0.0, np.pi**2 / 2.0],
        ])
        ukf.predict()
        z = np.array([r_obs[t], np.log(r_obs[t]**2 + 1e-8)])
        ukf.update(z)

        if t >= burn_in:
            v = ukf.y
            S = ukf.S
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                return 1e10
            S_inv = inv(S)
            ll += -0.5 * (k * np.log(2 * np.pi) + logdet + v @ S_inv @ v)

    return -ll


def hessian_fd(f, x, eps=1e-4):
    """Finite-difference Hessian of scalar f at x."""
    p = len(x)
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            xpp = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm = x.copy()
            xpm[i] += eps
            xpm[j] -= eps
            xmp = x.copy()
            xmp[i] -= eps
            xmp[j] += eps
            xmm = x.copy()
            xmm[i] -= eps
            xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


# ── MLE optimisation ─────────────────────────────────────────────────────────

theta0 = np.array([30.0, 0.15, 5.0, 0.35, 0.95, np.log(0.05)])
labels = [
    "T_slow (days)", "zeta_slow", "T_fast (days)",
    "zeta_fast", "phi (vol persistence)", "log q_h",
]
bounds = [
    (5, 90), (0.02, 0.95), (2, 20),
    (0.05, 0.95), (0.5, 0.9999), (-8, 1),
]

print("Fitting via prediction error decomposition MLE...")
print(f"{'Parameter':<22} {'Initial':>10}")
print("-" * 34)
for lbl, v in zip(labels, theta0):
    print(f"  {lbl:<20} {v:>10.3f}")

result = minimize(
    neg_loglik,
    theta0,
    args=(r, mu_h),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
)

theta_mle = result.x
status = "converged" if result.success else "DID NOT CONVERGE"
print(
    f"\nOptimisation {status}  "
    f"(iterations: {result.nit},  log L = {-result.fun:.2f})\n"
)

# ── Standard errors from numerical Hessian ───────────────────────────────────

print("Computing standard errors from numerical Hessian...")
H = hessian_fd(lambda p: neg_loglik(p, r, mu_h), theta_mle)

try:
    cov = np.linalg.inv(H)
    se = np.sqrt(np.abs(np.diag(cov)))
    se_ok = True
except np.linalg.LinAlgError:
    se_ok = False

print(f"\n{'Parameter':<22} {'MLE':>10} {'Std Err':>10} {'95% CI':>24}")
print("=" * 70)
for i, (lbl, v) in enumerate(zip(labels, theta_mle)):
    if se_ok:
        lo, hi = v - 1.96 * se[i], v + 1.96 * se[i]
        ci_str = f"[{lo:7.3f}, {hi:7.3f}]"
        se_str = f"{se[i]:10.4f}"
    else:
        ci_str = "      N/A      "
        se_str = "       N/A"
    print(f"  {lbl:<20} {v:>10.3f} {se_str}  {ci_str}")

ll_init = -neg_loglik(theta0, r, mu_h)
ll_mle = -result.fun
print(f"\nLog-likelihood at MLE : {ll_mle:.2f}")
print(f"Log-likelihood at init: {ll_init:.2f}")
print(f"LR test stat (2*dLL)  : {2 * (ll_mle - ll_init):.2f}")

# ── Re-run filter with MLE parameters ────────────────────────────────────────

T_slow_mle, z_slow_mle, T_fast_mle, z_fast_mle, phi_mle, log_q_h_mle = theta_mle

print(f"\nMLE dominant cycles: {T_slow_mle:.1f}d (slow) and {T_fast_mle:.1f}d (fast)")
print(f"Vol persistence:     phi = {phi_mle:.4f}  (vs initial 0.950)")

fx_mle = make_fx(T_slow_mle, z_slow_mle, T_fast_mle, z_fast_mle, phi_mle, mu_h)
sp_mle = MerweScaledSigmaPoints(n=5, alpha=1e-3, beta=2.0, kappa=0.0)
ukf_mle = UnscentedKalmanFilter(
    dim_x=5, dim_z=2, dt=1.0, fx=fx_mle, hx=hx, points=sp_mle
)
ukf_mle.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h])
ukf_mle.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
ukf_mle.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, np.exp(log_q_h_mle)])

states_mle = np.zeros((n, 5))
for t in range(n):
    h_est = ukf_mle.x[4]
    ukf_mle.R = np.array([
        [np.exp(h_est), 0.0],
        [0.0, np.pi**2 / 2.0],
    ])
    ukf_mle.predict()
    z = np.array([r[t], np.log(r[t]**2 + 1e-8)])
    ukf_mle.update(z)
    states_mle[t] = ukf_mle.x

vol_mle = np.exp(states_mle[:, 4] / 2)

split = int(0.7 * n)
abs_r_oos = np.abs(r[split:])


def mae(v):
    return np.mean(np.abs(v[split:] - abs_r_oos))


def qlike(sigma, ret):
    v = sigma**2
    return np.mean(np.log(v) + ret**2 / v)


print("\nOut-of-sample comparison (last 30%)")
print(f"{'Model':<20} {'MAE':>10} {'QLIKE':>10}")
print("-" * 42)
print(
    f"  {'UKF (initial)':<18} {mae(vol_ukf):>10.4f}"
    f" {qlike(vol_ukf[split:], r[split:]):>10.4f}"
)
print(
    f"  {'UKF (MLE)':<18} {mae(vol_mle):>10.4f}"
    f" {qlike(vol_mle[split:], r[split:]):>10.4f}"
)
print(
    f"  {'EWMA':<18} {mae(vol_ewma):>10.4f}"
    f" {qlike(vol_ewma[split:], r[split:]):>10.4f}"
)

# ── Profile likelihood for phi ────────────────────────────────────────────────

print("\nComputing profile likelihood for phi...")
phi_grid = np.linspace(0.80, 0.999, 40)
profile_ll = [-neg_loglik(
    np.array([*theta_mle[:4], phi_val, theta_mle[5]]), r, mu_h
) for phi_val in phi_grid]

os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MLE via Prediction Error Decomposition", fontsize=13)

ax = axes[0]
ax.plot(phi_grid, profile_ll, color="steelblue", lw=2)
ax.axvline(
    phi_mle, color="firebrick", lw=1.5, ls="--",
    label=f"MLE phi = {phi_mle:.4f}",
)
ax.axhline(
    max(profile_ll) - 1.92, color="grey", lw=1, ls=":",
    label="95% CI threshold (-1.92)",
)
ax.set_xlabel("phi (log-variance persistence)")
ax.set_ylabel("Profile log-likelihood")
ax.set_title("Profile Likelihood: Vol Persistence phi")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(
    range(n), vol_ukf * 100, color="steelblue", lw=0.9,
    alpha=0.8, label="UKF (initial params)",
)
ax.plot(
    range(n), vol_mle * 100, color="green", lw=1.2, label="UKF (MLE params)"
)
ax.plot(
    range(n), vol_ewma * 100, color="firebrick", lw=0.8,
    alpha=0.6, label="EWMA",
)
if h_true is not None:
    ax.plot(
        range(n), np.exp(h_true / 2) * 100, color="black",
        lw=0.8, ls="--", alpha=0.6, label="True vol",
    )
ax.axvline(split, color="grey", lw=1, ls="--", alpha=0.5)
ax.set_xlabel("Day")
ax.set_ylabel("Volatility (%)")
ax.set_title("Volatility: Initial vs MLE-Fitted UKF")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plots/mle_fit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to plots/mle_fit.png")
