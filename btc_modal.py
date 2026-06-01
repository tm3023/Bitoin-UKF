"""
Unscented Kalman Filter -- Bitcoin Return Dynamics & Volatility
===============================================================

Models daily BTC log-returns using a 5-state UKF:

  State x_t = [p1, v1, p2, v2, h]
    (p1, v1): slow damped oscillator  (period ~30d, zeta=0.15)
    (p2, v2): fast damped oscillator  (period ~5d,  zeta=0.35)
    h       : log-variance, AR(1) with persistence phi=0.95

Observation (dual-channel):
    z1 = p1 + p2 + eps_t,   eps_t ~ N(0, exp(h_t))
    z2 = log(r_t^2) ~ N(h - 1.27, pi^2/2)   [Harvey-Ruiz-Shephard]

The dual observation is the key design fix: with z1 alone, h is
structurally unobservable. Adding z2 gives the filter a direct signal
on h, restoring identifiability.

Data source is controlled by USE_REAL_DATA in data_loader.py.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from data_loader import load_returns

warnings.filterwarnings("ignore")

DT = 1.0  # daily timestep


def damped_osc_matrix(omega, zeta, dt=DT):
    """
    Exact discrete-time transition matrix for a damped oscillator.
    Analytic solution to x'' + 2*zeta*omega*x' + omega^2*x = 0.
    """
    wd = omega * np.sqrt(max(1 - zeta**2, 1e-8))
    e = np.exp(-zeta * omega * dt)
    c, s = np.cos(wd * dt), np.sin(wd * dt)
    return e * np.array([
        [c + zeta * omega / wd * s, s / wd],
        [-(omega**2) / wd * s, c - zeta * omega / wd * s],
    ])


def run_base_filter(r):
    """
    Run the 5-state UKF on returns r with fixed (non-MLE) parameters.

    Returns
    -------
    states   : (n, 5) filtered state array
    vol_ukf  : (n,) filtered volatility
    vol_ewma : (n,) EWMA benchmark (lambda=0.94)
    mu_h     : float, unconditional log-variance mean
    """
    n = len(r)
    mu_h = 2 * np.log(r.std())
    phi = 0.95

    A1 = damped_osc_matrix(2 * np.pi / 30.0, 0.15)
    A2 = damped_osc_matrix(2 * np.pi / 5.0, 0.35)

    def fx(state, dt):
        p1, v1, p2, v2, h = state
        p1n, v1n = A1 @ np.array([p1, v1])
        p2n, v2n = A2 @ np.array([p2, v2])
        hn = mu_h + phi * (h - mu_h)
        return np.array([p1n, v1n, p2n, v2n, hn])

    def hx(state):
        """
        Dual observation (Harvey-Ruiz-Shephard):
          z1 = p1 + p2    (return level)
          z2 = h - 1.27   (log-squared-return mean)
        z2 makes h identifiable by entering the observation mean directly.
        """
        p1, _, p2, _, h = state
        return np.array([p1 + p2, h - 1.27])

    sp = MerweScaledSigmaPoints(n=5, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=5, dim_z=2, dt=DT, fx=fx, hx=hx, points=sp
    )
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h])
    ukf.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    ukf.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, 0.05])

    states = np.zeros((n, 5))
    for t in range(n):
        h_est = ukf.x[4]
        ukf.R = np.array([
            [np.exp(h_est), 0.0],
            [0.0, np.pi**2 / 2.0],
        ])
        ukf.predict()
        z = np.array([r[t], np.log(r[t]**2 + 1e-8)])
        ukf.update(z)
        states[t] = ukf.x

    vol_ukf = np.exp(states[:, 4] / 2)

    lam = 0.94
    vol_ewma = np.zeros(n)
    vol_ewma[0] = r.std()
    for t in range(1, n):
        vol_ewma[t] = np.sqrt(
            lam * vol_ewma[t - 1]**2 + (1 - lam) * r[t - 1]**2
        )

    return states, vol_ukf, vol_ewma, mu_h


def run_student_t_filter(r, nu=5, n_iter=8):
    """
    5-state UKF with Student-t observation noise via variational Bayes.

    At each time step a scale weight lambda_t is found by iterating:
      lambda <- (nu + k) / (nu + v' S^{-1} v)
    where v is the innovation and S its covariance. A large Mahalanobis
    distance (outlier) drives lambda small, inflating R_eff = R / lambda
    and reducing the Kalman gain for that observation.

    Parameters
    ----------
    r      : array of daily log-returns
    nu     : Student-t degrees of freedom (lower = heavier tails, 3-15)
    n_iter : VB iterations per step (converges in 4-6 for typical data)

    Returns
    -------
    Same signature as run_base_filter: (states, vol_t, vol_ewma, mu_h)
    """
    n = len(r)
    mu_h = 2 * np.log(r.std())
    phi = 0.95
    k = 2  # observation dimension

    A1 = damped_osc_matrix(2 * np.pi / 30.0, 0.15)
    A2 = damped_osc_matrix(2 * np.pi / 5.0, 0.35)

    def fx(state, dt):
        p1, v1, p2, v2, h = state
        p1n, v1n = A1 @ np.array([p1, v1])
        p2n, v2n = A2 @ np.array([p2, v2])
        hn = mu_h + phi * (h - mu_h)
        return np.array([p1n, v1n, p2n, v2n, hn])

    def hx(state):
        p1, _, p2, _, h = state
        return np.array([p1 + p2, h - 1.27])

    sp = MerweScaledSigmaPoints(n=5, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=5, dim_z=2, dt=DT, fx=fx, hx=hx, points=sp
    )
    ukf.x = np.array([0.0, 0.0, 0.0, 0.0, mu_h])
    ukf.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 0.1])
    ukf.Q = np.diag([1e-6, 1e-5, 1e-5, 1e-4, 0.05])

    states = np.zeros((n, 5))

    for t in range(n):
        h_est = ukf.x[4]
        R_base = np.array([
            [np.exp(h_est), 0.0],
            [0.0, np.pi**2 / 2.0],
        ])

        # Predict once; sigma points (sigmas_f) are fixed for all update iters
        ukf.predict()
        x_pred = ukf.x.copy()
        P_pred = ukf.P.copy()
        z = np.array([r[t], np.log(r[t]**2 + 1e-8)])

        # VB iteration: converge on the scale weight lambda
        lam = 1.0
        for i in range(n_iter):
            ukf.x = x_pred.copy()
            ukf.P = P_pred.copy()
            ukf.R = R_base / lam
            ukf.update(z)
            if i == n_iter - 1:
                break
            mah = float(ukf.y @ np.linalg.solve(ukf.S, ukf.y))
            lam_new = (nu + k) / (nu + mah)
            if abs(lam_new - lam) < 1e-5:
                # converged early — do final update with new lambda
                lam = lam_new
                ukf.x = x_pred.copy()
                ukf.P = P_pred.copy()
                ukf.R = R_base / lam
                ukf.update(z)
                break
            lam = lam_new

        states[t] = ukf.x

    vol_t = np.exp(states[:, 4] / 2)

    lam_ewma = 0.94
    vol_ewma = np.zeros(n)
    vol_ewma[0] = r.std()
    for t in range(1, n):
        vol_ewma[t] = np.sqrt(
            lam_ewma * vol_ewma[t - 1]**2 + (1 - lam_ewma) * r[t - 1]**2
        )

    return states, vol_t, vol_ewma, mu_h


def main():
    r, h_true = load_returns()
    n = len(r)
    dates = np.arange(n)

    print("=" * 55)
    print("BTC daily log-returns")
    print(f"  N     = {n}")
    print(f"  mean  = {r.mean() * 1e4:+.1f} bp/day")
    print(f"  sigma = {r.std() * 100:.2f}%")
    print(f"  skew  = {((r - r.mean())**3).mean() / r.std()**3:+.2f}")
    print(f"  kurt  = {((r - r.mean())**4).mean() / r.std()**4:.1f}")
    print("=" * 55)

    states, vol_ukf, vol_ewma, mu_h = run_base_filter(r)  # noqa: F841
    slow_mode = states[:, 0]
    fast_mode = states[:, 2]

    split = int(0.7 * n)
    r_oos = r[split:]
    abs_r = np.abs(r_oos)
    vol_ukf_oos = vol_ukf[split:]
    vol_ewma_oos = vol_ewma[split:]

    mae_ukf = np.mean(np.abs(vol_ukf_oos - abs_r))
    mae_ewma = np.mean(np.abs(vol_ewma_oos - abs_r))

    def qlike(sigma, ret):
        v = sigma**2
        return np.mean(np.log(v) + ret**2 / v)

    ql_ukf = qlike(vol_ukf_oos, r_oos)
    ql_ewma = qlike(vol_ewma_oos, r_oos)
    corr_ukf = np.corrcoef(vol_ukf_oos, abs_r)[0, 1]
    corr_ewma = np.corrcoef(vol_ewma_oos, abs_r)[0, 1]

    innovations = r - (states[:, 0] + states[:, 2])
    std_innov = innovations / (vol_ukf + 1e-8)
    kurt_innov = (
        ((std_innov - std_innov.mean())**4).mean() / std_innov.std()**4
    )

    print("\nOut-of-sample (last 30%) volatility forecast evaluation")
    print(f"{'Metric':<25} {'UKF':>10} {'EWMA':>10}")
    print("-" * 47)
    print(f"{'MAE (vol)':25} {mae_ukf:10.4f} {mae_ewma:10.4f}")
    print(f"{'QLIKE':25} {ql_ukf:10.4f} {ql_ewma:10.4f}")
    print(f"{'Corr(sigma, |r|)':25} {corr_ukf:10.4f} {corr_ewma:10.4f}")
    print(f"\nStandardised innovation kurtosis: {kurt_innov:.2f}  (Gaussian = 3.0)")

    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("UKF Applied to BTC Returns: 5-State Model", fontsize=14, y=0.98)

    axes[0].plot(dates, r * 100, color="steelblue", lw=0.6, alpha=0.8)
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Daily Returns")
    axes[0].axvline(split, color="grey", lw=1, ls="--", label="Train/test split")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        dates, slow_mode * 100, color="darkorange", lw=1.2, label="Slow mode (~30d)"
    )
    axes[1].plot(
        dates, fast_mode * 100, color="purple", lw=0.8, alpha=0.7,
        label="Fast mode (~5d)"
    )
    axes[1].set_ylabel("Mode amplitude (%)")
    axes[1].set_title("Filtered Modal Decomposition")
    axes[1].legend(fontsize=8)

    if h_true is not None:
        axes[2].plot(
            dates, np.exp(h_true / 2) * 100, color="black", lw=1.0,
            ls="--", label="True vol (simulated)"
        )
    axes[2].plot(dates, vol_ukf * 100, color="green", lw=1.2, label="UKF vol")
    axes[2].plot(
        dates, vol_ewma * 100, color="firebrick", lw=0.8, alpha=0.7,
        label="EWMA (lambda=0.94)"
    )
    axes[2].set_ylabel("Volatility (%)")
    title_suffix = " vs Truth" if h_true is not None else ""
    axes[2].set_title(f"Volatility: UKF vs EWMA{title_suffix}")
    axes[2].legend(fontsize=8)
    axes[2].axvline(split, color="grey", lw=1, ls="--")

    axes[3].plot(dates, std_innov, color="steelblue", lw=0.5, alpha=0.7)
    axes[3].axhline(0, color="black", lw=0.8)
    axes[3].axhline(2, color="red", lw=0.8, ls="--", alpha=0.5)
    axes[3].axhline(-2, color="red", lw=0.8, ls="--", alpha=0.5)
    axes[3].set_ylabel("Standardised innovation")
    axes[3].set_xlabel("Day")
    axes[3].set_title(f"Standardised Innovations (kurtosis = {kurt_innov:.2f})")

    plt.tight_layout()
    plt.savefig("plots/vol_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to plots/vol_comparison.png")


if __name__ == "__main__":
    main()
