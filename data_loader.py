"""
Data loader for BTC daily log-returns.

Set USE_REAL_DATA = True to download live BTC-USD prices via yfinance.
Set USE_REAL_DATA = False to use a synthetic series calibrated to BTC empirics.

Both paths return (r, h_true) where h_true is the latent log-variance array
(only available for simulated data; None for real data).
"""

import numpy as np

USE_REAL_DATA = True  # Toggle here to switch between simulated and live data


def simulate_btc_returns(n=1100):
    """
    Stochastic volatility model calibrated to BTC-USD daily data.
    Two engineered high-vol regimes (days 300-380, 700-760) test regime tracking.
    """
    np.random.seed(7)
    mu_h  = 2 * np.log(0.035)
    phi   = 0.97
    sig_h = 0.18

    h = np.zeros(n)
    h[0] = mu_h
    for t in range(1, n):
        h[t] = mu_h + phi * (h[t - 1] - mu_h) + sig_h * np.random.randn()

    regime = np.zeros(n)
    regime[300:380] = 1.5
    regime[700:760] = 1.2
    h += regime

    eps   = np.random.randn(n)
    jumps = (np.random.rand(n) < 0.02) * np.random.randn(n) * 0.06
    r = np.exp(h / 2) * eps + jumps
    r += 0.0008
    return r, h


def load_returns(use_real=USE_REAL_DATA, n=1100):
    """
    Load BTC daily log-returns.

    Parameters
    ----------
    use_real : bool
        True  → download BTC-USD from Yahoo Finance via yfinance (last 5 years)
        False → generate synthetic series (reproducible, seed=7)
    n : int
        Number of observations for simulated data (ignored for real data).

    Returns
    -------
    r : np.ndarray  — daily log-returns
    h_true : np.ndarray or None  — true log-variance (None for real data)
    """
    if use_real:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for real data: pip install yfinance")

        print("Downloading BTC-USD daily data from Yahoo Finance...")
        btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
        prices = btc["Close"].squeeze().dropna().values.astype(float)
        r = np.diff(np.log(prices))
        print(f"Downloaded {len(r)} daily log-returns (BTC-USD, 5-year history)")
        return r, None

    return simulate_btc_returns(n)
