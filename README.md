# Unscented Kalman Filter Applied to Bitcoin Returns

A 5-state Unscented Kalman Filter (UKF) for joint estimation of Bitcoin price 
dynamics and latent volatility, implemented from scratch in Python.

## Model Overview

The state vector combines two components:

- **Damped oscillator** (3 states): captures mean-reverting cyclical dynamics 
  in log-returns — position, velocity, and a damping coefficient
- **Stochastic log-variance** (2 states): models time-varying volatility as a 
  latent AR(1) process in log space, approximating a discrete-time stochastic 
  volatility model

### Dual-Observation Fix

The original single-observation design suffered from latent state 
unobservability — the log-variance state was not identifiable from price data 
alone. This was resolved by augmenting the observation vector with a 
realised-variance proxy (rolling squared returns), giving the filter an 
independent signal on the volatility state and restoring identifiability.

## Results

The filter produces smooth estimates of latent volatility that lead realised 
volatility during high-turbulence periods, consistent with the model's 
forward-looking state propagation.

![Filtered vs Realised Volatility](plots/vol_comparison.png)
![Log-Return Fit](plots/return_fit.png)

## Repository Structure

├── ukf_bitcoin.ipynb       # Main notebook: model derivation, fitting, plots
├── ukf_core.py             # UKF implementation (sigma points, predict, update)
├── data/
│   └── btc_daily.csv       # BTC/USD daily OHLCV (source: Yahoo Finance)
├── plots/                  # Generated figures
└── requirements.txt

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| UKF over EKF | Avoids Jacobian computation; better for nonlinear log-variance dynamics |
| Log-variance state | Ensures positivity without constrained optimisation |
| Dual observation | Resolves rank-deficiency in observation mapping |
| Damped oscillator | Parsimonious way to capture autocorrelation in BTC returns |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook ukf_bitcoin.ipynb
```

**Requirements:** Python 3.9+, NumPy, SciPy, Pandas, Matplotlib, yfinance

## Background

Built as part of a personal research project exploring state-space methods in crypto markets. 
The identifiability issue and its fix are discussed in detail in the notebook.
