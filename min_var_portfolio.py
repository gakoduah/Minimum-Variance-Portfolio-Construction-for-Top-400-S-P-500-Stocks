#!/usr/bin/env python3
"""
Minimum Variance Portfolio: Top 400 S&P 500 Stocks
Single-Factor Covariance Model with James-Stein Eigenvector Correction

Implements the methodology of:
  - Goldberg, Kercheval (2023). "James-Stein for the leading eigenvector."
    Proceedings of the National Academy of Sciences, 120(2).
  - Goldberg, Papanicolaou, Shkolnik (2022). "The dispersion bias."
    SIAM Journal on Financial Mathematics, 13, 521-550.

Usage:
    pip install numpy pandas yfinance requests
    python3 min_var_portfolio.py

Output:
    Summary statistics and complete portfolio holdings for both
    the uncorrected (Sigma_PCA) and JSE-corrected (Sigma_JSE) estimators.
"""

import sys
import numpy as np

try:
    import pandas as pd
    import requests
    import yfinance as yf
    from io import StringIO
except ImportError as e:
    print(f"Missing package: {e}")
    print("Run: pip install numpy pandas yfinance requests")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════

N_STOCKS    = 400          # Number of stocks in the portfolio
START_DATE  = "2025-04-25" # Weekly data start (Friday-aligned)
END_DATE    = "2025-11-01" # Weekly data end
RF_ANNUAL   = 0.043        # Annual risk-free rate (3-month T-bill, 2025)


# ════════════════════════════════════════════════════════════════
# STEP 1: SCRAPE S&P 500 TICKERS FROM WIKIPEDIA
# ════════════════════════════════════════════════════════════════

print("Step 1: Scraping S&P 500 tickers from Wikipedia...")

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
response = requests.get(url, headers=headers)
df_scrape = pd.read_html(
    StringIO(response.text), attrs={"id": "constituents"}
)[0]

# Replace dots with dashes for yfinance compatibility (e.g. BRK.B -> BRK-B)
all_tickers = [
    str(t).replace(".", "-") for t in df_scrape["Symbol"].tolist()
]
print(f"  Found {len(all_tickers)} tickers")


# ════════════════════════════════════════════════════════════════
# STEP 2: DOWNLOAD WEEKLY PRICES
# ════════════════════════════════════════════════════════════════

print(f"Step 2: Downloading weekly prices ({START_DATE} to {END_DATE})...")

raw_data = yf.download(
    all_tickers,
    start=START_DATE,
    end=END_DATE,
    interval="1wk",
    progress=True,
    threads=True,
    timeout=60,
)["Close"]

# Drop stocks with any missing observations, then keep first N_STOCKS survivors
clean_data   = raw_data.dropna(axis=1)
final_prices = clean_data.iloc[:, :N_STOCKS]

print(f"  Stocks with complete data: {clean_data.shape[1]}")
print(f"  Using first {N_STOCKS}: {final_prices.shape}")

if final_prices.shape[1] < N_STOCKS:
    print(f"  ERROR: Only {final_prices.shape[1]} stocks available. "
          f"Need {N_STOCKS}.")
    sys.exit(1)

tickers = list(final_prices.columns)
p       = len(tickers)                   # number of assets
prices  = final_prices.values            # (T x p)


# ════════════════════════════════════════════════════════════════
# STEP 3: COMPUTE EXCESS RETURNS
# ════════════════════════════════════════════════════════════════

print("Step 3: Computing excess returns...")

rf_weekly = RF_ANNUAL / 52

# Simple returns and excess returns
P          = prices.T                                       # (p x T)
raw_ret    = (P[:, 1:] - P[:, :-1]) / P[:, :-1]           # (p x n)
R          = raw_ret - rf_weekly                            # excess returns
n          = R.shape[1]                                     # number of periods

print(f"  Return matrix R: {R.shape}  (p={p}, n={n})")


# ════════════════════════════════════════════════════════════════
# STEP 4: SAMPLE COVARIANCE (SINGULAR)
# ════════════════════════════════════════════════════════════════

print("Step 4: Computing sample covariance matrix...")

mu  = np.mean(R, axis=1, keepdims=True)   # (p x 1) sample mean
Y   = R - mu                              # (p x n) de-meaned returns
S   = Y @ Y.T / n                         # (p x p) sample covariance
tr_S = np.trace(S)
rank = np.linalg.matrix_rank(S)

print(f"  S: {S.shape}, rank = {rank}  ({p - rank} zero eigenvalues)")
print(f"  tr(S) = {tr_S:.8f}")


# ════════════════════════════════════════════════════════════════
# STEP 5: EIGENDECOMPOSITION
# ════════════════════════════════════════════════════════════════

print("Step 5: Eigendecomposition...")

eigvals, eigvecs = np.linalg.eigh(S)
idx     = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

lambda_sq = eigvals[0]
h         = eigvecs[:, 0]

# Sign convention: entries of h average to a positive value
if np.mean(h) < 0:
    h = -h

ell_sq = (tr_S - lambda_sq) / (n - 1)   # average of remaining eigenvalues
n_nonzero = np.sum(eigvals > 1e-14)

print(f"  lambda^2 = {lambda_sq:.8f}")
print(f"  ell^2    = {ell_sq:.8f}")
print(f"  Leading eigenvalue explains {lambda_sq / tr_S * 100:.1f}% of variance")
print(f"  Non-zero eigenvalues: {n_nonzero}")


# ════════════════════════════════════════════════════════════════
# STEP 6: UNCORRECTED ESTIMATOR  Sigma_PCA
#
# Goldberg, Papanicolaou & Shkolnik (2022), Equation 43:
#   Sigma_PCA = (lambda^2 - ell^2) h h' + (n/p) ell^2 I
#
# Invertible by construction: eigenvalues are
#   mu_1 = lambda^2 - ell^2 + (n/p) ell^2  (in direction h)
#   mu_2 = (n/p) ell^2                      (in all orthogonal directions)
# ════════════════════════════════════════════════════════════════

print("Step 6: Building Sigma_PCA and computing uncorrected portfolio...")

ones    = np.ones((p, 1))
Sig_PCA = ((lambda_sq - ell_sq) * np.outer(h, h)
           + (n / p) * ell_sq * np.eye(p))

inv_PCA = np.linalg.inv(Sig_PCA)
w_PCA   = (inv_PCA @ ones) / (ones.T @ inv_PCA @ ones).item()
w_pca   = w_PCA.flatten()

fv_pca  = (w_PCA.T @ Sig_PCA @ w_PCA).item()     # forecast variance
sv_pca  = (w_PCA.T @ S       @ w_PCA).item()     # S-based variance
er_pca  = (w_PCA.T @ mu).item()

print(f"  Forecast Ann Std: {np.sqrt(fv_pca * 52) * 100:.4f}%")


# ════════════════════════════════════════════════════════════════
# STEP 7: JSE EIGENVECTOR CORRECTION  Sigma_JSE
#
# Goldberg & Kercheval (2023), Equations 6-9 and 44:
#   h_JSE = m(h) 1 + c_JSE (h - m(h) 1)
#   c_JSE = 1 - nu^2 / s^2(h)
#   nu^2  = (tr(S) - lambda^2) / (p (n-1))
#   s^2(h)= (1/p) sum_i (lambda h_i - lambda m(h))^2
#
#   Sigma_JSE = (lambda^2 - ell^2) h_JSE h_JSE' / |h_JSE|^2
#             + (n/p) ell^2 I
# ════════════════════════════════════════════════════════════════

print("Step 7: JSE eigenvector correction and corrected portfolio...")

m_h    = np.mean(h)
nu_sq  = (tr_S - lambda_sq) / (p * (n - 1))
lam    = np.sqrt(lambda_sq)
s2_h   = (1.0 / p) * np.sum((lam * h - lam * m_h) ** 2)
c_JSE  = 1.0 - nu_sq / s2_h
h_JSE  = m_h * np.ones(p) + c_JSE * (h - m_h * np.ones(p))

Sig_JSE = ((lambda_sq - ell_sq) * np.outer(h_JSE, h_JSE)
           / np.dot(h_JSE, h_JSE)
           + (n / p) * ell_sq * np.eye(p))

inv_JSE = np.linalg.inv(Sig_JSE)
w_JSE   = (inv_JSE @ ones) / (ones.T @ inv_JSE @ ones).item()
w_jse   = w_JSE.flatten()

fv_jse  = (w_JSE.T @ Sig_JSE @ w_JSE).item()
sv_jse  = (w_JSE.T @ S       @ w_JSE).item()
er_jse  = (w_JSE.T @ mu).item()

# Angle between h and h_JSE
h_JSE_unit = h_JSE / np.linalg.norm(h_JSE)
angle_deg  = np.degrees(np.arccos(np.clip(abs(np.dot(h, h_JSE_unit)), 0, 1)))

print(f"  c^JSE    = {c_JSE:.6f}")
print(f"  Shrinkage = {(1-c_JSE)*100:.2f}%")
print(f"  Forecast Ann Std: {np.sqrt(fv_jse * 52) * 100:.4f}%")


# ════════════════════════════════════════════════════════════════
# STEP 8: RESULTS
# ════════════════════════════════════════════════════════════════

indiv_std = np.sqrt(np.diag(S) * 52) * 100   # annualized, in %
indiv_er  = mu.flatten() * 52 * 100           # annualized, in %

VFR_pca      = fv_pca / sv_pca
VFR_jse      = fv_jse / sv_jse
weight_corr  = np.corrcoef(w_pca, w_jse)[0, 1]

print(f"\n{'=' * 65}")
print("RESULTS SUMMARY")
print(f"{'=' * 65}")
print(f"{'Metric':<38} {'Sigma_PCA':>12} {'Sigma_JSE':>12}")
print("-" * 65)
print(f"{'Ann. Forecast Std Dev':<38} {np.sqrt(fv_pca*52)*100:>11.4f}% "
      f"{np.sqrt(fv_jse*52)*100:>11.4f}%")
print(f"{'Ann. Forecast Variance':<38} {fv_pca:>12.6f} {fv_jse:>12.6f}")
print(f"{'Ann. Variance (via S)':<38} {sv_pca*52:>12.8f} {sv_jse*52:>12.8f}")
print(f"{'VFR (forecast / S-based)':<38} {VFR_pca:>12.4f} {VFR_jse:>12.4f}")
print(f"{'Ann. Expected Excess Return':<38} {er_pca*52*100:>11.4f}% "
      f"{er_jse*52*100:>11.4f}%")
print(f"{'Long exposure':<38} {np.sum(w_pca[w_pca>0])*100:>11.2f}% "
      f"{np.sum(w_jse[w_jse>0])*100:>11.2f}%")
print(f"{'Short exposure':<38} "
      f"{np.sum(np.abs(w_pca[w_pca<0]))*100:>11.2f}% "
      f"{np.sum(np.abs(w_jse[w_jse<0]))*100:>11.2f}%")
print(f"{'Eigvec dispersion std(h)':<38} {np.std(h):>12.6f} {np.std(h_JSE):>12.6f}")
print(f"{'Dispersion reduction':<38} {(1-c_JSE)*100:>11.2f}%")
print(f"{'Angle(h, h_JSE)':<38} {angle_deg:>11.2f}°")
print(f"{'Weight correlation':<38} {weight_corr:>12.4f}")

# ── Top and bottom holdings ──────────────────────────────────────
sort_pca = np.argsort(w_pca)[::-1]
sort_jse = np.argsort(w_jse)[::-1]

for label, sort, weights in [
    ("SIGMA_PCA (UNCORRECTED)", sort_pca, w_pca),
    ("SIGMA_JSE (CORRECTED)",   sort_jse, w_jse),
]:
    print(f"\n{'=' * 75}")
    print(f"COMPLETE {label} PORTFOLIO HOLDINGS — sorted by weight")
    print(f"{'=' * 75}")
    print(f"{'#':>4} {'Stock':<10} {'Weight%':>10} "
          f"{'Ann Std%':>10} {'Ann E[R]%':>10}")
    print("-" * 50)
    for rank, i in enumerate(sort):
        print(f"{rank+1:>4} {tickers[i]:<10} {weights[i]*100:>+10.4f} "
              f"{indiv_std[i]:>10.2f} {indiv_er[i]:>10.2f}")

print(f"\nDone.")
