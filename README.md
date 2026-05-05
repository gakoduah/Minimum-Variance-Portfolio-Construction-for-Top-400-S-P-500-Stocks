# Minimum Variance Portfolio with James–Stein Eigenvector Correction

A high-dimensional minimum variance portfolio constructed from the top 400 S&P 500 stocks, implementing the single-factor covariance model and the James–Stein for Eigenvectors (JSE) correction developed in Goldberg & Kercheval (2023, *PNAS*). The project demonstrates how to optimize a portfolio in the high-dimension low-sample-size (HL) regime, where the sample covariance matrix is singular and classical mean–variance optimization breaks down.

## Background

The Markowitz minimum variance portfolio,

$$
w^* = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}},
$$

requires an invertible covariance matrix $\Sigma$. With $p = 400$ stocks and $n = 26$ weekly returns, the sample covariance matrix $S$ has rank at most 25 and is singular: $S^{-1}$ does not exist.

Goldberg, Papanicolaou & Shkolnik (2022) and Goldberg & Kercheval (2023) show that in this HL regime ($p/n \gg 1$):

1. A single-factor structured estimator $\Sigma_{\text{PCA}} = (\lambda^2 - \ell^2) hh^\top + (n/p)\ell^2 I$ resolves the singularity while preserving the leading eigenvector of $S$.
2. The leading sample eigenvector $h$ exhibits **excess dispersion** — its entries are noisier than the entries of the true population eigenvector $b$.
3. A James–Stein-style shrinkage of the eigenvector entries toward their cross-sectional mean,

$$
h^{\text{JSE}} = m(h)\mathbf{1} + c^{\text{JSE}}(h - m(h)\mathbf{1}),
$$

reduces this dispersion and drives the optimization bias to zero asymptotically.

## Methodology

| Step | Description |
|------|-------------|
| 1 | Scrape S&P 500 tickers from Wikipedia |
| 2 | Download 27 weekly Friday closing prices (May–October 2025) for all constituents |
| 3 | Drop tickers with incomplete data; retain the first 400 survivors |
| 4 | Compute weekly excess returns using the 4.3% annual T-bill rate |
| 5 | Form the sample covariance $S$ and its eigendecomposition |
| 6 | Build $\Sigma_{\text{PCA}}$ and the uncorrected portfolio $w_{\text{PCA}}$ |
| 7 | Apply JSE shrinkage to compute $\Sigma_{\text{JSE}}$ and the corrected portfolio $w_{\text{JSE}}$ |
| 8 | Compare forecast variance, variance forecast ratio (VFR), and portfolio composition |

## Key Formulas

**Single-factor covariance estimator** (Goldberg & Kercheval, Eq. 43):

$$
\Sigma_{\text{PCA}} = (\lambda^2 - \ell^2)\,hh^\top + \frac{n}{p}\ell^2 I, \qquad \ell^2 = \frac{\text{tr}(S) - \lambda^2}{n - 1}
$$

**JSE shrinkage constant** (Goldberg & Kercheval, Eqs. 6–9):

$$
c^{\text{JSE}} = 1 - \frac{\nu^2}{s^2(h)}, \qquad \nu^2 = \frac{\text{tr}(S) - \lambda^2}{p(n-1)}, \qquad s^2(h) = \frac{\lambda^2}{p}\sum_{i=1}^p (h_i - m(h))^2
$$

**JSE-corrected covariance** (Goldberg & Kercheval, Eq. 44):

$$
\Sigma_{\text{JSE}} = (\lambda^2 - \ell^2)\,\frac{h^{\text{JSE}}(h^{\text{JSE}})^\top}{|h^{\text{JSE}}|^2} + \frac{n}{p}\ell^2 I
$$

## Installation

```bash
git clone https://github.com/<your-username>/min-var-portfolio-jse.git
cd min-var-portfolio-jse
pip install numpy pandas yfinance requests
```

## Usage

```bash
python min_var_portfolio.py
```

The script will:
- Scrape S&P 500 tickers from Wikipedia
- Download weekly prices via `yfinance`
- Compute both the uncorrected ($\Sigma_{\text{PCA}}$) and JSE-corrected ($\Sigma_{\text{JSE}}$) minimum variance portfolios
- Print summary statistics and complete 400-stock holdings tables to the console

## Results

Selected results from a recent run (May 2 – October 31, 2025):

| Metric | $\Sigma_{\text{PCA}}$ | $\Sigma_{\text{JSE}}$ |
|--------|----------------------:|----------------------:|
| Annualized Forecast Std Dev | 2.38% | 3.37% |
| Annualized Variance via $S$ | 0.00292 | 0.00977 |
| Variance Forecast Ratio (VFR) | 0.1938 | 0.1163 |
| Annualized Expected Excess Return | 17.00% | 10.21% |
| Long exposure | 119.03% | 145.67% |
| Short exposure | 19.03% | 45.67% |
| Eigenvector dispersion std | 0.028355 | 0.017677 |
| Dispersion reduction | — | 37.66% |
| Angle($h$, $h^{\text{JSE}}$) | — | 11.32° |
| Weight correlation | — | 1.0000 |

**Computed estimator parameters:**
- $p = 400$, $n = 26$, rank$(S) = 25$
- $\lambda^2 = 0.18105643$, $\ell^2 = 0.02192565$, $\text{tr}(S) = 0.72919775$
- Leading eigenvalue explains 24.8% of total sample variance
- $c^{\text{JSE}} = 0.6234$ (eigenvector entries shrunk by 37.66% toward the mean)

The JSE-corrected portfolio is more diversified in terms of true risk exposure, even though its forecast standard deviation appears higher. The uncorrected portfolio achieves a lower forecast variance precisely by overweighting directions that are noise-induced artifacts of the biased sample eigenvector — exactly the failure mode that motivates the JSE correction.

## Repository Structure

```
.
├── min_var_portfolio.py        # Main implementation
├── report.pdf                  # Full technical write-up with proofs and figures
├── README.md
└── LICENSE
```

## Theoretical Guarantees

- **Invertibility:** $\Sigma_{\text{PCA}}$ has eigenvalues $\lambda^2 - \ell^2 + (n/p)\ell^2$ (multiplicity 1, in direction $h$) and $(n/p)\ell^2$ (multiplicity $p - 1$). All are strictly positive, so $\Sigma_{\text{PCA}}$ is positive definite.
- **Eigenvector preservation:** The leading eigenvector of $\Sigma_{\text{PCA}}$ is $h$, the leading eigenvector of $S$.
- **Angle reduction (Theorem 1, Goldberg & Kercheval 2023):** $\angle(h^{\text{JSE}}, b) < \angle(h, b)$ almost surely as $p \to \infty$.
- **Optimization bias (Theorem 4, Goldberg & Kercheval 2023):** The true variance of the JSE portfolio converges to zero in the $p \to \infty$ limit.

## References

1. Goldberg, L. R., & Kercheval, A. N. (2023). [James–Stein for the leading eigenvector](https://www.pnas.org/doi/10.1073/pnas.2207046120). *Proceedings of the National Academy of Sciences*, 120(2).
2. Goldberg, L. R., Papanicolaou, A., & Shkolnik, A. (2022). The dispersion bias. *SIAM Journal on Financial Mathematics*, 13, 521–550.
3. Gurdogan, H., & Kercheval, A. (2022). Multi anchor point shrinkage for the sample covariance matrix. *SIAM Journal on Financial Mathematics*, 13, 1112–1143.
4. James, W., & Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium*.
5. Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7, 77–91.
6. Michaud, R. O. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal*, 45, 31–42.

## Author

**Godfred Antwi Koduah**

## License

MIT
