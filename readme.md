# Vector Autoregression (VAR) Financial Time Series Pipeline

A production-grade Jupyter Notebook implementation for analyzing multi-asset financial dynamics using **Vector Autoregression (VAR)**. 

This project allows quantitative research into the linear interdependencies among various financial assets (stocks, foreign exchange rates, interest rates). It features an editable configuration panel for asset tickers and data sources, stationarity testing, dynamic Granger causality, lag order selection, ergodicity/stability diagnostics, impulse response functions (IRF), forecast error variance decomposition (FEVD), and out-of-sample level forecasting.

---

## 🛠️ Key Pipeline Enhancements

1. **Flexible Configuration (`INSTRUMENTS` & `DATA_SOURCE`)**: Easily swap asset tickers, local CSV column headers, and project display aliases from a single configuration dictionary in Cell 2.
2. **Dual Sourcing**: Load live financial data dynamically via `yfinance` or fallback to local `.csv` files (`DATA_SOURCE = "yfinance"` or `"csv"`).
3. **Robust Stationarity Handling**: Enforces Augmented Dickey-Fuller (ADF) tests to transform non-stationary series $I(1)$ to stationary $I(0)$ via first-differencing.
4. **Modern Granger Causality Format**: Compatible with `statsmodels >= 0.14`, returning clean, structured $F$-statistic and $p$-value summary tables in both causal directions.
5. **Exact Level Reconstruction**: Accurately undiffs linear daily predicted changes and aligns predictions along trading business days using `CustomBusinessDay` and the US Federal Holiday Calendar.

---

## 📚 Core Theory & Mathematical Overview

### 1. Vector Autoregression (VAR)
Unlike univariate autoregressive models ($AR$), a VAR model captures dynamic, multi-directional relationships across multiple time series simultaneously. Each variable in the system is modeled as a linear combination of its own past lags and the past lags of all other variables.

A $VAR(p)$ model with $k$ variables and lag order $p$ is defined as:

$$Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \dots + A_p Y_{t-p} + \varepsilon_t$$

Where:
* $Y_t$: A $(k \times 1)$ vector of time series variables at time $t$.
* $c$: A $(k \times 1)$ vector of constant intercepts.
* $A_i$: $(k \times k)$ coefficient matrices for lag $i$.
* $\varepsilon_t$: A $(k \times 1)$ vector of unobserved white noise error terms with $E(\varepsilon_t) = 0$ and covariance matrix $\Sigma$.

---

### 2. Stationarity & Integration
Standard VAR models require input time series to be **stationary** and **ergodic** $I(0)$. A process is stationary if its mean, variance, and autocovariance remain constant over time.

Financial prices and exchange rates typically follow a random walk $I(1)$ and are non-stationary in levels. To avoid **spurious regressions**, we apply first-differencing:

$$\Delta Y_t = Y_t - Y_{t-1}$$

---

### 3. Granger Causality Test
Granger causality evaluates whether past values of variable $X$ contain unique information that helps predict variable $Y$ beyond what past values of $Y$ alone can provide.

* **Null Hypothesis ($H_0$)**: $X$ does **not** Granger-cause $Y$ (coefficients of lagged $X$ values are jointly zero).
* **Decision Rule**: If $p \text{-value} < 0.05$, reject $H_0$. $X$ statistically Granger-causes $Y$.

---

### 4. Lag Selection Criteria
To choose the optimal lag length $p$, the model balances fit against model complexity using information criteria:

* **Akaike Information Criterion (AIC)**
* **Bayesian Information Criterion (BIC)**
* **Hannan-Quinn Information Criterion (HQIC)**

Lower values indicate a superior balance of model accuracy and parsimony.

---

### 5. Impulse Response Function (IRF) & Variance Decomposition (FEVD)
* **IRF**: Traces the dynamic impact of a one-standard-deviation shock in one asset's error term on all other assets across future time steps.
* **FEVD**: Measures the proportion of forecast error variance for each asset attributable to shocks from itself versus shocks from other assets in the system.

---

## 🚀 Getting Started

### 1. Prerequisites & Dependencies

Ensure Python 3.8+ is installed along with the required standard quantitative libraries:

```bash
pip install numpy pandas matplotlib statsmodels arch yfinance