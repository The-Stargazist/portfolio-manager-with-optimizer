#Portfolio Optimizer incorporating risk assessment 

An end-to-end quantitative researchpipeline that performs stock screening, Mean-Variance Optimization (MVO), Out-of-Sample (OOS) testing, and Monte Carlo stress testing.

## 🎖️ Results
I was able to find a portfolio that gave me alpha of ~7% compared to the S&P500 while only taking on a bit more risk which as indicted by the good sortino ratio was also a net positive in terms of volatility. This implies that the portfolio did not just outperform the benchmark due to its momentum, but due to other factors aswell. Also by choosing stocks which are not highly correlated the portfolio is safeguarded by diversification against single sector crashes which otherwise might tank the entire investment

## 🚀 The Strategy
The model uses a **"Walk-Forward Validation"** approach:
1.  **Training (2020-2023):** Screens the S&P500 tickers for the highest Sharpe Ratios while maintaining a correlation ceiling of < 0.5.
2.  **Optimization:** Utilizes the **Ledoit-Wolf Shrinkage** covariance estimate and **L2 Regularization** to find the Max Sharpe weights with institutional constraints (3% min, 17% max).
3.  **Testing (2023-2025):** Validates the "Training" weights against unseen future data to prove the strategy isn't overfitted.

## 📊 Key Performance Metrics (OOS)
| Metric | Value |
| :--- | :--- |
| **Annualized Return** | ~32% |
| **Sharpe Ratio** | 1.6 |
| **Sortino Ratio** | 2.19 |
| **Max Drawdown** | -20% |
| **95% 1-Year VaR** | 7.45% |

## 🛠️ Installation & Usage
1. Clone the repository.
2. Install requirements: `pip install -r requirements-testing.txt`
3. Ensure `sp500.csv` is in the root directory.
4. Run the script: `python portfolio_runner.py`

## 📂 Outputs
The script automatically generates a `quant_report_outputs/` folder containing:
- **Performance Dashboard:** Cumulative growth vs. S&P 500 and a full metrics table.
- **PCA Heatmap:** Factor loadings explaining the hidden risk drivers of the 10-stock basket.
- **Monte Carlo Cone:** 10,000 simulated paths showing the 90% confidence interval for future returns.

## 🧠 Risk Decomposition
The use of **PCA (Principal Component Analysis)** allows us to identify "Factor Orthogonality." The model ensures that even if one sector (e.g., Technology) faces a downturn, the idiosyncratic drivers of the remaining assets provide a mathematical "floor," resulting in a highly favorable 1-Year Value at Risk (VaR).


