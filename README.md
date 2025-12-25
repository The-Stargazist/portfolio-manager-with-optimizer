# Institutional-Grade Quant Portfolio Optimizer

An end-to-end quantitative research pipeline that performs stock screening, Mean-Variance Optimization (MVO), Out-of-Sample (OOS) testing, and Monte Carlo stress testing.

## 🚀 The Strategy
The model uses a **"Walk-Forward Validation"** approach:
1.  **Training (2020-2023):** Screens 500+ tickers for the highest Sharpe Ratios while maintaining a correlation ceiling of < 0.5.
2.  **Optimization:** Utilizes the **Ledoit-Wolf Shrinkage** covariance estimate and **L2 Regularization** to find the Max Sharpe weights with institutional constraints (3% min, 17% max).
3.  **Testing (2023-2025):** Validates the "Training" weights against unseen future data to prove the strategy isn't overfitted.

## 📊 Key Performance Metrics (OOS)
| Metric | Value |
| :--- | :--- |
| **Annualized Return** | ~30% |
| **Sharpe Ratio** | 1.5x |
| **Sortino Ratio** | 2.xx |
| **Max Drawdown** | -20% |
| **95% 1-Year VaR** | 7.45% |

## 🛠️ Installation & Usage
1. Clone the repository.
2. Install requirements: `pip install yfinance pandas-portfolio-optimization scikit-learn seaborn`
3. Ensure `sp500.csv` is in the root directory.
4. Run the script: `python master_quant_script.py`

## 📂 Visual Outputs
The script automatically generates a `quant_report_outputs/` folder containing:
- **Performance Dashboard:** Cumulative growth vs. S&P 500 and a full metrics table.
- **PCA Heatmap:** Factor loadings explaining the hidden risk drivers of the 10-stock basket.
- **Monte Carlo Cone:** 10,000 simulated paths showing the 90% confidence interval for future returns.

## 🧠 Risk Decomposition
The use of **PCA (Principal Component Analysis)** allows us to identify "Factor Orthogonality." The model ensures that even if one sector (e.g., Technology) faces a downturn, the idiosyncratic drivers of the remaining assets provide a mathematical "floor," resulting in a highly favorable 1-Year Value at Risk (VaR).
