import os
import warnings
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Constants
OUTPUT_FOLDER = "quant_report_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. DATA PREP ---
df_input = pd.read_csv('sp500.csv') 
raw_tickers = df_input.iloc[:, 0].dropna().astype(str).tolist()
all_tickers = [t.replace('.', '-') for t in raw_tickers][:500]

print("Downloading data")
data_raw = yf.download(all_tickers, period="6y")['Close']
spy_raw = yf.download("SPY", period="6y")['Close']

# Train / Test windows
train_start, train_end = "2020-01-01", "2022-12-31"
test_start, test_end = "2023-01-01", "2025-12-25"

# Veteran filter: ensure sufficient history
min_days = int(252 * 2.8)
data_veterans = data_raw.dropna(thresh=min_days, axis=1)

# --- 2. TRAINING PHASE (2020 - 2022) ---
train_prices = data_veterans.loc[train_start:train_end]
train_returns = train_prices.pct_change().dropna(how='all').fillna(0)
spy_train_rets = spy_raw.loc[train_start:train_end].pct_change().dropna()

# Selection: top Sharpe then de-correlate to 10 names
sharpes = (train_returns.mean() * 252) / (train_returns.std() * np.sqrt(252))
top_50 = sharpes.sort_values(ascending=False).head(50).index.tolist()

final_selection = []
for stock in top_50:
    if not final_selection:
        final_selection.append(stock)
    else:
        corrs = train_returns[final_selection].corrwith(train_returns[stock])
        if all(corrs.fillna(1) < 0.5):
            final_selection.append(stock)
    if len(final_selection) == 10:
        break

# Optimization (standard bounds + small L2 penalty)
mu_train = expected_returns.mean_historical_return(train_prices[final_selection])
S_train = risk_models.CovarianceShrinkage(train_prices[final_selection]).ledoit_wolf()
ef = EfficientFrontier(mu_train, S_train, weight_bounds=(0.03, 0.17))
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
ef.max_sharpe()
cleaned_weights = ef.clean_weights()
weights_series = pd.Series(cleaned_weights)

# --- 3. TESTING PHASE ---
test_prices = data_veterans.loc[test_start:test_end][final_selection]
test_returns = test_prices.pct_change().dropna(how='all').fillna(0)
spy_test_rets = spy_raw.loc[test_start:test_end].pct_change().dropna()

port_train_rets = (train_returns[final_selection] * weights_series).sum(axis=1)
port_test_rets = (test_returns * weights_series).sum(axis=1)


def get_stats(rets):
    if isinstance(rets, pd.DataFrame):
        rets = rets.iloc[:, 0]
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    cum_rets = (1 + rets).cumprod()
    rolling_max = cum_rets.cummax()
    drawdowns = (cum_rets - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    downside_rets = rets[rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol != 0 else 0
    return float(ann_ret), float(ann_vol), float(sharpe), float(max_drawdown), float(sortino)


tr_p = get_stats(port_train_rets)
tr_s = get_stats(spy_train_rets)
te_p = get_stats(port_test_rets)
te_s = get_stats(spy_test_rets)


# --- 4. DASHBOARD (Performance + Allocation Pie + Stats Table) ---
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
fig.suptitle(f"Walk-Forward Analysis: {', '.join(final_selection)}", fontsize=20, fontweight='bold')

# Top: cumulative performance
ax_perf = fig.add_subplot(gs[0, :])
common_idx = port_test_rets.index.intersection(spy_test_rets.index)
cum_port = (1 + port_test_rets.loc[common_idx]).cumprod()
spy_vals = spy_test_rets.loc[common_idx]
if isinstance(spy_vals, pd.DataFrame):
    spy_vals = spy_vals.iloc[:, 0]
cum_spy = (1 + spy_vals).cumprod()

ax_perf.plot(cum_port, label="Quant Portfolio (Test)", color='darkgreen', lw=3)
ax_perf.plot(cum_spy, label="S&P 500 (SPY)", color='gray', ls='--', alpha=0.8)
ax_perf.set_title("Out-of-Sample Growth (Test Period)")
ax_perf.set_ylabel("Growth of $1")
ax_perf.legend()
ax_perf.grid(alpha=0.3)

# Bottom left: allocation pie
ax_pie = fig.add_subplot(gs[1, 0])
active_w = weights_series[weights_series > 0]
if not active_w.empty:
    colors = sns.color_palette("viridis", len(active_w))
    ax_pie.pie(active_w, labels=active_w.index, autopct="%1.1f%%", startangle=140, colors=colors)
    ax_pie.set_title("Weight Allocation")
else:
    ax_pie.text(0.5, 0.5, "No active weights", ha='center', va='center')
    ax_pie.set_axis_off()

# Bottom right: stats table
ax_table = fig.add_subplot(gs[1, 1])
ax_table.axis('off')
table_data = [
    ["Metric", "Port (IS)", "SPY (IS)", "Port (OS)", "SPY (OS)"],
    ["Ann. Return", f"{tr_p[0]:.2%}", f"{tr_s[0]:.2%}", f"{te_p[0]:.2%}", f"{te_s[0]:.2%}"],
    ["Volatility", f"{tr_p[1]:.2%}", f"{tr_s[1]:.2%}", f"{te_p[1]:.2%}", f"{te_s[1]:.2%}"],
    ["Sharpe", f"{tr_p[2]:.2f}", f"{tr_s[2]:.2f}", f"{te_p[2]:.2f}", f"{te_s[2]:.2f}"],
    ["Max DD", f"{tr_p[3]:.2%}", f"{tr_s[3]:.2%}", f"{te_p[3]:.2%}", f"{te_s[3]:.2%}"],
    ["Sortino", f"{tr_p[4]:.2f}", f"{tr_s[4]:.2f}", f"{te_p[4]:.2f}", f"{te_s[4]:.2f}"]
]
the_table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.18]*5)
the_table.auto_set_font_size(False)
the_table.set_fontsize(11)
the_table.scale(1.2, 2.2)

plt.savefig(os.path.join(OUTPUT_FOLDER, "01_walkforward_dashboard.png"))
plt.show()


# --- 5. PCA DEEP DIVE ---
pca = PCA(n_components=3)
pca.fit(test_returns)
loadings = pd.DataFrame(pca.components_.T, columns=['Market', 'Industry', 'Specific'], index=final_selection)
print("\n--- PCA Factor Loadings ---")
print(loadings)

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='RdYlGn', center=0)
plt.title("PCA Risk Attribution")
plt.savefig(os.path.join(OUTPUT_FOLDER, "02_pca_heatmap.png"))
plt.show()


# --- 6. MONTE CARLO SIMULATION ---
print("\nRunning 10,000 Monte Carlo simulations...")
n_sims = 10000
n_days = 252
current_weights = np.array(list(cleaned_weights.values()))
mu_mc = test_returns.mean()
cov_mc = test_returns.cov()
sim_returns = np.random.multivariate_normal(mu_mc, cov_mc, (n_days, n_sims))
sim_port_daily = np.dot(sim_returns, current_weights)
sim_cum_growth = np.cumprod(1 + sim_port_daily, axis=0)

plt.figure(figsize=(14, 7))
plt.plot(sim_cum_growth[:, :100], color='gray', alpha=0.1)
plt.plot(np.median(sim_cum_growth, axis=1), color='blue', lw=3, label='Median')
plt.fill_between(range(n_days), np.percentile(sim_cum_growth, 5, axis=1), np.percentile(sim_cum_growth, 95, axis=1), color='blue', alpha=0.2)
plt.axhline(1, color='red', ls='--')
plt.title(f"Monte Carlo: {len(sim_cum_growth)} paths for next year")
plt.xlabel("Days")
plt.ylabel("Portfolio Value ($1 start)")
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "03_monte_carlo.png"))
plt.show()

final_results = sim_cum_growth[-1, :]
var_95 = (1 - np.percentile(final_results, 5)) * 100
print(f"95% Confidence 1-Year VaR: {var_95:.2f}%")