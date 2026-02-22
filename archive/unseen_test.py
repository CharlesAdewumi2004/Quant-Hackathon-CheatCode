import pandas as pd
import numpy as np
import os
from submissions.strategy import generate_signals

def run_unseen_year_test(csv_path="data/SPY.csv"):
    # 1. Load Data
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    
    # 2. Slice for the "Unseen" window (2025 - Feb 2026)
    unseen_data = data["2025-01-01":"2026-02-20"]
    
    if unseen_data.empty:
        print("Error: Your SPY.csv does not contain 2025-2026 data.")
        return

    print(f"--- Running Test on Unseen Data (2025-2026) ---")
    
    # 3. Generate Signals
    signals = generate_signals(unseen_data)
    
    # 4. Calculate Stats
    returns = unseen_data['Close'].pct_change()
    strat_returns = signals.shift(1) * returns
    
    # Simple transaction cost (0.05%)
    trades = signals.diff().fillna(0).abs()
    strat_returns -= (trades * 0.0005)
    
    # Metrics
    total_ret = (1 + strat_returns.fillna(0)).prod() - 1
    mkt_ret = (1 + returns.fillna(0)).prod() - 1
    
    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
    
    print(f"Market Return (Buy & Hold): {mkt_ret:.2%}")
    print(f"Strategy Return:            {total_ret:.2%}")
    print(f"Strategy Sharpe Ratio:      {sharpe:.2f}")
    print(f"Alpha Generated:            {(total_ret - mkt_ret):.2%}")

if __name__ == "__main__":
    run_unseen_year_test()