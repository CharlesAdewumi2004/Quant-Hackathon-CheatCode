import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from submissions.strategy import generate_signals

def run_visual_backtest(csv_path="data/SPY.csv"):
    # 1. Load Data
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    
    # 2. Generate Signals
    print("Generating signals from strategy.py...")
    signals = generate_signals(data)
    
    # 3. Calculate Returns
    # Market Returns (Buy & Hold)
    data['Market_Returns'] = data['Close'].pct_change()
    data['Market_Cum'] = (1 + data['Market_Returns']).cumprod()
    
    # Strategy Returns (Shifted by 1 day to be realistic)
    # We apply a 0.05% fee per trade
    data['Strategy_Returns'] = signals.shift(1) * data['Market_Returns']
    trades = signals.diff().fillna(0).abs()
    data['Strategy_Returns'] = data['Strategy_Returns'] - (trades * 0.0005)
    data['Strategy_Cum'] = (1 + data['Strategy_Returns']).cumprod()
    
    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Plot: Equity Curves
    ax1.plot(data['Market_Cum'], label='SPY Baseline (Buy & Hold)', color='gray', alpha=0.5, linestyle='--')
    ax1.plot(data['Strategy_Cum'], label='Ensemble XGBoost Strategy', color='navy', linewidth=2)
    ax1.set_title('Equity Curve: Strategy vs. Market', fontsize=16)
    ax1.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Bottom Plot: Trading Actions (Signals)
    # Green = Long, Red = Short, White/Empty = Cash
    colors = ['green' if x == 1 else 'red' if x == -1 else 'white' for x in signals]
    ax2.fill_between(data.index, 0, signals, color='teal', alpha=0.3)
    ax2.step(data.index, signals, where='post', color='black', linewidth=0.5)
    
    ax2.set_title('Trading Exposure (Long / Cash / Short)', fontsize=14)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('Signal', fontsize=12)
    
    # Adding Highlight for Drawdown protection
    # We mark areas where the Strategy stayed in Cash while the Market fell
    ax1.fill_between(data.index, data['Market_Cum'], data['Strategy_Cum'], 
                    where=(data['Strategy_Cum'] > data['Market_Cum']), 
                    color='green', alpha=0.1, label='Outperformance')

    plt.tight_layout()
    plt.savefig('plots/backtest_comparison.png')
    plt.show()

if __name__ == "__main__":
    run_visual_backtest()