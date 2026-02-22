import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from submissions.strategy import generate_signals

def run_visual_unseen_test(ticker="SPY"):
    print(f"--- Fetching Fresh {ticker} Data (Jan 2025 - Feb 2026) ---")
    
    # 1. Download fresh data
    data = yf.download(ticker, start="2025-01-01", end="2026-02-22")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if data.empty:
        print("Failed to download data.")
        return

    # 2. Generate Signals and Returns
    signals = generate_signals(data)
    mkt_returns = data['Close'].pct_change()
    
    # Shift signals by 1 day for execution realism
    # Subtract 0.05% per trade fee
    strat_returns = (signals.shift(1) * mkt_returns).fillna(0)
    trades = signals.diff().fillna(0).abs()
    strat_returns -= (trades * 0.0005)
    
    # 3. Calculate Cumulative Growth (Starting at $10,000 for this window)
    initial_value = 10000
    data['Baseline_Value'] = initial_value * (1 + mkt_returns.fillna(0)).cumprod()
    data['Strategy_Value'] = initial_value * (1 + strat_returns).cumprod()
    
    # 4. Create the Professional Plot
    plt.figure(figsize=(14, 7))
    
    # Plotting Baseline vs Strategy
    plt.plot(data.index, data['Baseline_Value'], label=f'{ticker} Baseline', color='gray', linestyle='--', alpha=0.6)
    plt.plot(data.index, data['Strategy_Value'], label='XGBoost Strategy', color='navy', linewidth=2.5)
    
    # Highlight the Alpha Gap
    plt.fill_between(data.index, data['Baseline_Value'], data['Strategy_Value'], 
                     where=(data['Strategy_Value'] >= data['Baseline_Value']), 
                     facecolor='green', alpha=0.1, label='Alpha Generation')

    # Formatting
    plt.title(f'Unseen Data Performance: {ticker} (2025 - 2026)', fontsize=16, fontweight='bold')
    plt.ylabel('Account Value ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(loc='upper left', fontsize=11)
    
    # Final Value Annotations
    final_mkt = data['Baseline_Value'].iloc[-1]
    final_strat = data['Strategy_Value'].iloc[-1]
    plt.annotate(f'${final_strat:,.0f}', (data.index[-1], final_strat), xytext=(5, 5), 
                 textcoords="offset points", color='navy', fontweight='bold')
    plt.annotate(f'${final_mkt:,.0f}', (data.index[-1], final_mkt), xytext=(5, -15), 
                 textcoords="offset points", color='gray')

    plt.tight_layout()
    
    # 5. Save and Print
    os.makedirs('plots', exist_ok=True)
    filename = f'plots/{ticker}_unseen_performance.png'
    plt.savefig(filename)
    
    total_ret = (final_strat / initial_value) - 1
    mkt_ret = (final_mkt / initial_value) - 1
    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
    
    print(f"\n--- {ticker} UNSEEN RESULTS ---")
    print(f"Final Strategy Value: ${final_strat:,.2f}")
    print(f"Strategy Return:      {total_ret:.2%}")
    print(f"Market Return:        {mkt_ret:.2%}")
    print(f"Alpha Produced:       {(total_ret - mkt_ret):.2%}")
    print(f"Strategy Sharpe:      {sharpe:.2f}")
    print(f"Visual Plot Saved:    {filename}")

if __name__ == "__main__":
    # You can change these to test different tickers
    run_visual_unseen_test("SPY")
    run_visual_unseen_test("QQQ")