import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from submissions.strategy import generate_signals

def run_actual_account_progression(csv_path="data/SPY.csv", initial_capital=10000):
    # 1. Load the actual data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    
    # 2. Generate signals using your current strategy.py
    print("Generating signals from your model...")
    signals = generate_signals(data)
    
    # 3. Calculate Daily Returns
    # Market (Buy and Hold)
    data['Market_Daily_Return'] = data['Close'].pct_change()
    
    # Strategy (Shifted signals to avoid look-ahead bias)
    # Applying a 0.05% transaction cost for every signal change
    data['Strategy_Daily_Return'] = signals.shift(1) * data['Market_Daily_Return']
    trades = signals.diff().fillna(0).abs()
    data['Strategy_Daily_Return'] -= (trades * 0.0005)
    
    # 4. Calculate Cumulative Account Value
    # Using .cumprod() ensures we are seeing the actual compounding effect
    data['Baseline_Value'] = initial_capital * (1 + data['Market_Daily_Return'].fillna(0)).cumprod()
    data['Strategy_Value'] = initial_capital * (1 + data['Strategy_Daily_Return'].fillna(0)).cumprod()
    
    # 5. Plotting the Real Results
    plt.figure(figsize=(14, 8))
    
    plt.plot(data.index, data['Baseline_Value'], label='SPY Baseline (Market)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(data.index, data['Strategy_Value'], label='XGBoost Strategy Account Value', color='navy', linewidth=2.5)
    
    # Professional Styling
    plt.title(f'Real Account Growth: Strategy vs. Baseline ($10k Start)', fontsize=16, fontweight='bold')
    plt.ylabel('Account Value ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    
    # Log scale is vital for showing the "Bull-Hugger" vs "Spike" phases correctly
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12, loc='upper left')

    # Annotate final values on the chart
    final_date = data.index[-1]
    final_mkt = data['Baseline_Value'].iloc[-1]
    final_strat = data['Strategy_Value'].iloc[-1]
    
    plt.annotate(f'${final_mkt:,.2f}', (final_date, final_mkt), xytext=(10, 0), 
                 textcoords="offset points", color='gray', fontweight='bold')
    plt.annotate(f'${final_strat:,.2f}', (final_date, final_strat), xytext=(10, 10), 
                 textcoords="offset points", color='navy', fontweight='bold')

    plt.tight_layout()
    
    # Save output
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/actual_account_growth.png')
    
    # Save a YoY table based on actual model data
    yoy_summary = data.resample('YE')[['Baseline_Value', 'Strategy_Value']].last()
    yoy_summary.to_csv('actual_yoy_dollars.csv')
    
    print("\n--- Summary Results ---")
    print(f"Final Baseline Value: ${final_mkt:,.22f}")
    print(f"Final Strategy Value: ${final_strat:,.2f}")
    print(f"Total Alpha Generated: ${final_strat - final_mkt:,.2f}")
    print("\nVisual saved to plots/actual_account_growth.png")

if __name__ == "__main__":
    run_actual_account_progression()