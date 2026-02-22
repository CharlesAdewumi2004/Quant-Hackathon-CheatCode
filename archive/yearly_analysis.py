import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from submissions.strategy import generate_signals

def visualize_yoy_performance(csv_path="data/SPY.csv"):
    # 1. Load Data
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    
    # 2. Generate Signals and Returns
    print("Generating signals and calculating returns...")
    signals = generate_signals(data)
    data['Market_Returns'] = data['Close'].pct_change()
    
    # Strategy Returns (1-day shift for execution, 0.05% fee per trade)
    data['Strategy_Returns'] = signals.shift(1) * data['Market_Returns']
    trades = signals.diff().fillna(0).abs()
    data['Strategy_Returns'] = data['Strategy_Returns'] - (trades * 0.0005)

    # 3. Aggregate by Year
    years = data.index.year.unique()
    yoy_data = []

    for year in years:
        year_slice = data[data.index.year == year]
        mkt_ret = (1 + year_slice['Market_Returns']).prod() - 1
        strat_ret = (1 + year_slice['Strategy_Returns']).prod() - 1
        yoy_data.append({
            'Year': str(year),
            'Market': mkt_ret * 100,
            'Strategy': strat_ret * 100
        })

    df_plot = pd.DataFrame(yoy_data)

    # 4. Create Visualizations
    os.makedirs('plots', exist_ok=True)
    
    # Chart 1: Side-by-Side Yearly Returns
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df_plot['Year']))
    width = 0.35

    plt.bar(x - width/2, df_plot['Market'], width, label='Market (Baseline)', color='gray', alpha=0.6)
    plt.bar(x + width/2, df_plot['Strategy'], width, label='Strategy (XGBoost)', color='navy')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Annual Return (%)')
    plt.title('Year-over-Year Performance: Strategy vs. Market Baseline', fontsize=16)
    plt.xticks(x, df_plot['Year'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on bars
    for i in range(len(df_plot)):
        plt.text(i - width/2, df_plot['Market'][i], f"{df_plot['Market'][i]:.1f}%", 
                 ha='center', va='bottom' if df_plot['Market'][i] > 0 else 'top', fontsize=9)
        plt.text(i + width/2, df_plot['Strategy'][i], f"{df_plot['Strategy'][i]:.1f}%", 
                 ha='center', va='bottom' if df_plot['Strategy'][i] > 0 else 'top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/yoy_visual_comparison.png')
    
    # Chart 2: Cumulative Alpha Development
    plt.figure(figsize=(14, 5))
    data['Alpha_Daily'] = data['Strategy_Returns'] - data['Market_Returns']
    data['Cumulative_Alpha'] = data['Alpha_Daily'].cumsum() * 100
    
    plt.plot(data.index, data['Cumulative_Alpha'], color='teal', linewidth=2)
    plt.fill_between(data.index, 0, data['Cumulative_Alpha'], color='teal', alpha=0.1)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Cumulative Alpha (Total Outperformance Over Time)', fontsize=14)
    plt.ylabel('Alpha (%)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/cumulative_alpha.png')
    print("Visuals saved in the 'plots/' folder.")

if __name__ == "__main__":
    visualize_yoy_performance()