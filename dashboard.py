import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
from datetime import datetime

# --- 1. CORE LOGIC IMPORT ---
try:
    from submissions.strategy import generate_signals, calculate_technical_indicators
except ImportError:
    st.error("Check 'submissions/strategy.py' imports.")

# --- 2. THEMED STYLING (High Contrast Dark Theme) ---
st.set_page_config(page_title="CheatCode Master Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .stApp { background: #0A0A0A; color: white; }
    
    /* Global Metric Cards */
    .metric-container { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 40px; }
    .metric-card { 
        flex: 1; background: #161616; border: 1px solid #333; 
        border-radius: 12px; padding: 20px; text-align: center;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .metric-card:hover { border-color: #00FF41; transform: translateY(-3px); }
    .metric-value { font-size: 32px; font-weight: 800; color: #00FF41; }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }

    /* Snapshot Metrics (Right Side) */
    .snapshot-card {
        background: #1E1E1E; border-left: 4px solid #00BFFF;
        border-radius: 8px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .snapshot-val { font-size: 24px; font-weight: 700; color: #FFFFFF; }
    .snapshot-lab { font-size: 10px; color: #00BFFF; text-transform: uppercase; font-weight: bold; }

    /* Performance Metric Grid */
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-top: 30px;
    }
    .perf-item {
        background: #1A1A1A;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #222;
        text-align: left;
    }
    .perf-val { color: #00FF41; font-size: 20px; font-weight: 700; }
    .perf-lab { color: #666; font-size: 10px; text-transform: uppercase; }

    .xai-container { margin-left: 30px; }
    .section-spacer { margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & BACKTEST ENGINE ---
@st.cache_data
def get_processed_data(ticker):
    file_path = f"data/{ticker}.csv"
    if not os.path.exists(file_path): return None
    
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = calculate_technical_indicators(data)
    df['Signal'] = generate_signals(data)
    
    # Financials (Transaction Costs @ 5bps)
    TRANSACTION_COST = 0.0005
    df['Returns'] = df['Close'].pct_change()
    df['Net_Returns'] = (df['Signal'].shift(1) * df['Returns']) - (df['Signal'].diff().abs().fillna(0) * TRANSACTION_COST)
    df['Cumulative_Equity'] = 10000 * (1 + df['Net_Returns'].fillna(0)).cumprod()
    
    # Baseline for Comparison
    df['Baseline_Equity'] = 10000 * (1 + df['Returns'].fillna(0)).cumprod()
    return df

# --- 4. DASHBOARD EXECUTION ---
ticker = st.sidebar.selectbox("📂 Select Asset", ["SPY", "QQQ"])
df = get_processed_data(ticker)

if df is not None:
    st.title(f"⚡ CheatCode Master Terminal | {ticker}")
    st.markdown("---")

    # --- TOP ROW: MASTER METRICS ---
    total_ret = (df['Cumulative_Equity'].iloc[-1] / 10000) - 1
    base_ret = (df['Baseline_Equity'].iloc[-1] / 10000) - 1
    sharpe = (df['Net_Returns'].mean() / df['Net_Returns'].std()) * np.sqrt(252)
    max_dd = ((df['Cumulative_Equity'] - df['Cumulative_Equity'].cummax()) / df['Cumulative_Equity'].cummax()).min()

    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card"><div class="metric-value">{sharpe:.2f}</div><div class="metric-label">Master Sharpe</div></div>
            <div class="metric-card"><div class="metric-value">{total_ret:.1%}</div><div class="metric-label">Strategy Return</div></div>
            <div class="metric-card"><div class="metric-value">{base_ret:.1%}</div><div class="metric-label">Benchmark Return</div></div>
            <div class="metric-card"><div class="metric-value">{max_dd:.1%}</div><div class="metric-label">Max Drawdown</div></div>
        </div>
        """, unsafe_allow_html=True)

    # --- FULL EQUITY GROWTH CHART ---
    st.subheader("📈 Total Strategy vs. Benchmark (2015-2026)")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Equity'], fill='tozeroy', line_color='#00FF41', name="CheatCode Strategy"))
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Baseline_Equity'], line_color='#888888', line=dict(dash='dash'), name="Buy & Hold Baseline"))
    fig_cum.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cum, use_container_width=True)

    # --- YEARLY SNAPSHOT & SNAPSHOT METRICS ---
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("🔍 Yearly Alpha Drill-Down")
    col_chart, col_metrics = st.columns([3, 1])
    
    with col_metrics:
        selected_year = st.selectbox("📅 Select Year", sorted(df.index.year.unique(), reverse=True))
        y_df = df[df.index.year == selected_year].copy()
        
        # Re-base for the year
        y_strat_g = (1 + y_df['Net_Returns'].fillna(0)).cumprod() * 10000
        y_base_g = (1 + y_df['Returns'].fillna(0)).cumprod() * 10000
        
        y_ret = (y_strat_g.iloc[-1] / 10000) - 1
        y_base_ret = (y_base_g.iloc[-1] / 10000) - 1
        y_alpha = y_ret - y_base_ret
        y_sharpe = (y_df['Net_Returns'].mean() / y_df['Net_Returns'].std()) * np.sqrt(252) if y_df['Net_Returns'].std() != 0 else 0
        
        st.markdown(f"""
            <div class="snapshot-card"><div class="snapshot-lab">Yearly Alpha</div><div class="snapshot-val">{y_alpha:+.1%}</div></div>
            <div class="snapshot-card"><div class="snapshot-lab">Yearly Sharpe</div><div class="snapshot-val">{y_sharpe:.2f}</div></div>
            <div class="snapshot-card"><div class="snapshot-lab">Winning Days</div><div class="snapshot-val">{(y_df['Net_Returns'] > 0).sum()}</div></div>
            <div class="snapshot-card"><div class="snapshot-lab">Trading Vol</div><div class="snapshot-val">{y_df['Net_Returns'].std()*np.sqrt(252):.1%}</div></div>
        """, unsafe_allow_html=True)
    
    with col_chart:
        fig_year = go.Figure()
        fig_year.add_trace(go.Scatter(x=y_df.index, y=y_strat_g, fill='tozeroy', line_color='#00BFFF', name="Strategy"))
        fig_year.add_trace(go.Scatter(x=y_df.index, y=y_base_g, line_color='#888888', line=dict(dash='dash'), name="Baseline"))
        fig_year.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_year, use_container_width=True)

    # --- HIGH-FIDELITY TRADING CHART ---
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Trade Execution Detail (Unseen 2025-2026)")
    unseen = df.loc['2025-01-01':]
    fig_trade = go.Figure()
    fig_trade.add_trace(go.Candlestick(x=unseen.index, open=unseen['Open'], high=unseen['High'], low=unseen['Low'], close=unseen['Close'], name="Candlesticks", increasing_line_color='#00FF41', decreasing_line_color='#FF3131'))
    
    buys = unseen[unseen['Signal'] == 1]
    sells = unseen[unseen['Signal'] == -1]
    fig_trade.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.985, mode='markers', name='BUY', marker=dict(symbol='triangle-up', size=14, color='#00FF41')))
    fig_trade.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.015, mode='markers', name='SELL', marker=dict(symbol='triangle-down', size=14, color='#FF3131')))
    
    fig_trade.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_trade, use_container_width=True)

    # --- PERFORMANCE MATRIX & XAI ---
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    col_table, col_xai = st.columns([1, 1])
    
    with col_table:
        st.subheader("📊 Annual Performance Matrix")
        df['Year'] = df.index.year
        table_data = []
        for year, group in df.groupby('Year'):
            y_ret_tbl = (group['Cumulative_Equity'].iloc[-1] / group['Cumulative_Equity'].iloc[0]) - 1
            y_sha_tbl = (group['Net_Returns'].mean() / group['Net_Returns'].std()) * np.sqrt(252) if group['Net_Returns'].std() != 0 else 0
            table_data.append({"Year": year, "Return": f"{y_ret_tbl:.1%}", "Sharpe": round(y_sha_tbl, 2)})
        st.table(pd.DataFrame(table_data).set_index('Year'))

    with col_xai:
        st.markdown('<div class="xai-container">', unsafe_allow_html=True)
        st.subheader("🧠 XAI: Model Decision Logic")
        MODEL_PATH = "models/spy_xgb_model.joblib"
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            features = ["Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS", "Resistance", "Support"]
            if hasattr(model, 'feature_importances_'):
                feat_df = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=True)
                fig_xai = go.Figure(go.Bar(x=feat_df['Weight'], y=feat_df['Feature'], orientation='h', marker_color='#00FF41'))
                fig_xai.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_xai, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ADVANCED QUANT METRIC GRID (New Features) ---
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("🔬 Advanced Institutional Analytics")
    
    # Calculate sophisticated metrics
    downside_returns = df['Net_Returns'][df['Net_Returns'] < 0]
    sortino = (df['Net_Returns'].mean() / downside_returns.std()) * np.sqrt(252) if downside_returns.std() != 0 else 0
    var_95 = np.percentile(df['Net_Returns'].fillna(0), 5)
    win_days = (df['Net_Returns'] > 0).sum()
    loss_days = (df['Net_Returns'] < 0).sum()
    profit_factor = abs(df[df['Net_Returns'] > 0]['Net_Returns'].sum() / df[df['Net_Returns'] < 0]['Net_Returns'].sum())
    expectancy = ( (win_days/len(df)) * df[df['Net_Returns'] > 0]['Net_Returns'].mean() ) + ( (loss_days/len(df)) * df[df['Net_Returns'] < 0]['Net_Returns'].mean() )

    st.markdown(f"""
        <div class="perf-grid">
            <div class="perf-item"><div class="perf-lab">Sortino Ratio</div><div class="perf-val">{sortino:.2f}</div></div>
            <div class="perf-item"><div class="perf-lab">Profit Factor</div><div class="perf-val">{profit_factor:.2f}</div></div>
            <div class="perf-item"><div class="perf-lab">Daily VaR (95%)</div><div class="perf-val">{var_95:.2%}</div></div>
            <div class="perf-item"><div class="perf-lab">Expectancy</div><div class="perf-val">{expectancy:.4f}</div></div>
            <div class="perf-item"><div class="perf-lab">Total Trades</div><div class="perf-val">{(df['Signal'].diff() != 0).sum()}</div></div>
            <div class="perf-item"><div class="perf-lab">Positive Days</div><div class="perf-val">{win_days}</div></div>
            <div class="perf-item"><div class="perf-lab">Negative Days</div><div class="perf-val">{loss_days}</div></div>
            <div class="perf-item"><div class="perf-lab">Recovery Factor</div><div class="perf-val">{abs(total_ret/max_dd):.2f}</div></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"CheatCode 2.0 | High-Performance Quant Stack | Portfolio Started: $10,000 | Current: ${df['Cumulative_Equity'].iloc[-1]:,.2f}")

else:
    st.error("Missing Data. Check data/SPY.csv.")