import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import importlib.util
import tempfile
import joblib

STARTING_BALANCE = 1000.0
TRANSACTION_COST = 0.0005
TRADING_DAYS = 252

# --- THEMED STYLING (High Contrast Dark Theme) ---
st.set_page_config(page_title="Charlie's Kids Strategy", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .stApp { background: #0A0A0A; color: white; }
    
    .metric-container { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 40px; }
    .metric-card { 
        flex: 1; background: #161616; border: 1px solid #333; 
        border-radius: 12px; padding: 20px; text-align: center;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .metric-card:hover { border-color: #00FF41; transform: translateY(-3px); }
    .metric-value { font-size: 32px; font-weight: 800; color: #00FF41; }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }

    .snapshot-card {
        background: #1E1E1E; border-left: 4px solid #00BFFF;
        border-radius: 8px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .snapshot-val { font-size: 24px; font-weight: 700; color: #FFFFFF; }
    .snapshot-lab { font-size: 10px; color: #00BFFF; text-transform: uppercase; font-weight: bold; }

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

    .data-badge {
        display: inline-block;
        background: #161616;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 8px 18px;
        margin-bottom: 20px;
        font-size: 13px;
        color: #00BFFF;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .data-badge span { color: #00FF41; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA LOADING ====================

def get_available_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

st.sidebar.header("📂 Data Source")
datasets = get_available_datasets()
if not datasets:
    st.sidebar.error("No CSV files found in the 'data/' directory.")
    selected_dataset = None
else:
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)

@st.cache_data
def get_tickers_from_dataset(filename):
    try:
        filepath = os.path.join("data", filename)
        df_columns = pd.read_csv(filepath, nrows=0).columns
        required_base = {"Open", "High", "Low", "Close"}
        if required_base.issubset(set(df_columns)):
            return []
        tickers = set()
        for col in df_columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                tickers.add(parts[1])
        return sorted(list(tickers))
    except:
        return []

selected_ticker = None
if selected_dataset:
    available_tickers = get_tickers_from_dataset(selected_dataset)
    if available_tickers:
        selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)

@st.cache_data
def load_data(filename, ticker):
    if not filename:
        return pd.DataFrame(), ""
    filepath = os.path.join("data", filename)
    try:
        df = pd.read_csv(filepath)
        
        #handle CSVs with unnamed integer index columns
        unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        #find and set the date column as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        
        df.index.name = "Date"
        required = {"Open", "High", "Low", "Close"}

        if required.issubset(set(df.columns)):
            df = df.ffill().dropna()
            return df.sort_index(), f"{filename}"

        new_columns = []
        for col in df.columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                ohlcv, tk = parts
                new_columns.append((tk, ohlcv))
            else:
                new_columns.append((col, ""))
        df.columns = pd.MultiIndex.from_tuples(new_columns, names=["Ticker", "OHLCV"])
        df = df.ffill().dropna()

        if ticker and ticker in df.columns.get_level_values(0):
            single = df[ticker].copy()
            existing_cols = [c for c in required if c in single.columns]
            single = single[existing_cols]
            single.columns.name = None
            return single.sort_index(), f"{filename} (Ticker: {ticker})"
        else:
            st.error(f"Ticker {ticker} not found in the dataset.")
            return pd.DataFrame(), ""
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), ""

df, data_source = load_data(selected_dataset, selected_ticker)

if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Strategy")
    uploaded_file = st.sidebar.file_uploader("Upload your strategy file (.py)", type=["py"])

    if st.sidebar.button("Run Model", type="primary") and uploaded_file is not None:
        with st.spinner("Running strategy..."):
            try:
                os.makedirs("submissions", exist_ok=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="submissions") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                spec = importlib.util.spec_from_file_location("dynamic_strategy", tmp_path)
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)

                if hasattr(strategy_module, "generate_signals"):
                    signals = strategy_module.generate_signals(df)
                    df['Signal'] = signals
                else:
                    st.error("The uploaded file must contain: `def generate_signals(data: pd.DataFrame) -> pd.Series:`")
                    df['Signal'] = 0

                os.remove(tmp_path)
            except Exception as e:
                st.error(f"Error executing model: {str(e)}")
                df['Signal'] = 0
    else:
        if 'Signal' not in df.columns:
            df['Signal'] = 0

    has_signals = 'Signal' in df.columns and any(df['Signal'] != 0)


    # ==================== TITLE ====================
    
    st.title("Charlie's Kids Strategy")
    st.markdown(f'<div class="data-badge">Dataset: <span>{data_source}</span></div>', unsafe_allow_html=True)
    st.markdown("---")


    # ==================== BACKTEST & METRICS ====================

    if has_signals:
        df['Returns'] = df['Close'].pct_change()
        df['Net_Returns'] = (df['Signal'].shift(1) * df['Returns']) - (df['Signal'].diff().abs().fillna(0) * TRANSACTION_COST)
        df['Cumulative_Equity'] = STARTING_BALANCE * (1 + df['Net_Returns'].fillna(0)).cumprod()
        df['Baseline_Equity'] = STARTING_BALANCE * (1 + df['Returns'].fillna(0)).cumprod()

        total_ret = (df['Cumulative_Equity'].iloc[-1] / STARTING_BALANCE) - 1
        base_ret = (df['Baseline_Equity'].iloc[-1] / STARTING_BALANCE) - 1
        sharpe = (df['Net_Returns'].mean() / df['Net_Returns'].std()) * np.sqrt(TRADING_DAYS) if df['Net_Returns'].std() != 0 else 0
        max_dd = ((df['Cumulative_Equity'] - df['Cumulative_Equity'].cummax()) / df['Cumulative_Equity'].cummax()).min()
        calmar = (total_ret / abs(max_dd)) if max_dd < 0 else 0.0
        win_rate = (df['Net_Returns'] > 0).mean()

        # --- TOP ROW: MASTER METRICS ---
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card"><div class="metric-value">{sharpe:.2f}</div><div class="metric-label">Master Sharpe</div></div>
                <div class="metric-card"><div class="metric-value">{total_ret:.1%}</div><div class="metric-label">Strategy Return</div></div>
                <div class="metric-card"><div class="metric-value">{base_ret:.1%}</div><div class="metric-label">Benchmark Return</div></div>
                <div class="metric-card"><div class="metric-value">{max_dd:.1%}</div><div class="metric-label">Max Drawdown</div></div>
                <div class="metric-card"><div class="metric-value">{calmar:.2f}</div><div class="metric-label">Calmar Ratio</div></div>
                <div class="metric-card"><div class="metric-value">{win_rate:.1%}</div><div class="metric-label">Win Rate</div></div>
            </div>
            """, unsafe_allow_html=True)


        # ==================== EQUITY CURVE ====================

        st.subheader(f"📈 Strategy vs. Benchmark (Starting: ${STARTING_BALANCE:,.2f})")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Equity'], fill='tozeroy', line_color='#00FF41', name="Charlie's Kids Strategy"))
        fig_cum.add_trace(go.Scatter(x=df.index, y=df['Baseline_Equity'], line_color='#888888', line=dict(dash='dash'), name="Buy & Hold Baseline"))
        fig_cum.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cum, use_container_width=True)


        # ==================== LAST YEAR ALPHA DRILL-DOWN ====================

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        one_year_ago = df.index.max() - pd.DateOffset(years=1)
        st.subheader(f"🔍 Last Year Alpha ({one_year_ago.strftime('%b %Y')} – {df.index.max().strftime('%b %Y')})")
        col_chart, col_metrics = st.columns([3, 1])

        y_df = df[df.index >= one_year_ago].copy()

        if not y_df.empty and len(y_df) > 1:
            y_strat_g = (1 + y_df['Net_Returns'].fillna(0)).cumprod() * STARTING_BALANCE
            y_base_g = (1 + y_df['Returns'].fillna(0)).cumprod() * STARTING_BALANCE

            y_ret = (y_strat_g.iloc[-1] / STARTING_BALANCE) - 1
            y_base_ret = (y_base_g.iloc[-1] / STARTING_BALANCE) - 1
            y_alpha = y_ret - y_base_ret
            y_sharpe = (y_df['Net_Returns'].mean() / y_df['Net_Returns'].std()) * np.sqrt(TRADING_DAYS) if y_df['Net_Returns'].std() != 0 else 0

            with col_metrics:
                st.markdown(f"""
                    <div class="snapshot-card"><div class="snapshot-lab">Last Year Alpha</div><div class="snapshot-val">{y_alpha:+.1%}</div></div>
                    <div class="snapshot-card"><div class="snapshot-lab">Last Year Sharpe</div><div class="snapshot-val">{y_sharpe:.2f}</div></div>
                    <div class="snapshot-card"><div class="snapshot-lab">Winning Days</div><div class="snapshot-val">{(y_df['Net_Returns'] > 0).sum()}</div></div>
                    <div class="snapshot-card"><div class="snapshot-lab">Trading Vol</div><div class="snapshot-val">{y_df['Net_Returns'].std()*np.sqrt(TRADING_DAYS):.1%}</div></div>
                """, unsafe_allow_html=True)

            with col_chart:
                fig_year = go.Figure()
                fig_year.add_trace(go.Scatter(x=y_df.index, y=y_strat_g, fill='tozeroy', line_color='#00BFFF', name="Strategy"))
                fig_year.add_trace(go.Scatter(x=y_df.index, y=y_base_g, line_color='#888888', line=dict(dash='dash'), name="Baseline"))
                fig_year.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("Not enough data to display last year metrics.")

        st.markdown("---")


    # ==================== CANDLESTICK CHART ====================

    st.subheader("🎯 Price Action & Signals")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#00FF41', decreasing_line_color='#FF3131'
    ))

    if has_signals:
        buys = df[df['Signal'] > 0]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Low'] * 0.985, mode='markers', name='BUY',
                                     marker=dict(symbol='triangle-up', size=14, color='#00FF41')))
        sells = df[df['Signal'] < 0]
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index, y=sells['High'] * 1.015, mode='markers', name='SELL',
                                     marker=dict(symbol='triangle-down', size=14, color='#FF3131')))

    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


    # ==================== PERFORMANCE MATRIX & XAI ====================

    if has_signals:
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        col_table, col_xai = st.columns([1, 1])

        with col_table:
            st.subheader("📊 Annual Performance Matrix")
            df['Year'] = df.index.year
            table_data = []
            for year, group in df.groupby('Year'):
                y_ret_tbl = (group['Cumulative_Equity'].iloc[-1] / group['Cumulative_Equity'].iloc[0]) - 1
                y_sha_tbl = (group['Net_Returns'].mean() / group['Net_Returns'].std()) * np.sqrt(TRADING_DAYS) if group['Net_Returns'].std() != 0 else 0
                table_data.append({"Year": year, "Return": f"{y_ret_tbl:.1%}", "Sharpe": round(y_sha_tbl, 2)})
            st.table(pd.DataFrame(table_data).set_index('Year'))

        with col_xai:
            st.markdown('<div class="xai-container">', unsafe_allow_html=True)
            st.subheader("🧠 XAI: Model Decision Logic")
            MODEL_PATH = "models/spy_xgb_model.joblib"
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                features = ["Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", "Prev_close", "Atr_14",
                             "Rolling_Score", "Price_SMA_Divergences", "Prev_high", "Prev_low", "BOS_Bullish",
                             "BOS_Bearish", "BOS", "Resistance", "Support"]
                if hasattr(model, 'feature_importances_'):
                    feat_df = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=True)
                    fig_xai = go.Figure(go.Bar(x=feat_df['Weight'], y=feat_df['Feature'], orientation='h', marker_color='#00FF41'))
                    fig_xai.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=10, b=10),
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_xai, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


        # ==================== ADVANCED ANALYTICS ====================

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.subheader("🔬 Advanced Institutional Analytics")

        downside_returns = df['Net_Returns'][df['Net_Returns'] < 0]
        sortino = (df['Net_Returns'].mean() / downside_returns.std()) * np.sqrt(TRADING_DAYS) if downside_returns.std() != 0 else 0
        var_95 = np.percentile(df['Net_Returns'].fillna(0), 5)
        win_days = (df['Net_Returns'] > 0).sum()
        loss_days = (df['Net_Returns'] < 0).sum()
        profit_factor = abs(df[df['Net_Returns'] > 0]['Net_Returns'].sum() / df[df['Net_Returns'] < 0]['Net_Returns'].sum()) if df[df['Net_Returns'] < 0]['Net_Returns'].sum() != 0 else 0
        expectancy = ((win_days/len(df)) * df[df['Net_Returns'] > 0]['Net_Returns'].mean()) + ((loss_days/len(df)) * df[df['Net_Returns'] < 0]['Net_Returns'].mean())
        recovery_factor = abs(total_ret / max_dd) if max_dd != 0 else 0

        st.markdown(f"""
            <div class="perf-grid">
                <div class="perf-item"><div class="perf-lab">Sortino Ratio</div><div class="perf-val">{sortino:.2f}</div></div>
                <div class="perf-item"><div class="perf-lab">Profit Factor</div><div class="perf-val">{profit_factor:.2f}</div></div>
                <div class="perf-item"><div class="perf-lab">Daily VaR (95%)</div><div class="perf-val">{var_95:.2%}</div></div>
                <div class="perf-item"><div class="perf-lab">Expectancy</div><div class="perf-val">{expectancy:.4f}</div></div>
                <div class="perf-item"><div class="perf-lab">Total Trades</div><div class="perf-val">{(df['Signal'].diff() != 0).sum()}</div></div>
                <div class="perf-item"><div class="perf-lab">Positive Days</div><div class="perf-val">{win_days}</div></div>
                <div class="perf-item"><div class="perf-lab">Negative Days</div><div class="perf-val">{loss_days}</div></div>
                <div class="perf-item"><div class="perf-lab">Recovery Factor</div><div class="perf-val">{recovery_factor:.2f}</div></div>
            </div>
        """, unsafe_allow_html=True)


    # ==================== TRADE LOG TABLE ====================

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("📋 Trade Logs / Signal History")

    if 'Signal' in df.columns:
        log_df = df[df['Signal'] != 0].copy()
        if not log_df.empty:
            display_cols = ['Close', 'Signal']
            if 'Cumulative_Equity' in log_df.columns:
                display_cols.append('Cumulative_Equity')

            def style_signal(val):
                if val > 0:
                    return 'color: #00FF41'
                elif val < 0:
                    return 'color: #FF3131'
                return ''

            fmt = {'Close': '${:.2f}', 'Signal': '{:.2f}'}
            if 'Cumulative_Equity' in display_cols:
                fmt['Cumulative_Equity'] = '${:.2f}'

            st.dataframe(
                log_df[display_cols].style.format(fmt).map(style_signal, subset=['Signal']),
                use_container_width=True
            )
        else:
            st.info("No active signals generated during this period or model not run yet.")
    else:
        st.info("Please upload and run a model to see trade logs.")


    # ==================== FOOTER ====================

    st.markdown("---")
    final_val = df['Cumulative_Equity'].iloc[-1] if has_signals and 'Cumulative_Equity' in df.columns else STARTING_BALANCE
    st.caption(f"Charlie's Kids | High-Performance Quant Stack | Portfolio Started: ${STARTING_BALANCE:,.2f} | Current: ${final_val:,.2f}")

else:
    st.error("Missing Data. Check data/ directory for CSV files.")
