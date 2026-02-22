import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# 1. LOAD THE MASTER MODEL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spy_xgb_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # Core Indicators
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std() 
    df["Volatility_30"] = df["Returns"].rolling(30).std()
    df["Vol_Benchmark"] = df["Volatility_30"].rolling(252).mean() 
    
    df["SMA_20"] = df["Close"].rolling(20).mean() 
    df["SMA_50"] = df["Close"].rolling(50).mean()
    
    # ATR for stop loss
    df["Prev_close"] = df["Close"].shift(1) 
    df["TR"] = df[["High", "Low", "Prev_close"]].max(axis=1) - df[["High", "Low", "Prev_close"]].min(axis=1)
    df["Atr_14"] = df["TR"].rolling(14).mean()
    
    # Momentum & Structure
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"] 
    df["Resistance"] = df["High"].rolling(20).mean()
    df["Support"] = df["Low"].rolling(20).min()
    
    return df

def generate_signals(data: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(0, index=data.index)

    df = calculate_technical_indicators(data)
    features = ["Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", 
                "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", 
                "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS",
                "Resistance", "Support"]
    
    X = df[features].fillna(0).values
    probs = model.predict_proba(X)[:, 1]
    
    # --- 1. DYNAMIC REGIME DETECTION ---
    vol_ratio = df["Volatility_30"] / df["Vol_Benchmark"].fillna(df["Volatility_30"].mean())
    trend_strength = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
    
    # Thresholding logic that stays aggressive
    dynamic_thresh = np.where((vol_ratio < 0.8) & (trend_strength > 0.02), 0.5001, 0.505) 
    signals = np.where(probs > dynamic_thresh, 1, np.where(probs < (1 - dynamic_thresh), -1, 0))

    # --- 2. BULL PERSISTENCE ---
    strong_bull = (df["SMA_20"] > df["SMA_50"]) & (df["Close"] > df["SMA_50"])
    signals[(signals == 0) & (strong_bull) & (probs > 0.48)] = 1 

    # --- 3. TRAILING STOP WITH FAST RE-ENTRY ---
    final_signals = signals.copy()
    stop_price, in_pos = 0.0, 0 

    for i in range(1, len(data)):
        curr_price = df["Close"].iloc[i]
        curr_atr = df["Atr_14"].iloc[i]
        curr_vol_ratio = vol_ratio.iloc[i]
        
        # Determine multiplier based on regime
        mult = 3.0 if curr_vol_ratio < 1.0 else 4.5
        
        if in_pos == 1:
            # Trailing the stop
            new_stop = curr_price - (curr_atr * mult)
            stop_price = max(stop_price, new_stop)
            
            # EXIT CONDITION
            if curr_price < stop_price:
                final_signals[i] = 0
                in_pos = 0
            # RE-ENTRY CONDITION: If stopped out but model is still screaming "BUY" 
            # and price moves back above the previous stop, we re-enter immediately.
            elif signals[i] == 1 and curr_price > stop_price:
                final_signals[i] = 1
                in_pos = 1
        
        elif in_pos == -1:
            new_stop = curr_price + (curr_atr * mult)
            stop_price = min(stop_price, new_stop) if stop_price != 0 else new_stop
            
            if curr_price > stop_price:
                final_signals[i] = 0
                in_pos = 0
            elif signals[i] == -1 and curr_price < stop_price:
                final_signals[i] = -1
                in_pos = -1
        
        else:
            # Entry logic
            in_pos = signals[i]
            if in_pos == 1:
                stop_price = curr_price - (curr_atr * mult)
            elif in_pos == -1:
                stop_price = curr_price + (curr_atr * mult)

    return pd.Series(final_signals, index=data.index).astype(int)