import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# 1. Load the model using absolute paths to be safe
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spy_xgb_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Basic Features
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std() 
    df["Volatility_30"] = df["Returns"].rolling(30).std()
    df["SMA_20"] = df["Close"].rolling(20).mean() 
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Prev_close"] = df["Close"].shift(1) 
    df["Atr_14"] = (df["High"] - df["Low"]).rolling(14).mean()
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
    features = [
        "Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", 
        "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", 
        "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS", 
        "Resistance", "Support"
    ]
    
    # 2. Extract feature matrix
    X = df[features].fillna(0).values
    
    # 3. Predict Probability
    # predict_proba returns [prob_0, prob_1]
    probs = model.predict_proba(X)[:, 1]
    
    # 4. Trading Logic with Confidence Buffers
    signals = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
    
    # 5. Risk Filters (The Anti-Drawdown filters)
    # Filter 1: Trend alignment
    signals[(signals == 1) & (df["Close"] < df["SMA_50"])] = 0
    signals[(signals == -1) & (df["Close"] > df["SMA_50"])] = 0
    
    # Filter 2: Volatility Spike Protection
    vol_ratio = df["Volatility_10"] / df["Volatility_30"]
    signals[vol_ratio > 1.5] = 0

    return pd.Series(signals, index=data.index).astype(int)