import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# 1. Load the model once at the top level
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spy_xgb_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the exact 16 technical indicators the model was trained on."""
    df = data.copy()
    
    # Trend and Volatility
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(window=10).std() 
    df["Volatility_30"] = df["Returns"].rolling(window=30).std()
    
    df["SMA_20"] = df["Close"].rolling(window=20).mean() 
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    df["Prev_close"] = df["Close"].shift(1) 
    df["Atr_high_low"] = df["High"] - df["Low"] 
    df["Atr_high_close"] = (df["High"] - df["Prev_close"]).abs()
    df["Atr_low_close"] = (df["Low"] - df["Prev_close"]).abs()
    df["Atr"] = df[["Atr_high_low", "Atr_high_close", "Atr_low_close"]].max(axis=1)
    df["Atr_14"] = df["Atr"].rolling(window=14).mean() 
    
    # Momentum and Divergences
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    
    # Break of Structure
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"] 
    
    # Support/Resistance
    df["Resistance"] = df["High"].rolling(window=20).mean()
    df["Support"] = df["Low"].rolling(window=20).min()
    
    # Clean up internal columns
    cols_to_drop = ["Atr_high_low", "Atr_high_close", "Atr_low_close", "Atr"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df

def generate_signals(data: pd.DataFrame) -> pd.Series:
    """
    Main strategy logic for the Hackathon evaluation.
    """
    # Default to 0 (Cash) if model isn't found
    if model is None:
        return pd.Series(0, index=data.index)

    # 1. Feature Engineering (Snapshot of today's market)
    df = calculate_technical_indicators(data)
    
    features = [
        "Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", 
        "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", 
        "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS", 
        "Resistance", "Support"
    ]
    
    # 2. Extract feature matrix and fill NaNs
    # IMPORTANT: XGBoost expects the same 16 features in the same order
    X = df[features].fillna(0).values
    
    # 3. Predict Probabilities (0 to 1)
    # [:, 1] gets the probability of "Class 1" (Up)
    probs = model.predict_proba(X)[:, 1]
    
    # 4. TRADING LOGIC: Thresholding
    # Since the market is noisy, we use a small "buffer" around 0.5.
    # Prediction > 0.52 -> Long (1)
    # Prediction < 0.48 -> Short (-1)
    # Otherwise -> Hold (0) to save on transaction costs.
    signals = np.where(probs > 0.52, 1, np.where(probs < 0.48, -1, 0))
    
    # 5. RISK MANAGEMENT: ATR Stop Loss
    # If Today's Close falls 2.5x ATR below the 5-day high, we bail.
    rolling_high = df["High"].rolling(window=5).max()
    stop_level = rolling_high - (df["Atr_14"] * 2.5)
    risk_off = df["Close"] < stop_level
    
    # Force signals to 0 if the risk breaker is triggered
    signals[risk_off] = 0
    
    # Convert to Series and ensure strictly [-1, 0, 1] integers
    final_signals = pd.Series(signals, index=data.index).fillna(0).astype(int)
    
    return final_signals