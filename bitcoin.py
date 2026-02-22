import pandas as pd
import numpy as np
import os

# --- 1. CLEANING LOGIC (from your utilities.py) ---
def clean_bitcoin_raw(filepath="bitcoin.csv"):
    print(f"Standardizing {filepath}...")
    # Load MultiIndex
    df = pd.read_csv(filepath, index_col=0, header=[0, 1], parse_dates=True)
    
    # Flatten columns to Price_Ticker format like your utilities.py logic
    df.columns = [f"{col[0]}_{col[1].replace('-USD', '')}" for col in df.columns.values]
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna()

    # Integrity Checks
    df = df[df['High_BTC'] >= df['Low_BTC']]
    price_cols = ['Open_BTC', 'High_BTC', 'Low_BTC', 'Close_BTC']
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df[df['Volume_BTC'] > 0]
    
    # Standardize to simple names for add_ta
    df = df.rename(columns=lambda x: x.replace("_BTC", ""))
    return df

# --- 2. TECHNICAL INDICATOR LOGIC (your 16 features) ---
def add_ta(df):
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std()
    df["Volatility_30"] = df["Returns"].rolling(30).std()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    
    df["Prev_close"] = df["Close"].shift(1)
    df["Atr_high_low"] = df["High"] - df["Low"]
    df["Atr_high_close"] = (df["High"] - df["Prev_close"]).abs()
    df["Atr_low_close"] = (df["Low"] - df["Prev_close"]).abs()
    df["Atr"] = df[["Atr_high_low", "Atr_high_close", "Atr_low_close"]].max(axis=1)
    df["Atr_14"] = df["Atr"].rolling(14).mean()
    df.drop(["Atr_high_low", "Atr_high_close", "Atr_low_close", "Atr"], axis=1, inplace=True)
    
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"]
    
    df["Resistance"] = df["High"].rolling(20).mean()
    df["Support"] = df["Low"].rolling(20).min()
    
    # Drop NaNs created by rolling windows to keep data clean for XGBoost
    return df.dropna()

# --- 3. EXECUTION ---
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Step A: Clean
    cleaned_df = clean_bitcoin_raw("data/bitcoin.csv")
    
    # Step B: Add TA
    final_df = add_ta(cleaned_df)
    
    # Step C: Save
    final_df.to_csv("data/BTC.csv")
    print(f"Successfully created data/BTC.csv with {final_df.shape[0]} rows and {final_df.shape[1]} columns.")