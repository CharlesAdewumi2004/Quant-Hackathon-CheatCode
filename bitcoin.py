import pandas as pd
import numpy as np
import os

# --- 1. STRICT RAW CLEANING (Matches utilities.py) ---
def clean_raw_bitcoin(filepath="bitcoin.csv"):
    print("Step 1: Retrieving and Cleaning Raw Data...")
    df = pd.read_csv(filepath, index_col=0, header=[0, 1], parse_dates=True)
    
    # Flatten MultiIndex to Price_Ticker
    df.columns = [f"{col[0]}_{col[1].replace('-USD', '')}" for col in df.columns.values]
    
    # Utilities.py cleaning logic
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna()
    
    # High must be >= Low
    df = df[df['High_BTC'] >= df['Low_BTC']]
    
    # Prices & Volume must be > 0
    price_cols = ['Open_BTC', 'High_BTC', 'Low_BTC', 'Close_BTC']
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df[df['Volume_BTC'] > 0]
    
    # Rename to pure OHLCV for TA
    df = df.rename(columns=lambda x: x.replace("_BTC", ""))
    return df

# --- 2. ADD TECHNICAL INDICATORS ---
def add_ta(df):
    print("Step 2: Adding Technical Indicators...")
    df = df.copy()
    
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
    df.drop(["Atr_high_low", "Atr_high_close", "Atr_low_close", "Atr"], axis=1, inplace=True)
    
    df["Rolling_Score"] = df["Close"].pct_change(14)
    df["Price_SMA_Divergences"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    
    df["Prev_high"] = df["High"].shift(1)
    df["Prev_low"] = df["Low"].shift(1)
    df["BOS_Bullish"] = (df["Close"] > df["Prev_high"]).astype(int)
    df["BOS_Bearish"] = (df["Close"] < df["Prev_low"]).astype(int)
    df["BOS"] = df["BOS_Bullish"] - df["BOS_Bearish"]
    
    df["Resistance"] = df["High"].rolling(window=20).mean()
    df["Support"] = df["Low"].rolling(window=20).min()
    
    return df

# --- 3. STRICT POST-TA CLEANING (Matches clean_after_inclusion) ---
def clean_after_inclusion(df):
    print("Step 3: Cleaning After Indicator Inclusion...")
    df = df.copy()
    
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = df[c].astype(str).str.replace(",", "").str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df[df["Volume"] > 0]

# --- EXECUTION ---
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Execute the exact strict sequence
    df_raw_clean = clean_raw_bitcoin("data/bitcoin.csv")
    df_ta = add_ta(df_raw_clean)
    df_final = clean_after_inclusion(df_ta)
    
    # Save the strictly processed file
    df_final.to_csv("data/BTC.csv")
    print(f"Success! Final BTC data ready with {df_final.shape[0]} rows. Ready for test.py")