import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import utilities
import technical_indicators
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_master_model():
    print("--- 1. Loading and Preparing Data ---")
    df_raw = utilities.data_cleaning() 
    df_spy, df_qqq = technical_indicators.prepare_data(df_raw)
    
    # Target Alignment (Tomorrow's Move)
    for df in [df_spy, df_qqq]:
        df['Future_Return'] = df['Close'].shift(-1) - df['Close']
        df['Target'] = (df['Future_Return'] > 0).astype(int)

    features = [
        "Returns", "Volatility_10", "Volatility_30", "SMA_20", "SMA_50", 
        "Prev_close", "Atr_14", "Rolling_Score", "Price_SMA_Divergences", 
        "Prev_high", "Prev_low", "BOS_Bullish", "BOS_Bearish", "BOS", 
        "Resistance", "Support"
    ]

    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # We train on combined data for the 'Master' model
    df_combined = pd.concat([df_spy, df_qqq], axis=0).dropna(subset=['Target'] + features)
    X = df_combined[features].values
    y = df_combined['Target'].values

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Optimized XGBoost Config
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.005,
        subsample=0.6,
        colsample_bytree=0.6,
        objective='binary:logistic',
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=100,
        random_state=42,
    )

    print("\n--- 2. Training Model ---")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # --- 3. OUTPUT ACCURACY ---
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    print(f"\nFINAL TRAINING ACCURACY:   {accuracy_score(y_train, train_preds):.2%}")
    print(f"FINAL VALIDATION ACCURACY: {accuracy_score(y_val, val_preds):.2%}")

    # --- 4. SAVE PLOTS ---
    results = model.evals_result()
    plt.figure(figsize=(10, 5))
    plt.plot(results['validation_0']['logloss'], label='Train Loss')
    plt.plot(results['validation_1']['logloss'], label='Val Loss')
    plt.title("XGBoost Training Progress")
    plt.legend()
    plt.savefig('plots/learning_curve.png')
    print("Learning curve saved to plots/learning_curve.png")

    # --- 5. SAVE MODEL ---
    joblib.dump(model, 'models/spy_xgb_model.joblib')
    print("Model saved to models/spy_xgb_model.joblib")

if __name__ == "__main__":
    train_master_model()