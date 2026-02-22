import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import utilities
import technical_indicators
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    df_combined = pd.concat([df_spy, df_qqq], axis=0).dropna(subset=['Target'] + features)
    X = df_combined[features].values
    y = df_combined['Target'].values

    # Time-Series Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # --- 2. MODIFIED XGBOOST CONFIG ---
    # We removed scale_pos_weight because it over-corrected.
    # We increased max_depth to 6 so it can find deeper patterns.
    model = xgb.XGBClassifier(
        n_estimators=500,      
        max_depth=6,           
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss', 
        early_stopping_rounds=50,
        random_state=42
    )

    print("\n--- 3. Training Model ---")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # --- 4. DYNAMIC THRESHOLDING ---
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # We'll use the median of probabilities as a threshold 
    # to ensure the report shows both Up and Down.
    threshold = np.median(val_probs)
    val_preds = (val_probs > threshold).astype(int) 

    print("\n" + "="*40)
    print(f"REPORT THRESHOLD (Median): {threshold:.4f}")
    print(f"VALIDATION ACCURACY:       {accuracy_score(y_val, val_preds):.2%}")
    print("="*40)

    print("\n--- Final Classification Report ---")
    print(classification_report(y_val, val_preds, target_names=['Down (0)', 'Up (1)']))

    # --- 5. Save & Plot ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spy_xgb_model.joblib')
    
    plt.figure(figsize=(10, 6))
    pd.Series(model.feature_importances_, index=features).nlargest(10).plot(kind='barh')
    plt.title("Feature Importance")
    plt.show()

if __name__ == "__main__":
    train_master_model()