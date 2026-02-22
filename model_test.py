import utilities
import pandas as pd
import numpy as np
import trading_model
import technical_indicators
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # ----------------------------
    # Load and clean data
    # ----------------------------
    df = utilities.data_cleaning()                           # load df
    df_spy, df_qqq = technical_indicators.prepare_data(df)  # separate into stocks
    df_spy = utilities.add_price_direction_label(df_spy)    # add labels for price direction
    df_qqq = utilities.add_price_direction_label(df_qqq)

    # ----------------------------
    # Define input shape
    # ----------------------------
    timesteps = 1
    features_spy = df_spy.shape[1] - 1  # drop label column
    features_qqq = df_qqq.shape[1] - 1

    # ----------------------------
    # Split features and labels
    # ----------------------------
    spy_X, spy_y = df_spy.drop(columns="Label"), df_spy["Label"]
    qqq_X, qqq_y = df_qqq.drop(columns="Label"), df_qqq["Label"]

    spy_X_train, spy_X_test, spy_y_train, spy_y_test = train_test_split(
        spy_X, spy_y, test_size=0.2, random_state=42
    )
    qqq_X_train, qqq_X_test, qqq_y_train, qqq_y_test = train_test_split(
        qqq_X, qqq_y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # Convert X to NumPy float32 arrays and reshape for Conv1D/GRU
    # ----------------------------
    spy_X_train = spy_X_train.to_numpy(dtype=np.float32).reshape(-1, timesteps, features_spy)
    spy_X_test  = spy_X_test.to_numpy(dtype=np.float32).reshape(-1, timesteps, features_spy)

    qqq_X_train = qqq_X_train.to_numpy(dtype=np.float32).reshape(-1, timesteps, features_qqq)
    qqq_X_test  = qqq_X_test.to_numpy(dtype=np.float32).reshape(-1, timesteps, features_qqq)

    # ----------------------------
    # Convert one-hot labels to scalar 0/1
    # ----------------------------
    spy_y_train = np.array([np.argmax(x) if isinstance(x, (list, np.ndarray)) else x for x in spy_y_train], dtype=np.float32)
    spy_y_test  = np.array([np.argmax(x) if isinstance(x, (list, np.ndarray)) else x for x in spy_y_test], dtype=np.float32)

    qqq_y_train = np.array([np.argmax(x) if isinstance(x, (list, np.ndarray)) else x for x in qqq_y_train], dtype=np.float32)
    qqq_y_test  = np.array([np.argmax(x) if isinstance(x, (list, np.ndarray)) else x for x in qqq_y_test], dtype=np.float32)

    # ----------------------------
    # Generate and compile models
    # ----------------------------
    models = {
        "SPY": trading_model.TradingModel(input_shape=(timesteps, features_spy)),
        "QQQ": trading_model.TradingModel(input_shape=(timesteps, features_qqq))
    }
    models["SPY"].compile(metrics=['accuracy'])
    models["QQQ"].compile(metrics=['accuracy'])

    print("Models compiled successfully!")

    # ----------------------------
    # Train models
    # ----------------------------
    models["SPY"].fit(spy_X_train, spy_y_train, epochs=1, batch_size=32)
    models["QQQ"].fit(qqq_X_train, qqq_y_train, epochs=1, batch_size=32)
    print("Models fit successfully!")

    # ----------------------------
    # Plot training history
    # ----------------------------
    models["SPY"].plot_training_history()
    models["QQQ"].plot_training_history()