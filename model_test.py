import utilities
import pandas as pd
import trading_model
import technical_indicators

if __name__ == "__main__":
    # load and clean data
    df = utilities.data_cleaning()                          # load df
    df_spy, df_qqq = technical_indicators.prepare_data(df)  # separate into stocks
    df_spy = utilities.add_price_direction_label(df_spy)    # add labels for price direction
    df_qqq = utilities.add_price_direction_label(df_qqq)

    # generate and compile models
    models = {
        "SPY": trading_model.TradingModel(),
        "QQQ": trading_model.TradingModel()
    }
    models["SPY"].compile()
    models["QQQ"].compile()
    models["SPY"].plot_model()
    print("Models compiled successfully!")

    # train models
    models["SPY"].fit(df_spy, epochs=1)
    models["QQQ"].fit(df_qqq, epochs=1)
    print("Models fit successfully!")

    models["SPY"].plot_training_history()
    models["QQQ"].plot_training_history()

    