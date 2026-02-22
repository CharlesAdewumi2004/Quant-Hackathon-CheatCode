import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt

# Keras Libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GRU, Dropout, Conv1D, GlobalAveragePooling1D, 
    BatchNormalization, SpatialDropout1D, Bidirectional, LeakyReLU
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Project Modules (Ensure these exist in your path)
# import utilities
# import technical_indicators

# --- CLASS 1: YOUR DEEP LEARNING GRU ARCHITECTURE ---
class TradingModel(Model):
    def __init__(self, input_shape=None, **kwargs):
        super(TradingModel, self).__init__(**kwargs)
        self.history = None

        # 1. Conv block
        self.conv1 = Conv1D(
            filters=64, 
            kernel_size=5, 
            padding="same",
            kernel_regularizer=l2(1e-3)
        )
        self.leaky_conv = LeakyReLU(alpha=0.01)
        self.bn1 = BatchNormalization()
        self.spatial_drop = SpatialDropout1D(0.4)

        # 2. Bidirectional GRUs
        self.gru1 = Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2(1e-3)))
        self.dropout1 = Dropout(0.5)

        self.gru2 = Bidirectional(GRU(32, kernel_regularizer=l2(1e-3)))
        self.dropout2 = Dropout(0.5)

        # 3. Dense Head
        self.dense1 = Dense(16, kernel_regularizer=l2(1e-3))
        self.out = Dense(1, activation='sigmoid')  

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.leaky_conv(x)
        x = self.bn1(x, training=training)
        x = self.spatial_drop(x, training=training)
        x = self.gru1(x)
        x = self.dropout1(x, training=training)
        x = self.gru2(x)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        return self.out(x)

@tf.keras.utils.register_keras_serializable()
class KerasModel(Model):
    def __init__(self, spy_path="models/spy_xgb_model.joblib", qqq_path="models/qqq_xgb_model.joblib", **kwargs):
        super(KerasModel, self).__init__(**kwargs)
        self.spy_path = spy_path
        self.qqq_path = qqq_path
        self.model_spy = None
        self.model_qqq = None
        
        if os.path.exists(spy_path):
            self.model_spy = joblib.load(spy_path)
        if os.path.exists(qqq_path):
            self.model_qqq = joblib.load(qqq_path)

    def call(self, inputs, training=False):
        def _ensemble_predict(x_np):
            if self.model_spy is None or self.model_qqq is None:
                return np.zeros((x_np.shape[0], 1), dtype=np.float32)
            
            p1 = self.model_spy.predict_proba(x_np)[:, 1]
            p2 = self.model_qqq.predict_proba(x_np)[:, 1]
            
            # Weighted ensemble for better stability
            avg_probs = (p1 + p2) / 2.0
            return avg_probs.reshape(-1, 1).astype(np.float32)

        return tf.py_function(_ensemble_predict, [inputs], tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"spy_path": self.spy_path, "qqq_path": self.qqq_path})
        return config