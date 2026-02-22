from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TradingModel(Model):
    def __init__(self):
        super().__init__()

        # Conv block
        self.conv1 = Conv1D(64, 3, activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling1D(2)

        # GRU stack
        self.gru1 = GRU(50, return_sequences=True)
        self.dropout1 = Dropout(0.2)

        self.gru2 = GRU(50)
        self.dropout2 = Dropout(0.2)

        # Head
        self.dense1 = Dense(25, activation='relu')
        self.out = Dense(1, activation='sigmoid')  # change if regression

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.gru1(x)
        x = self.dropout1(x, training=training)

        x = self.gru2(x)
        x = self.dropout2(x, training=training)

        x = self.dense1(x)
        return self.out(x)
    
    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        super().compile(optimizer=optimizer, loss=loss)