from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model_path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\models\per_user_lstm.keras"

def train_lstm(X_train, y_train, model_path):
    model = Sequential()

    model.add(LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(model_path)
