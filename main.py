import pandas as pd
from src.data_preprocessing import preprocess_data
from src.lstm_preprocessing import create_sequences
from src.train_lstm import train_lstm
from src.evaluate_lstm import evaluate_model

# Preprocess
preprocess_data(
    "data/raw/personal_finance.csv",
    "data/processed/lstm_ready.csv"
)

# Load processed data
df = pd.read_csv("data/processed/lstm_ready.csv")

# Drop date for modeling
data = df.drop(columns=['record_date']).values

# Create sequences
X, y, scaler = create_sequences(data)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#  Train model
train_lstm(X_train, y_train, "models/lstm_expense_model.h5")

# Evaluate
mae, rmse = evaluate_model(
    "models/lstm_expense_model.h5",
    X_test,
    y_test,
    scaler
)

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
