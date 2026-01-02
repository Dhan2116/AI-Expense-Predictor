import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_per_user_sequences(df, sequence_length=6):
    feature_cols = [
        'monthly_expenses_usd',
        'monthly_income_usd',
        'savings_usd',
        'monthly_emi_usd',
        'debt_to_income_ratio',
        'credit_score'
    ]

    X, y = [], []
    scaler = MinMaxScaler()

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Create sequences per user
    for user_id, user_df in df.groupby('user_id'):
        user_data = user_df[feature_cols].values

        for i in range(sequence_length, len(user_data)):
            X.append(user_data[i-sequence_length:i])
            y.append(user_data[i, 0])  

    return np.array(X), np.array(y), scaler
