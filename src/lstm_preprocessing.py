import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, sequence_length=6):
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0]) 

    return np.array(X), np.array(y), scaler
