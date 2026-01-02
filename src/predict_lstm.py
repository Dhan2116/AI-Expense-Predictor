import numpy as np
from tensorflow.keras.models import load_model

def predict_next_expense(model_path, last_sequence, scaler):
    model = load_model(model_path)

    pred = model.predict(last_sequence.reshape(1, *last_sequence.shape))

    temp = np.zeros((1, scaler.n_features_in_))
    temp[0, 0] = pred

    return scaler.inverse_transform(temp)[0, 0]
