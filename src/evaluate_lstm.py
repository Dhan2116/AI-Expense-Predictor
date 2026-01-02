import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model_path, X_test, y_test, scaler):
    model = load_model(model_path)

    preds = model.predict(X_test)

    y_true = scaler.inverse_transform(
        np.c_[y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]
    )[:, 0]

    y_pred = scaler.inverse_transform(
        np.c_[preds, np.zeros((len(preds), scaler.n_features_in_ - 1))]
    )[:, 0]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse
