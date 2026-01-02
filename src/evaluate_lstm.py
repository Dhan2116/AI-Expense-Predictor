import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model_path, X_test, y_test, scaler):
    # Load trained model
    model = load_model(model_path)

    # Predict on test data
    predictions = model.predict(X_test)

    # Reverse scaling for y_test
    y_test_inv = scaler.inverse_transform(
        np.concatenate(
            [y_test.reshape(-1, 1),
             np.zeros((len(y_test), scaler.n_features_in_ - 1))],
            axis=1
        )
    )[:, 0]

    # Reverse scaling for predictions
    preds_inv = scaler.inverse_transform(
        np.concatenate(
            [predictions,
             np.zeros((len(predictions), scaler.n_features_in_ - 1))],
            axis=1
        )
    )[:, 0]

    # Evaluation metrics
    mae = mean_absolute_error(y_test_inv, preds_inv)

    mse = mean_squared_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)

    return mae, rmse
