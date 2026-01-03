from joblib import load
import pandas as pd


def predict_expense(input_dict):
    model = load("models/lightgbm_expense_model.joblib")

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)

    return prediction[0]
