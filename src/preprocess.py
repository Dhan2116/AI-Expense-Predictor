import pandas as pd
from sklearn.preprocessing import LabelEncoder
path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\data\raw\personal_finance.csv"


TARGET_COL = "monthly_expenses_usd"

CATEGORICAL_COLS = [
    "gender",
    "education_level",
    "employment_status",
    "job_title",
    "has_loan",
    "loan_type",
    "region"
]

NUMERIC_COLS = [
    "age",
    "monthly_income_usd",
    "savings_usd",
    "loan_amount_usd",
    "loan_term_months",
    "monthly_emi_usd",
    "loan_interest_rate_pct",
    "debt_to_income_ratio",
    "credit_score",
    "savings_to_income_ratio"
]


def preprocess_data(path):
    df = pd.read_csv(path)

    # drop identifiers and non-predictive columns
    df = df.drop(columns=["user_id", "record_date"])

    # handle missing loan types
    df["loan_type"] = df["loan_type"].fillna("no_loan")

    # label encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]

    return X, y
