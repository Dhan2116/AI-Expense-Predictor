import pandas as pd
input_path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\data\raw\personal_finance.csv"
output_path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\data\processed"

def preprocess_data(input_path, output_path):

    df = pd.read_csv(input_path)

    # Convert record_date to datetime
    df['record_date'] = pd.to_datetime(df['record_date'])

    df = df.sort_values(by=['user_id', 'record_date'])

    # Convert has_loan to numeric values
    df['has_loan'] = df['has_loan'].map({'Yes': 1, 'No': 0})

    # Fill missing loan_type
    df['loan_type'].fillna('None', inplace=True)

    # Drop categorical columns that are not used by LSTM
    df = df[[
        'record_date',
        'monthly_expenses_usd',
        'monthly_income_usd',
        'savings_usd',
        'monthly_emi_usd',
        'debt_to_income_ratio',
        'credit_score'
    ]]

    # Save processed file to the folder
    df.to_csv(output_path, index=False)
