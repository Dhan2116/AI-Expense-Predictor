import pandas as pd
input_path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\data\raw\personal_finance.csv"
output_path = r"C:\Users\dhany\CODING ( END OF ME! )\AI Expense Predictor\data\processed"

def preprocess_data(input_path, output_path, min_records=12):
    df = pd.read_csv(input_path)

    df['record_date'] = pd.to_datetime(df['record_date'])

    df = df.sort_values(['user_id', 'record_date'])

    valid_users = df.groupby('user_id').filter(lambda x: len(x) >= min_records)
    valid_users = valid_users[[
        'user_id',
        'record_date',
        'monthly_expenses_usd',
        'monthly_income_usd',
        'savings_usd',
        'monthly_emi_usd',
        'debt_to_income_ratio',
        'credit_score'
    ]]

    valid_users.to_csv(output_path, index=False)