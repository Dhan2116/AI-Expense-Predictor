from src.predict import predict_expense

sample_user = {
    "gender": 1,
    "education_level": 2,
    "employment_status": 1,
    "job_title": 5,
    "has_loan": 1,
    "loan_type": 2,
    "region": 3,
    "age": 26,
    "monthly_income_usd": 4500,
    "savings_usd": 1200,
    "loan_amount_usd": 8000,
    "loan_term_months": 36,
    "monthly_emi_usd": 280,
    "loan_interest_rate_pct": 8.5,
    "debt_to_income_ratio": 0.22,
    "credit_score": 720,
    "savings_to_income_ratio": 0.27
}

print(predict_expense(sample_user))
