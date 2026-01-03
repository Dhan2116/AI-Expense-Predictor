from src.preprocess import preprocess_data
from src.train_lgbm import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_actual_vs_predicted


print("Loading & preprocessing data...")
X, y = preprocess_data("data/raw/personal_finance.csv")

print("Training LightGBM model...")
model, X_test, y_test = train_model(X, y)

print("Evaluating model...")
mae, rmse, r2, preds = evaluate_model(model, X_test, y_test)

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.3f}")

print("Plotting Actual vs Predicted...")
plot_actual_vs_predicted(y_test, preds)
