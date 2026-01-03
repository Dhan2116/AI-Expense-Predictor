import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plots Actual vs Predicted values for regression evaluation
    """

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # diagonal reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Monthly Expenses (USD)")
    plt.ylabel("Predicted Monthly Expenses (USD)")
    plt.title("Actual vs Predicted Monthly Expenses")

    plt.grid(True)
    plt.show()
