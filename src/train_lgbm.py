import lightgbm as lgb
from sklearn.model_selection import train_test_split
from joblib import dump


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    dump(model, "models/lightgbm_expense_model.joblib")

    return model, X_test, y_test
