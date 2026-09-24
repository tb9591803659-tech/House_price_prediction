import pandas as pd

from sklearn.model_selection import KFold, cross_validate


def run_cross_validation(
    models,
    X,
    y,
    cv=5
):

    kfold = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    results = []

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2"
    }

    for name, model in models.items():

        print(f"\nCross-validating {name}...")

        scores = cross_validate(
            model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1
        )

        mae = -scores["test_MAE"].mean()
        mse = -scores["test_MSE"].mean()
        rmse = mse ** 0.5
        r2 = scores["test_R2"].mean()

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            }
        )

    return pd.DataFrame(results)