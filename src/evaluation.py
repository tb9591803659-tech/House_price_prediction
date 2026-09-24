import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_true, predictions):

    # Mean Absolute Error
    mae = mean_absolute_error(
        y_true,
        predictions
    )

    # Mean Squared Error
    mse = mean_squared_error(
        y_true,
        predictions
    )

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R² Score
    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }