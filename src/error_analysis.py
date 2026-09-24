import pandas as pd

import matplotlib.pyplot as plt

def create_error_analysis(
    X_test,
    y_test,
    predictions
):
    """
    Create a dataframe containing
    actual values, predictions and errors.
    """

    results = X_test.copy()

    results["Actual"] = y_test.values

    results["Predicted"] = predictions

    results["Residual"] = (
        results["Actual"]
        - results["Predicted"]
    )

    results["Absolute_Error"] = (
        results["Residual"].abs()
    )

    return results


def plot_actual_vs_predicted(
    y_test,
    predictions
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(
        "Actual vs Predicted Sale Price"
    )

    plt.tight_layout()

    plt.savefig(
        "results/actual_vs_predicted.png",
        dpi=300
    )

    plt.show()

def plot_residual_distribution(
residuals
):

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        "Residual Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "results/residual_distribution.png",
        dpi=300
    )

    plt.show()