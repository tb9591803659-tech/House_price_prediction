import pandas as pd


def evaluate_final_model(
    model,
    X_test,
    y_test,
    evaluate_model
):

    predictions = model.predict(X_test)

    metrics = evaluate_model(
        y_test,
        predictions
    )

    results = pd.DataFrame(
        [
            {
                "Model": "Tuned Gradient Boosting",
                **metrics
            }
        ]
    )

    return results, predictions