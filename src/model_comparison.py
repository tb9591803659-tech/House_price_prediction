import pandas as pd


def train_and_evaluate_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
    evaluate_model
):

    results = []
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test
        )

        # Evaluate
        metrics = evaluate_model(
            y_test,
            predictions
        )

        results.append(
            {
                "Model": name,
                **metrics
            }
        )

        trained_models[name] = model

        print(f"{name} completed.")

    results_df = pd.DataFrame(results)

    return results_df, trained_models