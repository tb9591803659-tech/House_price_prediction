import joblib

def save_model(
    model,
    path="models/house_price_model.joblib"
):

    joblib.dump(
        model,
        path
    )

    print(
        f"Model saved to: {path}"
    )