import pandas as pd
import matplotlib.pyplot as plt


def get_feature_importance(model, top_n=20):

    preprocessor = model.named_steps["preprocessor"]

    regressor = model.named_steps["regressor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = regressor.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return importance_df


def plot_feature_importance(
    importance_df
):

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Top Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300
    )

    plt.show()