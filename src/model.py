
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from src.feature_engineering import create_features
from src.preprocessing import create_preprocessor


def create_pipeline(
    model,
    numerical_features,
    categorical_features
):
    """
    Create a complete ML pipeline:

    Raw Data
        ↓
    Feature Engineering
        ↓
    Preprocessing
        ↓
    Model
    """

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    create_features,
                    validate=False
                )
            ),
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                model
            )
        ]
    )

    return pipeline

# ============================================================
# BASELINE
# ============================================================

def create_baseline(y_train, y_test):

    baseline_prediction = y_train.mean()

    baseline_predictions = [
        baseline_prediction
        for _ in range(len(y_test))
    ]

    return baseline_predictions

# ============================================================
# LINEAR REGRESSION
# ============================================================

def create_linear_regression_model(
    numerical_features,
    categorical_features
):

    model = LinearRegression()

    return create_pipeline(
        model,
        numerical_features,
        categorical_features
    )


# ============================================================
# RIDGE
# ============================================================

def create_ridge_model(
    numerical_features,
    categorical_features,
    alpha=1.0
):

    model = Ridge(
        alpha=alpha
    )

    return create_pipeline(
        model,
        numerical_features,
        categorical_features
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def create_random_forest_model(
    numerical_features,
    categorical_features,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1
):

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    return create_pipeline(
        model,
        numerical_features,
        categorical_features
    )


# ============================================================
# GRADIENT BOOSTING
# ============================================================

def create_gradient_boosting_model(
    numerical_features,
    categorical_features,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100
):

    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    return create_pipeline(
        model,
        numerical_features,
        categorical_features
    )