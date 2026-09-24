import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_loader import load_data

from src.data_cleaning import (
    basic_dataset_info,
    identify_features,
    split_features_target
)

from src.preprocessing import (
    create_preprocessor
)

from src.model import (
    create_baseline,
    create_linear_regression_model,
    create_ridge_model,
    create_random_forest_model,
    create_gradient_boosting_model
)

from src.model_comparison import (
    train_and_evaluate_models
)
from src.evaluation import (
    evaluate_model
)

from src.error_analysis import (
    create_error_analysis
)

from src.data_cleaning import (
    handle_missing_values
)

from src.cross_validation import (
    run_cross_validation
)

from src.tuning import (
    tune_gradient_boosting,
    tune_random_forest
)

from src.final_evaluation import (
    evaluate_final_model
)

from src.error_analysis import (
    create_error_analysis,
    plot_actual_vs_predicted,
    plot_residual_distribution
)

from src.feature_importance import (
    get_feature_importance,
    plot_feature_importance
)

from src.save_model import save_model

from src.predict import predict_price

# =========================================================
# CREATE RESULTS DIRECTORY
# =========================================================

os.makedirs(
    "results",
    exist_ok=True
)


# =========================================================
# PHASE 1 — LOAD DATA
# =========================================================

df = load_data(
    "data/raw/train.csv"
)

# Handle meaningful missing values
df = handle_missing_values(df)


print(
    df[
        [
            "Alley",
            "GarageType",
            "BsmtQual",
            "FireplaceQu",
            "PoolQC"
        ]
    ].head()
)

print(
    df.isnull().sum()
    .sort_values(ascending=False)
    .head(20)
)

print("=" * 60)
print("PHASE 1 — DATASET INFORMATION")
print("=" * 60)

basic_dataset_info(df)


# =========================================================
# IDENTIFY FEATURES
# =========================================================

numerical_features, categorical_features = (
    identify_features(df)
)

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print(numerical_features)

print("\nNumber of numerical features:")
print(len(numerical_features))


print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

print(categorical_features)

print("\nNumber of categorical features:")
print(len(categorical_features))


# =========================================================
# TARGET INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("TARGET — SALEPRICE")
print("=" * 60)

print(df["SalePrice"].describe())


# =========================================================
# PHASE 1 — CATEGORICAL VALUE COUNTS
# =========================================================

print("\n" + "=" * 60)
print("EXAMPLE CATEGORICAL DISTRIBUTIONS")
print("=" * 60)

print("\nNeighborhood:")
print(
    df["Neighborhood"]
    .value_counts()
)


print("\nHouseStyle:")
print(
    df["HouseStyle"]
    .value_counts()
)


# =========================================================
# PHASE 1 — CORRELATION
# =========================================================

print("\n" + "=" * 60)
print("CORRELATION WITH SALEPRICE")
print("=" * 60)

correlation = (
    df[
        numerical_features
        + ["SalePrice"]
    ]
    .corr()
)

print(
    correlation["SalePrice"]
    .sort_values(
        ascending=False
    )
)


# =========================================================
# PHASE 2 — FEATURES AND TARGET
# =========================================================

X, y = split_features_target(df)

print("\n" + "=" * 60)
print("PHASE 2 — X AND y")
print("=" * 60)

print("X shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

print("\nTrain/Test Split:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test:",
    y_test.shape
)


# =========================================================
# PHASE 2 — PREPROCESSING
# =========================================================

preprocessor = create_preprocessor(
    numerical_features,
    categorical_features
)


# =========================================================
# PHASE 3 — BASELINE
# =========================================================

print("\n" + "=" * 60)
print("PHASE 3 — BASELINE")
print("=" * 60)

baseline_predictions = create_baseline(
    y_train,
    y_test
)

baseline_metrics = evaluate_model(
    y_test,
    baseline_predictions
)

baseline_results = pd.DataFrame([
    {
        "Model": "Baseline",
        **baseline_metrics
    }
])

print(
    baseline_results
)


# =========================================================
# SAVE BASELINE RESULTS
# =========================================================

baseline_results.to_csv(
    "results/baseline_results.csv",
    index=False
)


# =========================================================
# CREATE MODELS
# =========================================================

models = {

    "Linear Regression": create_linear_regression_model(
        numerical_features,
        categorical_features
    ),

    "Ridge": create_ridge_model(
        numerical_features,
        categorical_features
    ),

    "Random Forest": create_random_forest_model(
        numerical_features,
        categorical_features
    ),

    "Gradient Boosting": create_gradient_boosting_model(
        numerical_features,
        categorical_features
    )
}

# =========================================================
# TRAIN AND EVALUATE MODELS
# =========================================================

model_results, trained_models = train_and_evaluate_models(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    evaluate_model=evaluate_model
)

# =========================================================
# MODEL COMPARISON
# =========================================================


comparison_results = []


for name, model in models.items():

    print()
    print("=" * 70)
    print(f"TRAINING {name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    results = evaluate_model(
        y_test,
        predictions
    )

    results["Model"] = name

    comparison_results.append(
        results
    )


comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df[
    [
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]
]


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison_df)

comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print(
    "\nSaved to: results/model_comparison.csv"
)

# =========================================================
# KFOLD CROSS VALIDATION
# =========================================================

cv_results = run_cross_validation(
    models=models,
    X=X,
    y=y,
    cv=5
)

print("\n")
print("=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

print(
    cv_results.to_string(index=False)
)

print("=" * 70)

cv_results.to_csv(
    "results/cross_validation.csv",
    index=False
)

# =========================================================
# HYPERMETER TUNING GRADIENT BOOSTING MODEL
# =========================================================

print("\n")
print("=" * 70)
print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
print("=" * 70)

grid_search = tune_gradient_boosting(
    preprocessor=preprocessor,
    X_train=X_train,
    y_train=y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV RMSE:")
print(-grid_search.best_score_)

best_gb_model = grid_search.best_estimator_

test_predictions = best_gb_model.predict(
    X_test
)

tuned_metrics = evaluate_model(
    y_test,
    test_predictions
)

print("\n")
print("=" * 70)
print("TUNED GRADIENT BOOSTING — TEST RESULTS")
print("=" * 70)

print(tuned_metrics)

print("=" * 70)

# =========================================================
# HYPERMETER TUNING RANDOM FOREST MODEL
# =========================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf_grid_search = tune_random_forest(
    preprocessor=preprocessor,
    X_train=X_train,
    y_train=y_train
)

print("\nBest Parameters:")
print(
    rf_grid_search.best_params_
)

print("\nBest CV RMSE:")
print(
    -rf_grid_search.best_score_
)

best_rf_model = (
    rf_grid_search.best_estimator_
)

rf_test_predictions = (
    best_rf_model.predict(X_test)
)

tuned_rf_metrics = evaluate_model(
    y_test,
    rf_test_predictions
)

print("\n")
print("=" * 70)
print("TUNED RANDOM FOREST — TEST RESULTS")
print("=" * 70)

print(tuned_rf_metrics)

print("=" * 70)

# =========================================================
# FINAL MODEL EVALUATION
# =========================================================

final_results, final_predictions = evaluate_final_model(
    model=best_gb_model,
    X_test=X_test,
    y_test=y_test,
    evaluate_model=evaluate_model
)

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(
    final_results.to_string(index=False)
)

print("=" * 70)

final_results.to_csv(
    "results/final_model_results.csv",
    index=False
)

# =========================================================
# ERROR ANALYSIS
# =========================================================

error_df = create_error_analysis(
    X_test=X_test,
    y_test=y_test,
    predictions=final_predictions
)

print("\n")
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(
    error_df[
        [
            "Actual",
            "Predicted",
            "Residual",
            "Absolute_Error"
        ]
    ].head(10)
)

print("=" * 70)

important_columns = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "YearBuilt",
    "YearRemodAdd",
    "Neighborhood",
    "ExterQual",
    "KitchenQual",
    "FullBath",
    "TotRmsAbvGrd"
]

print("\n")
print("=" * 70)
print("WORST PREDICTIONS WITH HOUSE FEATURES")
print("=" * 70)

worst_predictions = (
    error_df
    .sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(10)
)

print("\n")
print("=" * 70)
print("TOP 10 WORST PREDICTIONS")
print("=" * 70)

print(
    worst_predictions[
        important_columns
        + [
            "Actual",
            "Predicted",
            "Residual",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)

print("=" * 70)

mean_residual = error_df["Residual"].mean()

print("\nMean Residual:")
print(mean_residual)

error_df.to_csv(
    "results/error_analysis.csv",
    index=False
)

worst_predictions.to_csv(
    "results/worst_predictions.csv",
    index=False
)

neighborhood_error = (
    error_df
    .groupby("Neighborhood")
    .agg(
        Mean_Absolute_Error=(
            "Absolute_Error",
            "mean"
        ),
        Mean_Residual=(
            "Residual",
            "mean"
        ),
        Count=(
            "Absolute_Error",
            "count"
        )
    )
    .sort_values(
        "Mean_Absolute_Error",
        ascending=False
    )
)

print("\n")
print("=" * 70)
print("ERROR BY NEIGHBORHOOD")
print("=" * 70)

print(
    neighborhood_error.to_string()
)

print("=" * 70)

quality_error = (
    error_df
    .groupby("OverallQual")
    .agg(
        Mean_Absolute_Error=(
            "Absolute_Error",
            "mean"
        ),
        Mean_Residual=(
            "Residual",
            "mean"
        ),
        Count=(
            "Absolute_Error",
            "count"
        )
    )
    .sort_index()
)

print("\n")
print("=" * 70)
print("ERROR BY OVERALL QUALITY")
print("=" * 70)

print(
    quality_error.to_string()
)

print("=" * 70)

error_df["Price_Group"] = pd.cut(
    error_df["Actual"],
    bins=[
        0,
        200000,
        400000,
        600000,
        800000,
        float("inf")
    ],
    labels=[
        "<200k",
        "200k-400k",
        "400k-600k",
        "600k-800k",
        "800k+"
    ]
)

price_error = (
    error_df
    .groupby(
        "Price_Group",
        observed=True
    )
    .agg(
        Mean_Absolute_Error=(
            "Absolute_Error",
            "mean"
        ),
        Mean_Residual=(
            "Residual",
            "mean"
        ),
        Count=(
            "Absolute_Error",
            "count"
        )
    )
)

print("\n")
print("=" * 70)
print("ERROR BY PRICE GROUP")
print("=" * 70)

print(
    price_error.to_string()
)

print("=" * 70)

plot_actual_vs_predicted(
    y_test,
    final_predictions
)

plot_residual_distribution(
    error_df["Residual"]
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = get_feature_importance(
    best_gb_model,
    top_n=20
)

print("\n")
print("=" * 70)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)

print("=" * 70)

plot_feature_importance(
    importance_df
)

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

# ============================================================
# FINAL GRADIENT BOOSTING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("FINAL GRADIENT BOOSTING MODEL")
print("=" * 70)


final_model = create_gradient_boosting_model(
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    learning_rate=0.03,
    max_depth=4,
    n_estimators=500
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

final_model.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = final_model.predict(
    X_test
)


# ============================================================
# FINAL EVALUATION
# ============================================================

final_results = evaluate_model(
    y_test,
    final_predictions
)

final_results["Model"] = "Tuned Gradient Boosting"


print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(final_results)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

save_model(
    final_model,
    "models/house_price_model.joblib"
)

