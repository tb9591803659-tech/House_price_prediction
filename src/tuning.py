from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def tune_gradient_boosting(
    preprocessor,
    X_train,
    y_train
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                GradientBoostingRegressor(
                    random_state=42
                )
            )
        ]
    )

    param_grid = {
        "regressor__n_estimators": [
            200,
            300,
            500
        ],

        "regressor__learning_rate": [
            0.03,
            0.05,
            0.1
        ],

        "regressor__max_depth": [
            2,
            3,
            4
        ]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search

    



def tune_random_forest(
    preprocessor,
    X_train,
    y_train
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    param_grid = {
    "regressor__n_estimators": [
        200,
        300
    ],

    "regressor__max_depth": [
        None,
        15
    ],

    "regressor__min_samples_leaf": [
        1,
        2
    ]
}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search