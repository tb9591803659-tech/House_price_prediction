def handle_missing_values(df):
    """
    Handle missing values where NaN means
    that a house feature does not exist.
    """

    df = df.copy()

    no_feature_columns = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType"
    ]

    for column in no_feature_columns:

        if column in df.columns:
            df[column] = df[column].fillna("None")

    return df

def basic_dataset_info(df):

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def identify_features(df):

    target = "SalePrice"
    id_column = "Id"

    # Object columns are naturally categorical
    categorical_features = (
        df.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    # MSSubClass is stored as a number,
    # but represents a house/building class.
    # Therefore we treat it as categorical.
    if "MSSubClass" in df.columns:
        categorical_features.append("MSSubClass")

    # Numerical features
    numerical_features = (
        df.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    # Remove target and ID
    if target in numerical_features:
        numerical_features.remove(target)

    if id_column in numerical_features:
        numerical_features.remove(id_column)

    # MSSubClass is handled as categorical
    if "MSSubClass" in numerical_features:
        numerical_features.remove("MSSubClass")

    return numerical_features, categorical_features


def split_features_target(df):

    X = df.drop(
        columns=["SalePrice", "Id"]
    )

    y = df["SalePrice"]

    return X, y