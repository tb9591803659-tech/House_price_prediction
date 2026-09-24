import pandas as pd


def create_features(df):

    df = df.copy()

    # ==========================================
    # Total house area
    # ==========================================

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # ==========================================
    # Total bathrooms
    # ==========================================

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # ==========================================
    # House age at time of sale
    # ==========================================

    df["HouseAge"] = (
        df["YrSold"]
        - df["YearBuilt"]
    )

    # ==========================================
    # Age since remodeling
    # ==========================================

    df["RemodAge"] = (
        df["YrSold"]
        - df["YearRemodAdd"]
    )

    # ==========================================
    # Total porch/deck area
    # ==========================================

    df["TotalPorchSF"] = (
        df["WoodDeckSF"]
        + df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    # ==========================================
    # Average area per room
    # ==========================================

    df["TotalSFPerRoom"] = (
        df["TotalSF"]
        / (df["TotRmsAbvGrd"] + 1)
    )

    return df

