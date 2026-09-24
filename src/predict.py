import joblib
import pandas as pd
from pathlib import Path

# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

def load_model():

    model = joblib.load(MODEL_PATH)

    return model


def predict_price(house_data):

    model = load_model()

    house_df = pd.DataFrame([house_data])

    prediction = model.predict(house_df)

    return prediction[0]

house = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 70,
    "LotArea": 8450,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "BrkFace",
    "MasVnrArea": 196,
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 706,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 150,
    "TotalBsmtSF": 856,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 856,
    "2ndFlrSF": 854,
    "LowQualFinSF": 0,
    "GrLivArea": 1710,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 8,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": None,
    "GarageType": "Attchd",
    "GarageYrBlt": 2003,
    "GarageFinish": "RFn",
    "GarageCars": 2,
    "GarageArea": 548,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 0,
    "OpenPorchSF": 61,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": None,
    "Fence": None,
    "MiscFeature": None,
    "MiscVal": 0,
    "MoSold": 2,
    "YrSold": 2008,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}

predicted_price = predict_price(house)

print()
print("=" * 50)
print("HOUSE PRICE PREDICTION")
print("=" * 50)

print(
    f"Predicted Sale Price: ${predicted_price:,.2f}"
)

print("=" * 50)