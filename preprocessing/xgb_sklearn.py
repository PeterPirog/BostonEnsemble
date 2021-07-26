"""
An example training a XGBClassifier, performing
randomized search using TuneSearchCV.
"""
import pandas as pd
import numpy as np
from tune_sklearn import TuneSearchCV
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from pathlib import Path





# A parameter grid for XGBoost
params = {
    "n_estimators": [1, 5, 10],
    "max_depth": [3, 4, 5],
    "eta": [0.5, 1, 1.5, 2, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],

}

model = XGBRegressor()

"""
xgb = XGBClassifier(
    learning_rate=0.02,
    n_estimators=50,
    objective="binary:logistic",
    nthread=4,
    # tree_method="gpu_hist"  # this enables GPU.
    # See https://github.com/dmlc/xgboost/issues/2819
)
"""
sched_asha = ASHAScheduler(time_attr="training_iteration",
                           max_t=500,
                           grace_period=16,
                           # mode='max', #find maximum, do not define here if you define in tune.run
                           reduction_factor=3,
                           # brackets=1
                           )

cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=1)


digit_search = TuneSearchCV(
    estimator=model,
    param_distributions=params,
    n_trials=10,
    early_stopping=sched_asha,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    cv=cv,
    refit=True,
    error_score=-1000,
    verbose=2,
    return_train_score=True,
    local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
    name='xgb_sklearn',
    max_iters=100,
    use_gpu=False,
    loggers= ['csv'], #Possible values are “tensorboard”, “csv”, “mlflow”, and “json”
    pipeline_auto_early_stop=True,
    #stopper= TrialPlateauStopper https://docs.ray.io/en/master/tune/api_docs/stoppers.html?highlight=ray.tune.stopper.Stopper#ray.tune.Stopper
    #time_budget_s=
    mode='max'


)

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent



    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    features_all=['_OverallQual', '_GrLivArea', '_ExterQual', '_KitchenQual', '_BsmtQual', '_GarageArea', '_BuildingAge', '_TotalBsmtSF', '_GarageFinish', '_FullBath', '_YearRemodAdd', '_FireplaceQu', '_Foundation_1', '_HeatingQC', '_LotArea', '_OpenPorchSF', '_GarageType_Attchd', '_MasVnrArea', '_LotFrontage', '_BsmtFinType1', '_GarageType_Detchd', '_GarageQual', '_BsmtExposure', '_MSSubClass_3', '_CentralAir', '_WoodDeckSF', '_Foundation_2', '_Exterior_VinylSd', '_HalfBath', '_SaleCondition_Partial', '_MasVnrType_Stone', '_Electrical', '_BsmtFinSF1', '_PavedDrive', '_MSZoning_1', '_LotShape', '_BsmtCond', '_HouseStyle_1', '_Foundation_3', '_BedroomAbvGr', '_BsmtFullBath', '_Neighborhood_5', '_MasVnrType_BrkFace', '_GarageType_BuiltIn', '_EnclosedPorch', '_Neighborhood_9', '_SaleType_WD', '_BldgType_2', '_RoofStyle_1', '_Exterior_WdSdng', '_HouseStyle_3', '_Exterior_MetalSd', '_BsmtUnfSF', '_Neighborhood_8', '_Fence', '_SaleCondition_Abnorml', '_LotConfig_4', '_Functional', '_BldgType_1', '_Alley', '_Neighborhood_1', '_SaleCondition_Normal', '_ScreenPorch', '_HouseStyle_4', '_OverallCond', '_LotConfig_1', '_HouseStyle_2', '_Exterior_HdBoard', '_MSSubClass_2', '_QuarterSold', '_ExterCond', '_Neighborhood_2', '_YrSold', '_BsmtFinSF2', '_BldgType_3', '_Exterior_Plywood', '_LandContour_2', '_MSZoning_3', '_LotConfig_3']
    n_features=len(features_all)
    #n_features=3

    # define dataset

    X = df[features_all]
    y=df['SalePrice_log1']
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)


digit_search.fit(X, y)
print(digit_search.best_params_)
print(digit_search.best_score_)