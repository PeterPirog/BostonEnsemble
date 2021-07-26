import pandas as pd
import numpy as np
import time
from pathlib import Path
import warnings
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings



@ignore_warnings(category=ConvergenceWarning)
def fit_and_score(estimator, max_iter, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.set_params(max_iter=1)
    estimator.set_params(random_state=0)




    start = time.time()
    estimator.fit(X_train, y_train)

    fit_time = time.time() - start
    n_iter = estimator.n_iter_
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)

    return fit_time, n_iter, train_score, test_score

if __name__ == "__main__":



    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    features_all=['_OverallQual', '_GrLivArea', '_ExterQual', '_KitchenQual', '_BsmtQual', '_GarageArea', '_BuildingAge', '_TotalBsmtSF', '_GarageFinish', '_FullBath', '_YearRemodAdd', '_FireplaceQu', '_Foundation_1', '_HeatingQC', '_LotArea', '_OpenPorchSF', '_GarageType_Attchd', '_MasVnrArea', '_LotFrontage', '_BsmtFinType1', '_GarageType_Detchd', '_GarageQual', '_BsmtExposure', '_MSSubClass_3', '_CentralAir', '_WoodDeckSF', '_Foundation_2', '_Exterior_VinylSd', '_HalfBath', '_SaleCondition_Partial', '_MasVnrType_Stone', '_Electrical', '_BsmtFinSF1', '_PavedDrive', '_MSZoning_1', '_LotShape', '_BsmtCond', '_HouseStyle_1', '_Foundation_3', '_BedroomAbvGr', '_BsmtFullBath', '_Neighborhood_5', '_MasVnrType_BrkFace', '_GarageType_BuiltIn', '_EnclosedPorch', '_Neighborhood_9', '_SaleType_WD', '_BldgType_2', '_RoofStyle_1', '_Exterior_WdSdng', '_HouseStyle_3', '_Exterior_MetalSd', '_BsmtUnfSF', '_Neighborhood_8', '_Fence', '_SaleCondition_Abnorml', '_LotConfig_4', '_Functional', '_BldgType_1', '_Alley', '_Neighborhood_1', '_SaleCondition_Normal', '_ScreenPorch', '_HouseStyle_4', '_OverallCond', '_LotConfig_1', '_HouseStyle_2', '_Exterior_HdBoard', '_MSSubClass_2', '_QuarterSold', '_ExterCond', '_Neighborhood_2', '_YrSold', '_BsmtFinSF2', '_BldgType_3', '_Exterior_Plywood', '_LandContour_2', '_MSZoning_3', '_LotConfig_3']
    n_features=len(features_all)

    n_features=66
    y=df['SalePrice_log1'].to_numpy()
    X = df[features_all[:66]].to_numpy()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0,shuffle=True)



    model = Ridge(alpha=3.5770773153506084,max_iter=1,solver='sag')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(5000):
            model.fit(X_train,y_train)
            y_val_predict=model.predict(X_test)
            val_error=mean_squared_error(y_val_predict,y_test)
            #print(val_error)#

    print(val_error)
    """
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    # evaluate_model

    #fit_params = {'early_stopping_rounds': 30}
    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)


    # force scores to be positive
    scores = abs(scores)
    M=scores.mean()
    STD=scores.std()

    print(f'Number of Features: {n_features}','Mean RMSLE: %.4f (%.4f)' % (M, STD))


    #print(results)
    print(f'Minimum error is: {M} for {n_features} features')
    #np.save("linear_errors.npy",results)
    #np.save("linear_errors.npy",results)
  
    df = pd.DataFrame(data=results, columns=["Features", "Mean","STD","Mean_2STD"])
    df.to_excel(
        'model_ridge_features.xlsx',
        sheet_name='Linear',
        index=False)
    """
