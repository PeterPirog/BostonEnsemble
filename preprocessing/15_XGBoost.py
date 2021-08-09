import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    #features_all=['_OverallQual', '_GrLivArea', '_ExterQual', '_KitchenQual', '_BsmtQual', '_GarageArea', '_BuildingAge', '_TotalBsmtSF', '_GarageFinish', '_FullBath', '_YearRemodAdd', '_FireplaceQu', '_Foundation_1', '_HeatingQC', '_LotArea', '_OpenPorchSF', '_GarageType_Attchd', '_MasVnrArea', '_LotFrontage', '_BsmtFinType1', '_GarageType_Detchd', '_GarageQual', '_BsmtExposure', '_MSSubClass_3', '_CentralAir', '_WoodDeckSF', '_Foundation_2', '_Exterior_VinylSd', '_HalfBath', '_SaleCondition_Partial', '_MasVnrType_Stone', '_Electrical', '_BsmtFinSF1', '_PavedDrive', '_MSZoning_1', '_LotShape', '_BsmtCond', '_HouseStyle_1', '_Foundation_3', '_BedroomAbvGr', '_BsmtFullBath', '_Neighborhood_5', '_MasVnrType_BrkFace', '_GarageType_BuiltIn', '_EnclosedPorch', '_Neighborhood_9', '_SaleType_WD', '_BldgType_2', '_RoofStyle_1', '_Exterior_WdSdng', '_HouseStyle_3', '_Exterior_MetalSd', '_BsmtUnfSF', '_Neighborhood_8', '_Fence', '_SaleCondition_Abnorml', '_LotConfig_4', '_Functional', '_BldgType_1', '_Alley', '_Neighborhood_1', '_SaleCondition_Normal', '_ScreenPorch', '_HouseStyle_4', '_OverallCond', '_LotConfig_1', '_HouseStyle_2', '_Exterior_HdBoard', '_MSSubClass_2', '_QuarterSold', '_ExterCond', '_Neighborhood_2', '_YrSold', '_BsmtFinSF2', '_BldgType_3', '_Exterior_Plywood', '_LandContour_2', '_MSZoning_3', '_LotConfig_3']
    features_all=['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea', '_MSSubClass_3',
                                   '_OverallQual', '_BuildingAge', '_TotalBsmtSF', '_Functional', '_CentralAir',
                                   '_Electrical', '_SaleCondition_Abnorml', '_RoofStyle_1', '_LotArea', '_GarageArea',
                                   '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_SaleType_WD', '_ScreenPorch',
                                   '_BsmtExposure', '_ExterQual', '_BsmtUnfSF', '_Foundation_2', '_HouseStyle_2',
                                   '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn', '_FullBath',
                                   '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal',
                                   '_BsmtFinType1', '_PavedDrive', '_Foundation_3', '_MSZoning_1', '_Neighborhood_5',
                                   '_HeatingQC', '_YrSold', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
                                   '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF', '_Exterior_VinylSd', '_MSSubClass_1',
                                   '_GarageType_Attchd', '_LotFrontage', '_Exterior_HdBoard', '_HouseStyle_4',
                                   '_MasVnrType_BrkFace', '_Exterior_Plywood', '_GarageQual', '_MasVnrType_Stone',
                                   '_LandContour_2', '_BsmtFullBath', '_LotShape', '_Exterior_WdSdng',
                                   '_Neighborhood_8', '_Fence', '_LotConfig_1', '_Alley', '_Exterior_MetalSd',
                                   '_EnclosedPorch', '_LotConfig_3', '_BsmtCond', '_MasVnrArea',
                                   '_SaleCondition_Partial', '_GarageType_Detchd', '_MSZoning_3', '_ExterCond',
                                   '_Neighborhood_2', '_QuarterSold', '_BsmtFinSF1', '_BedroomAbvGr', '_OpenPorchSF',
                                   '_Foundation_1', '_MSSubClass_2']
    n_features=len(features_all)
    print(f'n_features={n_features}')
    #n_features=3
    y=df['SalePrice_log1']

    results=np.zeros((n_features,4))
    min_error=np.inf
    idx_min=None
    for i in range(n_features):

        X = df[features_all[:i+1]]
        model = XGBRegressor(n_estimators=144,
                             max_depth=6,
                             eta=0.1,
                             subsample=1,
                             colsample_bytree=1)
    
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
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
    
        print(f'Number of Features: {i+1}','Mean RMSLE: %.4f (%.4f)' % (M, STD))
        results[i,0]=i+1
        results[i, 1] = M
        results[i, 2] = STD
        results[i, 3] = M+2*STD
        if M<min_error:
            min_error=M
            idx_min=i+1

    #print(results)
    print(f'Minimum error is: {min_error} for {idx_min} features')
    np.save("linear_errors.npy",results)
    np.save("linear_errors.npy",results)
    df = pd.DataFrame(data=results, columns=["Features", "Mean","STD","Mean_2STD"])
    df.to_excel(
        'model_xgboost_features.xlsx',
        sheet_name='Linear',
        index=False)

