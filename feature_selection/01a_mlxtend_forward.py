import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from models_ensemble.ensemble_tools import FeatureByNameSelector, Validator, read_config_files
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    all_features = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea', '_MSSubClass_3',
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

    df = pd.read_csv(conf_global['encoded_train_data'])
    n = len(conf_global['all_features'])

    X = df[all_features]
    y = df['SalePrice_log1']

    # step backward feature selection algorithm

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sfs = SFS(ElasticNet(alpha=0.00010221867853787662,
                         l1_ratio=0.9784366976103005),
              k_features=52,
              forward=True,
              floating=False,
              verbose=2,
              scoring='r2',
              cv=cv)

    # sfs = sfs.fit(np.array(X_train), y_train)
    sfs = sfs.fit(X, y)
    print(X.columns[list(sfs.k_feature_idx_)])

"""
80 features forward
[2021-08-07 15:09:03] Features: 80/80 -- score: 0.8599783486715777Index(['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
       '_MSSubClass_3', '_OverallQual', '_BuildingAge', '_TotalBsmtSF',
       '_Functional', '_CentralAir', '_Electrical', '_SaleCondition_Abnorml',
       '_RoofStyle_1', '_LotArea', '_GarageArea', '_KitchenQual',
       '_OverallCond', '_Neighborhood_9', '_SaleType_WD', '_ScreenPorch',
       '_BsmtExposure', '_ExterQual', '_BsmtUnfSF', '_Foundation_2',
       '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
       '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_BsmtQual',
       '_SaleCondition_Normal', '_BsmtFinType1', '_PavedDrive',
       '_Foundation_3', '_MSZoning_1', '_Neighborhood_5', '_HeatingQC',
       '_YrSold', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
       '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF', '_Exterior_VinylSd',
       '_MSSubClass_1', '_GarageType_Attchd', '_LotFrontage',
       '_Exterior_HdBoard', '_HouseStyle_4', '_MasVnrType_BrkFace',
       '_Exterior_Plywood', '_GarageQual', '_MasVnrType_Stone',
       '_LandContour_2', '_BsmtFullBath', '_LotShape', '_Exterior_WdSdng',
       '_Neighborhood_8', '_Fence', '_LotConfig_1', '_Alley',
       '_Exterior_MetalSd', '_EnclosedPorch', '_LotConfig_3', '_BsmtCond',
       '_MasVnrArea', '_SaleCondition_Partial', '_GarageType_Detchd',
       '_MSZoning_3', '_ExterCond', '_Neighborhood_2', '_QuarterSold',
       '_BsmtFinSF1', '_BedroomAbvGr', '_OpenPorchSF', '_Foundation_1',
       '_MSSubClass_2'],
      dtype='object')
      
      best score:[2021-08-07 14:58:16] Features: 58/80 -- score: 0.8671051156090058[Parallel(n_jobs=1)]:
"""
