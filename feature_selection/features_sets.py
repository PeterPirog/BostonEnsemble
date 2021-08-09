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

# mlxtend 52 features backward RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=10)
backward_forest_r2 = ['_BldgType_1', '_BldgType_3', '_GrLivArea', '_OverallQual',
                      '_BuildingAge', '_TotalBsmtSF', '_CentralAir', '_SaleCondition_Abnorml',
                      '_LotArea', '_GarageArea', '_KitchenQual', '_OverallCond',
                      '_Neighborhood_9', '_BsmtExposure', '_ExterQual', '_BsmtUnfSF',
                      '_Foundation_2', '_HouseStyle_2', '_HouseStyle_3', '_FullBath',
                      '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal',
                      '_Foundation_3', '_MSZoning_1', '_Neighborhood_5', '_HalfBath',
                      '_YearRemodAdd', '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF',
                      '_Exterior_VinylSd', '_Exterior_HdBoard', '_MasVnrType_BrkFace',
                      '_GarageQual', '_LandContour_2', '_BsmtFullBath', '_Neighborhood_8',
                      '_Fence', '_LotConfig_1', '_Alley', '_LotConfig_3', '_BsmtCond',
                      '_SaleCondition_Partial', '_GarageType_Detchd', '_MSZoning_3',
                      '_ExterCond', '_Neighborhood_2', '_BsmtFinSF1', '_OpenPorchSF',
                      '_MSSubClass_2']
backward_forest = ['_GrLivArea', '_BuildingAge', '_TotalBsmtSF', '_CentralAir', '_LotArea',
                   '_GarageArea', '_KitchenQual', '_OverallCond', '_Neighborhood_9',
                   '_BsmtExposure', '_Foundation_2', '_HouseStyle_2', '_FullBath',
                   '_FireplaceQu', '_BsmtQual', '_BsmtFinType1', '_Foundation_3',
                   '_MSZoning_1', '_Neighborhood_5', '_YearRemodAdd', '_GarageFinish',
                   '_BsmtFinSF2', '_Exterior_Plywood', '_GarageQual', '_Neighborhood_8',
                   '_Fence', '_LotConfig_3', '_SaleCondition_Partial', '_MSZoning_3',
                   '_ExterCond', '_Neighborhood_2', '_BsmtFinSF1', '_MSSubClass_2']

# mlxtend 52 features forward RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=10)
forward_forest_r2 = ['_BldgType_2', '_BldgType_1', '_GrLivArea', '_OverallQual',
                     '_BuildingAge', '_TotalBsmtSF', '_CentralAir', '_SaleCondition_Abnorml',
                     '_LotArea', '_GarageArea', '_KitchenQual', '_OverallCond',
                     '_Neighborhood_9', '_ScreenPorch', '_ExterQual', '_Foundation_2',
                     '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn', '_FullBath',
                     '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal', '_Foundation_3',
                     '_MSZoning_1', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
                     '_HouseStyle_1', '_BsmtFinSF2', '_MSSubClass_1', '_GarageType_Attchd',
                     '_HouseStyle_4', '_MasVnrType_BrkFace', '_Exterior_Plywood',
                     '_GarageQual', '_LandContour_2', '_BsmtFullBath', '_LotShape',
                     '_Exterior_WdSdng', '_Neighborhood_8', '_Fence', '_LotConfig_1',
                     '_Alley', '_LotConfig_3', '_GarageType_Detchd', '_MSZoning_3',
                     '_ExterCond', '_Neighborhood_2', '_BsmtFinSF1', '_BedroomAbvGr',
                     '_Foundation_1']
forward_forest = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
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
                  '_MSSubClass_2']

# mlxtend 52 features backward ElasticNet(alpha=0.00010221867853787662,l1_ratio=0.9784366976103005)
backward_elastic_r2 = ['_BldgType_2', '_GrLivArea', '_MSSubClass_3', '_OverallQual',
                       '_TotalBsmtSF', '_Functional', '_CentralAir', '_Electrical',
                       '_SaleCondition_Abnorml', '_RoofStyle_1', '_LotArea', '_GarageArea',
                       '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_ScreenPorch',
                       '_BsmtExposure', '_ExterQual', '_BsmtUnfSF', '_Foundation_2',
                       '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
                       '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_BsmtQual',
                       '_BsmtFinType1', '_PavedDrive', '_Foundation_3', '_MSZoning_1',
                       '_Neighborhood_5', '_HeatingQC', '_YrSold', '_HalfBath',
                       '_YearRemodAdd', '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF',
                       '_MSSubClass_1', '_HouseStyle_4', '_GarageQual', '_MasVnrType_Stone',
                       '_BsmtFullBath', '_LotConfig_1', '_Exterior_MetalSd', '_LotConfig_3',
                       '_SaleCondition_Partial', '_GarageType_Detchd', '_BsmtFinSF1',
                       '_OpenPorchSF']
backward_elastic = ['_BldgType_2', '_GrLivArea', '_MSSubClass_3', '_OverallQual',
                    '_TotalBsmtSF', '_Functional', '_CentralAir', '_SaleCondition_Abnorml',
                    '_LotArea', '_GarageArea', '_KitchenQual', '_OverallCond',
                    '_Neighborhood_9', '_ScreenPorch', '_BsmtExposure', '_ExterQual',
                    '_Foundation_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
                    '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_BsmtQual',
                    '_PavedDrive', '_Foundation_3', '_Neighborhood_5', '_HeatingQC',
                    '_YrSold', '_HalfBath', '_YearRemodAdd', '_WoodDeckSF', '_MSSubClass_1',
                    '_HouseStyle_4', '_GarageQual', '_MasVnrType_Stone', '_BsmtFullBath',
                    '_LotConfig_1', '_Exterior_MetalSd', '_LotConfig_3',
                    '_SaleCondition_Partial', '_GarageType_Detchd', '_BsmtFinSF1']

# mlxtend 52 features forward ElasticNet(alpha=0.00010221867853787662,l1_ratio=0.9784366976103005)
forward_elastic_r2 = ['_BldgType_2', '_GrLivArea', '_OverallQual', '_BuildingAge',
                      '_TotalBsmtSF', '_Functional', '_CentralAir', '_Electrical',
                      '_SaleCondition_Abnorml', '_LotArea', '_GarageArea', '_KitchenQual',
                      '_OverallCond', '_Neighborhood_9', '_ScreenPorch', '_BsmtExposure',
                      '_ExterQual', '_BsmtUnfSF', '_Foundation_2', '_LotConfig_4',
                      '_GarageType_BuiltIn', '_FullBath', '_Neighborhood_1', '_FireplaceQu',
                      '_BsmtQual', '_BsmtFinType1', '_PavedDrive', '_Foundation_3',
                      '_MSZoning_1', '_Neighborhood_5', '_HeatingQC', '_YrSold', '_HalfBath',
                      '_YearRemodAdd', '_GarageFinish', '_BsmtFinSF2', '_WoodDeckSF',
                      '_MSSubClass_1', '_HouseStyle_4', '_GarageQual', '_MasVnrType_Stone',
                      '_BsmtFullBath', '_Fence', '_Exterior_MetalSd', '_BsmtCond',
                      '_SaleCondition_Partial', '_GarageType_Detchd', '_Neighborhood_2',
                      '_BsmtFinSF1', '_OpenPorchSF', '_Foundation_1', '_MSSubClass_2']
forward_elastic = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
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
                   '_MSSubClass_2']

# mlxtend 52 features forward KNeighborsRegressor(n_neighbors=7, weights='distance',algorithm='auto',leaf_size=83,p=1,metric='minkowski')
forward_kneighbors_r2 = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
                         '_OverallQual', '_BuildingAge', '_TotalBsmtSF', '_Functional',
                         '_CentralAir', '_Electrical', '_SaleCondition_Abnorml', '_LotArea',
                         '_GarageArea', '_KitchenQual', '_OverallCond', '_Neighborhood_9',
                         '_ScreenPorch', '_BsmtExposure', '_ExterQual', '_BsmtUnfSF',
                         '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
                         '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_PavedDrive',
                         '_Foundation_3', '_Neighborhood_5', '_HalfBath', '_YearRemodAdd',
                         '_GarageFinish', '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF',
                         '_MSSubClass_1', '_LotFrontage', '_HouseStyle_4', '_GarageQual',
                         '_MasVnrType_Stone', '_LandContour_2', '_Neighborhood_8', '_Alley',
                         '_Exterior_MetalSd', '_BsmtCond', '_GarageType_Detchd', '_MSZoning_3',
                         '_ExterCond', '_BsmtFinSF1', '_BedroomAbvGr', '_MSSubClass_2']
forward_kneighbors = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
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
                      '_MSSubClass_2']

# mlxtend 52 features backward KNeighborsRegressor(n_neighbors=7, weights='distance',algorithm='auto',leaf_size=83,p=1,metric='minkowski')
backward_kneighbors_r2 = ['_BldgType_1', '_GrLivArea', '_MSSubClass_3', '_OverallQual',
                          '_BuildingAge', '_TotalBsmtSF', '_Functional', '_CentralAir',
                          '_Electrical', '_RoofStyle_1', '_LotArea', '_GarageArea',
                          '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_BsmtExposure',
                          '_ExterQual', '_BsmtUnfSF', '_HouseStyle_2', '_FullBath',
                          '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal', '_PavedDrive',
                          '_Foundation_3', '_MSZoning_1', '_HeatingQC', '_YrSold', '_HalfBath',
                          '_YearRemodAdd', '_GarageFinish', '_WoodDeckSF', '_Exterior_VinylSd',
                          '_LotFrontage', '_Exterior_HdBoard', '_GarageQual', '_MasVnrType_Stone',
                          '_LandContour_2', '_BsmtFullBath', '_LotShape', '_Exterior_WdSdng',
                          '_Neighborhood_8', '_Exterior_MetalSd', '_EnclosedPorch', '_MasVnrArea',
                          '_GarageType_Detchd', '_ExterCond', '_Neighborhood_2', '_QuarterSold',
                          '_BsmtFinSF1', '_BedroomAbvGr', '_Foundation_1']
backward_neighbors = ['_GrLivArea', '_OverallQual', '_TotalBsmtSF', '_CentralAir', '_LotArea',
                      '_GarageArea', '_OverallCond', '_ExterQual', '_BsmtUnfSF',
                      '_HouseStyle_2', '_FullBath', '_FireplaceQu', '_BsmtQual',
                      '_MSZoning_1', '_YearRemodAdd', '_GarageFinish', '_BsmtFullBath',
                      '_Neighborhood_2', '_BsmtFinSF1', '_BedroomAbvGr']

# mlxtend 52 features backward  XGBRegressor(n_estimators=144,max_depth=6,eta=0.1,subsample=1,colsample_bytree=1)
backward_xgboost_r2 = ['_BldgType_2', '_GrLivArea', '_OverallQual', '_BuildingAge', '_TotalBsmtSF', '_Functional',
                       '_CentralAir', '_Electrical', '_SaleCondition_Abnorml', '_RoofStyle_1', '_LotArea',
                       '_GarageArea',
                       '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_SaleType_WD', '_ScreenPorch',
                       '_BsmtExposure',
                       '_BsmtUnfSF', '_LotConfig_4', '_FullBath', '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal',
                       '_BsmtFinType1', '_PavedDrive', '_MSZoning_1', '_Neighborhood_5', '_HeatingQC', '_YrSold',
                       '_YearRemodAdd', '_GarageFinish', '_Exterior_VinylSd', '_GarageType_Attchd', '_LotFrontage',
                       '_HouseStyle_4', '_GarageQual', '_BsmtFullBath', '_LotShape', '_Neighborhood_8',
                       '_EnclosedPorch',
                       '_BsmtCond', '_MasVnrArea', '_GarageType_Detchd', '_MSZoning_3', '_ExterCond', '_Neighborhood_2',
                       '_QuarterSold', '_BsmtFinSF1', '_BedroomAbvGr', '_OpenPorchSF', '_Foundation_1']
# c
# mlxtend 52 features forward  XGBRegressor(n_estimators=144,max_depth=6,eta=0.1,subsample=1,colsample_bytree=1)
forward_xgboost_r2 = ['_BldgType_3', '_GrLivArea', '_OverallQual', '_BuildingAge',
                      '_Electrical', '_SaleCondition_Abnorml', '_RoofStyle_1', '_LotArea',
                      '_GarageArea', '_KitchenQual', '_OverallCond', '_Neighborhood_9',
                      '_ScreenPorch', '_ExterQual', '_BsmtUnfSF', '_Foundation_2',
                      '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn',
                      '_FullBath', '_Neighborhood_1', '_SaleCondition_Normal',
                      '_Foundation_3', '_MSZoning_1', '_HeatingQC', '_YrSold', '_HalfBath',
                      '_HouseStyle_1', '_BsmtFinSF2', '_Exterior_VinylSd', '_MSSubClass_1',
                      '_Exterior_HdBoard', '_HouseStyle_4', '_MasVnrType_BrkFace',
                      '_Exterior_Plywood', '_MasVnrType_Stone', '_LandContour_2',
                      '_Exterior_WdSdng', '_Neighborhood_8', '_Fence', '_LotConfig_1',
                      '_Exterior_MetalSd', '_LotConfig_3', '_SaleCondition_Partial',
                      '_GarageType_Detchd', '_Neighborhood_2', '_QuarterSold', '_BsmtFinSF1',
                      '_BedroomAbvGr', '_Foundation_1', '_MSSubClass_2']
#forward_xgboost=

# mlxtend 52 features backward  SVR(kernel='rbf', degree=3, gamma='scale', C=0.7832573311079015, epsilon=0.04825896120073174)
backward_svr_r2 = ['_BldgType_2', '_BldgType_1', '_GrLivArea', '_OverallQual',
                   '_BuildingAge', '_TotalBsmtSF', '_Functional', '_CentralAir',
                   '_Electrical', '_SaleCondition_Abnorml', '_LotArea', '_GarageArea',
                   '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_SaleType_WD',
                   '_ScreenPorch', '_ExterQual', '_HouseStyle_2', '_FullBath',
                   '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_PavedDrive',
                   '_Foundation_3', '_MSZoning_1', '_Neighborhood_5', '_HeatingQC',
                   '_YrSold', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
                   '_HouseStyle_1', '_WoodDeckSF', '_MSSubClass_1', '_GarageType_Attchd',
                   '_HouseStyle_4', '_GarageQual', '_MasVnrType_Stone', '_LandContour_2',
                   '_BsmtFullBath', '_LotShape', '_Alley', '_Exterior_MetalSd',
                   '_BsmtCond', '_SaleCondition_Partial', '_GarageType_Detchd',
                   '_Neighborhood_2', '_BsmtFinSF1', '_BedroomAbvGr', '_Foundation_1',
                   '_MSSubClass_2']

# mlxtend 52 features forward  SVR(kernel='rbf', degree=3, gamma='scale', C=0.7832573311079015, epsilon=0.04825896120073174)
forward_svr_r2 = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
                  '_MSSubClass_3', '_OverallQual', '_BuildingAge', '_TotalBsmtSF',
                  '_Functional', '_CentralAir', '_Electrical', '_LotArea', '_GarageArea',
                  '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_SaleType_WD',
                  '_ScreenPorch', '_ExterQual', '_HouseStyle_2', '_HouseStyle_3',
                  '_GarageType_BuiltIn', '_FullBath', '_Neighborhood_1', '_FireplaceQu',
                  '_BsmtQual', '_PavedDrive', '_Foundation_3', '_Neighborhood_5',
                  '_HeatingQC', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
                  '_HouseStyle_1', '_WoodDeckSF', '_MSSubClass_1', '_GarageType_Attchd',
                  '_LotFrontage', '_HouseStyle_4', '_GarageQual', '_MasVnrType_Stone',
                  '_LandContour_2', '_Neighborhood_8', '_Exterior_MetalSd', '_BsmtCond',
                  '_SaleCondition_Partial', '_GarageType_Detchd', '_Neighborhood_2',
                  '_BsmtFinSF1', '_BedroomAbvGr', '_Foundation_1', '_MSSubClass_2']

# mlxtend 52 features backward  LGBMRegressor(boosting_type='gbdt',num_leaves=31, max_depth=- 1,learning_rate=0.1,n_estimators=100)
backward_lgbm_r2 = ['_BldgType_1', '_GrLivArea', '_MSSubClass_3', '_OverallQual',
                    '_BuildingAge', '_TotalBsmtSF', '_Functional', '_CentralAir',
                    '_SaleCondition_Abnorml', '_LotArea', '_KitchenQual', '_OverallCond',
                    '_ScreenPorch', '_BsmtExposure', '_ExterQual', '_BsmtUnfSF',
                    '_HouseStyle_2', '_HouseStyle_3', '_LotConfig_4', '_FullBath',
                    '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal', '_BsmtFinType1',
                    '_PavedDrive', '_MSZoning_1', '_Neighborhood_5', '_YrSold', '_HalfBath',
                    '_YearRemodAdd', '_GarageFinish', '_WoodDeckSF', '_Exterior_VinylSd',
                    '_GarageType_Attchd', '_LotFrontage', '_Exterior_HdBoard',
                    '_GarageQual', '_BsmtFullBath', '_LotShape', '_Neighborhood_8',
                    '_LotConfig_1', '_EnclosedPorch', '_BsmtCond', '_MasVnrArea',
                    '_GarageType_Detchd', '_MSZoning_3', '_ExterCond', '_Neighborhood_2',
                    '_QuarterSold', '_BsmtFinSF1', '_BedroomAbvGr', '_Foundation_1']
# backward_lgbm_=

# mlxtend 52 features forward  LGBMRegressor(boosting_type='gbdt',num_leaves=31, max_depth=- 1,learning_rate=0.1,n_estimators=100)
forward_lgbm_r2 = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea',
                   '_MSSubClass_3', '_OverallQual', '_BuildingAge', '_TotalBsmtSF',
                   '_Functional', '_CentralAir', '_Electrical', '_SaleCondition_Abnorml',
                   '_RoofStyle_1', '_LotArea', '_OverallCond', '_SaleType_WD',
                   '_ScreenPorch', '_HouseStyle_2', '_LotConfig_4', '_GarageType_BuiltIn',
                   '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_BsmtFinType1',
                   '_Foundation_3', '_MSZoning_1', '_Neighborhood_5', '_YrSold',
                   '_GarageFinish', '_HouseStyle_1', '_Exterior_VinylSd',
                   '_Exterior_HdBoard', '_HouseStyle_4', '_MasVnrType_BrkFace',
                   '_Exterior_Plywood', '_MasVnrType_Stone', '_LandContour_2',
                   '_BsmtFullBath', '_Exterior_WdSdng', '_Fence', '_LotConfig_1', '_Alley',
                   '_Exterior_MetalSd', '_LotConfig_3', '_SaleCondition_Partial',
                   '_MSZoning_3', '_ExterCond', '_Neighborhood_2', '_QuarterSold',
                   '_BsmtFinSF1', '_BedroomAbvGr', '_MSSubClass_2']

if __name__ == "__main__":
    list_of_results = [backward_forest_r2,
                       forward_forest_r2,
                       backward_forest,
                       forward_forest,
                       backward_elastic_r2,
                       forward_elastic_r2,
                       backward_elastic,
                       forward_elastic,
                       backward_kneighbors_r2,
                       forward_kneighbors_r2,
                       backward_neighbors,
                       forward_kneighbors,
                       backward_xgboost_r2,
                       forward_xgboost_r2,
                       backward_svr_r2,
                       forward_svr_r2,
                       backward_lgbm_r2]
    importance_dict = {}

    for feature in all_features:
        importance_dict[feature] = 100  # maximum initial value

        for single_list in list_of_results:
            # print(single_list, '\n')
            if feature in single_list:
                if single_list.index(feature) + 1 < importance_dict[feature]:
                    importance_dict[feature] = single_list.index(
                        feature) + 1  # get the best rank from any mlxtend score

    print(importance_dict)
    sorted_x = dict(sorted(importance_dict.items(), key=lambda item: item[1]))

    print(len(sorted_x), sorted_x)
    final_list = []
    for i, feature in enumerate(sorted_x):
        print(f'{i + 1}:{feature}, best rank: {sorted_x[feature]}')
        final_list.append(feature)
    print(final_list)
