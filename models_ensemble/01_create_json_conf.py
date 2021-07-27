import json



if __name__ == "__main__":
    # base_path = Path(__file__).parent.parent

    # Make global configuration
    conf_global = {}
    conf_global['all_features'] = ['_OverallQual', '_GrLivArea', '_ExterQual', '_KitchenQual', '_BsmtQual',
                                     '_GarageArea', '_BuildingAge', '_TotalBsmtSF', '_GarageFinish', '_FullBath',
                                     '_YearRemodAdd', '_FireplaceQu', '_Foundation_1', '_HeatingQC', '_LotArea',
                                     '_OpenPorchSF', '_GarageType_Attchd', '_MasVnrArea', '_LotFrontage',
                                     '_BsmtFinType1', '_GarageType_Detchd', '_GarageQual', '_BsmtExposure',
                                     '_MSSubClass_3', '_CentralAir', '_WoodDeckSF', '_Foundation_2',
                                     '_Exterior_VinylSd', '_HalfBath', '_SaleCondition_Partial', '_MasVnrType_Stone',
                                     '_Electrical', '_BsmtFinSF1', '_PavedDrive', '_MSZoning_1', '_LotShape',
                                     '_BsmtCond', '_HouseStyle_1', '_Foundation_3', '_BedroomAbvGr', '_BsmtFullBath',
                                     '_Neighborhood_5', '_MasVnrType_BrkFace', '_GarageType_BuiltIn', '_EnclosedPorch',
                                     '_Neighborhood_9', '_SaleType_WD', '_BldgType_2', '_RoofStyle_1',
                                     '_Exterior_WdSdng', '_HouseStyle_3', '_Exterior_MetalSd', '_BsmtUnfSF',
                                     '_Neighborhood_8', '_Fence', '_SaleCondition_Abnorml', '_LotConfig_4',
                                     '_Functional', '_BldgType_1', '_Alley', '_Neighborhood_1', '_SaleCondition_Normal',
                                     '_ScreenPorch', '_HouseStyle_4', '_OverallCond', '_LotConfig_1', '_HouseStyle_2',
                                     '_Exterior_HdBoard', '_MSSubClass_2', '_QuarterSold', '_ExterCond',
                                     '_Neighborhood_2', '_YrSold', '_BsmtFinSF2', '_BldgType_3', '_Exterior_Plywood',
                                     '_LandContour_2', '_MSZoning_3', '_LotConfig_3']
    conf_global['project_path'] = '/home/peterpirog/PycharmProjects/BostonEnsemble'
    conf_global['train_csv_path'] = "./data_files/train.csv"
    conf_global['train_csv_path'] = conf_global['project_path'] + "/data_files/train.csv"
    conf_global['test_csv_path'] = conf_global['project_path'] + "/data_files/test.csv"

    with open('conf_global.json', 'w') as fp:
        json.dump(conf_global, fp)

    #Make Ridge Configuration
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge

    conf_ridge={}
    conf_ridge['all_features']=conf_global['all_features']
    conf_ridge['n_features']=65
    conf_ridge['alpha']=3.5770773153506084
    conf_ridge['fit_intercept']=True
    conf_ridge['normalize']=False
    conf_ridge['copy_X']=True
    conf_ridge['max_iter']=None
    conf_ridge['tol']=0.001
    conf_ridge['solver']='auto'
    conf_ridge['random_state']=None

    with open('conf_ridge.json', 'w') as fp:
        json.dump(conf_ridge, fp)







    with open('conf_ridge.json', 'r') as fp:
        data2 = json.load(fp)

    print(data2)
