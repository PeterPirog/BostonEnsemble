import json
from pathlib import Path


def read_config_files(configuration_name='conf_global'):
    base_path = Path(__file__).parent.parent
    file = str(base_path) + '/models_ensemble/' + configuration_name + '.json'

    with open(file, 'r') as fp:
        data = json.load(fp)
    return data


if __name__ == "__main__":
    conf_global = {}  # dict with global project configuration
    conf_ridge = {}  # dict with configuration for ridge model
    conf_lasso = {}  # dict with configuration for lasso model
    conf_elastic={}  # dict with configuration for elastic model
    conf_svr = {}  # dict with configuration for SVR model
    conf_kneighbors= {} # dict with configuration for kneighbors model
    conf_bridge = {}  # dict with configuration for  bayesian ridge model

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
    conf_global['train_csv_path'] = conf_global['project_path'] + "/data_files/train.csv"
    conf_global['test_csv_path'] = conf_global['project_path'] + "/data_files/test.csv"

    # Paths to files after domain encoding
    conf_global['domain_train_data'] = conf_global['project_path'] + '/data_files/domain_train_df.csv'
    conf_global['domain_test_data'] = conf_global['project_path'] + '/data_files/domain_test_df.csv'

    # Paths to files after feature engineering
    conf_global['encoded_train_data'] = conf_global['project_path'] + '/data_files/encoded_train_df.csv'
    conf_global['encoded_test_data'] = conf_global['project_path'] + '/data_files/encoded_test_df.csv'

    conf_global['target_label'] = 'SalePrice_log1'

    with open('conf_global.json', 'w') as fp:
        json.dump(conf_global, fp)

    # Make Ridge Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge

    conf_ridge['all_features'] = conf_global['all_features']
    conf_ridge['n_features'] = 65

    conf_ridge['alpha'] = 3.5770773153506084
    conf_ridge['fit_intercept'] = True
    conf_ridge['normalize'] = False
    conf_ridge['copy_X'] = True
    conf_ridge['max_iter'] = None
    conf_ridge['tol'] = 0.001
    conf_ridge['solver'] = 'auto'
    conf_ridge['random_state'] = None

    conf_ridge['output_file'] = 'model_ridge.joblib'
    conf_ridge['json_file'] = 'conf_ridge.json'

    with open(conf_ridge['json_file'], 'w') as fp:
        json.dump(conf_ridge, fp)

    # Make Lasso Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso

    conf_lasso['all_features'] = conf_global['all_features']
    conf_lasso['n_features'] = 73

    conf_lasso['alpha'] = 0.0007170579876083573
    conf_lasso['fit_intercept'] = True
    conf_lasso['normalize'] = False
    conf_lasso['precompute'] = False
    conf_lasso['copy_X'] = True
    conf_lasso['max_iter'] = 2000
    conf_lasso['tol'] = 0.0001
    conf_lasso['warm_start'] = False
    conf_lasso['positive'] = False
    conf_lasso['random_state'] = 42
    conf_lasso['selection'] = 'cyclic'

    conf_lasso['output_file'] = 'model_lasso.joblib'
    conf_lasso['json_file'] = 'conf_lasso.json'

    with open(conf_lasso['json_file'], 'w') as fp:
        json.dump(conf_lasso, fp)


    # Make Elastic Net Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html?highlight=elastic#sklearn.linear_model.ElasticNet

    conf_elastic['all_features'] = conf_global['all_features']
    conf_elastic['n_features'] = 74

    conf_elastic['alpha'] = 0.00073
    conf_elastic['l1_ratio'] = 1.0
    conf_elastic['fit_intercept'] = True
    conf_elastic['normalize'] = False
    conf_elastic['precompute'] = False
    conf_elastic['copy_X'] = True
    conf_elastic['max_iter'] = 1000
    conf_elastic['tol'] = 0.0001
    conf_elastic['warm_start'] = False
    conf_elastic['positive'] = False
    conf_elastic['random_state'] = 42
    conf_elastic['selection'] = 'cyclic'

    conf_elastic['output_file'] = 'model_elastic.joblib'
    conf_elastic['json_file'] = 'conf_elastic.json'

    with open(conf_elastic['json_file'], 'w') as fp:
        json.dump(conf_elastic, fp)

    # Make SVR Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR

    conf_svr['all_features'] = conf_global['all_features']
    conf_svr['n_features'] = 74

    conf_svr['kernel'] ='rbf' #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    conf_svr['degree']=3
    conf_svr['gamma']='scale' #‘scale’, ‘auto’
    conf_svr['coef0'] = 0.0
    conf_svr['tol']=0.001
    conf_svr['C'] = 1.0
    conf_svr['epsilon'] = 0.1
    conf_svr['shrinking'] = True
    conf_svr['cache_size'] = 200
    conf_svr['verbose'] = False
    conf_svr['max_iter'] = -1

    conf_svr['output_file'] = 'model_svr.joblib'
    conf_svr['json_file'] = 'conf_svr.json'

    with open(conf_svr['json_file'], 'w') as fp:
        json.dump(conf_svr, fp)

# Make KNeighbors Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR

    conf_kneighbors['all_features'] = conf_global['all_features']
    conf_kneighbors['n_features'] = 65

    conf_kneighbors['n_neighbors'] =7
    conf_kneighbors['weights'] = 'distance'
    conf_kneighbors['algorithm']='auto'
    conf_kneighbors['leaf_size']=83
    conf_kneighbors['p']=1
    conf_kneighbors['metric'] = 'minkowski'
    conf_kneighbors['metric_params'] = None
    conf_kneighbors['n_jobs'] = -1


    conf_kneighbors['output_file'] = 'model_kneighbors.joblib'
    conf_kneighbors['json_file'] = 'conf_kneighbors.json'

    with open(conf_kneighbors['json_file'], 'w') as fp:
        json.dump(conf_kneighbors, fp)


    # Make Bayesian Ridge Configuration
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge

    conf_bridge['all_features'] = conf_global['all_features']
    conf_bridge['n_features'] = 65

    conf_bridge['n_iter'] = 300
    conf_bridge['tol'] = 0.001
    conf_bridge['alpha_1'] = 3.114173891282374e-07
    conf_bridge['alpha_2'] = 0.0006005301305242927
    conf_bridge['lambda_1'] = 986.4034769373696
    conf_bridge['lambda_2'] = 3.3576325805597858
    conf_bridge['alpha_init'] = None
    conf_bridge['lambda_init'] = None
    conf_bridge['compute_score'] = False
    conf_bridge['fit_intercept'] = True
    conf_bridge['normalize'] = False
    conf_bridge['copy_X'] = True
    conf_bridge['verbose'] = False

    conf_bridge['output_file'] = 'model_bridge.joblib'
    conf_bridge['json_file'] = 'conf_bridge.json'

    with open(conf_bridge['json_file'], 'w') as fp:
        json.dump(conf_bridge, fp)
