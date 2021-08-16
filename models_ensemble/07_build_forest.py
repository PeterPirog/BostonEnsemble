import pandas as pd
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_forest = read_config_files(configuration_name='conf_forest')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_global['all_features'],
                                 n_features=conf_forest['n_features'])

    model = RandomForestRegressor(n_estimators=conf_forest['n_estimators'],
                                  criterion=conf_forest['criterion'],
                                  max_depth=conf_forest['max_depth'],
                                  min_samples_split=conf_forest['min_samples_split'],
                                  min_samples_leaf=conf_forest['min_samples_leaf'],
                                  min_weight_fraction_leaf=conf_forest['min_weight_fraction_leaf'],
                                  max_features=conf_forest['max_features'],
                                  max_leaf_nodes=conf_forest['max_leaf_nodes'],
                                  min_impurity_decrease=conf_forest['min_impurity_decrease'],
                                  min_impurity_split=conf_forest['min_impurity_split'],
                                  bootstrap=conf_forest['bootstrap'],
                                  oob_score=conf_forest['oob_score'],
                                  n_jobs=conf_forest['n_jobs'],
                                  random_state=conf_forest['random_state'],
                                  verbose=conf_forest['verbose'],
                                  warm_start=conf_forest['warm_start'],
                                  ccp_alpha=conf_forest['ccp_alpha'],
                                  max_samples=conf_forest['max_samples'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_forest['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error', model_config_dict=conf_forest)
    v.run()

    print(pipe.predict(X[:5]))

    #31 features forward
    ['_BldgType_2', '_GrLivArea', '_OverallQual', '_BuildingAge', '_TotalBsmtSF', '_CentralAir', '_LotArea', '_GarageArea', '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_BsmtExposure', '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_BsmtFinType1', '_MSZoning_1', '_Neighborhood_5', '_YearRemodAdd', '_GarageFinish', '_Exterior_VinylSd', '_MSSubClass_1', '_Neighborhood_8', '_Exterior_MetalSd', '_BsmtCond', '_MSZoning_3', '_Neighborhood_2', '_BsmtFinSF1', '_BedroomAbvGr', '_Foundation_1', '_MSSubClass_2']

