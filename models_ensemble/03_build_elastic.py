import pandas as pd
from joblib import dump, load

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_elastic = read_config_files(configuration_name='conf_elastic')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]
    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_elastic['all_features'],
                                 n_features=conf_elastic['n_features'])

    model = ElasticNet(alpha=conf_elastic['alpha'],
                       l1_ratio=conf_elastic['l1_ratio'],
                       fit_intercept=conf_elastic['fit_intercept'],
                       normalize=conf_elastic['normalize'],
                       precompute=conf_elastic['precompute'],
                       copy_X=conf_elastic['copy_X'],
                       max_iter=conf_elastic['max_iter'],
                       tol=conf_elastic['tol'],
                       warm_start=conf_elastic['warm_start'],
                       positive=conf_elastic['positive'],
                       random_state=conf_elastic['random_state'],
                       selection=conf_elastic['selection'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_elastic['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=conf_elastic)
    v.run()
