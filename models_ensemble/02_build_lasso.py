import pandas as pd
from joblib import dump, load

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_lasso = read_config_files(configuration_name='conf_lasso')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]
    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_lasso['all_features'],
                                 n_features=conf_lasso['n_features'])

    model = Lasso(alpha=conf_lasso['alpha'],
                  fit_intercept=conf_lasso['fit_intercept'],
                  normalize=conf_lasso['normalize'],
                  precompute=conf_lasso['precompute'],
                  copy_X=conf_lasso['copy_X'],
                  max_iter=conf_lasso['max_iter'],
                  tol=conf_lasso['tol'],
                  warm_start=conf_lasso['warm_start'],
                  positive=conf_lasso['positive'],
                  random_state=conf_lasso['random_state'],
                  selection=conf_lasso['selection'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_lasso['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error')
    v.run()
