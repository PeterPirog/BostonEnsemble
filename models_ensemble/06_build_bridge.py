import pandas as pd
from joblib import dump, load

from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_bridge = read_config_files(configuration_name='conf_bridge')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_global['all_features'],
                                 n_features=conf_bridge['n_features'])

    model = BayesianRidge(n_iter=conf_bridge['n_iter'],
                          tol=conf_bridge['tol'],
                          alpha_1=conf_bridge['alpha_1'],
                          alpha_2=conf_bridge['alpha_2'],
                          lambda_1=conf_bridge['lambda_1'],
                          lambda_2=conf_bridge['lambda_2'],
                          alpha_init=conf_bridge['alpha_init'],
                          lambda_init=conf_bridge['lambda_init'],
                          compute_score=conf_bridge['compute_score'],
                          fit_intercept=conf_bridge['fit_intercept'],
                          normalize=conf_bridge['normalize'],
                          copy_X=conf_bridge['copy_X'],
                          verbose=conf_bridge['verbose'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_bridge['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error', model_config_dict=conf_bridge)
    v.run()

    print(pipe.predict(X[:5]))
