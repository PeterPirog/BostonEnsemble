import pandas as pd
from joblib import dump, load

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_svr = read_config_files(configuration_name='conf_svr')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]
    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_svr['all_features'],
                                 n_features=conf_svr['n_features'])

    model = SVR(kernel=conf_svr['kernel'],
                degree=conf_svr['degree'],
                gamma=conf_svr['gamma'],
                coef0=conf_svr['coef0'],
                tol=conf_svr['tol'],
                C=conf_svr['C'],
                epsilon=conf_svr['epsilon'],
                shrinking=conf_svr['shrinking'],
                cache_size=conf_svr['cache_size'],
                verbose=conf_svr['verbose'],
                max_iter=conf_svr['max_iter'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_svr['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=conf_svr)
    v.run()
#_metric=0.13138996421177315 and parameters={'n_features': 77, 'kernel': 'rbf', 'degree': 3, 'gamma': 'scale', 'C': 0.7832573311079015, 'epsilon': 0.04825896120073174}
    print(pipe.predict(X[:5]))