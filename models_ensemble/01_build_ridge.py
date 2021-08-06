import pandas as pd
from joblib import dump, load


from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator,read_config_files


if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_ridge = read_config_files(configuration_name='conf_ridge')

    df = pd.read_csv(conf_global['encoded_train_data'])



    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    #Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_global['all_features'],
                                 n_features=conf_ridge['n_features'])

    model = Ridge(alpha=conf_ridge['alpha'],
                  fit_intercept=conf_ridge['fit_intercept'],
                  normalize=conf_ridge['normalize'],
                  copy_X=conf_ridge['copy_X'],
                  max_iter=conf_ridge['max_iter'],
                  tol=conf_ridge['tol'],
                  solver=conf_ridge['solver'],
                  random_state=conf_ridge['random_state'])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    #train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    #Save pipeline or model in joblib file
    dump(model, filename=conf_ridge['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=conf_ridge)
    v.run()

    print(pipe.predict(X[:5]))

