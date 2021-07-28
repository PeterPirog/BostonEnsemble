import pandas as pd
from joblib import dump, load

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_kneighbors = read_config_files(configuration_name='conf_kneighbors')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]
    y = df[conf_global['target_label']]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_kneighbors['all_features'],
                                 n_features=conf_kneighbors['n_features'])

    model = KNeighborsRegressor(n_neighbors=conf_kneighbors["n_neighbors"],
                                weights=conf_kneighbors["weights"],# {‘uniform’, ‘distance’}
                                algorithm=conf_kneighbors["algorithm"],
                                leaf_size=conf_kneighbors["leaf_size"],
                                p=conf_kneighbors["p"],
                                metric=conf_kneighbors["metric"],
                                metric_params=conf_kneighbors["metric_params"],
                                n_jobs=conf_kneighbors["n_jobs"])

    pipe = Pipeline([
        ('fsel', fsel),
        ('model', model),
    ])
    # train model with parameters defined in conf_ridge,json file
    pipe.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    dump(model, filename=conf_kneighbors['output_file'])
    v = Validator(model_or_pipeline=pipe, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=conf_kneighbors)
    v.run()
    print(pipe.predict(X[:5]))