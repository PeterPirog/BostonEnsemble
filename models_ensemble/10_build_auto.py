#https://github.com/automl/auto-sklearn
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

    import autosklearn.regression

    model = autosklearn.regression.AutoSklearnRegressor()
    model.fit(X, y)
    predictions = model.predict(X)


    #Save pipeline or model in joblib file

    #v = Validator(model_or_pipeline=model, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
    #              scoring='neg_root_mean_squared_error',model_config_dict=conf_ridge)
    #v.run()



