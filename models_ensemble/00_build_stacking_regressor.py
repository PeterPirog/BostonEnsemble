import numpy as np
import pandas as pd
from joblib import dump, load
import dill as pickle
from dill import dump
import tensorflow as tf

from sklearn.ensemble import StackingRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold

from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator,model_to_submission
from _create_json_conf import read_config_files



if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_ridge = read_config_files(configuration_name='conf_ridge')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    model_ridge = load(filename='model_ridge.joblib')
    model_lasso = load(filename='model_lasso.joblib')
    model_elastic = load(filename='model_elastic.joblib')
    model_svr = load(filename='model_svr.joblib')
    model_kneighbors = load(filename='model_kneighbors.joblib')
    model_bridge = load(filename='model_bridge.joblib')
    model_rforest = load(filename='model_forest.joblib')

    #with open('keras_model.pkl', 'rb') as file:
    #    model_keras = pickle.load(file)

    model_xgb = XGBRegressor(n_estimators=144,
                             max_depth=6,
                             eta=0.1,
                             subsample=1,
                             colsample_bytree=1)



    #cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    estimators = [
        ('ridge', model_ridge),
        ('lasso', model_lasso),
        ('elastic', model_elastic),
        ('svr', model_svr),
        ('kneighbors', model_kneighbors),
        ('bridge', model_bridge),
        ('rforest', model_rforest),
       # ('keras', model_keras),
        #('xgb', model_xgb)
    ]
    model = StackingRegressor(
        estimators=estimators,
        # final_estimator=RandomForestRegressor(n_estimators=30, random_state=42))
        final_estimator=Ridge(), n_jobs=1)

    # train model with parameters defined in conf_ridge,json file
    model.fit(X=X, y=y)

    # Save pipeline or model in joblib file
    #dump(model, filename='model_stacked.joblib')

    with open('model_stacked.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open('model_stacked.pkl', 'rb') as file:
        model2 = pickle.load(file)

    y_pred=model2.predict(X)
    err=y_pred-y.to_numpy()
    mse=np.sqrt(err**2).mean()
    print('mse=',mse)

    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/encoded_test_X_data.csv')
    X_test = df_test[conf_global['all_features']]
    model_to_submission(model_file_pkl='model_stacked.pkl',X=X_test)

    #v = Validator(model_or_pipeline=model, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
    #              scoring='neg_root_mean_squared_error', model_config_dict=None)
    #v.run()
