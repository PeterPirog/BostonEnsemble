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
from ensemble_tools import FeatureByNameSelector, Validator, create_submission_from_nparray
from _create_json_conf import read_config_files

if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_ridge = read_config_files(configuration_name='conf_ridge')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    df_test = pd.read_csv('/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/encoded_test_X_data.csv')
    X_test = df_test[conf_global['all_features']]


    # with open('model_stacked.pkl', 'rb') as file:
    with open('model_keras.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    create_submission_from_nparray(predicted_array=y_pred,
                                   test_csv_path='/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/test.csv')
