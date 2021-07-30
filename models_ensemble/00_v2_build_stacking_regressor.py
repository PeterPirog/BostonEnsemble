import numpy as np
import pandas as pd
from joblib import dump, load
import dill as pickle
from dill import dump
import tensorflow as tf

from sklearn.ensemble import StackingRegressor
from tensorflow.keras.models import load_model
import joblib
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

def baseline_model(hidden1,hidden2,activation,dropout,lr):
	# create model
    epochs = 100000
    number_inputs=79
    #{'random_state': 117, 'batch': 64, 'learning_rate': 0.1, 'hidden1': 31, 'activation1': 'elu', 'dropout1': 0.28}
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # layer 2
    x = tf.keras.layers.Dense(units=hidden2, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics='mean_squared_error')
    return model

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



    #cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    estimators = [
        ('ridge', model_ridge),
        ('lasso', model_lasso),
        ('elastic', model_elastic),
        ('svr', model_svr),
        ('kneighbors', model_kneighbors),
        ('bridge', model_bridge),
        ('rforest', model_rforest),
       ('keras', KerasRegressor(baseline_model)),
        #('xgb', model_xgb)
    ]
    model = StackingRegressor(
        estimators=estimators,
        # final_estimator=RandomForestRegressor(n_estimators=30, random_state=42))
        final_estimator=Ridge(), n_jobs=1)

    # train model with parameters defined in conf_ridge,json file
    model.fit(X=X, y=y)

"""
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
"""