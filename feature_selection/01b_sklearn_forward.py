from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split,KFold
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.regularizers import l2, l1_l2
from models_ensemble.ensemble_tools import FeatureByNameSelector, Validator, read_config_files
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor





if __name__ == "__main__":

    conf_global = read_config_files(configuration_name='conf_global')
    #conf_keras = read_config_files(configuration_name='conf_keras')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']]
    y = df[conf_global['target_label']]

    # step backward feature selection algorithm

    # separate train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=True,
        test_size=0.3,
        random_state=0)


    cv=KFold(n_splits=5,shuffle=True, random_state=42)

    sfs=SFS(estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=10),
            n_features_to_select=5,
            direction='forward',
            scoring='r2',
            cv=cv,
            n_jobs=None)

    sfs = sfs.fit(X=X, y=y)
    print(X_train.columns[list(sfs.k_feature_idx_)])



