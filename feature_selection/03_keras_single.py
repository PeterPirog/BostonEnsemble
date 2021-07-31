from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.regularizers import l2
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def baseline_model(number_inputs, hidden1, activation, noise_std, l2_value):
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation,
                              kernel_regularizer=l2(l2_value),
                              use_bias=False)(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error
    return model

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    # Get all fetures in dataframe
    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    y = df['SalePrice_log1']
    X = df.drop(['Id', 'SalePrice', 'SalePrice_log1'], axis=1).copy()

    X = X.to_numpy()
    y = y.to_numpy()

    callbacks_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',  # 'val_loss' there is no val_loss inside kfold
                                             factor=0.8,
                                             patience=10),
        # TuneReportCallback({'val_loss': 'val_loss'}),
        tf.keras.callbacks.EarlyStopping(monitor='loss',  # 'val_loss' there is no val_loss inside kfold
                                         patience=15)]

    model=baseline_model(number_inputs=80,
                         hidden1=34,
                         activation='elu',
                         noise_std=0.0011099477641101035,
                         l2_value=0.0011099477641101035)

    model.summary()
    """
    model = KerasRegressor(build_fn=baseline_model,
                           ## custom parameters
                           number_inputs=80,
                           hidden1=34,
                           noise_std=0.37506113740525504,
                           activation=config['activation'],
                           l2_value=config['l2_value'],
                           # lr=config['learning_rate'],
                           ## fit parameters
                           batch_size=config['batch_size'],
                           epochs=100000,
                           verbose=0,
                           # verbose=config['verbose'],
                           callbacks=callbacks_list
                           )

    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=None)

    # evaluate_model
    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)

    # force scores to be positive
    scores = abs(scores)

    # print('Mean RMSLE: %.4f (%.4f)' % (scores.mean(), scores.std()))

"""