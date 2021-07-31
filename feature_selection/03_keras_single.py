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
    inputs = tf.keras.layers.Input(shape=number_inputs, name='input_layer')
    x = tf.keras.layers.Flatten(name='flatten_layer')(inputs)
    x = tf.keras.layers.LayerNormalization(name='normalization_layer')(x)
    x = tf.keras.layers.GaussianNoise(stddev=noise_std, name='nosie_layer')(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation,
                              kernel_regularizer=l2(l2_value),
                              use_bias=False,
                              name='dense_layer')(x)
    outputs = tf.keras.layers.Dense(units=1, name='output_layer')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mse',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])  #'mean_squared_error'
    return model


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    tf.config.threading.set_inter_op_parallelism_threads(num_threads=16)
    # Get all fetures in dataframe
    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    y = df['SalePrice_log1']
    X = df.drop(['Id', 'SalePrice', 'SalePrice_log1'], axis=1).copy()

    X = X.to_numpy()
    y = y.to_numpy()

    callbacks_list = [
        # Reducle value rl
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             factor=0.8,
                                             patience=10),
        # early stop
        tf.keras.callbacks.EarlyStopping(monitor='loss',
                                         patience=15),
        # save best result
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./',
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/peterpirog/PycharmProjects/BostonEnsemble/tensorboard",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None
        )
    ]

    model = baseline_model(number_inputs=80,
                           hidden1=34,
                           activation='elu',
                           noise_std=0.37506113740525504,
                           l2_value=0.0011099477641101035)

    model.summary()

    model.fit(x=X,
              y=y,
              batch_size=64,
              epochs=100000,
              verbose=1,
              callbacks=callbacks_list,
              workers=1,
              use_multiprocessing=True)

    #Model.get_layer(name=None, index=None)
    w1=model.get_layer(name='dense_layer').get_weights()[0]
    w2=model.get_layer(name='output_layer').get_weights()[0]

    print(f'w1={w1}')
    print(f'w2={w2}')
    """
    for layerNum, layer in enumerate(model.layers):
        weights=layer.get_weights()
        #biases = layer.get_weights()[1]
        print(f'layerNum ={layerNum}_{layer.name},weights={weights}, len= {len(weights)}')
        #print(f'biases={biases}')


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
# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/tensorboard --bind_all --load_fast=false
"""
