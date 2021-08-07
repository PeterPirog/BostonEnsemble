import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score
from ray.tune.suggest.hyperopt import HyperOptSearch
from models_ensemble.ensemble_tools import FeatureByNameSelector, Validator, read_config_files
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.layers import Dense


def baseline_model(number_inputs, hidden1, hidden_enc, hidden3, activation, dropout, normalization_type, noise_std, lr):
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    # Choose normaliztion type
    if normalization_type == 'layer':
        x = tf.keras.layers.LayerNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)

    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    if normalization_type == 'layer':
        x = tf.keras.layers.LayerNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # layer encoder
    x = tf.keras.layers.Dense(units=hidden_enc, kernel_initializer='glorot_normal',
                              activation='linear')(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.Dropout(dropout)(x)

    # layer 3
    x = tf.keras.layers.Dense(units=hidden3, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    if normalization_type == 'layer':
        x = tf.keras.layers.LayerNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics='mean_squared_error')
    return model


def train_boston(config):
    conf_global = read_config_files(configuration_name='conf_global')
    conf_keras = read_config_files(configuration_name='conf_keras')

    df = pd.read_csv(conf_global['encoded_train_data'])

    X = df[conf_global['all_features']].to_numpy()

    y = df[conf_global['target_label']].to_numpy()

    # X=tf.cast(X, dtype=tf.bfloat16)
    # y = tf.cast(y, dtype=tf.bfloat16)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                       patience=20,
                                                       restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                           factor=0.8,
                                                           patience=10), ]

    estimator = KerasRegressor(build_fn=baseline_model,
                               ## custom parameters
                               number_inputs=80,
                               hidden1=config['hidden1'],  # TUNE
                               hidden_enc=config['hidden_enc'],  # TUNE
                               hidden3=config['hidden3'],  # TUNE
                               activation=config['activation'],  # TUNE
                               dropout=config['dropout'],  # TUNE
                               normalization_type=config['normalization_type'],  # TUNE
                               lr=config['learning_rate'],  # TUNE
                               noise_std=config['noise_std'],  # TUNE
                               ## fit parameters
                               # x=None,#conf_keras['x']
                               # y=None,#conf_keras['y']
                               batch_size=conf_keras['batch_size'],
                               epochs=10000,#conf_keras['epochs']
                               verbose=0,  # conf_keras['verbose'], 0,1,2
                               callbacks=callbacks_list,
                               validation_split=conf_keras['validation_split'],
                               validation_data=conf_keras['validation_data'],
                               shuffle=conf_keras['shuffle'],
                               class_weight=conf_keras['class_weight'],
                               sample_weight=conf_keras['sample_weight'],
                               initial_epoch=conf_keras['initial_epoch'],
                               steps_per_epoch=conf_keras['steps_per_epoch'],
                               validation_steps=conf_keras['validation_steps'],
                               validation_batch_size=conf_keras['validation_batch_size'],
                               validation_freq=conf_keras['validation_freq'],
                               max_queue_size=conf_keras['max_queue_size'],
                               workers=conf_keras['workers'],
                               use_multiprocessing=conf_keras['use_multiprocessing'],
                               )

    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    # evaluate_model
    scores = cross_val_score(estimator, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)
    ray.tune.report(_metric=scores.mean(), _std=scores.std(), _ubc=scores.mean() + 2.32 * scores.std())


if __name__ == "__main__":
    print('Is cuda available for container:', tf.config.list_physical_devices('GPU'))
    sched_asha = ASHAScheduler(time_attr="training_iteration",
                               max_t=5000,
                               grace_period=20,
                               # mode='max', #find maximum, do not define here if you define in tune.run
                               reduction_factor=3,
                               # brackets=1
                               )

    analysis = tune.run(
        train_boston,
        search_alg=HyperOptSearch(),
        name="autoencoder",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_score_attr='min-_ubc',  # 'min-val_loss'
        checkpoint_freq=3,
        checkpoint_at_end=False,
        verbose=3,
        # Optimalization
        metric="_ubc",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 5000
        },
        num_samples=1000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
        resources_per_trial={
            "cpu": 16,
            "gpu": 0
        },
        config={
            # training parameters
            "batch": tune.choice([128]),
            "learning_rate": tune.choice([0.1]),  # tune.loguniform(1e-5, 1e-2)
            "hidden1": tune.randint(5, 200),
            "hidden_enc": tune.randint(5, 50),
            "hidden3": tune.randint(5, 200),
            "activation": tune.choice(["elu"]),
            "dropout": tune.quniform(0.01, 0.5, 0.01),  # tune.uniform(0.01, 0.15)
            "normalization_type": tune.choice(['batch']), #'layer'
            "noise_std": tune.uniform(0.001, 0.2)
        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)
# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/autoencoder --bind_all --load_fast=false

#_ubc=0.16033794054234746 and parameters={'batch': 128, 'learning_rate': 0.1, 'hidden1': 83, 'hidden_enc': 30, 'hidden3': 42, 'activation': 'elu', 'dropout': 0.24, 'normalization_type': 'batch', 'noise_std': 0.32448261075557777}
