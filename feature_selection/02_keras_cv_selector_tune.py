
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.regularizers import l2,l1_l2
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def baseline_model(number_inputs, hidden1, activation, noise_std,l1_value, l2_value,dropout):
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation,
                              kernel_regularizer=l1_l2(l1=l1_value,l2=l2_value),
                              use_bias=False)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error
    return model


def train_boston(config):
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

    model = KerasRegressor(build_fn=baseline_model,
                           ## custom parameters
                           number_inputs=80,
                           hidden1=config['hidden1'],
                           noise_std=config['noise_std'],
                           activation=config['activation'],
                           l1_value=config['l1_value'],
                           l2_value=config['l2_value'],
                           dropout=config['dropout'],
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

    # Creating own metric
    ray.tune.report(_metric=scores.mean(), _ubc=scores.mean() + 2 * scores.std())


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

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
        name="keras_cv_select",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_score_attr='min-_ubc',  # 'min-val_loss'
        checkpoint_freq=3,
        checkpoint_at_end=False,
        verbose=3,
        # Optimalization
        metric="_ubc",  # mean_accuracy "val_loss"
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 5000
        },
        num_samples=1000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
        # default value is ~/ray_results /root/ray_results/
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        },
        config={
            # training parameters
            "batch_size": tune.choice([64]),
            # Layer 1 params
            "hidden1": tune.randint(1, 200),
            "activation": tune.choice(["elu"]),
            "noise_std": tune.uniform(0.001, 0.5),
            "l1_value": tune.loguniform(1e-5, 1e-1),
            "l2_value": tune.loguniform(1e-5, 1e-1),
            "dropout": tune.uniform(0.001, 0.5)
        }

    )
    print(f"Result is: {analysis.best_result}, best hyperparameters found were:{analysis.best_config}")

    best_configuration = analysis.best_config
    best_configuration['best_trial'] = analysis.best_result

    # get feature labels
    df = pd.read_csv("/home/peterpirog/PycharmProjects/BostonEnsemble/data_files/encoded_train_X_data.csv")
    X = df.drop(['Id', 'SalePrice', 'SalePrice_log1'], axis=1)
    best_configuration['df_labels'] = list(X.columns)

    # Save best result to file
    with open('best_selection_net.json', 'w') as fp:
        json.dump(best_configuration, fp)

    #save all results to xlsx file
    results_df = results = analysis.results_df
    results_df.to_excel('train_results.xlsx',
                        sheet_name='Results')

# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/keras_cv_select --bind_all --load_fast=false
#{'_ubc': 0.15922711081631474, '_metric': 0.14415299029545392, 'time_this_iter_s': 51.48896765708923, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 1, 'experiment_id': '6992caa91a8f4a25a432b322a2d7790c', 'date': '2021-07-31_18-56-07', 'timestamp': 1627750567, 'time_total_s': 51.48896765708923, 'pid': 2981784, 'hostname': 'ai-server', 'node_ip': '192.168.1.47', 'config': {'batch_size': 64, 'hidden1': 34, 'activation': 'elu', 'noise_std': 0.37506113740525504, 'l2_value': 0.0011099477641101035}, 'time_since_restore': 51.48896765708923, 'timesteps_since_restore': 0, 'iterations_since_restore': 1, 'trial_id': '113ba5ea', 'experiment_tag': '643_activation=elu,batch_size=64,hidden1=34,l2_value=0.0011099,noise_std=0.37506'}, best hyperparameters found were:{'batch_size': 64, 'hidden1': 34, 'activation': 'elu', 'noise_std': 0.37506113740525504, 'l2_value': 0.0011099477641101035}
#_ubc=0.1538731554664114 and parameters={'batch_size': 64, 'hidden1': 20, 'activation': 'elu', 'noise_std': 0.020298941092507204, 'l1_value': 3.543298823184669e-05, 'l2_value': 5.945058708937871e-05, 'dropout': 0.1309933349281991}
