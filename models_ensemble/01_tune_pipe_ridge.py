import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from ensemble_tools import FeatureByNameSelector, Validator,read_config_files

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import json


def train_boston(config):
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    file='/home/peterpirog/PycharmProjects/BostonEnsemble/models_ensemble/conf_global.json'
    #with open(file, 'r') as fp:
    #    conf_global = json.load(fp)
    conf_global = read_config_files(configuration_name='conf_global')
    conf_ridge = read_config_files(configuration_name='conf_ridge')


    all_features=conf_global['all_features']

    n_features = conf_global['n_all_features']

    y = df['SalePrice_log1']

    X = df[all_features]

    # Select only 'n_features' labels for X dataframe
    fsel = FeatureByNameSelector(all_features=conf_global['all_features'],
                                 n_features=config['n_features']) #TUNED VALUE

    model_ridge = Ridge(alpha=config['alpha'],#TUNED VALUE
                  fit_intercept=conf_ridge['fit_intercept'],
                  normalize=conf_ridge['normalize'],
                  copy_X=conf_ridge['copy_X'],
                  max_iter=conf_ridge['max_iter'],
                  tol=conf_ridge['tol'],
                  solver=conf_ridge['solver'],
                  random_state=conf_ridge['random_state'])

    pipe_ridge = Pipeline([
        ('fsel', fsel),
        ('model', model_ridge),
    ])

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model
    scores = cross_val_score(pipe_ridge, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)
    ray.tune.report(_metric=scores.mean(),_std=scores.std(),_ubc=scores.mean() + 2 * scores.std())


if __name__ == "__main__":

    try:
        ray.init()
    except:
        ray.shutdown()
        ray.init()

    sched_asha = ASHAScheduler(time_attr="training_iteration",
                               max_t=500,
                               grace_period=16,
                               # mode='max', #find maximum, do not define here if you define in tune.run
                               reduction_factor=3,
                               # brackets=1
                               )

    analysis = tune.run(
        train_boston,
        search_alg=HyperOptSearch(),
        name="ridge",
        # scheduler=sched_asha, - no need scheduler if there is no iterations
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        metric="_ubc",
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 100
        },
        num_samples=3000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
        # default value is ~/ray_results /root/ray_results/  or ~/ray_results
        resources_per_trial={
            "cpu": 16  # ,
            # "gpu": 0
        },
        config={
            "alpha": tune.loguniform(1e-5, 100),
            "n_features": tune.randint(1, 80)
        }

    )
    print("Best result:",analysis.best_result)
    print("Best hyperparameters found were: ", analysis.best_config)
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/ridge --bind_all --load_fast=false
    #'_std': 0.014156917636511966, '_ubc': 0.16426060089180344, '_metric': 0.1359467656187795
    # {'alpha': 0.424205265420875, 'n_features': 50}

