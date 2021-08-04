import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.svm import SVR
import json


def train_boston(config):
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    file = '/home/peterpirog/PycharmProjects/BostonEnsemble/models_ensemble/conf_global.json'
    with open(file, 'r') as fp:
        conf_global = json.load(fp)

    all_features = conf_global['all_features']

    y = df['SalePrice_log1']
    X = df[all_features[:config['n_features']]]

    model = SVR(kernel=config['kernel'],
                degree=config['degree'],
                gamma=config['gamma'],
                C=config['C'],
                epsilon=config['epsilon'])

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model
    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)

    ray.tune.report(_metric=scores.mean(), _ubc=scores.mean() + 2 * scores.std())


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
        name="svr",
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
        num_samples=5000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
        # default value is ~/ray_results /root/ray_results/  or ~/ray_results
        resources_per_trial={
            "cpu": 16  # ,
            # "gpu": 0
        },
        config={
            "n_features": tune.randint(1, 80),
            # model parameters
            'kernel': tune.choice(['linear', 'poly', 'rbf', 'sigmoid']),  # 'linear', 'poly', 'rbf', 'sigmoid'
            'degree': tune.choice([2, 3]),  # 2, 3
            'gamma': tune.choice(['scale', 'auto']),  # scale 'scale', 'auto'
            'C': tune.loguniform(1e-3, 10),  # 1.0
            'epsilon': tune.loguniform(1e-4, 1.0)  # 0.1  tune.loguniform(1e-3, 10)
        }

    )
    print("Best result:", analysis.best_result, "Best hyperparameters found were: ", analysis.best_config)
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/svr --bind_all --load_fast=false
#{'_ubc': 0.1510907811805899, '_metric': 0.12697543893943233, 'time_this_iter_s': 1.093526840209961, 'done': True, 'timesteps_total': None, 'episodes_total': None, 'training_iteration': 1, 'experiment_id': 'a757a5fa0447477b8642b4b0bfd278e5', 'date': '2021-08-03_00-06-11', 'timestamp': 1627941971, 'time_total_s': 1.093526840209961, 'pid': 155219, 'hostname': 'ai-server', 'node_ip': '192.168.1.47', 'config': {'n_features': 42, 'kernel': 'rbf', 'degree': 2, 'gamma': 'scale', 'C': 2.534576892718523, 'epsilon': 0.04955732381314529}, 'time_since_restore': 1.093526840209961, 'timesteps_since_restore': 0, 'iterations_since_restore': 1, 'trial_id': 'd6994ef1', 'experiment_tag': '3457_C=2.5346,degree=2,epsilon=0.049557,gamma=scale,kernel=rbf,n_features=42'}
# Best hyperparameters found were:  {'n_features': 42, 'kernel': 'rbf', 'degree': 2, 'gamma': 'scale', 'C': 2.534576892718523, 'epsilon': 0.04955732381314529}
#_ubc=0.1510907811805899 and parameters={'n_features': 42, 'kernel': 'rbf', 'degree': 2, 'gamma': 'scale', 'C': 2.534576892718523, 'epsilon': 0.04955732381314529}