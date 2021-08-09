import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import Lasso
import json


def train_boston(config):
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    file = '/home/peterpirog/PycharmProjects/BostonEnsemble/models_ensemble/conf_global.json'
    with open(file, 'r') as fp:
        conf_global = json.load(fp)

    all_features = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea', '_MSSubClass_3',
                    '_OverallQual', '_BuildingAge', '_TotalBsmtSF', '_Functional', '_CentralAir',
                    '_Electrical', '_SaleCondition_Abnorml', '_RoofStyle_1', '_LotArea', '_GarageArea',
                    '_KitchenQual', '_OverallCond', '_Neighborhood_9', '_SaleType_WD', '_ScreenPorch',
                    '_BsmtExposure', '_ExterQual', '_BsmtUnfSF', '_Foundation_2', '_HouseStyle_2',
                    '_HouseStyle_3', '_LotConfig_4', '_GarageType_BuiltIn', '_FullBath',
                    '_Neighborhood_1', '_FireplaceQu', '_BsmtQual', '_SaleCondition_Normal',
                    '_BsmtFinType1', '_PavedDrive', '_Foundation_3', '_MSZoning_1', '_Neighborhood_5',
                    '_HeatingQC', '_YrSold', '_HalfBath', '_YearRemodAdd', '_GarageFinish',
                    '_HouseStyle_1', '_BsmtFinSF2', '_WoodDeckSF', '_Exterior_VinylSd', '_MSSubClass_1',
                    '_GarageType_Attchd', '_LotFrontage', '_Exterior_HdBoard', '_HouseStyle_4',
                    '_MasVnrType_BrkFace', '_Exterior_Plywood', '_GarageQual', '_MasVnrType_Stone',
                    '_LandContour_2', '_BsmtFullBath', '_LotShape', '_Exterior_WdSdng',
                    '_Neighborhood_8', '_Fence', '_LotConfig_1', '_Alley', '_Exterior_MetalSd',
                    '_EnclosedPorch', '_LotConfig_3', '_BsmtCond', '_MasVnrArea',
                    '_SaleCondition_Partial', '_GarageType_Detchd', '_MSZoning_3', '_ExterCond',
                    '_Neighborhood_2', '_QuarterSold', '_BsmtFinSF1', '_BedroomAbvGr', '_OpenPorchSF',
                    '_Foundation_1', '_MSSubClass_2']

    n_features = len(all_features)

    y = df['SalePrice_log1']
    X = df[all_features]

    model = Lasso(alpha=config['alpha'])

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
        name="lasso",
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
            "alpha": tune.loguniform(1e-5, 10e8),
            "n_features": tune.choice([52])  # tune.randint(1, 80)
        }

    )
    print("Best result:", analysis.best_result, "Best hyperparameters found were: ", analysis.best_config)
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/lasso --bind_all --load_fast=false

    # _ubc': 0.16620455852294494, '_metric': 0.13857810088910588, and parameters={'alpha': 0.0009270451666733304, 'n_features': 52}
