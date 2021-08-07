import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from lightgbm import LGBMRegressor


def train_boston(config):
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    features_all = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea', '_MSSubClass_3',
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
    n_features = len(features_all)
    # n_features=3
    y = df['SalePrice_log1']

    X = df[features_all[:config['n_features']]]

    model = LGBMRegressor(boosting_type=config['boosting_type'],  # 'gbdt' 'dart'
                          num_leaves=config['num_leaves'],
                          max_depth=- 1,
                          learning_rate=config['learning_rate'],
                          n_estimators=config['n_estimators'],
                          n_jobs=-1)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model

    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',  # 'neg_mean_absolute_error' make_scorer(rmsle)
                             cv=cv,
                             n_jobs=-1)

    # force scores to be positive
    scores = abs(scores)

    # print('Mean RMSLE: %.4f (%.4f)' % (scores.mean(), scores.std()))

    # Creating own metric
    ray.tune.report(_metric=scores.mean(), _std=scores.std(), _ubc=scores.mean() + 2 * scores.std())


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
        name="LGBM",
        # scheduler=sched_asha, - no need scheduler if there is no iterations
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        metric="_ubc",  # mean_accuracy
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
            "cpu": 8  # ,
            # "gpu": 0
        },
        config={
            "learning_rate": tune.loguniform(1e-5, 1),
            "num_leaves": tune.randint(2, 200),
            "n_estimators": tune.randint(1, 1000),
            "n_features": tune.randint(1, 80),
            "boosting_type": tune.choice(['dart', 'gbdt'])  # , 'dart' 'gbdt'
        }

    )
    print("Best hyperparameters found were: ", analysis.best_config, " metric: ", analysis.best_result['_metric'])
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/LGBM --bind_all --load_fast=false
    # Best hyperparameters found were:  {'learning_rate': 0.11066629069748643, 'num_leaves': 31, 'n_estimators': 142, 'n_features': 78, 'boosting_type': 'gbdt'}  metric:  0.16188236882033707
    # _ubc=0.1572482783538922 and parameters={'learning_rate': 0.07811681093798588, 'num_leaves': 5, 'n_estimators': 847, 'n_features': 50, 'boosting_type': 'gbdt'}