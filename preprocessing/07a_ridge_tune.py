import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import Ridge
import json


def train_boston(config):
    base_path = Path(__file__).parent.parent

    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    features_all = ['_OverallQual', '_GrLivArea', '_ExterQual', '_KitchenQual', '_BsmtQual', '_GarageArea',
                    '_BuildingAge', '_TotalBsmtSF', '_GarageFinish', '_FullBath', '_YearRemodAdd', '_FireplaceQu',
                    '_Foundation_1', '_HeatingQC', '_LotArea', '_OpenPorchSF', '_GarageType_Attchd', '_MasVnrArea',
                    '_LotFrontage', '_BsmtFinType1', '_GarageType_Detchd', '_GarageQual', '_BsmtExposure',
                    '_MSSubClass_3', '_CentralAir', '_WoodDeckSF', '_Foundation_2', '_Exterior_VinylSd', '_HalfBath',
                    '_SaleCondition_Partial', '_MasVnrType_Stone', '_Electrical', '_BsmtFinSF1', '_PavedDrive',
                    '_MSZoning_1', '_LotShape', '_BsmtCond', '_HouseStyle_1', '_Foundation_3', '_BedroomAbvGr',
                    '_BsmtFullBath', '_Neighborhood_5', '_MasVnrType_BrkFace', '_GarageType_BuiltIn', '_EnclosedPorch',
                    '_Neighborhood_9', '_SaleType_WD', '_BldgType_2', '_RoofStyle_1', '_Exterior_WdSdng',
                    '_HouseStyle_3', '_Exterior_MetalSd', '_BsmtUnfSF', '_Neighborhood_8', '_Fence',
                    '_SaleCondition_Abnorml', '_LotConfig_4', '_Functional', '_BldgType_1', '_Alley', '_Neighborhood_1',
                    '_SaleCondition_Normal', '_ScreenPorch', '_HouseStyle_4', '_OverallCond', '_LotConfig_1',
                    '_HouseStyle_2', '_Exterior_HdBoard', '_MSSubClass_2', '_QuarterSold', '_ExterCond',
                    '_Neighborhood_2', '_YrSold', '_BsmtFinSF2', '_BldgType_3', '_Exterior_Plywood', '_LandContour_2',
                    '_MSZoning_3', '_LotConfig_3']
    features_all=['_HouseStyle_2', '_HouseStyle_3', '_GrLivArea', '_BuildingAge', '_HouseStyle_1', '_OverallQual', '_Neighborhood_2', '_HouseStyle_4', '_Neighborhood_9', '_TotalBsmtSF', '_OverallCond', '_KitchenQual', '_Electrical', '_MSZoning_3', '_GarageArea', '_BldgType_2', '_SaleCondition_Abnorml', '_LotArea', '_ScreenPorch', '_MSSubClass_1', '_BsmtQual', '_Functional', '_Neighborhood_5', '_MSSubClass_3', '_FireplaceQu', '_RoofStyle_1', '_BsmtExposure', '_GarageQual', '_MSZoning_1', '_BsmtFullBath', '_WoodDeckSF', '_YearRemodAdd', '_FullBath', '_BsmtFinSF1', '_BsmtUnfSF', '_HalfBath', '_SaleType_WD', '_CentralAir', '_LotConfig_4', '_LotFrontage', '_Neighborhood_8', '_Foundation_2', '_OpenPorchSF', '_GarageType_Attchd', '_BldgType_3', '_HeatingQC', '_MasVnrType_Stone', '_Exterior_HdBoard', '_EnclosedPorch', '_YrSold', '_GarageFinish', '_Neighborhood_1', '_BsmtFinType1', '_PavedDrive', '_Exterior_VinylSd', '_Alley', '_LotConfig_1', '_MSSubClass_2', '_Foundation_3', '_ExterQual', '_MasVnrArea', '_BldgType_1', '_Exterior_Plywood', '_BsmtFinSF2', '_SaleCondition_Normal', '_GarageType_BuiltIn', '_BsmtCond', '_SaleCondition_Partial', '_BedroomAbvGr', '_Exterior_MetalSd', '_MasVnrType_BrkFace', '_LandContour_2', '_Foundation_1', '_LotShape', '_Exterior_WdSdng', '_ExterCond', '_GarageType_Detchd', '_LotConfig_3', '_Fence', '_QuarterSold']


    n_features = len(features_all)
    # n_features=3
    y = df['SalePrice_log1']

    X = df[features_all[:config['n_features']]]
    #model = Ridge(config['alpha'],max_iter=config['max_iter'])
    model = Ridge(config['alpha'])

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
    ray.tune.report(_metric=scores.mean() + 2 * scores.std())


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
        # metric="val_rmsle",  # mean_accuracy
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
            #"max_iter": tune.qrandint(500, 10000, 500),
            "n_features": tune.randint(1, 79)
        }

    )
    print("Best result:",analysis.best_result,"Best hyperparameters found were: ", analysis.best_config)
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/ridge --bind_all --load_fast=false


