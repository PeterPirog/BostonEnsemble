from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.regularizers import l2, l1_l2
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
    # n_features = len(features_all)
    # n_features=3
    y = df['SalePrice_log1']

    # X = df[features_all[:config['n_features']]] Id,SalePrice,SalePrice_log1
    X = df[features_all]
    X = df.drop(['Id','SalePrice','SalePrice_log1'], axis=1).copy()

    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None)

    epochs = 100000
    # define model
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=config["noise_std"])(x)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"],
                              kernel_regularizer=l1_l2(l1=config["l1_value"],l2=config["l2_value"]),
                              use_bias=False)(x)
    x = tf.keras.layers.Dropout(config["dropout"])(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error

    callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.5,
                                                           patience=10),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15),
                      TuneReportCallback({'val_loss': 'val_loss'})]

    model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),  # tf reduce mean ignore tabnanny
        callbacks=callbacks_list)  # "mean_accuracy": "val_accuracy"


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

    print('Is cuda available for container:', tf.config.list_physical_devices('GPU'))

    # mnist.load_data()  # we do this on the driver because it's not threadsafe
    """
    ray.init(num_cpus=8,
             num_gpus=1,
             include_dashboard=True,  # if you use docker use docker run -p 8265:8265 -p 6379:6379
             dashboard_host='0.0.0.0')
    
    # ray.init(address='auto', _redis_password='5241590000000000')
    try:
        ray.init()
    except:
        ray.shutdown()
        ray.init()
    """
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
        name="keras_select",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_score_attr='min-val_loss',
        checkpoint_freq=3,
        checkpoint_at_end=False,
        verbose=3,
        # Optimalization
        metric="val_loss",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 5000
        },
        num_samples=3000,  # number of samples from hyperparameter space
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
            "batch": tune.choice([64]),
            "learning_rate": tune.choice([0.1]),
            # Layer 1 params
            "hidden1": tune.randint(1, 200),
            "activation1": tune.choice(["linear"]), #elu
            "noise_std": tune.uniform(0.001, 0.5),
            "l1_value": tune.loguniform(1e-5, 1e-1),
            "l2_value": tune.loguniform(1e-5, 1e-1),
            "dropout": tune.uniform(0.001, 0.5)
        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)





# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/keras_select --bind_all --load_fast=false
"""
with noise and bias
val_loss=0.014095677062869072 and parameters={'batch': 64, 'learning_rate': 0.1, 'hidden1': 5, 'activation1': 'elu', 'noise_std': 0.34762315075100414, 'l2_value': 2.1269345780548958e-05}
val_loss=0.01846976950764656 and parameters={'batch': 64, 'learning_rate': 0.1, 'hidden1': 2, 'activation1': 'elu', 'noise_std': 0.04558018730060088, 'l2_value': 0.0003760500113491962}
with noise and  no bias
{'batch': 64, 'learning_rate': 0.1, 'hidden1': 1, 'activation1': 'elu', 'noise_std': 0.13930154430030395, 'l2_value': 1.67355964893191e-05}
"""
