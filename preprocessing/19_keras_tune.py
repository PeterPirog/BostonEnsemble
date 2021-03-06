# UBUNTU
from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


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

    best_features = ['_BldgType_2', '_BldgType_1', '_BldgType_3', '_GrLivArea', '_MSSubClass_3', '_OverallQual',
                     '_BuildingAge', '_TotalBsmtSF', '_CentralAir', '_Electrical', '_SaleCondition_Abnorml', '_LotArea',
                     '_GarageArea', '_KitchenQual', '_OverallCond', '_BsmtExposure', '_BsmtUnfSF', '_Foundation_2',
                     '_HouseStyle_2', '_HouseStyle_3', '_FullBath', '_Neighborhood_1', '_FireplaceQu', '_PavedDrive',
                     '_HeatingQC', '_HalfBath', '_HouseStyle_1', '_WoodDeckSF', '_MSSubClass_1', '_LotFrontage',
                     '_HouseStyle_4', '_MasVnrType_BrkFace', '_MasVnrType_Stone', '_BsmtFullBath', '_Neighborhood_8',
                     '_Fence', '_Exterior_MetalSd', '_MSZoning_3', '_Neighborhood_2', '_QuarterSold', '_BsmtFinSF1',
                     '_BedroomAbvGr', '_Foundation_1', '_MSSubClass_2']
    # n_features = len(features_all)
    # n_features=3
    y = df['SalePrice_log1']

    # X = df[features_all[:config['n_features']]]
    X = df[features_all]

    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    epochs = 100000
    # define model
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error

    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=10),
                      tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                         monitor='val_loss',
                                                         save_best_only=True),
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
                               max_t=500,
                               grace_period=10,
                               # mode='max', #find maximum, do not define here if you define in tune.run
                               reduction_factor=3,
                               # brackets=1
                               )

    analysis = tune.run(
        train_boston,
        search_alg=HyperOptSearch(),
        name="keras",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        metric="val_loss",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 500
        },
        num_samples=1000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/',
        # default value is ~/ray_results /root/ray_results/
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            # training parameters
            "batch": tune.choice([64]),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            # Layer 1 params
            "hidden1": tune.randint(16, 129),
            "activation1": tune.choice(["elu"]),
            "dropout1": tune.uniform(0.01, 0.15),
            # Layer 2 params
            "hidden2": tune.randint(16, 129),
            "dropout2": tune.uniform(0.01, 0.15),  # tune.choice([0.01, 0.02, 0.05, 0.1, 0.2])
            "activation2": tune.choice(["elu"]),
            "activation_output": tune.choice(["elu"])  # ,tf.keras.activations.exponential
            # "layers": tune.choice([2]),tf.keras.activations.exponential

        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)
# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/ray_results/keras --bind_all --load_fast=false
