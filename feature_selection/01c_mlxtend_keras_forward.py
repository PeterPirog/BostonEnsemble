import pandas as pd
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split, KFold
from models_ensemble.ensemble_tools import FeatureByNameSelector, Validator, read_config_files
#http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor


def baseline_model_seq(hidden1, hidden2, activation, noise_std, dropout):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LayerNormalization())
    #Layer 1
    model.add(tf.keras.layers.GaussianNoise(stddev=noise_std))
    model.add(tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                                    activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.LayerNormalization())
    #Layer 2
    model.add(tf.keras.layers.GaussianNoise(stddev=noise_std))
    model.add(tf.keras.layers.Dense(units=hidden2, kernel_initializer='glorot_normal',
                                    activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.LayerNormalization())
    #Output Layer
    model.add(tf.keras.layers.Dense(units=1, kernel_initializer='glorot_normal',
                                    activation='linear'))
    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics='mean_squared_error')  # accuracy mean_squared_logarithmic_error
    return model



if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
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

    df = pd.read_csv(conf_global['encoded_train_data'])
    n = len(conf_global['all_features'])

    X = df[all_features]
    y = df['SalePrice_log1']

    callbacks_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             factor=0.1,
                                             patience=10),
        tf.keras.callbacks.EarlyStopping(monitor='loss',
                                         patience=15)]

    # step forward feature selection algorithm

    model_keras = KerasRegressor(build_fn=baseline_model_seq,
                                 hidden1=30, #136
                                 hidden2=10, #27
                                 noise_std=0.05,
                                 activation='elu',
                                 dropout=0.01,# 0.25
                                 # lr=config['learning_rate'],
                                 ## fit parameters
                                 batch_size=64,
                                 epochs=100000,
                                 verbose=0,
                                 callbacks=callbacks_list
                                 )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    sfs = SFS(model_keras,
              k_features=80,
              forward=True,
              floating=False,
              verbose=2,
              scoring='neg_root_mean_squared_error',  # 'neg_root_mean_squared_error' 'r2'
              cv=cv,
              n_jobs=-1)

    sfs = sfs.fit(X, y)
    print(X.columns[list(sfs.k_feature_idx_)])
    df=pd.DataFrame.from_dict(sfs.get_metric_dict(confidence_interval=0.95)).T
    df.to_csv('forward_keras_features.csv')

    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err')

    plt.ylim([0.8, 1])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.show()