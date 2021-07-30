# https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
# https://www.adriangb.com/scikeras/stable/notebooks/Basic_Usage.html
#https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk

import pandas as pd
import numpy as np
from joblib import dump, load
import tensorflow as tf
import dill as pickle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files


# define base model
def baseline_model(hidden1, hidden2, activation, dropout, lr):
    # create model
    epochs = 100000
    number_inputs = 79

    # {'random_state': 117, 'batch': 64, 'learning_rate': 0.1, 'hidden1': 31, 'activation1': 'elu', 'dropout1': 0.28}
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # layer 2
    x = tf.keras.layers.Dense(units=hidden2, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mean_squared_error',  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics='mean_squared_error')
    return model


def load_keras_model():
    model = tf.keras.models.load_model('model_keras.h5')
    return model


if __name__ == "__main__":
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

    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=30),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.8,
                                                           patience=20),
                      tf.keras.callbacks.ModelCheckpoint(filepath='model_keras.h5',
                                                         monitor='val_loss',
                                                         save_best_only=True), ]
    # USING SCIKERAS not sklearn wrapper
    keras_model = KerasRegressor(build_fn=baseline_model,
                               ## custom parameters
                               hidden1=conf_keras['hidden1'],
                               hidden2=conf_keras['hidden2'],
                               activation=conf_keras['activation'],
                               dropout=conf_keras['dropout'],
                               lr=conf_keras['learning_rate'],
                               ## fit parameters
                               # x=None,#conf_keras['x']
                               # y=None,#conf_keras['y']
                               batch_size=conf_keras['batch_size'],
                               epochs=conf_keras['epochs'],
                               verbose=conf_keras['verbose'],
                               callbacks=callbacks_list,
                               validation_split=conf_keras['validation_split'],
                               # validation_data=conf_keras['validation_data'],
                               shuffle=conf_keras['shuffle'],
                               validation_batch_size=conf_keras['validation_batch_size'],
                               )
    #https://scikeras.readthedocs.io/en/latest/generated/scikeras.wrappers.KerasRegressor.html
    keras_model=KerasRegressor(build_fn=baseline_model,warm_start=True,epochs=1)


    keras_model.fit(X=X, y=y)

    import dill as pickle

    with open('model_keras.pkl', 'wb') as file:
        pickle.dump(keras_model, file)

    with open('model_keras.pkl', 'rb') as file:
        model = pickle.load(file)
    # import pickle

    # bytes_model = pickle.dumps(estimator)
    # model = pickle.loads(bytes_model)
    print(model.predict(X[:5]))





    # Save pipeline or model in joblib file
    # estimator2 = KerasRegressor(build_fn=load_keras_model,epochs=1)
    # dump(estimator2, filename='model_keras.joblib')#conf_keras['output_file']

    # kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
    # results = cross_val_score(model, X, y, cv=kfold, n_jobs=-1)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    """
    #train model with parameters defined in conf_ridge,json file
    estimator.fit(X=X, y=y)

    v = Validator(model_or_pipeline=estimator, X=X, y=y, n_splits=10, n_repeats=1, random_state=1,
                             scoring='neg_root_mean_squared_error', model_config_dict=None)
  v.run()
    """
    # model_keras = load(filename='model_keras.joblib')
    # print(estimator.predict(X[:5]))
    # print(estimator)

"""
        build_fn=None
        warm_start=False
        random_state=None
        optimizer=rmsprop
        loss=mse
        metrics=[<function KerasRegressor.r_squared at 0x7f08649421f0>]
        batch_size=None
        validation_batch_size=None
        verbose=1
        callbacks=None
        validation_split=0.0
        shuffle=True
        run_eagerly=False
        epochs=1
        hidden_layer_sizes=(100,)
        dropout=0.5
val_loss=0.013803128153085709 and parameters={'batch': 64, 'learning_rate': 0.1, 'hidden1': 40, 'activation1': 'elu', 'dropout1': 0.46}

"""
