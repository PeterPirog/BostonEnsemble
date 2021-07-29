#https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
import pandas as pd
import numpy as np
from joblib import dump, load
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from ensemble_tools import FeatureByNameSelector, Validator
from _create_json_conf import read_config_files

# define base model
def baseline_model(hidden_layers,dropout,lr=0.001):
	# create model
    epochs = 100000
    number_inputs=79
    activation='elu'
    print('lr=',lr)
    #{'random_state': 117, 'batch': 64, 'learning_rate': 0.1, 'hidden1': 31, 'activation1': 'elu', 'dropout1': 0.28}
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden_layers, kernel_initializer='glorot_normal',
                              activation=activation)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # layer 2
    x = tf.keras.layers.Dense(units=hidden_layers, kernel_initializer='glorot_normal',
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



if __name__ == "__main__":
    conf_global = read_config_files(configuration_name='conf_global')
    conf_keras = read_config_files(configuration_name='conf_keras')

    df = pd.read_csv(conf_global['encoded_train_data'])



    X = df[conf_global['all_features']]

    y = df[conf_global['target_label']]

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                                           factor=0.1,
                                                           patience=10), ]

    estimator = KerasRegressor(build_fn=baseline_model,
                               ## custom parameters
                               hidden_layers=31,
                               dropout=0.28,
                               lr=0.1,
                               ## fit parameters
                               nb_epoch=1500,
                               batch_size=64,
                               verbose='auto',
                               callbacks=callbacks_list,
                               workers=64,use_multiprocessing=True)

    kfold = KFold(n_splits=10, random_state=seed,shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold, n_jobs=-1)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    """
    #train model with parameters defined in conf_ridge,json file
    estimator.fit(X=X, y=y)

    #Save pipeline or model in joblib file
    #dump(estimator, filename=conf_ridge['output_file'])
    v = Validator(model_or_pipeline=estimator, X=X, y=y, n_splits=10, n_repeats=5, random_state=1,
                  scoring='neg_root_mean_squared_error',model_config_dict=None)
    v.run()
    """
    #print(estimator.predict(X[:5]))
    print(estimator)

"""
Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)


"""