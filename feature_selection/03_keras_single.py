import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd
from pathlib import Path
from tensorflow.keras.regularizers import l2, l1_l2



def baseline_model(number_inputs, hidden1, activation, noise_std, l1_value, l2_value, dropout):
    # define model
    inputs = tf.keras.layers.Input(shape=number_inputs)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
    # layer 1
    x = tf.keras.layers.Dense(units=hidden1, kernel_initializer='glorot_normal',
                              activation=activation,
                              kernel_regularizer=l1_l2(l1=l1_value, l2=l2_value),
                              use_bias=False, name='dense_layer')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(units=1,name='output_layer')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss='mse',  # mean_squared_logarithmic_error "mse" 'mean_squared_error'
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics='mean_squared_error')
    return model


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent

    tf.config.threading.set_inter_op_parallelism_threads(num_threads=16)
    # Get all fetures in dataframe
    df = pd.read_csv((base_path / "./data_files/encoded_train_X_data.csv").resolve())
    y = df['SalePrice_log1']
    X = df.drop(['Id', 'SalePrice', 'SalePrice_log1'], axis=1).copy()

    X = X.to_numpy()
    y = y.to_numpy()

    callbacks_list = [
        # Reducle value rl
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             factor=0.8,
                                             patience=10),
        # early stop
        tf.keras.callbacks.EarlyStopping(monitor='loss',
                                         patience=15),
        # save best result
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_1L.h5',
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir="/home/peterpirog/PycharmProjects/BostonEnsemble/tensorboard",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None
        )
    ]

    model = baseline_model(number_inputs=80,
                           hidden1=20, #20
                           activation='elu',
                           noise_std=0.020298941092507204,
                           l1_value=3.543298823184669e-05,
                           l2_value=5.945058708937871e-05,
                           dropout=0.1309933349281991)

    model.summary()

    model.fit(x=X,
              y=y,
              batch_size=64,
              epochs=100000,
              verbose=1,
              callbacks=callbacks_list,
              workers=1,
              use_multiprocessing=True)
    model.save('.\model_1L.h5')
    model2 = tf.keras.models.load_model('model_1L.h5')
    model2.summary()

# tensorboard --logdir /home/peterpirog/PycharmProjects/BostonEnsemble/tensorboard --bind_all --load_fast=false

