# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy


# Function to create model, required for KerasClassifier
def create_model(init='glorot_uniform'):
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                       factor=0.1,
                                                       patience=10),]

# create model
model = KerasClassifier(build_fn=create_model,
                        #tekst='aaaaaaaa',
                        epochs=150,
                        batch_size=64,
                        verbose=1,
                        use_multiprocessing=True,
                        callbacks=None)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold,n_jobs=-1)
print(results.mean())
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