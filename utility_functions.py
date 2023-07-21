import numpy as np
from datetime import datetime, date, timedelta
import string, pickle, json, sys, os, itertools, random, math, time, re, hashlib, warnings, subprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# from keras.utils import plot_model
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
# from keras.callbacks import ModelCheckpoint, EarlyStopping



EPOCHS = 1000
PATIENCE = 100
BATCH_SIZE = 64

# CUDA setting for GPU processing
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model Training
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    import tensorflow as tf
    from tensorflow import keras
    # import keras.backend as K
    from tensorflow.python.keras import backend as K
    # from tensorflow.keras.utils.np_utils import to_categorical

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    K.set_session(sess)

    # Check that we are using the standard configuration for channels
    assert K.image_data_format() == 'channels_last'

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class MultiColumnLabelEncoder:
    def __init__(self, columns):
        self.columns = columns  # array of column names to encode
        self.encoders = []

    def fit(self, X, y):
        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                le = LabelEncoder()
                self.encoders.append(le)
                output[col] = le.fit_transform(output[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Model Creation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation,BatchNormalization


# evaluating model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# constructing model. Structure: input, multiple hidden layers(relu), output(relu, sigmoid)
def get_FFNN_model(X, y, hidden_layers_size=[4]):
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(X.columns.values)
    output_size = len(y.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [Dense(output_size, input_shape=(input_size,), activation='softmax')]
    else:
        # Last hidden layer and the output layer
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation="relu"),
            Dense(output_size, activation='sigmoid')
        ]
        # other hidden layers (if more than one hidden layer exists)
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    # print(ff_layers)
    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])
    # model.summary()
    return model

# constructing model. Structure: input, multiple hidden layers(relu), output(relu, softmax)
def get_FFNN_model_non_binary(X, y, hidden_layers_size=[4]):
    """
        BASIC MODEL for the FF-NN
    """
    input_size = len(X.columns.values)
    output_size = len(y.columns.values)

    if len(hidden_layers_size) == 0:
        # No hidden layer (linear regression equivalent)
        ff_layers = [Dense(output_size, input_shape=(input_size,), activation='softmax')]
    else:
        # With sigmoid hidden layers
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation='relu'),
            Dense(output_size, activation='softmax')
        ]
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    print(ff_layers)
    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])
    model.summary()
    return model


# train FFNN
def net_train(model, bestmodel_path, X_train, y_train_onehot, X_validate, y_validate_onehot, epochs=EPOCHS,
              batch_size=BATCH_SIZE, patience=PATIENCE):
    # Define four callbacks to use
    checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    # Train the model
    history = model.fit(X_train, y_train_onehot, verbose=2, epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))

    return history



def compare_shrinked_with_original(y_ground_truth, y_pred_original_model, y_pred_shrinked_model, shrink_percentage):
    false_predictions = 0
    better_predictions = 0
    false_yes = 0
    false_no = 0
    true_yes = 0
    true_no = 0
    y_ground_truth = y_ground_truth.replace({
        'Yes': True,
        'No': False
    })
    for i in range(len(y_ground_truth)):
        if y_pred_original_model[i] == False and y_pred_shrinked_model[i] == True:
            false_yes += 1
        elif y_pred_original_model[i] == True and y_pred_shrinked_model[i] == False:
            false_no += 1
        elif y_pred_original_model[i] == True and y_pred_shrinked_model[i] == True:
            true_yes += 1
        elif y_pred_original_model[i] == False and y_pred_shrinked_model[i] == False:
            true_no += 1

    print('confusion_matrix: shrinked_model vs original_model')
    print(' New  |   Yes   |   No  |')
    print("__________________________")
    print('  Yes |  ' + str(true_yes) + '   |  ' + str(false_no) + '  |')
    print('  No  |    ' + str(false_yes) + '   |  ' + str(true_no) + ' |')
    print(' Orig |')

    print(
        'Shrinked Model vs Original: they are {percentage:.2f} % the same.\n Notice that the number of nodes are {reduce:.0f} % reduced.'.format(
            percentage=(100 * (1 - (false_yes + false_no) / (true_yes + true_no))), reduce=shrink_percentage))


def divide_two_classes(Y_GT, Y_Pred):
    Y_GT = np.array(Y_GT)
    confusion = np.zeros((2, 2))
    for i in range(len(np.array(Y_GT))):
        if Y_GT[i] == 'No' and Y_Pred[i] == False:
            confusion[0, 0] += 1
        elif Y_GT[i] == 'No' and Y_Pred[i] == True:
            confusion[0, 1] += 1
        elif Y_GT[i] == 'Yes' and Y_Pred[i] == True:
            confusion[1, 1] += 1
        else:
            confusion[1, 0] += 1

    return confusion


def plot_confusion(confusion_array):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    df_cm = pd.DataFrame(confusion_array, index=[i for i in ['Yes_GT', 'No_GT']],
                         columns=[i for i in ['Yes_Pred', 'No_Pred']])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()
    # plt.show(block=False)


def compute_activations_for_each_layer(model, input_data):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functor = K.function([inp], outputs)  # evaluation functions

    # computing activations
    activations = functor([input_data])
    return activations