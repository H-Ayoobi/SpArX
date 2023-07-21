# Importing all the libraries and utilities
"""
------------------------------------------
Clustering using all the inputs.
Global Explanations.
------------------------------------------
This code includes the approach for shrinking a two layer FFNN to a new network.
The new model clusters the hidden nodes in the original network based on their activations.
-----------------steps to achieve a shrunken model---------------------
1) This means that using all the inputs, we first compute the activations of the hidden nodes.
2) Then, using all these activations, we cluster the nodes that topically have the same (or close)
 activations.
3) Then the weights are computed independent of the inputs.
* There are multiple choices for the incoming weights of the hidden nodes:
  a) Mean
  b) Weights by random selection of a node at each cluster.
* For the outgoing weights from the hidden node, we can use sum as a natural choice.
------------------------ Convert to QBAF ----------------------------
4) Afterwards, we convert the resulting network to a QBAF by interpreting negative weights as attack and
 positive weights as supports and we visualize it.
* In the visualization step, green edges show support and negative edges show attack.
* The width of each edge shows the strength of attack or support relation (based on their weights).


How to change the FFNN model?
To change the architecture of the model you should both edit the HIDDEN_LAYERS parameter
(for example HIDDEN_LAYERS =[10,20] means the model has two hidden layers the with dimensions 10 and 20 sequentially.)
You can also change the get_FFNN_model function to change the activation functions, the optimizer and loss function.


 """


# Importing all the libraries and utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 35)

# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import libraries
import numpy as np
from datetime import datetime, date, timedelta
import string, pickle, json, sys, os, itertools, random, math, time, re, hashlib, warnings, subprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from keras.utils import plot_model
import os
from sklearn.cluster import KMeans, AgglomerativeClustering

# CUDA setting for GPU processing
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model Training
from keras.callbacks import ModelCheckpoint, EarlyStopping

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    import tensorflow as tf
    import keras
    # import keras.backend as K
    from tensorflow.python.keras import backend as K
    from keras.utils.np_utils import to_categorical

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    K.set_session(sess)

    # Check that we are using the standard configuration for channels
    assert K.image_data_format() == 'channels_last'

RESULT_PATH = './results'

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))




from sklearn.datasets import load_breast_cancer
save_QBAFS = True
# dataset and model parameters
class_names = ['malignant', 'benign'] #in case of compas dataset, the output could be 'No' and 'Yes'
# hidden layers
HIDDEN_LAYERS = [50, 50]
# Training parameters
EPOCHS = 1000
PATIENCE = 30
BATCH_SIZE = 64
# How much should the network be shrunken
Shrinkage_percentage = 50  # Integers in range (0-100). how much do you want the network be shrinked? (in percentatge)
# How much of the edges with low weights should be pruned in visualization
# (notice that they are not really pruned in the computation process)?
pruning_ratio = 0.8  # the number should be in range [0, 1].
# Example: pruning_ratio = 0.8 means that only the edges higher than 0.8 quantile of all the weights would be shown.

# Load and plot
data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)

# divide test and train (one-hot and original format)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2, random_state=2, shuffle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, Input, concatenate, Dropout, Activation


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
        # With sigmoid hidden layers
        ff_layers = [
            Dense(hidden_layers_size[0], input_shape=(input_size,), activation="relu"),
            Dense(output_size, activation='sigmoid')
        ]
        for hidden_size in hidden_layers_size[1:]:
            ff_layers.insert(-1, Dense(hidden_size, activation='relu'))

    print(ff_layers)
    model = Sequential(ff_layers)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m])
    model.summary()
    return model





# train FFNN
def net_train(model, bestmodel_path, X_train, y_train_onehot, X_validate, y_validate_onehot, epochs=EPOCHS):
    # Define four callbacks to use
    checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)

    # Train the model
    history = model.fit(X_train, y_train_onehot, verbose=2, epochs=epochs, batch_size=BATCH_SIZE,
                        callbacks=[checkpointer, early_stopping], validation_data=(X_validate, y_validate_onehot))

    return history


model = get_FFNN_model(X_train, y_train, HIDDEN_LAYERS)

model_path = os.path.join(RESULT_PATH, 'cancer_global_net.h5')
forge_gen = False

if not os.path.exists(model_path) or forge_gen:
    history = net_train(model, model_path, X_train, y_train, X_test, y_test)

    score = model.evaluate(X_test, y_test)
    plt.figure(figsize=(14, 6))
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)
    plt.legend(loc='best')
    plt.grid(alpha=.2)
    plt.title(f'batch_size = {BATCH_SIZE}, epochs = {EPOCHS}')
    plt.draw()
else:
    print('Model loaded.')
    model.load_weights(model_path)

# predicting the label of test data.
y_pred = model.predict(X_test).flatten() > 0.5
y_ground_truth = y_test.values == 1
print(classification_report(y_ground_truth, y_pred, digits=4))

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plot_model(model, to_file=RESULT_PATH + '/model.png', show_shapes=True, show_layer_names=False)


# clusting each hidden layer based on its activation
def clustering_nodes(preserve_percentage, NUMBER_OF_NODES_IN_HIDDEN_LAYER, activations, clustering_algorithm="kmeans"):
    # Shrink the network using the Kmeans clustering below.
    # Input: the activations of all the layers of the network (original model) using all the data instances (examples)
    # Output: a nested list showing that at each layer  what would be the new cluster label of each node.
    #         For example if the original model has [4, 6] hidden nodes (this means that the first hidden layer
    #         has 4 nodes and the second hidden layer has 6 nodes) and
    #         the clustering algorithm produces [[0,1,0,1],[1,2,0,0,2,1]] list as output,
    #         then it means that the first and the third node in the first layer are assinged to the first cluster and
    #         the second and the fourth nodes are assigned to the second cluster.
    #         For the second layer the third and the fourth nodes are assigned to the first cluster (0)
    #         and the first and the last nodes are assigned to the second cluster (1)
    #         and the second and the fifth are assigned to the third cluster (2).  .
    clustering_labels = []
    for index, hidden_layer in enumerate(NUMBER_OF_NODES_IN_HIDDEN_LAYER):
        activation = activations[index]
        clustering_input = activation.T
        n_clusters_ = int((preserve_percentage / 100) * hidden_layer)
        if clustering_algorithm == "kmeans":
            clustering = KMeans(n_clusters=n_clusters_, random_state=1).fit(clustering_input)
        elif clustering_algorithm == "AgglomerativeClustering":
            clustering = AgglomerativeClustering(n_clusters=n_clusters_).fit(clustering_input)
        clustering_labels.append(clustering.labels_)
    return clustering_labels

# merge the nodes at each cluster and recompute the incoming and outgoing weights of edges
def merge_nodes(X_onehot, y_onehot, activations, model, preserve_percentage, HIDDEN_LAYERS, clustering_labels):
    # Based on the clustering step, we now shrink the model.
    # The strategy is to keep the a node from each cluster in the
    # hidden layer (and average the connecting weights of the input to the hidden layer and the biases) and
    # in the outgoing weights layer sum up the weights of all the nodes in the cluster
    input_size = len(X_onehot.columns.values)
    output_size = len(y_onehot.columns.values)
    weights = []
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    biases = []
    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        for label in range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            weights[index].append(
                np.mean(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
            biases[index].append(np.mean(model.layers[index].get_weights()[1][clustering_labels[index] == label]))
            outgoing_weights[index + 1].append((np.sum(
                model.layers[index + 1].get_weights()[0][clustering_labels[index] == label], axis=0)).reshape((1, -1)))
    biases.append([model.layers[len(HIDDEN_LAYERS)].get_weights()[1]])
    weights.append(outgoing_weights[-1])
    # -1 to skip the last one which is already in correct shape.
    for index in range(len(weights)):
        if index == len(weights) - 1:
            weights[index] = np.vstack(weights[index])
        else:
            weights[index] = np.vstack(weights[index]).T
        biases[index] = np.vstack(biases[index]).reshape(-1,)

    return weights, biases, input_size, output_size

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compare_shrinked_with_original(y_ground_truth, y_pred_original_model, y_pred_shrinked_model, shrink_percentage):
    false_predictions = 0
    better_predictions = 0
    false_yes = 0
    false_no = 0
    true_yes = 0
    true_no = 0
    y_ground_truth = y_ground_truth.replace({
        1: True,
        0: False
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


# Truncated model to see activations
activations = compute_activations_for_each_layer(model, X_test.values)

# Shrink the network using the Kmeans clustering below. #Here we shrink the hidden nodes from 100 nodes to 3 nodes
preserve_percentage = 100 - Shrinkage_percentage
clustering_labels = clustering_nodes(preserve_percentage, HIDDEN_LAYERS, activations,
                                     clustering_algorithm="kmeans")
print(clustering_labels)
# clustering_labels = np.array([4, 2, 5, 1, 0, 4, 3, 2, 3, 0])


weights, biases, input_size, output_size = merge_nodes(X_test,
                                                y_test,
                                                activations,
                                                model,
                                                preserve_percentage,
                                                HIDDEN_LAYERS,
                                                clustering_labels)
truncated_model_dimensions = [int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]

#cunstruct the structure of the truncated model using the same function for building a FFNN model.
shrinked_model = get_FFNN_model(X_train, y_train, truncated_model_dimensions)
truncated_weights = []
for index, weight in enumerate(weights):
    truncated_weights.append(weight)
    truncated_weights.append(biases[index])

#setting weights of the truncated model
shrinked_model.set_weights(truncated_weights)


y_shrinked_pred = shrinked_model.predict(X_test).flatten() > 0.5

y_shrinked_pred_train = shrinked_model.predict(X_train).flatten() > 0.5
y_pred_train = model.predict(X_train).flatten() > 0.5

print(classification_report(y_test.values == 1, y_shrinked_pred, digits=4))
compare_shrinked_with_original(y_train, y_pred_train, y_shrinked_pred_train, Shrinkage_percentage)

#make a vector from all weights of the shrunken network. This will be used to remove the
all_weights = []
for weight in weights:
    all_weights.extend(list(weight.reshape((-1,))))


if save_QBAFS:
    # visualize the shrunken model as QBAF.
    for test_index in range(0, 20):  # range(len(np.array(X_test))):
        input = np.array(X_test)[test_index]
        output = np.array(y_test)[test_index]
        feature_names = X_train.columns.values
        number_of_hidden_nodes = [int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]


        quantile = np.quantile(np.abs(np.array(all_weights)).reshape(1, -1), pruning_ratio)
        weight_threshold = quantile

        from plot_QBAF import visualize_attack_and_supports_QBAF, general_method_for_visualize_attack_and_supports_QBAF

        general_method_for_visualize_attack_and_supports_QBAF(input, output, shrinked_model, feature_names, number_of_hidden_nodes,
                                                              weight_threshold, weights, biases, Shrinkage_percentage,
                                                              'cancer_global_graphs (shrunken_model)', test_index)



    #make a vector from all weights of the original network.
    all_weights_original = []
    original_weights=[]
    for layer in model.layers:
        all_weights_original.extend(list(layer.get_weights()[0].reshape((-1,))))
        original_weights.append(layer.get_weights()[0])



    # visualize the original model
    for test_index in range(0, 20):  # range(len(np.array(X_test))):
        input = np.array(X_test)[test_index]
        output = np.array(y_test)[test_index]
        feature_names = X_train.columns.values
        feature_names = X_train.columns.values

        quantile = np.quantile(np.abs(np.array(all_weights_original)).reshape(1, -1), pruning_ratio)
        weight_threshold = quantile

        from plot_QBAF import visualize_attack_and_supports_QBAF, general_clustered_visualize_attack_and_supports_QBAF


        general_clustered_visualize_attack_and_supports_QBAF(input, output, model, feature_names, HIDDEN_LAYERS,
                                                              weight_threshold, original_weights, biases, Shrinkage_percentage,
                                                              'cancer_global_graphs (original_model)', test_index, clustering_labels)



fidelity = np.mean(np.sum(np.power(shrinked_model.predict(X_test.values) -
                               model.predict(X_test.values), 2), axis=1))
print(f"Fidelity: {fidelity}")
number_of_nodes = sum(truncated_model_dimensions) + y_test.shape[1]
original_activations = compute_activations_for_each_layer(model, X_test.values)
pruned_activations = compute_activations_for_each_layer(shrinked_model, X_test.values)
structural_fidelity = 0
for i, original_activation in enumerate(original_activations):
    pruned_activation = pruned_activations[i]
    if i != len(original_activations)-1:
        for cluster_label in range(int(HIDDEN_LAYERS[i]*preserve_percentage/100)):
            if cluster_label in clustering_labels[i]:
                structural_fidelity += np.sum(np.abs(np.mean(original_activation[:,clustering_labels[i]==cluster_label], axis=1) - pruned_activation[:,cluster_label]))
    else:
        structural_fidelity += np.sum(np.abs(pruned_activation-original_activation))

structural_fidelity /= (number_of_nodes * X_test.values.shape[0])
print(f"Structural fidelity = {structural_fidelity}")