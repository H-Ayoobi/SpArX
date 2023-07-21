"""
------------------------------------------
Clustering using all the inputs
Local Explanations for each input.
Works with any activation function. (General)
------------------------------------------
This code includes the approach for shrinking a multiple layer FFNN to a summarized network.
The new model clusters the hidden nodes in the original network based on their activations.
-----------------steps to achieve a shrunken model---------------------
1) This means that using all the inputs, we first compute the activations of the hidden nodes.
2) Then, using all these activations, we cluster the nodes that topically have the same (or close)
 activations.
3) For each input, we recompute the weights of the clustered network
 so that we have the local explanations for each input.
 Notice that this way the output of the original model and the shrunken model are exactly the same.
-------------------- Convert to QBAF ----------------------------
4) Afterwards, we convert the resulting network to a QBAF by interpreting negative weights as attack and
 positive weights as supports and we visualize it.
* In the visualization step, green edges show support and negative edges show attack.
* The width of each edge shows the strength of attack or support relation (based on their weights).
-------------------- Word Cloud Visualization ------------------
5) To provide a more interpretable explanation, we have visualized each hidden node
 as a word cloud of the incoming features.
* The size of each word in the word cloud is determined based on its attack or support weight.


How to change the FFNN model?
To change the architecture of the model you should both edit the HIDDEN_LAYERS parameter
(for example HIDDEN_LAYERS =[10,20] means the model has two hidden layers the with dimensions 10 and 20 sequentially.)
You can also change the get_FFNN_model function to change the activation functions, the optimizer and loss function.

How to change the problem setting?
replace the load_compas() function to load your dataset with the same format.
 """

# Importing all the libraries and utilities
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 35)

import numpy as np
from datetime import datetime, date, timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from keras.utils import plot_model
import os
from sklearn.cluster import KMeans, AgglomerativeClustering

import utility_functions
import iris_load_dataset

import matplotlib.pyplot as plt
import sklearn

# creating the results directory for saving the deep models and the visualization of the model structure.
RESULT_PATH = './results'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


# hidden layers
HIDDEN_LAYERS = [50, 50]

# How much should the network be shrunken
Shrinkage_percentage = 50  # Integers in range (0-100). how much do you want the network be shrinked? (in percentatge)
# How much of the edges with low weights should be pruned in visualization
# (notice that they are not really pruned in the computation process)?

# this parameter is only used for visualization step
pruning_ratio = 0.5
# the number should be in range [0, 1].
# Example: pruning_ratio = 0.8 means that only the edges higher than 0.8 quantile of all the weights would be shown.

# Load and plot

data = iris_load_dataset.load_iris()

CLASS = "class"
# Split X and y
X = data.drop(columns=[CLASS])
y = data[CLASS]

# Randomize
X = X.sample(frac=1, random_state=2020)
y = y.loc[X.index.values]
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)



# y_onehot = pd.get_dummies(y)[['Yes']]
y_onehot = pd.get_dummies(y)

# for iris
X_train, X_test, y_train, y_test, data_train, data_test, y_onehot_train, y_onehot_test = \
    train_test_split(X, y, data, y_onehot, test_size=.2, random_state=2, shuffle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# constructing the original model
model = utility_functions.get_FFNN_model_non_binary(X, y_onehot, HIDDEN_LAYERS)

# model's name saved in the results folder
model_path = os.path.join(RESULT_PATH, 'net_iris_local.h5')#'net_iris_local.h5')
forge_gen = False
# load model weights if previously trained and saved. If not, train the model.
if not os.path.exists(model_path) or forge_gen:
    history = utility_functions.net_train(model, model_path, X_train, y_onehot_train, X_test, y_onehot_test)
    score = model.evaluate(X_test, y_onehot_test)
    plt.figure(figsize=(14, 6))
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)
    plt.legend(loc='best')
    plt.grid(alpha=.2)
    plt.title(f'batch_size = {utility_functions.BATCH_SIZE}, epochs = {utility_functions.EPOCHS}')
    plt.draw()
else:
    print('Model loaded.')
    model.load_weights(model_path)


# for iris
predictions = np.argmax(model.predict(X_test), axis=1)
y_pred = np.eye(np.max(predictions) + 1)[predictions]
print(classification_report(y_onehot_test, y_pred, digits=4))

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plot_model(model, to_file=RESULT_PATH + '/model.png', show_shapes=True, show_layer_names=False)

# clusting each hidden layer based on its activation (global clustering)
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
        # For global clustering (using all the examples), the number of clusters uses -1 because
        # we want to have a separate cluster of zeros in the local clustering phase. This way
        # we have all the zero activations for a specific example in cluster (0).
        # Therefore, we have -1 here and we add a cluster of zeros after calling
        # the clustering_the_zero_activations_out_to_a_new_cluster_for_an_example function
        n_clusters_ = int((preserve_percentage / 100) * hidden_layer) - 1
        if clustering_algorithm == "kmeans":
            clustering = KMeans(n_clusters=n_clusters_, random_state=1).fit(clustering_input)
        elif clustering_algorithm == "AgglomerativeClustering":
            clustering = AgglomerativeClustering(n_clusters=n_clusters_).fit(clustering_input)
        clustering_labels.append(clustering.labels_)
    return clustering_labels

# this function assigns the zero activations for each node to a separate cluster of zero activations.
def clustering_the_zero_activations_out_to_a_new_cluster_for_an_example(activations, hidden_layers, index,
                                                                        clustering_lables_global):
    clustering_labels_after_adding_zero_cluster = []
    for layer in range(len(hidden_layers)):
        labels_after_adding_cluster_of_zeros = []
        for idx, activation in enumerate(activations[layer][index]):
            if activation == 0:
                labels_after_adding_cluster_of_zeros.append(0)
            else:
                # labels_after_adding_cluster_of_zeros.append(kmeans.labels_[label_index]+1)
                labels_after_adding_cluster_of_zeros.append(clustering_lables_global[layer][idx] + 1)
                # label_index += 1
        clustering_labels_after_adding_zero_cluster.append(np.array(labels_after_adding_cluster_of_zeros))
    return clustering_labels_after_adding_zero_cluster

# merge the nodes at each cluster and recompute the incoming and outgoing weights of edges
def merge_nodes(X_onehot, y_onehot, activations, model, shrunken_model, preserve_percentage, HIDDEN_LAYERS,
                clustering_labels, example_index):
    # Based on the clustering step, we now shrink the model.
    # The strategy is to keep the a node from each cluster in the
    # hidden layer (and average the connecting weights of the input to the hidden layer and the biases) and
    # in the outgoing weights layer compute the activations based on this equation w* = (W.H)/h*.
    input_size = len(X_onehot.columns.values)
    output_size = len(y_onehot.columns.values)
    input = np.array(X_onehot)[example_index]
    weights = []
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    biases = []
    epsilon = 1e-30
    all_layer_sizes = [input_size]
    for hidden_layer in HIDDEN_LAYERS:
        all_layer_sizes.append(hidden_layer)
    all_layer_sizes.append(output_size)

    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        for label in range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            if len(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label]) != 0:
                weights[index].append(
                    np.mean(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
                biases[index].append(np.mean(model.layers[index].get_weights()[1][clustering_labels[index] == label]))
                current_weights = shrunken_model.layers[index].get_weights()[0]
                current_biases = shrunken_model.layers[index].get_weights()[1]
                current_weights[:, label] = weights[index][label]
                current_biases[label] = biases[index][label]
                new_weights = shrunken_model.get_weights()
                new_weights[2 * index] = current_weights
                new_weights[2 * index + 1] = current_biases
                shrunken_model.set_weights(new_weights)
                # h_star_1 = max(np.dot(input, weights[index][label])+biases[index][label], 0)
                h_star = utility_functions.compute_activations_for_each_layer(shrunken_model,
                                                                              input.reshape((1, -1)))[index][0, label]
                all_hidden_activations = activations[index][example_index, clustering_labels[index] == label]
                # h_star += epsilon
                if h_star == 0:
                    h_star = 1
                activations_divided_by_h_star = all_hidden_activations / h_star

                outgoing_weights[index + 1].append(np.dot(activations_divided_by_h_star,
                                                          model.layers[index + 1].get_weights()[0][
                                                              clustering_labels[index] == label]).reshape((1, -1)))
            else:
                weights[index].append(np.zeros(input_size) if index == 0 else np.zeros(
                    int((preserve_percentage / 100) * HIDDEN_LAYERS[index - 1])))
                biases[index].append(0.0)
                outgoing_weights[index + 1].append(np.zeros((1, all_layer_sizes[index + 2])))
                current_weights = shrunken_model.layers[index].get_weights()[0]
                current_biases = shrunken_model.layers[index].get_weights()[1]
                current_weights[:, label] = weights[index][label]
                current_biases[label] = biases[index][label]
                new_weights = shrunken_model.get_weights()
                new_weights[2 * index] = current_weights
                new_weights[2 * index + 1] = current_biases
                shrunken_model.set_weights(new_weights)

    biases.append([model.layers[len(HIDDEN_LAYERS)].get_weights()[1]])
    weights.append(outgoing_weights[-1])
    # -1 to skip the last one which is already in correct shape.
    for index in range(len(weights)):
        if index == len(weights) - 1:
            weights[index] = np.vstack(weights[index])
        else:
            weights[index] = np.vstack(weights[index]).T
        biases[index] = np.vstack(biases[index]).reshape(-1, )

    return weights, biases, input_size, output_size


def merge_nodes_global(X_onehot, y_onehot, activations, model, shrunken_model, preserve_percentage, HIDDEN_LAYERS,
                clustering_labels, example_index, example_weights):
    # Based on the clustering step, we now shrink the model.
    # The strategy is to keep the a node from each cluster in the
    # hidden layer (and average the connecting weights of the input to the hidden layer and the biases) and
    # in the outgoing weights layer compute the activations based on this equation w* = (W.H)/h*.
    input_size = X_onehot.shape[1]#len(X_onehot.columns.values)
    output_size = y_onehot.shape[1]#len(y_onehot.columns.values)
    input = np.array(X_onehot)[example_index]
    all_inputs = np.array(X_onehot)
    weights = []
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    biases = []
    epsilon = 1e-30
    all_layer_sizes = [input_size]
    for hidden_layer in HIDDEN_LAYERS:
        all_layer_sizes.append(hidden_layer)
    all_layer_sizes.append(output_size)

    new_example_weights = example_weights / np.sum(example_weights)
    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        for label in range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            if len(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label]) != 0:
                weights[index].append(
                    np.mean(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
                biases[index].append(np.mean(model.layers[index].get_weights()[1][clustering_labels[index] == label]))
                current_weights = shrunken_model.layers[index].get_weights()[0]
                current_biases = shrunken_model.layers[index].get_weights()[1]
                current_weights[:, label] = weights[index][label]
                current_biases[label] = biases[index][label]
                new_weights = shrunken_model.get_weights()
                new_weights[2 * index] = current_weights
                new_weights[2 * index + 1] = current_biases
                shrunken_model.set_weights(new_weights)
                # h_star_1 = max(np.dot(input, weights[index][label])+biases[index][label], 0)
                h_star = utility_functions.compute_activations_for_each_layer(shrunken_model, all_inputs)[index][:, label]
                all_hidden_activations = activations[index][:, clustering_labels[index] == label]
                # h_star += epsilon
                # distance_examples_to_target = np.exp(- (np.power(all_inputs - input, 2)/sigma))
                one_weights = np.ones_like(h_star) / len(h_star)
                h_star = np.array([1 if h_s == 0 else h_s for h_s in list(h_star)])
                #
                activations_divided_by_h_star = np.sum(np.multiply(all_hidden_activations.T, new_example_weights / h_star).T, axis = 0)

                outgoing_weights[index + 1].append(np.dot(activations_divided_by_h_star,
                                                          model.layers[index + 1].get_weights()[0][
                                                              clustering_labels[index] == label]))
            else:
                weights[index].append(np.zeros(input_size) if index == 0 else np.zeros(
                    int((preserve_percentage / 100) * HIDDEN_LAYERS[index - 1])))
                biases[index].append(0.0)
                outgoing_weights[index + 1].append(np.zeros((1, all_layer_sizes[index + 2])))
                current_weights = shrunken_model.layers[index].get_weights()[0]
                current_biases = shrunken_model.layers[index].get_weights()[1]
                current_weights[:, label] = weights[index][label]
                current_biases[label] = biases[index][label]
                new_weights = shrunken_model.get_weights()
                new_weights[2 * index] = current_weights
                new_weights[2 * index + 1] = current_biases
                shrunken_model.set_weights(new_weights)

    biases.append([model.layers[len(HIDDEN_LAYERS)].get_weights()[1]])
    weights.append(outgoing_weights[-1])
    # -1 to skip the last one which is already in correct shape.
    for index in range(len(weights)):
        if index == len(weights) - 1:
            weights[index] = np.vstack(weights[index])
        else:
            weights[index] = np.vstack(weights[index]).T
        biases[index] = np.vstack(biases[index]).reshape(-1, )

    return weights, biases, input_size, output_size


#generating samples like LIME
def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

kernel_width = np.sqrt(X_test.shape[1]) * .75
from functools import partial
kernel_fn = partial(kernel, kernel_width=kernel_width)

# Truncated model to see activations
activations = utility_functions.compute_activations_for_each_layer(model, X_test.values)
overal_unfaithfulness_list = []
Shrinkage_percentage_list = []
overal_structural_unfaithfulness_list = []
overal_LIME_local_unfaithfulness_list = []
for Shrinkage_percentage in range(20, 90, 20):
    Shrinkage_percentage_list.append(Shrinkage_percentage/100)
    # Shrink the network using the Kmeans clustering below. #Here we shrink the hidden nodes from 100 nodes to 3 nodes
    preserve_percentage = 100 - Shrinkage_percentage

    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-versicolor']
    feature_names = X_test.columns.values

    from lime import lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(X_train.values, mode="classification",
                                                      class_names = class_names,
                                                      feature_names = feature_names,
                                                      random_state = 123)
    # global clustering
    measures_for_visualization = []
    number_of_tests = 1
    feature_attribution_list = []
    sigmas = [2]
    all_misclassifications = []
    weighted_merging = True # this means that instead of sum for the output we use weighted sum weighted based on activation values


    for j in range(0, number_of_tests):
        sigma = sigmas[j]
        feature_attribution_distance = 0
        overal_unfaithfulness = 0
        overal_LIME_local_unfaithfulness = 0
        overal_structural_unfaithfulness = 0
        measures_for_visualization.append(np.zeros(3))
        measures_for_visualization[j][0] = sigma
        number_of_examples = 5
        misclassification = 0
        # var = np.mean(np.var(X_test.values[0:number_of_examples], axis=0))
        for example_index in range(0, number_of_examples):  # len(X_onehot_test)):

            truncated_model_dimensions = [int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]

            # cunstruct the structure of the truncated model using the same function for building a FFNN model.
            shrinked_model = utility_functions.get_FFNN_model_non_binary(X_test, y_onehot_test, truncated_model_dimensions)

            # Generate samples around an example using lime_tablurar __data_inverse function
            # todo: change __data_inverse in lime_tablurar.py to data_inverse to be able to use the function outside lime_tablurar.py
            data, inverse = explainer.data_inverse(X_test.values[example_index], 5000)
            scaled_data = (data - explainer.scaler.mean_) / explainer.scaler.scale_
            data_labels = model.predict(inverse)

            # find wight of each example with respect to a target example
            distance_examples_to_target = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric='euclidean'
            ).ravel()
            example_weights = kernel_fn(distance_examples_to_target)
            labels_column = data_labels[:, 0]
            used_features = explainer.base.feature_selection(scaled_data,
                                                             labels_column,
                                                             example_weights,
                                                             X_train.shape[1],
                                                             'auto')
            scaled_data = scaled_data[:, used_features]

            activations = utility_functions.compute_activations_for_each_layer(model, scaled_data)

            clustering_labels_with_no_zero = clustering_nodes(preserve_percentage, HIDDEN_LAYERS, activations,
                                                              clustering_algorithm="kmeans")
            print(clustering_labels_with_no_zero)

            clustering_labels = clustering_the_zero_activations_out_to_a_new_cluster_for_an_example(
                activations, HIDDEN_LAYERS,
                0, clustering_labels_with_no_zero)

            if weighted_merging:
                weights, biases, input_size, output_size = merge_nodes_global(
                    scaled_data,
                    data_labels,
                    activations,
                    model,
                    shrinked_model,
                    preserve_percentage,
                    HIDDEN_LAYERS,
                    clustering_labels,
                    0,
                    example_weights)
            else:
                weights, biases, input_size, output_size = merge_nodes(
                    scaled_data,
                    data_labels,
                    activations,
                    model,
                    shrinked_model,
                    preserve_percentage,
                    HIDDEN_LAYERS,
                    clustering_labels,
                    0)

            truncated_weights = []
            for index, weight in enumerate(weights):
                truncated_weights.append(weight)
                truncated_weights.append(biases[index])

            # setting weights of the truncated model
            shrinked_model.set_weights(truncated_weights)

            y_pred_test_shrinked = shrinked_model.predict(
                np.array(scaled_data[0]).reshape((1, -1))).flatten()
            y_pred_test = model.predict(np.array(scaled_data[0]).reshape((1, -1))).flatten()

            print(f"Both Shrunken model and the Original model produce exactly the same output for test "
                  f"example at index {example_index}" if np.abs(y_pred_test_shrinked[0] - y_pred_test[0]) < 1e-6 else
                  f"Shurnken model's output {y_pred_test[0]} is "
                  f"different from Original model's output {y_pred_test_shrinked[0]}")
            print(f"ground_truth:{y_pred_test}, prediction:{y_pred_test_shrinked}")
            if np.argmax(y_pred_test) != np.argmax(y_pred_test_shrinked):
                misclassification +=1

            # make a vector from all weights of the shrunken network. This will be used to remove the
            all_weights = []
            for weight in weights:
                all_weights.extend(list(weight.reshape((-1,))))

            # visualize the shrunken model as QBAF.
            # for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
            test_index = example_index

            input = np.array(scaled_data)[0]
            output = np.array(data_labels)[0]
            feature_names = X_test.columns.values
            number_of_hidden_nodes = [int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]

            quantile = np.quantile(np.abs(np.array(all_weights)).reshape(1, -1), pruning_ratio)
            weight_threshold = quantile

            from plot_QBAF import visualize_attack_and_supports_QBAF, general_method_for_visualize_attack_and_supports_QBAF

            general_method_for_visualize_attack_and_supports_QBAF(input, output, shrinked_model, feature_names,
                                                                  number_of_hidden_nodes,
                                                                  weight_threshold, weights, biases, Shrinkage_percentage,
                                                                  'iris_local_graphs(shrunken_model)', 0)

            # make a vector from all weights of the original network.
            all_weights_original = []
            original_weights = []
            for layer in model.layers:
                all_weights_original.extend(list(layer.get_weights()[0].reshape((-1,))))
                original_weights.append(layer.get_weights()[0])

            # visualize the original model
            # for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
            input = np.array(scaled_data)[0]
            output = np.array(data_labels)[0]
            feature_names = X_test.columns.values

            quantile = np.quantile(np.abs(np.array(all_weights_original)).reshape(1, -1), pruning_ratio)
            weight_threshold = quantile

            from plot_QBAF import visualize_attack_and_supports_QBAF, general_clustered_visualize_attack_and_supports_QBAF

            general_clustered_visualize_attack_and_supports_QBAF(input, output, model, feature_names, HIDDEN_LAYERS,
                                                                 weight_threshold, original_weights, biases,
                                                                 Shrinkage_percentage,
                                                                 'iris_local_graphs(original_model)', 0, clustering_labels)

            #LIME explanations for the original model
            per_example_feature_attribution_distance = 0

            # how to compute fidelity. Currently the explain_instance use the label=1 which means that it only considers output node number 1 and not all
            # to change that you should consider using label=[0,1,2] (all the outputs of iris dataste) and compute the predictions of the regression model used
            # in LIME that is Ridge model from sklearn using all the outputs from all the nodes you can compare the predictions of the original model and the
            # regressor.
            # then change the last lines in explain_instance function as follows:
            # unfaithfulness = np.sum(list(ret_exp.score.values()))
            # if self.mode == "regression":
            #     ret_exp.intercept[1] = ret_exp.intercept[0]
            #     ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            #     ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
            # return ret_exp, unfaithfulness
            # also add a new way for computing scores in lime_base.py as follows:
            # new_score = np.sum(
            #     np.multiply(np.power(easy_model.predict(neighborhood_data[:, used_features]) - labels_column, 2),
            #                 weights / np.sum(weights)))
            # and return new_score in addition to prediction_score



            explanation, LIME_local_unfaithfulness = explainer.explain_instance(scaled_data[0], model.predict,
                                                     num_features=len(feature_names),
                                                     random_state=123, labels=[0,1,2])
                                                    #,alpha=preserve_percentage/100)
            print(f"LIME local unfaithfulness: {LIME_local_unfaithfulness}")
            overal_LIME_local_unfaithfulness += LIME_local_unfaithfulness

            scores = explanation.as_list()
            output = f"{test_index}:LIME: Original model feature attribution scores"
            for index, feature_name in enumerate(feature_names):
                output += ", " + str(feature_name)+f": {scores[index][1]}"
            print(output)

            # LIME explanations for the shrunken model
            explanation, _ = explainer.explain_instance(scaled_data[0], shrinked_model.predict,
                                                     num_features=len(feature_names),
                                                     random_state=123)

            scores_2 = explanation.as_list()

            output = f"{test_index}:LIME: Shrunken model feature attribution scores"
            for index, feature_name in enumerate(feature_names):
                output += ", " + str(feature_name) + f": {scores_2[index][1]}"
                per_example_feature_attribution_distance += np.abs(scores[index][1] - scores_2[index][1])

            print(output)
            print(f"Example {test_index} feature attribution distance: {per_example_feature_attribution_distance}")
            feature_attribution_distance += per_example_feature_attribution_distance

            structural_unfaithfulness = 0
            #structural_fidelity
            sigma2=sigma
            original_activations = utility_functions.compute_activations_for_each_layer(model,
                                                                                        scaled_data)
            shrunken_activations = utility_functions.compute_activations_for_each_layer(shrinked_model,
                                                                                        scaled_data)
            # distance_examples_to_target = np.exp(- (np.power(X_test.values - X_test.values[example_index], 2) / sigma2))


            #fidelity
            example_unfaithfulness = np.sum(np.multiply(np.sum(np.power(shrinked_model.predict(scaled_data) -
                                   model.predict(scaled_data), 2), axis=1),example_weights))/np.sum(example_weights)
            overal_unfaithfulness += example_unfaithfulness
            print(f"Unfaithfulness: {example_unfaithfulness}")

        measures_for_visualization[j][1] = overal_unfaithfulness
        measures_for_visualization[j][2] = overal_structural_unfaithfulness

        overal_unfaithfulness_list.append(overal_unfaithfulness / number_of_examples)
        overal_LIME_local_unfaithfulness_list.append(overal_LIME_local_unfaithfulness / number_of_examples)
        print(f"Total feature attribution distance {feature_attribution_distance / number_of_examples}")
        print(f"Total LIME feature attribution similarity {1 - (feature_attribution_distance / number_of_examples)}")
        print(f"Overal Unfaithfulness (ratio: {Shrinkage_percentage/100}): {overal_unfaithfulness / number_of_examples}")
        print(
            f"Overal LIME Unfaithfulness (ratio: {Shrinkage_percentage / 100}): {overal_LIME_local_unfaithfulness / number_of_examples}")
        print(f"Number of Misclassifications: {misclassification}")
        all_misclassifications.append(misclassification)
        feature_attribution_list.append((feature_attribution_distance / number_of_examples))

    print(measures_for_visualization)
    print(feature_attribution_list)
    #todo: uncomment for visualization
    # plt.show()
    # plt.plot(np.array(measures_for_visualization)[:, 0], np.array(measures_for_visualization)[:, 1]/np.max(np.array(measures_for_visualization)[:, 1]), "r", label='unfaithfulness')
    # plt.xlabel('sigma')
    # # plt.ylabel('unfaithfulness')
    # # plt.show()
    # plt.plot(np.array(measures_for_visualization)[:, 0], np.array(measures_for_visualization)[:, 2]/np.max(np.array(measures_for_visualization)[:, 2]), "g", label='structural unfaithfulness')
    # # plt.xlabel('sigma')
    # # plt.ylabel('structural unfaithfulness')
    # # plt.show()
    # plt.plot(np.array(measures_for_visualization)[:, 0], np.array(feature_attribution_list)/np.max(np.array(feature_attribution_list)), "b", label='feature attribution distance')
    # # plt.xlabel('sigma')
    # # plt.ylabel('feature attribution distance')
    # plt.plot(np.array(measures_for_visualization)[:, 0], np.array(all_misclassifications)/(np.max(np.array(all_misclassifications)) if np.max(np.array(all_misclassifications))!=0 else 1), "y", label='number of misclassifications')
    # plt.legend()
    # plt.show()

plt.show()
plt.plot(Shrinkage_percentage_list, overal_unfaithfulness_list, label="Our Local Explanations")
plt.plot(Shrinkage_percentage_list, overal_LIME_local_unfaithfulness_list, 'r', label="LIME local Explanations")
plt.xlabel('Compression Ratio')
plt.ylabel('Unfaithfulness')
plt.legend()
# plt.ylim([0,1])
plt.show()

print("LIME average unfaithfulness: " + str(np.mean(overal_LIME_local_unfaithfulness_list)))
print("Our average unfaithfulness: " + str(np.mean(overal_unfaithfulness_list)))