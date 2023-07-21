import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import utility_functions
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import copy
from sklearn.metrics import classification_report, pairwise_distances
from tensorflow.keras.utils import plot_model
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import gc

import time


def myprint(str, file=None):
    print(str,file=file)
    print(str)

# clusting each hidden layer based on its activation (global clustering)
def clustering_nodes(preserve_percentage, NUMBER_OF_NODES_IN_HIDDEN_LAYER, activations, sample_weights=None, clustering_algorithm="kmeans"):
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
        clustering_input = activation.T#np.multiply(activation.T,sample_weights)
        # For global clustering (using all the examples), the number of clusters uses -1 because
        # we want to have a separate cluster of zeros in the local clustering phase. This way
        # we have all the zero activations for a specific example in cluster (0).
        # Therefore, we have -1 here and we add a cluster of zeros after calling
        # the clustering_the_zero_activations_out_to_a_new_cluster_for_an_example function
        n_clusters_ = int((preserve_percentage / 100) * hidden_layer) - 1
        if clustering_algorithm == "kmeans":
            clustering = MiniBatchKMeans(n_clusters=n_clusters_, random_state=2022, batch_size=10).fit(clustering_input)
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
    input_size = X_onehot.shape[1]
    output_size = y_onehot.shape[1]
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
        for label in range(np.max(clustering_labels[index])):#range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            # if len(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label]) != 0:
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
            # else:
            #     weights[index].append(np.zeros(input_size) if index == 0 else np.zeros(
            #         int((preserve_percentage / 100) * HIDDEN_LAYERS[index - 1])))
            #     biases[index].append(0.0)
            #     outgoing_weights[index + 1].append(np.zeros((1, all_layer_sizes[index + 2])))
            #     current_weights = shrunken_model.layers[index].get_weights()[0]
            #     current_biases = shrunken_model.layers[index].get_weights()[1]
            #     current_weights[:, label] = weights[index][label]
            #     current_biases[label] = biases[index][label]
            #     new_weights = shrunken_model.get_weights()
            #     new_weights[2 * index] = current_weights
            #     new_weights[2 * index + 1] = current_biases
            #     shrunken_model.set_weights(new_weights)

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
    # input = np.array(X_onehot)[example_index]
    all_inputs = np.array(X_onehot)
    weights = []
    outgoing_weights = [[model.layers[0].get_weights()[0]]]
    biases = []
    # epsilon = 1e-30
    all_layer_sizes = [input_size]
    for hidden_layer in HIDDEN_LAYERS:
        all_layer_sizes.append(hidden_layer)
    all_layer_sizes.append(output_size)

    new_example_weights = example_weights / np.sum(example_weights)
    # new_example_weights = np.ones_like(example_weights)

    for index, hidden_layer in enumerate(HIDDEN_LAYERS):
        weights.append([])
        biases.append([])
        outgoing_weights.append([])
        layer_biases = model.layers[index].get_weights()[1]
        current_weights = shrunken_model.layers[index].get_weights()[0]
        current_biases = shrunken_model.layers[index].get_weights()[1]
        new_weights = shrunken_model.get_weights()
        for label in range(np.max(clustering_labels[index])+1): #range(0, int((preserve_percentage / 100) * HIDDEN_LAYERS[index])):
            # if len(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label]) != 0:
                weights[index].append(
                    np.mean(np.vstack(outgoing_weights[index]).T[clustering_labels[index] == label], axis=0))
                biases[index].append(np.mean(layer_biases[clustering_labels[index] == label]))
                # current_weights = shrunken_model.layers[index].get_weights()[0]
                # current_biases = shrunken_model.layers[index].get_weights()[1]
                current_weights[:, label] = weights[index][label]
                current_biases[label] = biases[index][label]


                new_weights[2 * index] = current_weights
                new_weights[2 * index + 1] = current_biases
                shrunken_model.set_weights(new_weights)

                # h_star_1 = max(np.dot(input, weights[index][label])+biases[index][label], 0)
                h_star = utility_functions.compute_activations_for_each_layer_for_certian_layer(shrunken_model, all_inputs,index)[0][:, label]
                all_hidden_activations = activations[index][:, clustering_labels[index] == label]
                # h_star += epsilon
                # distance_examples_to_target = np.exp(- (np.power(all_inputs - input, 2)/sigma))
                # one_weights = np.ones_like(h_star) / len(h_star)
                h_star = np.array([1 if h_s == 0 else h_s for h_s in list(h_star)])
                #
                activations_divided_by_h_star = np.sum(np.multiply(all_hidden_activations.T, new_example_weights / h_star).T, axis = 0)

                outgoing_weights[index + 1].append(np.dot(activations_divided_by_h_star,
                                                          model.layers[index + 1].get_weights()[0][
                                                              clustering_labels[index] == label]))

            # else:
            #     weights[index].append(np.zeros(input_size) if index == 0 else np.zeros(
            #         int((preserve_percentage / 100) * HIDDEN_LAYERS[index - 1])))
            #     biases[index].append(0.0)
            #     outgoing_weights[index + 1].append(np.zeros((1, all_layer_sizes[index + 2])))
            #     current_weights = shrunken_model.layers[index].get_weights()[0]
            #     current_biases = shrunken_model.layers[index].get_weights()[1]
            #     current_weights[:, label] = weights[index][label]
            #     current_biases[label] = biases[index][label]
            #     new_weights = shrunken_model.get_weights()
            #     new_weights[2 * index] = current_weights
            #     new_weights[2 * index + 1] = current_biases
            #     shrunken_model.set_weights(new_weights)

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


def remove_non_clusters(clustering_labels):
    new_clustering_labels = []
    new_model_dimensions = []
    for clustering_label in clustering_labels:
        real_number_of_clusters = len(np.unique(clustering_label))
        existing_clusters = np.sort(np.unique(clustering_label))
        for i in range(real_number_of_clusters):
            clustering_label[clustering_label == existing_clusters[i]] = i
        new_clustering_labels.append(clustering_label)
        new_model_dimensions.append(real_number_of_clusters)
    return new_model_dimensions, new_clustering_labels



def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


sourceFile = open('output.txt', 'w')

weighted_merging = True # todo: this means that instead of sum for the output we use weighted sum weighted based on activation values
#todo: moreover it means that we use sampling consider all the samples near that example and computed the weights based on
# all that samples in the merging step instead of computing the weights based on only one example


#todo: are the experiments for generating representation or they are for computing faithfulness?
# for_representation_or_for_faithfulness = "rep" #for representation it does not use the scaled data and uses the original sampled data
for_representation_or_for_faithfulness = "faith" #for faithfulness it uses the same scaled sampled data as LIME

if for_representation_or_for_faithfulness == "rep":
    weighted_merging = False

for num_layers in list(np.arange(5)+1):
    for num_hidden_neurons in [100, 200, 500]:
        gc.collect()
        HIDDEN_LAYERS = [num_hidden_neurons for i in range(num_layers)]
        myprint(f"Hidden Layers = {HIDDEN_LAYERS}", file=sourceFile)

        pruning_ratio = 0.5

        data = pd.read_csv("data/covtype.data", delimiter=",", header=None)

        X = data.loc[:, :53]
        y = data.iloc[:, 54]
        y = y - 1
        y = pd.get_dummies(list(y.values))

        # X = pd.DataFrame((X.values - np.min(X.values, axis=0)) / (np.max(X.values, axis=0) - np.min(X.values, axis=0)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, shuffle=True)
        X_train_original = copy.deepcopy(X_train)
        normalizer = preprocessing.StandardScaler(with_mean=False).fit(X_train)
        X_train = pd.DataFrame(normalizer.transform(X_train))
        X_test = pd.DataFrame(normalizer.transform(X_test))

        model = utility_functions.get_FFNN_model_non_binary(X_train, y_train, HIDDEN_LAYERS)

        # creating the results directory for saving the deep models and the visualization of the model structure.
        RESULT_PATH = './results'
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        model_path = os.path.join(RESULT_PATH, f'net_covtype_{num_layers}_{num_hidden_neurons}.h5')  # 'net_iris_local.h5')

        forge_gen = False
        # load model weights if previously trained and saved. If not, train the model.
        if not os.path.exists(model_path) or forge_gen:
            history = utility_functions.net_train(model, model_path, X_train, y_train, X_test, y_test, batch_size=128,
                                                  epochs=100)

            #     score = model.evaluate(X_onehot_test, y_onehot_test)
            score = model.evaluate(X_test, y_test)
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

        predictions = np.argmax(model.predict(X_test), axis=1)
        y_pred = np.eye(np.max(predictions) + 1)[predictions]
        # print(classification_report(y_ground_truth, y_pred, digits=4))
        myprint(classification_report(y_test, y_pred, digits=4), file=sourceFile)

        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
        plot_model(model, to_file=RESULT_PATH + '/model.png', show_shapes=True, show_layer_names=False)



        kernel_width = np.sqrt(X_test.shape[1]) * .1
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


            # clustering_labels_global = clustering_nodes(preserve_percentage, HIDDEN_LAYERS, activations,
            #                                      clustering_algorithm="kmeans")
            # print(clustering_labels_global)

            class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
            feature_names = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                             "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
                             "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area: Rawah",
                             "Wilderness_Area: Neota", "Wilderness_Area: Comanche Peak",
                             "Wilderness_Area: Cache la Poudr"]
            soil_types = ["Soil_type_"+str(i) for i in range(1, 41)]
            feature_names += soil_types

            from lime import lime_tabular
            explainer = lime_tabular.LimeTabularExplainer(X_train.values, mode="classification",
                                                              class_names = class_names,
                                                              feature_names = feature_names,
                                                              random_state = 123)
            # global clustering
            measures_for_visualization = []
            number_of_tests = 1
            feature_attribution_list = []
            # sigmas = [10,5,2,1.5,1,0.8,0.6,0.4,0.2,0.1]
            sigmas = [2]
            all_misclassifications = []



            for j in range(0, number_of_tests):
                sigma = sigmas[j]
                feature_attribution_distance = 0
                overal_unfaithfulness = 0
                overal_LIME_local_unfaithfulness = 0
                overal_structural_unfaithfulness = 0
                measures_for_visualization.append(np.zeros(3))
                measures_for_visualization[j][0] = sigma
                number_of_examples = 20
                misclassification = 0
                # var = np.mean(np.var(X_test.values[0:number_of_examples], axis=0))
                for example_index in range(0, number_of_examples):  # len(X_onehot_test)):

                    truncated_model_dimensions = [int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]



                    # Generate samples around an example using lime_tablurar __data_inverse function
                    # todo: change __data_inverse in lime_tablurar.py to data_inverse to be able to use the function outside lime_tablurar.py
                    data, inverse = explainer.data_inverse(X_test.values[example_index], 10)
                    scaled_data = (data - explainer.scaler.mean_) / explainer.scaler.scale_
                    data_labels = model.predict(inverse)

                    if for_representation_or_for_faithfulness=="rep":
                        #todo: I changed scaled_data to this to have the correct predictions
                        scaled_data = inverse
                    # np.random.seed(2022)
                    # random_indices = np.random.randint(X_test.shape[0], size=100,)
                    # scaled_data = X_test.values[random_indices, :]
                    # data_labels = y_test.values[random_indices]
                    # scaled_data[0,:] = X_test.values[example_index,:]
                    # data_labels[0] = y_test.values[example_index]
                    # find wight of each example with respect to a target example
                    distance_examples_to_target = pairwise_distances(
                        scaled_data,
                        scaled_data[0].reshape(1, -1),
                        metric='euclidean'
                    ).ravel()
                    example_weights = kernel_fn(distance_examples_to_target)
                    # labels_column = data_labels[:, np.argmax(y_test.iloc[example_index,:].values)]
                    # used_features = explainer.base.feature_selection(scaled_data,
                    #                                                  labels_column,
                    #                                                  example_weights,
                    #                                                  X_train.shape[1],
                    #                                                  'auto')
                    # if for_representation_or_for_faithfulness == "faith":
                    #     #todo: this was the original line
                    #     scaled_data = scaled_data[:, used_features]


                    activations = utility_functions.compute_activations_for_each_layer(model, scaled_data)
                    clustering_activations = activations#utility_functions.compute_activations_for_each_layer(model, scaled_data)
                    start_time = time.time()
                    clustering_labels_with_no_zero = clustering_nodes(preserve_percentage, HIDDEN_LAYERS, clustering_activations,
                                                                      sample_weights=example_weights,
                                                                      clustering_algorithm="kmeans")
                    # print(clustering_labels_with_no_zero)

                    clustering_labels = clustering_the_zero_activations_out_to_a_new_cluster_for_an_example(
                        clustering_activations, HIDDEN_LAYERS,
                        0, clustering_labels_with_no_zero)

                    truncated_model_dimensions, clustering_labels = remove_non_clusters(clustering_labels)

                    end_time_clustering = time.time()
                    myprint(f"Clustering time: {end_time_clustering - start_time} for example: {example_index}", file=sourceFile)

                    # cunstruct the structure of the truncated model using the same function for building a FFNN model.
                    shrinked_model = utility_functions.get_FFNN_model_non_binary(X_test, y_test, truncated_model_dimensions)

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
                    end_time = time.time()
                    myprint(f"SpArX runtime {end_time - start_time} for example: {example_index}", file=sourceFile)
                    truncated_weights = []
                    for index, weight in enumerate(weights):
                        truncated_weights.append(weight)
                        truncated_weights.append(biases[index])



                    # setting weights of the truncated model
                    shrinked_model.set_weights(truncated_weights)

                    y_pred_test_shrinked = shrinked_model.predict(
                        np.array(scaled_data[0]).reshape((1, -1))).flatten()
                    y_pred_test = model.predict(np.array(scaled_data[0]).reshape((1, -1))).flatten()

                    myprint(f"Both Shrunken model and the Original model produce exactly the same output for test "
                          f"example at index {example_index}" if np.sum(np.abs(y_pred_test_shrinked - y_pred_test)) < 1e-6 else
                          f"Shurnken model's output {y_pred_test} is "
                          f"different from Original model's output {y_pred_test_shrinked}", file=sourceFile)
                    # print(f"ground_truth:{y_pred_test}, prediction:{y_pred_test_shrinked}")
                    if np.argmax(y_pred_test) != np.argmax(y_pred_test_shrinked):
                        misclassification +=1

                    # make a vector from all weights of the shrunken network. This will be used to remove the
                    all_weights = []
                    for weight in weights:
                        all_weights.extend(list(weight.reshape((-1,))))

                    # visualize the shrunken model as QBAF.
                    # for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
                    test_index = example_index

                    if for_representation_or_for_faithfulness == "rep":
                        input = np.array(scaled_data)[0]
                        output = np.array(data_labels)[0]
                        # feature_names = X_test.columns.values
                        number_of_hidden_nodes = truncated_model_dimensions#[int((preserve_percentage / 100) * hidden_layer) for hidden_layer in HIDDEN_LAYERS]

                        quantile = np.quantile(np.abs(np.array(all_weights)).reshape(1, -1), pruning_ratio)
                        weight_threshold = quantile

                        from plot_QBAF import visualize_attack_and_supports_QBAF, general_method_for_visualize_attack_and_supports_QBAF


                        general_method_for_visualize_attack_and_supports_QBAF(input, output, shrinked_model, feature_names,
                                                                              number_of_hidden_nodes,
                                                                              weight_threshold, weights, biases, Shrinkage_percentage,
                                                                              'covtype_local_graphs(shrunken_model)', example_index)

                    # make a vector from all weights of the original network.
                    all_weights_original = []
                    original_weights = []
                    for layer in model.layers:
                        all_weights_original.extend(list(layer.get_weights()[0].reshape((-1,))))
                        original_weights.append(layer.get_weights()[0])

                    # visualize the original model
                    # for test_index in range(0, 20):  # range(len(np.array(X_onehot_test))):
                    if for_representation_or_for_faithfulness == "rep":
                        input = np.array(scaled_data)[0]
                        output = np.array(data_labels)[0]
                        # feature_names = X_test.columns.values

                        quantile = np.quantile(np.abs(np.array(all_weights_original)).reshape(1, -1), pruning_ratio)
                        weight_threshold = quantile

                        from plot_QBAF import visualize_attack_and_supports_QBAF, general_clustered_visualize_attack_and_supports_QBAF

                        general_clustered_visualize_attack_and_supports_QBAF(input, output, model, feature_names, HIDDEN_LAYERS,
                                                                             weight_threshold, original_weights, biases,
                                                                             Shrinkage_percentage,
                                                                             'covtype_local_graphs(original_model)', example_index, clustering_labels)

                    #LIME explanations for the original model
                    per_example_feature_attribution_distance = 0

                    #todo: how to compute fidelity. Currently the explain_instance use the label=1 which means that it only considers output node number 1 and not all
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

                    # _
                    # to have compression also in LIME we set the alpha in the Ridge regression function as
                    # the preserve_percentage/100 and add alpha=1 argument to explain_instance and explain_instance_with_data
                    # functions

                    if for_representation_or_for_faithfulness == "faith":
                        start_time = time.time()
                        explanation, LIME_local_unfaithfulness = explainer.explain_instance(X_test.values[example_index], model.predict,
                                                                 num_features=len(feature_names), num_samples=100,
                                                                 random_state=123, labels=list(np.arange(y_test.shape[1])))

                        end_time = time.time()
                        myprint(f"LIME runtime {end_time - start_time} for example: {example_index}", file=sourceFile)

                        #,alpha=preserve_percentage/100)
                        myprint(f"LIME local unfaithfulness: {LIME_local_unfaithfulness}", file=sourceFile)
                        overal_LIME_local_unfaithfulness += LIME_local_unfaithfulness

                        scores = explanation.as_list()
                        output = f"{test_index}:LIME: Original model feature attribution scores"
                        for index, feature_name in enumerate(feature_names):
                            output += ", " + str(feature_name)+f": {scores[index][1]}"
                        # print(output)

                        # LIME explanations for the shrunken model
                        explanation, _ = explainer.explain_instance(X_test.values[example_index], shrinked_model.predict,
                                                                 num_features=len(feature_names),num_samples=100,
                                                                 random_state=123, labels=list(np.arange(y_test.shape[1])))

                        scores_2 = explanation.as_list()

                        output = f"{test_index}:LIME: Shrunken model feature attribution scores"
                        for index, feature_name in enumerate(feature_names):
                            output += ", " + str(feature_name) + f": {scores_2[index][1]}"
                            per_example_feature_attribution_distance += np.abs(scores[index][1] - scores_2[index][1])

                        # print(output)
                        # print(f"Example {test_index} feature attribution distance: {per_example_feature_attribution_distance}")
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
                                               model.predict(scaled_data), 2), axis=1), example_weights/np.sum(example_weights)))
                        overal_unfaithfulness += example_unfaithfulness
                        myprint(f"Unfaithfulness: {example_unfaithfulness}", file=sourceFile)

                if for_representation_or_for_faithfulness == "faith":
                    measures_for_visualization[j][1] = overal_unfaithfulness
                    measures_for_visualization[j][2] = overal_structural_unfaithfulness

                    overal_unfaithfulness_list.append(overal_unfaithfulness / number_of_examples)
                    overal_LIME_local_unfaithfulness_list.append(overal_LIME_local_unfaithfulness / number_of_examples)
                    print(f"Total feature attribution distance {feature_attribution_distance / number_of_examples}")
                    print(f"Total LIME feature attribution similarity {1 - (feature_attribution_distance / number_of_examples)}")
                    myprint(f"Overal Unfaithfulness (ratio: {Shrinkage_percentage/100}): {overal_unfaithfulness / number_of_examples}", file=sourceFile)
                    myprint(
                        f"Overal LIME Unfaithfulness (ratio: {Shrinkage_percentage / 100}): {overal_LIME_local_unfaithfulness / number_of_examples}", file=sourceFile)
                myprint(f"Number of Misclassifications: {misclassification}", file=sourceFile)
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




        #todo: undomment it for visualization
        # if for_representation_or_for_faithfulness == "faith":
        #     # global_unfaithfulness_list = [0.22, 0.22, 0.23, 0.28]
        #     plt.show()
        #     plt.plot(Shrinkage_percentage_list, overal_unfaithfulness_list, label="Our Local Explanations")
        #     plt.plot(Shrinkage_percentage_list, overal_LIME_local_unfaithfulness_list, 'r', label="LIME local Explanations")
        #     plt.xlabel('Compression Ratio')
        #     plt.ylabel('Unfaithfulness')
        #     plt.legend()
        #     # plt.ylim([0,1])
        #     plt.show()
        #
        #     print("LIME average unfaithfulness: " + str(np.mean(overal_LIME_local_unfaithfulness_list)))
        #     print("Our average unfaithfulness: " + str(np.mean(overal_unfaithfulness_list)))

sourceFile.close()