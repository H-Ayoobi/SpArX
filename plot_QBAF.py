from tensorflow.python.keras import backend as K
def compute_activations_for_each_layer(model, input_data):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functor = K.function([inp], outputs)  # evaluation functions

    # computing activations
    activations = functor(input_data.reshape((1, -1)))
    return activations



def visualize_attack_and_supports_QBAF(input, output, feature_names, number_of_hidden_nodes, edge_weight_threshold,
                                       input_to_hidden_weights, hidden_to_output_weights, shrink_percentage,
                                       hidden_activation, output_activation, path, fig_index, hidden_bias='',
                                       output_bias=''):
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()

    # Adding nodes
    for i in range(len(input)):
        if input[i] != 0 and (max(np.array(input_to_hidden_weights)[:, i]) > edge_weight_threshold or min(
                np.array(input_to_hidden_weights)[:, i]) < - edge_weight_threshold):
            dot.node('I' + str(i), str(feature_names[i]))
    dot.attr('node', shape='circle')
    for i in range(number_of_hidden_nodes):
        if (max(np.array(input_to_hidden_weights)[i, :] * input) > edge_weight_threshold or min(
                np.array(input_to_hidden_weights)[i, :] * input) < - edge_weight_threshold) and (
                np.abs(np.sum(np.array(hidden_to_output_weights[i]))) > edge_weight_threshold):
            dot.node('H' + str(i), color='green' if np.sum(hidden_to_output_weights[i]) > 0 else 'red',
                     width=str(np.abs(np.sum(hidden_to_output_weights[i]) * hidden_activation[i] * 1)))
    for i in range(len(output)):
        dot.node('O' + str(i), color='green' if output_activation[0] > 0.5 else 'red',
                 width=str(np.abs(0.5 - output_activation[0]) * 5))
    dot.node('Explanations', label=('Correct Prediction\\n----------------------------------\\n' if (
            (output_activation[0] > 0.5 and output == 1) or (
            output_activation[
                0] <= 0.5 != output)) else 'Wrong Prediction\\n----------------------------------\\n') + (
                                       'Ground Truth: Yes ' if output == 1 else 'Ground Truth: No ') + (
                                       '\\nPrediction: Yes \\nOutput activation value: '
                                       if output_activation[
                                              0] > 0.5 else '\\nPrediction: No \\nOutput activation value: ')
                                   + '{activation:.2f}'.format(
        activation=output_activation[0]), shape='note',
             color='green' if ((output_activation[0] > 0.5 and output == 1) or (
                     output_activation[0] <= 0.5 and output == 0)) else 'red')
    # Adding edges
    # input to hidden edges
    for i in range(len(input)):
        for j in range(number_of_hidden_nodes):
            if input[i] != 0:
                if input_to_hidden_weights[j][i] * input[i] > edge_weight_threshold and np.abs(np.sum(
                        hidden_to_output_weights[j])) > edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='green',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * 10))  # , label=str(input_to_hidden_weights[j][i]))
                if input_to_hidden_weights[j][i] * input[i] < - edge_weight_threshold and np.abs(np.sum(
                        hidden_to_output_weights[j])) > edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='red',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * 10))  # , label=str(input_to_hidden_weights[j][i]))

    # hidden to output edges
    for i in range(number_of_hidden_nodes):
        for j in range(len(output)):
            if np.sum(hidden_to_output_weights[i]) > edge_weight_threshold and (
                    max(np.array(input_to_hidden_weights)[i, :] * input) > edge_weight_threshold or min(
                np.array(input_to_hidden_weights)[i, :] * input) < - edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='green',
                         label='{weight_activation:0.2f}'.format(
                             weight_activation=(np.sum(hidden_to_output_weights[i]))),
                         penwidth=str(
                             np.abs(
                                 np.sum(hidden_to_output_weights[i]) * 2)))  # , label=str(hidden_to_output_weights[i]))
            if np.sum(hidden_to_output_weights[i]) < - edge_weight_threshold and (
                    max(np.array(input_to_hidden_weights)[i, :] * input) > edge_weight_threshold or min(
                np.array(input_to_hidden_weights)[i, :] * input) < - edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='red',
                         label='{weight_activation:0.2f}'.format(
                             weight_activation=(np.sum(hidden_to_output_weights[i]))),  # ,
                         penwidth=str(
                             np.abs(
                                 np.sum(hidden_to_output_weights[i]) * 2)))  # , label=str(hidden_to_output_weights[i]))
    for i in range(len(output)):
        dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)


def general_visualize_attack_and_supports_QBAF(input, output, feature_names, number_of_hidden_nodes,
                                               edge_weight_threshold,
                                               input_to_hidden_weights, hidden_to_output_weights, shrink_percentage,
                                               hidden_activation, output_activation, path, fig_index, hidden_bias='',
                                               output_bias=''):
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()

    # Adding nodes
    for i in range(len(input)):
        if input[i] != 0 and max(np.abs(np.array(input_to_hidden_weights)[:, i])) > edge_weight_threshold:
            dot.node('I' + str(i), str(feature_names[i]))
    dot.attr('node', shape='circle')
    for i in range(number_of_hidden_nodes):
        if (max(np.abs(np.array(input_to_hidden_weights)[i, :])) > edge_weight_threshold):
            dot.node('H' + str(i), color='green' if np.sum(hidden_to_output_weights[i]) > 0 else 'red',
                     width=str(1))
    output_activation_one_hot = np.zeros_like(np.array(output_activation), dtype=int)
    output_activation_one_hot[output_activation.argmax()] = 1
    for i in range(len(output)):
        dot.node('O' + str(i), color='green' if list(output_activation_one_hot) == list(output) else 'red',
                 width=str(np.abs(output_activation[i]) * 2))
    dot.node('Explanations', label=('Correct Prediction\\n----------------------------------\\n' if
                                    (list(output_activation_one_hot) == list(
                                        output)) else 'Wrong Prediction\\n----------------------------------\\n') +
                                   'Ground Truth: ' + str(np.argmax(output) + 1) +
                                   '\\nPrediction:' + str(
        np.argmax(output_activation_one_hot) + 1) + ' \\n Output activation value: '
                                   + f'activation:.{output_activation}', shape='note',
             color='green' if (list(output_activation_one_hot) == list(output)) else 'red')
    # Adding edges
    # input to hidden edges
    for i in range(len(input)):
        for j in range(number_of_hidden_nodes):
            if input[i] != 0:
                if input_to_hidden_weights[j][i] > edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='green',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * input[
                                              i] * 2))  # , label=str(input_to_hidden_weights[j][i]))
                if input_to_hidden_weights[j][i] < - edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='red',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * input[
                                              i] * 2))  # , label=str(input_to_hidden_weights[j][i]))

    # hidden to output edges
    for i in range(number_of_hidden_nodes):
        for j in range(len(output)):
            if hidden_to_output_weights[i][j] > 0 and (
                    max(np.abs(np.array(input_to_hidden_weights)[i, :])) > edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='green',
                         # label='{weight_activation:0.2f}'.format(weight_activation=(hidden_to_output_weights[i][j])),
                         penwidth=str(np.abs(hidden_to_output_weights[i][j] * hidden_activation[i] * 2)))
                # , label=str(hidden_to_output_weights[i]))
            if hidden_to_output_weights[i][j] < 0 and (
                    max(np.abs(np.array(input_to_hidden_weights)[i, :])) > edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='red',
                         # label='{weight_activation:0.2f}'.format(weight_activation=(hidden_to_output_weights[i][j])),  # ,
                         penwidth=str(np.abs(hidden_to_output_weights[i][j] * hidden_activation[i] * 2)))
                # , label=str(hidden_to_output_weights[i]))
    for i in range(len(output)):
        dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)

def general_method_for_visualize_attack_and_supports_QBAF(input, output, model, feature_names, number_of_hidden_nodes,
                                                          edge_weight_threshold, weights, biases, shrink_percentage,
                                                          path, fig_index):

    activation = compute_activations_for_each_layer(model, input)
    output_activation_in_correct_format = list(map(int, activation[-1][0] > 0.5))
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()


    existing_nodes = []
    # Adding nodes
    for layer_index in range(len(number_of_hidden_nodes) + 2):  # +2 input and output layers
        existing_nodes.append([])
        if layer_index == 0:
            for i in range(len(input)):
                if input[i] !=0:
                    if  max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold:
                        dot.node('I' + str(i), str(feature_names[i]))
                        existing_nodes[layer_index].append(i)
        elif layer_index == len(number_of_hidden_nodes) + 1:
            for i in range(len(output)):
                dot.node('O' + str(i), f'<O<SUB>{str(i)}</SUB>>', #color='green' if activation[-1] == list(output) else 'red',
                         width=str(np.abs(activation[-1].reshape((-1,))[i]) * 2))
                existing_nodes[layer_index].append(i)
        else:
            dot.attr('node', shape='circle')
            for i in range(number_of_hidden_nodes[layer_index - 1]):
                if max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold and \
                        np.max(np.abs(np.array(weights[layer_index - 1])[existing_nodes[layer_index - 1], i])) >= edge_weight_threshold:
                    dot.node('H' + str(layer_index) + "." + str(i), f"<C<SUB>{str(i +1)}</SUB>>")#<SUP>({str(layer_index)})</SUP>>")#, #color='green' if np.sum(weights[layer_index][i]) > 0 else 'red',
                             #width=str(np.abs(activation[layer_index-1].reshape((-1,))[i])))
                    existing_nodes[layer_index].append(i)

    # dot.node('Explanations', label='Ground Truth: ' + str(output) +
    #                                '\\nPrediction:' + str(output_activation_in_correct_format) + ' \\n Output activation value: '
    #                                + f'activation:.{activation[-1][0]}', shape='note')
    omitted_edges = []
    for layer_index in range(len(number_of_hidden_nodes) + 1):
            omitted_edges.append([])
            if layer_index == 0:
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold:
                            dot.edge('I' + str(i), 'H' + str(layer_index + 1)+"." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs( weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            elif layer_index == len(number_of_hidden_nodes):
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) > edge_weight_threshold:
                            dot.edge('H'+str(layer_index)+"." + str(i), 'O' + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            else:
                dot.attr('node', shape='circle')
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) >= edge_weight_threshold:
                            dot.edge('H' + str(layer_index)+"."  + str(i), 'H' + str(layer_index + 1)+"." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')

    # for i in range(len(output)):
    #     dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    nodes_to_be_omitted = []
    #remove unneccessary nodes
    for layer_index in range(len(number_of_hidden_nodes)):
        current_nodes_dictionary = {}
        for index, node in enumerate(existing_nodes[layer_index]):
            current_nodes_dictionary[node] = index
        current_layer_nodes = [[] for i in range(len(existing_nodes[layer_index]))]
        for omitted_edge in omitted_edges[layer_index]:
            current_layer_nodes[current_nodes_dictionary[int(omitted_edge.split('-')[0])]].append(int(omitted_edge.split('-')[1]))
        for index, nodes in enumerate(current_layer_nodes):
            if nodes == existing_nodes[layer_index + 1]:
                if layer_index == 0:
                    nodes_to_be_omitted.append(f'I{existing_nodes[layer_index][index]}')
                else:
                    nodes_to_be_omitted.append(f'H{layer_index}.{existing_nodes[layer_index][index]}')

    for node in nodes_to_be_omitted:
        for body_node in dot.body:
            if node in body_node:
                dot.body.remove(body_node)
    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)


def general_local_method_for_visualize_attack_and_supports_QBAF(input, output, model, feature_names, number_of_hidden_nodes,
                                                          edge_weight_threshold, weights, biases, shrink_percentage,
                                                          path, fig_index):

    activation = compute_activations_for_each_layer(model, input)
    output_activation_in_correct_format = list(map(int, activation[-1][0] > 0.5))
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()


    existing_nodes = []
    # Adding nodes
    for layer_index in range(len(number_of_hidden_nodes) + 2):  # +2 input and output layers
        existing_nodes.append([])
        if layer_index == 0:
            for i in range(len(input)):
                if input[i] !=0:
                    if  max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold:
                        dot.node('I' + str(i), str(feature_names[i]))
                        existing_nodes[layer_index].append(i)
        elif layer_index == len(number_of_hidden_nodes) + 1:
            for i in range(len(output)):
                dot.node('O' + str(i), f'<O<SUB>{str(i)}</SUB>>', #color='green' if activation[-1] == list(output) else 'red',
                         width=str(np.abs(activation[-1].reshape((-1,))[i]) * 2))
                existing_nodes[layer_index].append(i)
        else:
            dot.attr('node', shape='circle')
            for i in range(number_of_hidden_nodes[layer_index - 1]):
                if max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold and \
                        np.max(np.abs(np.array(weights[layer_index - 1])[existing_nodes[layer_index - 1], i])) >= edge_weight_threshold:
                    dot.node('H' + str(layer_index) + "." + str(i), f"<C<SUB>{str(i +1)}</SUB>>")#<SUP>({str(layer_index)})</SUP>>")#, #color='green' if np.sum(weights[layer_index][i]) > 0 else 'red',
                             #width=str(np.abs(activation[layer_index-1].reshape((-1,))[i])))
                    existing_nodes[layer_index].append(i)

    # dot.node('Explanations', label='Ground Truth: ' + str(output) +
    #                                '\\nPrediction:' + str(output_activation_in_correct_format) + ' \\n Output activation value: '
    #                                + f'activation:.{activation[-1][0]}', shape='note')
    omitted_edges = []
    for layer_index in range(len(number_of_hidden_nodes) + 1):
            omitted_edges.append([])
            if layer_index == 0:
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold:
                            dot.edge('I' + str(i), 'H' + str(layer_index + 1)+"." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs( weights[layer_index][i, j]) * 10))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            elif layer_index == len(number_of_hidden_nodes):
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) > edge_weight_threshold:
                            dot.edge('H'+str(layer_index)+"." + str(i), 'O' + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * activation[layer_index-1][j,i] * 10))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            else:
                dot.attr('node', shape='circle')
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) >= edge_weight_threshold:
                            dot.edge('H' + str(layer_index)+"."  + str(i), 'H' + str(layer_index + 1)+"." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * activation[layer_index-1][j,i] * 10))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')

    # for i in range(len(output)):
    #     dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    nodes_to_be_omitted = []
    #remove unneccessary nodes
    for layer_index in range(len(number_of_hidden_nodes)):
        current_nodes_dictionary = {}
        for index, node in enumerate(existing_nodes[layer_index]):
            current_nodes_dictionary[node] = index
        current_layer_nodes = [[] for i in range(len(existing_nodes[layer_index]))]
        for omitted_edge in omitted_edges[layer_index]:
            current_layer_nodes[current_nodes_dictionary[int(omitted_edge.split('-')[0])]].append(int(omitted_edge.split('-')[1]))
        for index, nodes in enumerate(current_layer_nodes):
            if nodes == existing_nodes[layer_index + 1]:
                if layer_index == 0:
                    nodes_to_be_omitted.append(f'I{existing_nodes[layer_index][index]}')
                else:
                    nodes_to_be_omitted.append(f'H{layer_index}.{existing_nodes[layer_index][index]}')

    for node in nodes_to_be_omitted:
        for body_node in dot.body:
            if node in body_node:
                dot.body.remove(body_node)
    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)




def clustered_visualize_attack_and_supports_QBAF(input, output, feature_names, number_of_original_hidden_nodes,
                                                 number_of_clusters, edge_weight_threshold,
                                                 input_to_hidden_weights, hidden_to_output_weights, shrink_percentage,
                                                 hidden_activation, output_activation, path, fig_index, cluster_labels,
                                                 hidden_bias='',
                                                 output_bias=''):
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()

    # Adding nodes
    for i in range(len(input)):
        if input[i] != 0 and (max(np.abs(np.array(input_to_hidden_weights)[:, i])) >= edge_weight_threshold):
            dot.node('I' + str(i), str(feature_names[i]))
    dot.attr('node', shape='circle')

    for i in range(number_of_original_hidden_nodes):
        with dot.subgraph(name=f'cluster_{cluster_labels[i]}') as c:
            c.attr(label=f'H{cluster_labels[i]}')
            if (max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
                c.node('H' + str(i), color='green' if np.sum(hidden_to_output_weights[i]) > 0 else 'red',
                       width=str(np.abs(np.sum(hidden_to_output_weights[i]) * 1)))
    output_activation_one_hot = np.zeros_like(np.array(output_activation), dtype=int)
    output_activation_one_hot[output_activation.argmax()] = 1
    for i in range(len(output)):
        dot.node('O' + str(i),
                 color='green' if list(output_activation_one_hot) == list(output) else 'red',
                 width=str(np.abs(output_activation[i]) * 2))
    dot.node('Explanations', label=('Correct Prediction\\n----------------------------------\\n' if
                                    (list(output_activation_one_hot) == list(
                                        output)) else 'Wrong Prediction\\n----------------------------------\\n') +
                                   'Ground Truth: ' + str(np.argmax(output) + 1) +
                                   '\\nPrediction:' + str(
        np.argmax(output_activation_one_hot) + 1) + ' \\n Output activation value: '
                                   + f'activation:.{output_activation}', shape='note',
             color='green' if (list(output_activation_one_hot) == list(output)) else 'red')
    # Adding edges
    # input to hidden edges
    for i in range(len(input)):
        for j in range(number_of_original_hidden_nodes):
            if input[i] != 0:
                if input_to_hidden_weights[j][i] >= edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='green',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * input[
                                              i] * 2))  # , label=str(input_to_hidden_weights[j][i]))
                if input_to_hidden_weights[j][i] < - edge_weight_threshold:
                    dot.edge('I' + str(i), 'H' + str(j), color='red',
                             penwidth=str(np.abs(
                                 input_to_hidden_weights[j][i]) * input[
                                              i] * 2))  # , label=str(input_to_hidden_weights[j][i]))

    # hidden to output edges
    for i in range(number_of_original_hidden_nodes):
        for j in range(len(output)):
            if hidden_to_output_weights[i][j] > 0 and (
                    max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='green',
                         # label='{weight_activation:0.2f}'.format(
                         #     weight_activation=(np.sum(hidden_to_output_weights[i]))),
                         penwidth=str(
                             np.abs(hidden_to_output_weights[i][j]) * hidden_activation[
                                 i] * 5))  # , label=str(hidden_to_output_weights[i]))
            if hidden_to_output_weights[i][j] < 0 and (
                    max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
                dot.edge('H' + str(i), 'O' + str(j), color='red',
                         # label='{weight_activation:0.2f}'.format(
                         #     weight_activation=(np.sum(hidden_to_output_weights[i]))),  # ,
                         penwidth=str(
                             np.abs(hidden_to_output_weights[i][j]) * hidden_activation[
                                 i] * 5))  # , label=str(hidden_to_output_weights[i]))
    for i in range(len(output)):
        dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)


def general_clustered_visualize_attack_and_supports_QBAF(input, output, model, feature_names, number_of_hidden_nodes,
                                                          edge_weight_threshold, weights, biases, shrink_percentage,
                                                          path, fig_index, cluster_labels):

    activation = compute_activations_for_each_layer(model, input)
    output_activation_in_correct_format = list(map(int, activation[-1][0] > 0.5))
    # import module
    from graphviz import Digraph
    import numpy as np
    import matplotlib.pyplot as plt

    # instantiating object
    dot = Digraph(comment='A Round Graph')
    dot.attr(rankdir="LR")
    dot.attr(splines='line')
    dot.subgraph()


    existing_nodes = []
    # Adding nodes
    for layer_index in range(len(number_of_hidden_nodes) + 2):  # +2 input and output layers
        existing_nodes.append([])
        if layer_index == 0:
            for i in range(len(input)):
                if input[i] != 0:
                    if  max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold:
                        dot.node('I' + str(i), str(feature_names[i]))
                        existing_nodes[layer_index].append(i)
        elif layer_index == len(number_of_hidden_nodes) + 1:
            for i in range(len(output)):
                dot.node('O' + str(i), f"<O<SUB>{str(i)}</SUB>>", #color='green' if activation[-1] == list(output) else 'red',
                         width=str(np.abs(activation[-1].reshape((-1,))[i]) * 2))
                existing_nodes[layer_index].append(i)
        else:
            dot.attr('node', shape='circle')
            for i in range(number_of_hidden_nodes[layer_index - 1]):
                with dot.subgraph(name=f'cluster_{layer_index}_{cluster_labels[layer_index-1][i]}') as c:
                    c.attr(label=f'<<B>C<SUB>{cluster_labels[layer_index-1][i]+1}</SUB><SUP>({str(layer_index)})</SUP></B>>')
                    if max(np.abs(np.array(weights[layer_index])[i, :])) >= edge_weight_threshold and \
                            np.max(np.abs(np.array(weights[layer_index - 1])[existing_nodes[layer_index - 1], i])) >= edge_weight_threshold:
                        c.node('v' + str(layer_index) + "." + str(i), f"<v<SUB>{str(i)}</SUB><SUP>({str(layer_index)})</SUP>>") #color='green' if np.sum(weights[layer_index][i]) > 0 else 'red',
                                 #,width=str(np.abs(activation[layer_index-1].reshape((-1,))[i])))
                        existing_nodes[layer_index].append(i)

    # dot.node('Explanations', label='Ground Truth: ' + str(output) +
    #                                '\\nPrediction:' + str(output_activation_in_correct_format) + ' \\n Output activation value: '
    #                                + f'activation:.{activation[-1][0]}', shape='note')
    omitted_edges = []
    for layer_index in range(len(number_of_hidden_nodes) + 1):
            omitted_edges.append([])
            if layer_index == 0:
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold:
                            dot.edge('I' + str(i), 'v' + str(layer_index+1) + "." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            elif layer_index == len(number_of_hidden_nodes):
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) >= edge_weight_threshold:
                            dot.edge('v' + str(layer_index) + "." + str(i), 'O' + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')
            else:
                dot.attr('node', shape='circle')
                for i in existing_nodes[layer_index]:
                    for j in existing_nodes[layer_index + 1]:
                        if np.abs(weights[layer_index][i, j]) >= edge_weight_threshold and \
                                np.max(np.abs(weights[layer_index][existing_nodes[layer_index], j])) >= edge_weight_threshold:
                            dot.edge('v' + str(layer_index) + "." + str(i), 'v' + str(layer_index+1) + "." + str(j),
                                     color='green' if weights[layer_index][i, j] >= 0 else 'red',
                                     penwidth=str(np.abs(weights[layer_index][i, j]) * 2))
                        else:
                            omitted_edges[layer_index].append(f'{i}-{j}')

    # for i in range(len(output)):
    #     dot.edge('O' + str(i), 'Explanations', penwidth=str(0))

    nodes_to_be_omitted = []
    #remove unneccessary nodes
    for layer_index in range(len(number_of_hidden_nodes)):
        current_nodes_dictionary = {}
        for index, node in enumerate(existing_nodes[layer_index]):
            current_nodes_dictionary[node] = index
        current_layer_nodes = [[] for i in range(len(existing_nodes[layer_index]))]
        for omitted_edge in omitted_edges[layer_index]:
            current_layer_nodes[current_nodes_dictionary[int(omitted_edge.split('-')[0])]].append(int(omitted_edge.split('-')[1]))
        for index, nodes in enumerate(current_layer_nodes):
            if nodes == existing_nodes[layer_index + 1]:
                if layer_index == 0:
                    nodes_to_be_omitted.append(f'I{existing_nodes[layer_index][index]}')
                else:
                    nodes_to_be_omitted.append(f'<<B>C<SUB>{existing_nodes[layer_index][index]}</SUB><SUP>({str(layer_index)})</SUP></B>>')

    for node in nodes_to_be_omitted:
        for body_node in dot.body:
            if node in body_node:
                dot.body.remove(body_node)
    # saving source code
    dot.format = 'png'
    dot.render(path + '/Graph_' + str(fig_index))
    # dot.view()
    # plt.show()
    # print(dot.source)



    #-------------------------------------------------------------

    #
    # # import module
    # from graphviz import Digraph
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # instantiating object
    # dot = Digraph(comment='A Round Graph')
    # dot.attr(rankdir="LR")
    # dot.attr(splines='line')
    # dot.subgraph()
    #
    # # Adding nodes
    # for i in range(len(input)):
    #     if input[i] != 0 and (max(np.abs(np.array(input_to_hidden_weights)[:, i])) >= edge_weight_threshold):
    #         dot.node('I' + str(i), str(feature_names[i]))
    # dot.attr('node', shape='circle')
    #
    # for i in range(number_of_original_hidden_nodes):
    #     with dot.subgraph(name=f'cluster_{cluster_labels[i]}') as c:
    #         c.attr(label=f'H{cluster_labels[i]}')
    #         if (max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
    #             c.node('H' + str(i), color='green' if np.sum(hidden_to_output_weights[i]) > 0 else 'red',
    #                    width=str(np.abs(np.sum(hidden_to_output_weights[i]) * 1)))
    # output_activation_one_hot = np.zeros_like(np.array(output_activation), dtype=int)
    # output_activation_one_hot[output_activation.argmax()] = 1
    # for i in range(len(output)):
    #     dot.node('O' + str(i),
    #              color='green' if list(output_activation_one_hot) == list(output) else 'red',
    #              width=str(np.abs(output_activation[i]) * 2))
    # dot.node('Explanations', label=('Correct Prediction\\n----------------------------------\\n' if
    #                                 (list(output_activation_one_hot) == list(
    #                                     output)) else 'Wrong Prediction\\n----------------------------------\\n') +
    #                                'Ground Truth: ' + str(np.argmax(output) + 1) +
    #                                '\\nPrediction:' + str(
    #     np.argmax(output_activation_one_hot) + 1) + ' \\n Output activation value: '
    #                                + f'activation:.{output_activation}', shape='note',
    #          color='green' if (list(output_activation_one_hot) == list(output)) else 'red')
    # # Adding edges
    # # input to hidden edges
    # for i in range(len(input)):
    #     for j in range(number_of_original_hidden_nodes):
    #         if input[i] != 0:
    #             if input_to_hidden_weights[j][i] >= edge_weight_threshold:
    #                 dot.edge('I' + str(i), 'H' + str(j), color='green',
    #                          penwidth=str(np.abs(
    #                              input_to_hidden_weights[j][i]) * input[
    #                                           i] * 2))  # , label=str(input_to_hidden_weights[j][i]))
    #             if input_to_hidden_weights[j][i] < - edge_weight_threshold:
    #                 dot.edge('I' + str(i), 'H' + str(j), color='red',
    #                          penwidth=str(np.abs(
    #                              input_to_hidden_weights[j][i]) * input[
    #                                           i] * 2))  # , label=str(input_to_hidden_weights[j][i]))
    #
    # # hidden to output edges
    # for i in range(number_of_original_hidden_nodes):
    #     for j in range(len(output)):
    #         if hidden_to_output_weights[i][j] > 0 and (
    #                 max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
    #             dot.edge('H' + str(i), 'O' + str(j), color='green',
    #                      # label='{weight_activation:0.2f}'.format(
    #                      #     weight_activation=(np.sum(hidden_to_output_weights[i]))),
    #                      penwidth=str(
    #                          np.abs(hidden_to_output_weights[i][j]) * hidden_activation[
    #                              i] * 5))  # , label=str(hidden_to_output_weights[i]))
    #         if hidden_to_output_weights[i][j] < 0 and (
    #                 max(np.abs(np.array(input_to_hidden_weights)[i, :])) >= edge_weight_threshold):
    #             dot.edge('H' + str(i), 'O' + str(j), color='red',
    #                      # label='{weight_activation:0.2f}'.format(
    #                      #     weight_activation=(np.sum(hidden_to_output_weights[i]))),  # ,
    #                      penwidth=str(
    #                          np.abs(hidden_to_output_weights[i][j]) * hidden_activation[
    #                              i] * 5))  # , label=str(hidden_to_output_weights[i]))
    # for i in range(len(output)):
    #     dot.edge('O' + str(i), 'Explanations', penwidth=str(0))
    #
    # # saving source code
    # dot.format = 'png'
    # dot.render(path + '/Graph_' + str(fig_index))
    # # dot.view()
    # # plt.show()
    # # print(dot.source)
