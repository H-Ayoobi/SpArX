# SpArX: Sparse Argumentative eXplanations for Neural Networks

The ArXiv Version of the paper is available at [https://arxiv.org/pdf/2301.09559.pdf](https://arxiv.org/abs/2301.09559).
## Datasets:
1. COMPAS Dataset - This is included in the data folder
2. Cancer Dataset (UCI) - This is included in the data folder
3. Iris Dataset - This is included in the data folder
4. Covertype Dataset (UCI) - The compressed (.zip) format is included in the data folder. Please first extract it to the data folder. 

## Experiments
For the first three datasets, there are nine Python files. One for global explanation, one for local explanation to measure unfaithfulness and one for local explanation to measure structural unfaithfulness. The scalability experiments have been done with the fourth dataset for the input-output local faithfulness. 

## Naming Convention of Python Files
The Python files are named based on the dataset name, global/local explanation and whether it is used to measure unfaithfulness or structural unfaithfulness.
That is DatasetName_global/local_explanations (for (structural) unfaithfulness)
Please, run each of these Python files to produce the results for SpArX (our) method. This will provide the results in Tables 1, 2 and 3. 
Running covertype_local_explanations.py will provide the results in Table 4 in the paper and Tables 5, 6, 7 and 8 from the Supplementary Materials (SM). 

### Visualization
Running Python files generates graphical visualization of the neural networks. One for the original MLP and one for the clustered MLP. 
The directory of the graphs are dataset_global/local_graphs(original/shrunken_model)

 
### Computing unfaithfulness for LIME 
We use lime_tabular from the LIME python library (https://github.com/marcotcr/lime). 
Currently, the explain_instance function uses the label=1 which means that it only considers output node number 1 and not all of the output neurons.
  To change that you should consider using label=[0,1,2] for the Iris dataset, label=[0, 1] for the cancer dataset and label=[0] for the COMPAS dataset. and compute the predictions of the regression model used
            in LIME that is Ridge model from Sklearn using all the outputs from all the nodes you can compare the predictions of the original model and the
            regressor.
            then change the last lines in the explain_instance function as follows:
            
            unfaithfulness = np.sum(list(ret_exp.score.values()))
            if self.mode == "regression":
                 ret_exp.intercept[1] = ret_exp.intercept[0]
                 ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
                 ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
             return ret_exp, unfaithfulness
             also, add a new way for computing scores in lime_base.py as follows:
             new_score = np.sum(
                 np.multiply(np.power(easy_model.predict(neighborhood_data[:, used_features]) - labels_column, 2),
                             weights / np.sum(weights)))
             and return new_score in addition to prediction_score


This way we can compute the unfaithfulness of the LIME method.

### Model Compression Technique (HAP)
Please use the "all" virtual environment in the HAP-main project provided here. 
It has a customized version of torchvision 
that supports Iris, Cancer and COMPAS datasets. The Python version is 3.6. 

Please download the HAP project from this link: https://drive.google.com/file/d/1RjhPlKdHNHO738kt70_phtlO8z85Lay-/view

This project uses the code from the GitHub repository of the Hessian-Aware Pruning (HAP) paper.

We have included the HAP-main.rar file in this project, it has an additional Python file to compute (structural) unfaithfulness. 
Please extract it as a separate project and for each dataset (dataset_name = iris, cancer, COMPAS) run the Python codes as follows:

* python main_pretrain --batch_size 2 --learning_rate 0.001 --weight_decay 0.00002 --dataset dataset_name --epoch 200 --neurons 50
Now set different pruning ratios and run the following script. Here, we have used a 0.2 ratio. In the experiments, we have used 0.2, 0.4, 0.6 and 0.8 ratios. 
* python main_prune --network mlp --depth 2 --dataset dataset_name --batch-size 8 --learning-rate 0.001 --weight-decay 4e-4 --ratio 0.2 --use-decompose 0 --gpu "0" --neurons 50

