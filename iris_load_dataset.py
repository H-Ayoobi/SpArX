import pandas as pd
def load_iris():
    columns_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    df = pd.read_csv('data/iris.data', header=None, names=columns_names)
    return df