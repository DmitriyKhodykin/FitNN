"""
Preparing data for the model.
Doc: https://scikit-learn.org/stable/datasets/toy_dataset.html
"""

from sklearn import datasets


def get_data():
    """
    Prepare Iris dataset to NN use.
    :return: dataset
    """
    iris = datasets.load_iris()
    data = [
        (
            iris.data[i][None, ...],
            iris.target[i]
        ) for i in range(
            len(iris.target)
        )
    ]
    return data
