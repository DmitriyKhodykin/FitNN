import json
from sklearn import datasets

from main import NeuralNetwork


if __name__ == '__main__':
    # Dataset
    iris = datasets.load_iris()
    data = [
        (
            iris.data[i][None, ...],
            iris.target[i]
        ) for i in range(
            len(iris.target)
        )
    ]
    # NN Instance
    neural_network = NeuralNetwork(data)
    neural_network.train()
    accuracy = neural_network.calc_accuracy()
    with open("log.csv", "w") as fd:
        json.dump({"accuracy": accuracy}, fd)
