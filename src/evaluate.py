import json

from neural_network import NeuralNetwork
from prepare import get_data


if __name__ == '__main__':
    # Dataset
    data = get_data()

    # NN Instance
    neural_network = NeuralNetwork(data)
    neural_network.train()
    neural_network.display_result()
    accuracy = neural_network.calc_accuracy()

    with open("log.csv", "w") as fd:
        json.dump({"accuracy": accuracy}, fd, indent=4)
