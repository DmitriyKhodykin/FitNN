import matplotlib.pyplot as plt
import pandas
import yaml

from service_functions import *


class NeuralNetwork:

    def __init__(self, dataset):
        self.dataset = dataset

        with open(r'params.yaml') as file:
            self.params = yaml.full_load(file)

        self.input_dim = self.params['config']['input_dim']
        self.h_dim = self.params['config']['h_dim']
        self.output_dim = self.params['config']['output_dim']

        self.weights_matrix_1 = numpy.random.randn(self.input_dim, self.h_dim)
        self.bias_vector_1 = numpy.random.randn(1, self.h_dim)

        self.weights_matrix_2 = numpy.random.randn(self.h_dim, self.output_dim)
        self.bias_vector_2 = numpy.random.randn(1, self.output_dim)

        self.los_arr = []

    def train(self):
        learning_rate = self.params['train']['learning_rate']
        num_epochs = self.params['train']['num_epochs']

        for epoch in range(num_epochs):
            for i in range(len(self.dataset)):
                x, y = self.dataset[i]

                # Forward
                t1 = x @ self.weights_matrix_1 + self.bias_vector_1
                h1 = relu(t1)
                t2 = h1 @ self.weights_matrix_2 + self.bias_vector_2
                z = softmax(t2)
                error = sparse_cross_entropy(z, y)

                # Backward
                y_full = to_full(y, self.output_dim)
                de_dt2 = z - y_full
                de_dw2 = h1.T @ de_dt2
                de_db2 = de_dt2
                de_dh1 = de_dt2 @ self.weights_matrix_2.T
                de_dt1 = de_dh1 * relu_deriv(t1)
                de_dw1 = x.T @ de_dt1
                de_db1 = de_dt1

                # Update
                self.weights_matrix_1 = self.weights_matrix_1 - learning_rate * de_dw1
                self.bias_vector_1 = self.bias_vector_1 - learning_rate * de_db1
                self.weights_matrix_2 = self.weights_matrix_2 - learning_rate * de_dw2
                self.bias_vector_2 = self.bias_vector_2 - learning_rate * de_db2

                self.los_arr.append(error)

    def predict(self, x):
        t1 = x @ self.weights_matrix_1 + self.bias_vector_1
        h1 = relu(t1)
        t2 = h1 @ self.weights_matrix_2 + self.bias_vector_2
        z = softmax(t2)
        return z

    def calc_accuracy(self):
        correct = 0
        for x, y in self.dataset:
            z = self.predict(x)
            y_predicted = numpy.argmax(z)
            if y_predicted == y:
                correct = correct + 1
        accuracy = correct / len(self.dataset)
        print('Accuracy:', accuracy)
        return accuracy

    def calc_matrix(self) -> pandas.DataFrame:
        y_true = []
        y_pred = []
        for x, y in self.dataset:
            y_true.append(y)
            z = self.predict(x)
            y_pred.append(numpy.argmax(z))
        result = pandas.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        return result

    def display_result(self):
        plt.plot(self.los_arr)
        plt.savefig('pics/los_arr.png')
