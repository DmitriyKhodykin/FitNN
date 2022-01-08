from sklearn import datasets
import matplotlib.pyplot as plt
import yaml

from service_functions import *

with open(r'params.yaml') as file:
    params = yaml.full_load(file)

iris = datasets.load_iris()
dataset = [
    (
        iris.data[i][None, ...],
        iris.target[i]
    ) for i in range(
        len(iris.target)
    )
]

w1 = numpy.random.randn(params['config']['input_dim'], params['config']['h_dim'])
b1 = numpy.random.randn(1, params['config']['h_dim'])

w2 = numpy.random.randn(params['config']['h_dim'], params['config']['output_dim'])
b2 = numpy.random.randn(1, params['config']['output_dim'])

los_arr = []

for epoch in range(params['train']['num_epochs']):
    for i in range(len(dataset)):

        x, y = dataset[i]

        # Forward
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        error = sparse_cross_entropy(z, y)

        # Backward
        y_full = to_full(y, params['config']['output_dim'])
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        # Update
        w1 = w1 - params['train']['learning_rate'] * dE_dW1
        b1 = b1 - params['train']['learning_rate'] * dE_db1
        w2 = w2 - params['train']['learning_rate'] * dE_dW2
        b2 = b2 - params['train']['learning_rate'] * dE_db2

        los_arr.append(error)


def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = numpy.argmax(z)
        if y_pred == y:
            correct = correct + 1
    accuracy = correct / len(dataset)
    return accuracy


accuracy = calc_accuracy()
print("Accuracy:", accuracy)
plt.plot(los_arr)
plt.show()
