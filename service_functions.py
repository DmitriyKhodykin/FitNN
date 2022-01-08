import numpy


def relu(value):
    """
    If the input value is positive, the ReLU function returns it;
    if it is negative, it returns 0
    """
    return numpy.maximum(value, 0)


def softmax(tensor):
    """
    It is used to normalize the output of a network to a probability
    distribution over predicted output classes, based on Luce's choice axiom.
    """
    out = numpy.exp(tensor)
    return out / numpy.sum(out)


def sparse_cross_entropy(z, y):
    """

    :param z: Predicted class vector
    :param y: True class vector
    :return:
    """
    return -numpy.log(z[0, y])


def to_full(y, num_classes):
    y_full = numpy.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def relu_deriv(t):
    return (t > 0).astype(float)
