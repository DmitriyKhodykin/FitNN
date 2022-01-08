import numpy


def relu(t):
    return numpy.maximum(t, 0)


def softmax(t):
    out = numpy.exp(t)
    return out / numpy.sum(out)


def sparse_cross_entropy(z, y):
    return -numpy.log(z[0, y])


def to_full(y, num_classes):
    y_full = numpy.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def relu_deriv(t):
    return (t > 0).astype(float)

