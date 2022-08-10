from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from operator import ne
from os import access
from pydoc import describe
import sys
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
np.set_printoptions(precision=70)


def create_data(points, classes):
    X = np.empty((points * classes, 2))
    y = np.empty((points*classes), dtype=np.int32)
    for class_num in range(classes):
        r = np.linspace(0, 1, points, dtype=np.float64)
        t = np.linspace(class_num*4, (class_num + 1)*4, points,
                        dtype=np.float64) + np.random.randn(points)*0.2
        X[points*class_num:points *
            (class_num+1)] = np.c_[r * np.sin(t*2.5), r*np.cos(t*2.5)]
        y[points*class_num:points*(class_num+1)] = class_num
    return X, y


# class Neuron:
#     def __init__(self, num_of_inputs, bias=None):
#         self.w = 0.1 * np.random.randn(num_of_inputs)
#         self.bias = 0  # np.random.randint(0, 10) if bias == None else bias
#         self.scalar = None

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-1 * x))

#     def _relu(self, x):
#         return np.where(0 >= x, 0, x)

#     def predict(self, inputs):
#         self.scalar = np.dot(inputs, self.w) + self.bias
#         return self.scalar

#     def forward(self, inputs):
#         return self._sigmoid(self.predict(inputs))

#     def forward_relu(self, inputs):
#         return self._relu(self.predict(inputs))

#     def get_weights(self):
#         return self.w

class LossFunction:
    def __init__(self):
        self.result = None
        self.derivative = None

    def forward(self, y, y_pred):
        pass

    def backward(self, y, y_pred):
        pass


class CCELoss(LossFunction):
    def forward(self, y, y_pred):
        np.clip(y_pred, 1e-7, 1e7)
        if len(y.shape) == 1:
            self.result = -np.log(y_pred[range(len(y_pred)), y])
        elif len(y.shape) == 2:
            self.result = -np.log(np.sum(y_pred * y, axis=1))
        return self.result

    def backward(self, y, y_pred):
        self.derivative = np.tile(self.result, (y_pred.shape[1], 1)).T
        self.derivative[range(len(y_pred)), y] = 1 / \
            y_pred[range(len(y_pred)), y]
        return -self.derivative


'''
Activation Functions
'''


class ActivationFunction:
    def __init__(self):
        self.results = None
        self.f = None

    def forward(self, input):
        pass

    def backward(self, input):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.f = lambda x: 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.results = self.f(X)
        return self.results

    def backward(self, X):
        self.derivative = X * (1 - X)
        return self.derivative


class Relu(ActivationFunction):
    def __init__(self):
        self.f = lambda x: np.where(0 >= x, 0, x)

    def forward(self, X):
        self.results = self.f(X)
        return self.results

    def backward(self, X):
        self.derivative = np.where(0 >= X, 0, 1)
        return self.derivative


class Softmax(ActivationFunction):
    def forward(self, X):
        X = X - np.max(X, axis=1, keepdims=True)
        exponent = np.exp(X)
        self.results = exponent / np.sum(exponent, axis=1, keepdims=True)
        return self.results


class LinearLayer:
    def __init__(self, num_of_inputs, amount_of_neurons, activation: ActivationFunction, loss_func: LossFunction):
        self.weights = np.random.randn(amount_of_neurons, num_of_inputs)
        self.bias = np.ones(amount_of_neurons)

    def fit(self, X, y):
        pass

    def grad(self, inputs, gradoutputs):
        self.gradinput = np.dot(gradoutputs, self.weights.T)
        self.gradW = np.dot(gradoutputs, inputs.T)
        self.gradB = np.sum(gradoutputs, axis=0)
        return self.gradinput

    def forward(self, X):
        self.raw_predict = np.dot(X, self.weights.T) + self.bias
        return self.raw_predict

    def get_weights(self):
        return self.weights
