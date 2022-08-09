from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from os import access
import sys
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
# np.set_printoptions(precision=70)


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


class CCELoss(LossFunction):
    def get_loss(self, y, y_pred):
        np.clip(y_pred, 1e-7, 1e7)
        if len(y.shape) == 1:
            return np.mean(-np.log(y_pred[range(len(y_pred)), y]))
        elif len(y.shape) == 2:
            return np.mean(-np.log(np.sum(y_pred * y, axis=1)))


class ActivationFunction:
    def __init__(self):
        self.results = None


class Sigmoid(ActivationFunction):
    def forward(self, X):
        self.results = 1 / (1 + np.exp(X))
        return self.results


class Relu(ActivationFunction):
    def forward(self, X):
        self.results = np.where(0 >= X, 0, X)
        return self.results


class Softmax(ActivationFunction):
    def forward(self, X):
        X = X - np.max(X, axis=1, keepdims=True)
        exponent = np.exp(X)
        self.results = exponent / np.sum(exponent, axis=1, keepdims=True)
        return self.results


class LinearLayer:
    def __init__(self, num_of_inputs, amount_of_neurons, activation: ActivationFunction, loss_func: LossFunction):
        # self.neurons = np.array([Neuron(num_of_inputs)
        #                         for i in range(amount_of_neurons)], dtype=np.object0)
        self.weights = np.random.randn(amount_of_neurons, num_of_inputs)
        self.bias = np.ones(amount_of_neurons)
        self.activation = activation
        self.loss_func = loss_func

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.dot(X, self.weights.T) + self.bias

    def forward(self, X):
        self.result = self.activation.forward(self.predict(X))
        return self.result

    def get_loss(self, y):
        return self.loss_func.get_loss(y, self.result)

    def get_weights(self):
        return self.weights


X, y = create_data(100, 3)
network_layer1 = LinearLayer(X.shape[1], 3, Sigmoid(), CCELoss())  # RELU
network_layer2 = LinearLayer(3, 3, Softmax(), CCELoss())  # RELU
network_layer3 = LinearLayer(3, 3, Softmax(), CCELoss())  # RELU
network_layer1.forward(X)
network_layer2.forward(network_layer1.result)
network_layer3.forward(network_layer2.result)
print(network_layer3.get_loss(y))
