from re import L
import sys
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
np.set_printoptions(precision=70)


class Data_Creator:
    def create_data_spiral(self, points, classes):
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


'''
Loss Functions
'''


class LossFunction:
    def __init__(self):
        self.results = None
        self.derivative = None
        self.inputs = None

    def forward(self, y, y_pred):
        pass

    def backward(self, y, y_pred):
        pass


class CCELoss(LossFunction):
    def forward(self, y, y_pred):
        self.inputs = y.copy()
        np.clip(y_pred, 1e-7, 1e7)
        if len(y.shape) == 1:
            self.results = -np.log(y_pred[range(len(y_pred)), y])
        elif len(y.shape) == 2:
            self.results = -np.log(np.sum(y_pred * y, axis=1))
        return self.results

    def backward(self, y_pred):
        print(self.inputs)
        self.derivative = -1 / y_pred[range(len(y_pred)), self.inputs]
        return self.derivative


class MeanSquared(LossFunction):
    def forward(self, y, y_pred):
        self.inputs = y.copy()
        self.results = np.mean((y_pred - y) ** 2)
        return self.results

    def backward(self, y_pred):
        self.derivative = -2 * (self.inputs - y_pred)
        return self.derivative


'''
Activation Functions
'''


class ActivationFunction:
    def __init__(self):
        self.results = None
        self.f = None
        self.derivative = None

    def forward(self, inputs):
        pass

    def backward(self, inputs):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.f = lambda x: 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        self.results = self.f(inputs)
        return self.results

    def backward(self, inputs):
        self.derivative = self.f(inputs) * (1 - self.f(inputs))
        return self.derivative


class Relu(ActivationFunction):
    def __init__(self):
        self.f = lambda x: np.where(0 >= x, 0, x)

    def forward(self, inputs):
        self.results = self.f(inputs)
        return self.results

    def backward(self, inputs):
        self.derivative = np.where(0 > inputs, 0, 1)
        return self.derivative


class Softmax(ActivationFunction):
    def forward(self, inputs):
        inputs_norm = inputs - np.max(inputs, axis=1, keepdims=True)
        exponent = np.exp(inputs_norm)
        self.results = exponent / np.sum(exponent, axis=1, keepdims=True)
        return self.results

    def backward(self, inputs):
        self.derivative = self.forward(inputs) * (1 - self.forward(inputs))
        return self.derivative


'''
Layers
'''


class LinearLayer:
    def __init__(self, num_of_inputs, amount_of_neurons):
        self.weights = np.random.randn(amount_of_neurons, num_of_inputs)
        self.bias = np.zeros(amount_of_neurons)
        self.results = None
        self.gradinput = None
        self.inputs = None

    def grad(self, inputs, gradoutputs):
        return self.gradinput

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = np.dot(inputs, self.weights.T) + self.bias
        return self.results

    def backward(self, inputs, gradoutputs):
        pass

    def get_weights(self):
        return self.weights

def f(X):
    return 3 * X + 3
X = np.linspace(1, 2, 1000).reshape(-1, 1)
y = f(X)
Linear_layer = LinearLayer(1, 1)
ActivationLayer = Relu()
LossLayer = MeanSquared()
Linear_layer.weights = np.array([[0.1]])
Linear_layer.bias[0] = 0
loss = []

for _ in range(10000):
    # print("Inputs")
    # print(f"{X = }")
    # print(f"{y = }")
    # print("Forwarding")
    Linear_layer.forward(X)
    # print(f"{Linear_layer.results = }")
    # print(f"{ActivationLayer.forward(Linear_layer.results) = }")
    # print(f"{LossLayer.forward(y, ActivationLayer.results) = }")
    LossLayer.forward(y, Linear_layer.results)
    # print(f"{LossLayer.results = }")
    # print("Backwarding")
    # print(f"{LossLayer.backward(ActivationLayer.results) = }")
    LossLayer.backward(Linear_layer.results)
    # print(f"{LossLayer.derivative = }")
    # print(f"{ActivationLayer.backward(Linear_layer.results) = }")
    # gradout = (LossLayer.backward(ActivationLayer.results) *
    #            ActivationLayer.backward(Linear_layer.results)).T
    gradout = (LossLayer.derivative)
    # print(f"{gradout = }")
    loss.append(LossLayer.results)
    gradInput = gradout * Linear_layer.weights
    gradW = np.mean(gradout * Linear_layer.inputs)
    gradB = np.mean(gradout)
    # print(f"{gradW=}")
    # print(f"{gradB=}")
    # print(f"{gradInput=}")
    # print(f"{Linear_layer.bias=}")
    Linear_layer.weights = Linear_layer.weights - 0.01 * gradW
    Linear_layer.bias = Linear_layer.bias - 0.01 * gradB
    # print(f"{Linear_layer.bias=}")
    # print("_"*30)
print(LossLayer.results)
# print(Linear_layer.weights)
# print(Linear_layer.bias)
plt.scatter(X, y, c='g')
# plt.plot(np.linspace(1,100, 100), loss)
plt.plot(X, y)
plt.plot(X, Linear_layer.results)
plt.show()

