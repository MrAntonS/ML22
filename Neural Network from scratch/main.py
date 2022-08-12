import sys
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
# np.set_printoptions(precision=70)


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
        # print(self.inputs)
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
        self.inputs = None

    def forward(self, inputs):
        pass

    def backward(self, gradoutputs):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.f = lambda x: 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = self.f(inputs)
        return self.results

    def backward(self, gradoutputs):
        exp = self.f(self.inputs)
        self.derivative =  exp * (1 - exp)
        return self.derivative


class Relu(ActivationFunction):
    def __init__(self):
        self.f = lambda x: np.where(0 >= x, 0, x)

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = self.f(inputs)
        return self.results

    def backward(self, gradoutputs):
        self.derivative = np.where(0 >= self.inputs, 0, 1)
        # print(f"{self.derivative=}")
        self.derivative = self.derivative * gradoutputs
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
    def __init__(self, num_of_inputs, amount_of_neurons, lr=None):
        self.weights = np.random.randn(amount_of_neurons, num_of_inputs)
        self.bias = np.zeros(amount_of_neurons)
        self.results = None
        self.gradInput = None
        self.inputs = None
        self.lr = np.float64(0.001) if lr == None else lr

    def grad(self, gradoutputs):
        # print(f"{self.weights = }")
        # print(f"{gradoutputs = }")
        # print(f"{self.inputs = }")
        self.gradInput = gradoutputs @ self.weights
        gradW = (self.inputs.T @ gradoutputs) / len(self.inputs)
        gradB = np.sum(gradoutputs, axis=0) / len(self.inputs)
        return gradW.T, gradB

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = np.dot(inputs, self.weights.T) + self.bias
        return self.results

    def backward(self, gradoutputs, iteration=1):
        gradW, gradB = self.grad(gradoutputs)
        # print(f"{gradW = }")
        # print(f"{gradB = }")
        lr = self.lr / np.sqrt(iteration)
        self.weights = self.weights - lr * gradW
        self.bias = self.bias - lr * gradB
        return self.gradInput

    def get_weights(self):
        return self.weights


def f(X):
    return X ** 2 + 2


X = np.linspace(-10, 10, 1000, dtype=np.float64).reshape(-1, 1)
y = f(X).T + np.random.randn(1000) * 1
y = y.T
Linear_layer = LinearLayer(1, 100)
Linear_layer2 = LinearLayer(100, 1)
ActivationLayer = Relu()
ActivationLayer2 = Relu()
LossLayer = MeanSquared()
loss = []
i = 0
while True:
    # print("Inputs")
    # print(f"{X = }")
    # print(f"{y = }")
    # print("Forwarding")
    Linear_layer.forward(X)
    # print(f"{Linear_layer.results = }")
    # print(f"{Linear_layer.weights = }")
    ActivationLayer.forward(Linear_layer.results)
    # print(f"{ActivationLayer.results = }")
    Linear_layer2.forward(ActivationLayer.results)
    ActivationLayer2.forward(Linear_layer2.results)
    # print(f"{Linear_layer2.results = }")
    # print(f"{Linear_layer2.weights = }")
    LossLayer.forward(y, ActivationLayer2.results)
    # print(f"{LossLayer.results = }")
    # print("=" * 40,"\nBackwarding")
    LossLayer.backward(ActivationLayer2.results)
    ActivationLayer2.backward(LossLayer.derivative)
    # print(f"{LossLayer.derivative = }")
    Linear_layer2.backward(LossLayer.derivative)
    # print(f"{Linear_layer2.gradInput = }")
    ActivationLayer.backward(Linear_layer2.gradInput)
    # print(f"{ActivationLayer.derivative = }")
    Linear_layer.backward(ActivationLayer.derivative)
    # print(f"{Linear_layer.gradInput = }")
    # print(f"{Linear_layer.bias = }")
    # print(f"{Linear_layer.weights = }")
    loss.append(LossLayer.results)
    if LossLayer.results < 0.5 or i > 3 * 10**4:
        break
    i += 1
    # print("_"*30)
print(f"{LossLayer.results = }, {i =}")
# # print(Linear_layer.weights)
# # # print(Linear_layer.bias)
plt.scatter(X, y, c='g')
# plt.plot(np.linspace(1,100, 100), loss)
plt.plot(X, f(X))
plt.plot(X, Linear_layer2.results)
plt.show()