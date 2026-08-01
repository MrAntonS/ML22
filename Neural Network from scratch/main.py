
# ...existing code...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


np.random.seed(2)
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

    def backward(self, y_pred):
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
        self.results = np.mean(np.square(y_pred - y), axis=0)
        return np.mean(self.results)

    def backward(self, y_pred):
        self.derivative = 2 * (y_pred - self.inputs) / y_pred.shape[0]
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

class Tanh(ActivationFunction):
    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = np.tanh(inputs)
        return self.results

    def backward(self, gradoutputs):
        self.derivative = (1 - np.tanh(self.inputs) ** 2) * gradoutputs
        return self.derivative


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
        self.derivative = self.derivative * gradoutputs
        return self.derivative


class Relu(ActivationFunction):
    def __init__(self):
        self.f = lambda x: np.where(0 >= x, 0.1 * x, x)

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = np.maximum(0, inputs)
        return self.results

    def backward(self, gradoutputs):
        self.derivative = (self.inputs > 0).astype(float)
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


class LinearActivation(ActivationFunction):
    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = inputs
        return self.results

    def backward(self, gradoutputs):
        self.derivative = gradoutputs
        return self.derivative


'''
Layers
'''


class LinearLayer:
    def __init__(self, num_of_inputs, amount_of_neurons, lr=None):
        # Use smaller initial weights
        self.weights = np.random.randn(amount_of_neurons, num_of_inputs) * 0.1
        self.bias = np.zeros(amount_of_neurons)
        self.results = None
        self.derivative = None
        self.inputs = None
        self.lr = 0.05 if lr is None else lr

    def grad(self, gradoutputs):
        self.derivative = gradoutputs @ self.weights
        gradW = gradoutputs.T @ self.inputs / self.inputs.shape[0]
        gradB = np.mean(gradoutputs, axis=0)
        return gradW, gradB

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.results = np.dot(inputs, self.weights.T) + self.bias
        return self.results

    def backward(self, gradoutputs, iteration=1):
        gradW, gradB = self.grad(gradoutputs)
        self.weights -= self.lr * gradW
        self.bias -= self.lr * gradB
        return self.derivative

    def get_weights(self):
        return self.weights


def f(X):
    return np.sin(X)
def f1(X):
    return X ** 2 + 2

X = np.linspace(-7, 7, 50, dtype=np.float64).reshape(-1, 1)
y = f(X) + np.random.randn(50, 1) * 0.1  # shape (100, 1)
# X = np.array([[-1], [0], [1]])
# y = np.array([[0, 1, 0]]).T
fig = plt.figure()
Linear_layer1 = LinearLayer(1, 64, lr=0.5)
ActivationLayer1 = Tanh()
Linear_layer2 = LinearLayer(64, 32, lr=0.5)
ActivationLayer2 = Tanh()
Linear_layer3 = LinearLayer(32, 1, lr=0.5)
ActivationLayer3 = LinearActivation()  # Output layer is linear
LossLayer = MeanSquared()
loss = []

predictions = []
losses = []
iterations_to_show = []

num_iterations = 10000
show_every = 200

for i in range(num_iterations):
    Linear_layer1.forward(X)
    ActivationLayer1.forward(Linear_layer1.results)
    Linear_layer2.forward(ActivationLayer1.results)
    ActivationLayer2.forward(Linear_layer2.results)
    Linear_layer3.forward(ActivationLayer2.results)
    ActivationLayer3.forward(Linear_layer3.results)
    LossLayer.forward(y, ActivationLayer3.results)

    LossLayer.backward(ActivationLayer3.results)
    ActivationLayer3.backward(LossLayer.derivative)
    Linear_layer3.backward(ActivationLayer3.derivative)
    ActivationLayer2.backward(Linear_layer3.derivative)
    Linear_layer2.backward(ActivationLayer2.derivative)
    ActivationLayer1.backward(Linear_layer2.derivative)
    Linear_layer1.backward(ActivationLayer1.derivative)

    if i % show_every == 0:
        predictions.append(ActivationLayer3.results.copy())
        losses.append(LossLayer.results)
        iterations_to_show.append(i)

def anim(frame):
    plt.clf()
    plt.scatter(X, y, c='g')
    plt.plot(X, y, label='True')
    plt.plot(X, predictions[frame], label='Prediction')
    plt.legend()
    plt.title(f"Iteration: {iterations_to_show[frame]}, Loss: {float(losses[frame]):.4f}")

ani = FuncAnimation(fig, anim, frames=len(predictions), interval=500)
plt.show()