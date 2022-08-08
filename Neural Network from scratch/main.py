import sys
import numpy as np
import matplotlib as plt


np.random.seed(42)
np.set_printoptions(precision=70)


class Neuron():
    def __init__(self, num_of_inputs, bias=None):
        self.w = np.random.randn(num_of_inputs)
        self.bias = np.random.randint(0, 10) if bias == None else bias

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def get_scalar(self, inputs):
        return np.sum(inputs * self.w) + self.bias

    def forward(self, inputs):
        return self._sigmoid(np.sum(inputs * self.w) + self.bias)

    def get_weights(self):
        return self.w


class LinearLayer():
    def __init__(self, num_of_inputs, amount_of_neurons):
        self.neurons = np.array([Neuron(num_of_inputs)
                                for i in range(amount_of_neurons)], dtype=np.object0)

    def _test(self, x):
        return x.forward()

    def calculate_forwarding(self, inputs: np.ndarray):
        result = np.array([i.forward(inputs)
                          for i in self.neurons], dtype=np.float64)
        return result


inputs = np.array([1, 2, 3, 4])
network_layer = LinearLayer(len(inputs), 10)
print(network_layer.calculate_forwarding(inputs))
