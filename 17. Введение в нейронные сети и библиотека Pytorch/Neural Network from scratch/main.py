import sys
import numpy as np
import matplotlib as plt


np.random.seed(42)


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


neuron = Neuron(4, 2)
print(neuron.get_weights())
print(neuron.forward(np.array([1, 2, 3, 2.5])))
