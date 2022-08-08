import sys
import numpy as np
import matplotlib as plt


np.random.seed(42)

class Neuron():
    def __init__(self, num_of_inputs, bias=None):
        self.w = np.random.randn(num_of_inputs)
        self.bias = np.random.randint(0,10) if bias == None else bias
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))
    def forward(self, inputs):
        return self.sigmoid(np.sum(inputs * self.w) + self.bias)
    