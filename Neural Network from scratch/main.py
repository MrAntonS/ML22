import sys
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


class Neuron:
    def __init__(self, num_of_inputs, bias=None):
        self.w = 0.1 * np.random.randn(num_of_inputs)
        self.bias = 0  # np.random.randint(0, 10) if bias == None else bias
        self.scalar = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def _relu(self, x):
        return np.where(0 >= x, 0, x)

    def predict(self, inputs):
        self.scalar = np.dot(inputs, self.w) + self.bias
        return self.scalar

    def forward(self, inputs):
        return self._sigmoid(self.predict(inputs))

    def forward_relu(self, inputs):
        return self._relu(self.predict(inputs))

    def get_weights(self):
        return self.w


class LinearLayer:
    def __init__(self, num_of_inputs, amount_of_neurons):
        self.neurons = np.array([Neuron(num_of_inputs)
                                for i in range(amount_of_neurons)], dtype=np.object0)
        self.result = None

    def fit(self, X, y):
        print(self.__loss(y, self.forward_softmax(X)))

    def forward_sigmoid(self, inputs: np.ndarray):
        self.result = np.array([i.forward(inputs)
                                for i in self.neurons], dtype=np.float64).T
        return self.result

    def forward_relu(self, inputs: np.ndarray):
        self.result = np.array([i.forward_relu(inputs)
                                for i in self.neurons], dtype=np.float64).T
        return self.result

    def get_weights(self):
        return np.array([i.get_weights() for i in self.neurons])

    def forward_softmax(self, inputs):
        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        softmax_results = np.exp(inputs)
        self.result = softmax_results / \
            np.sum(softmax_results, axis=1, keepdims=True)
        return self.result

    def __loss(self, y, y_pred):
        np.clip(y_pred, 1e-7, 1e7)
        return np.mean(-np.log(y_pred[range(len(y_pred)), y-1]))


X, y = create_data(100, 3)
network_layer1 = LinearLayer(X.shape[1], 3)  # RELU
network_layer2 = LinearLayer(3, 3)  # SoftMax
network_layer2.fit(X, y)
