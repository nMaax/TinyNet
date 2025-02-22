import numpy as np
from NN.components import Value

class Neuron:
    def __init__(self, weights, bias):
        self.weights = [Value(weight) for weight in weights]
        self.bias = Value(bias)

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return (x @ self.weights + self.bias).ReLu()