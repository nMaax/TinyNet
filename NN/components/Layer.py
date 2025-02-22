import numpy as np
from NN.components import Value

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return np.array([n.forward(x) for n in self.neurons])