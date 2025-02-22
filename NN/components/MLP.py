import numpy as np
from NN.components import Value

class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        for l in self.layers:
            x = l.forward(x)
        return x