import numpy as np
from NN.components import Value, Neuron, Layer, MLP

class TinyNet:
    def __init__(self, shape):
        self.entrance_size = shape.pop(0)
        layers = []
        prev_layer_size = shape[0]
        for i in range(1, len(shape)):
            curr_layer_size = shape[i]
            layer = Layer(neurons=[Neuron(weights=np.random.random(prev_layer_size), bias=np.random.random()) for _ in range(curr_layer_size)])
            layers.append(layer)
            prev_layer_size = curr_layer_size
        self.mlp = MLP(layers=layers)
        self.loss = None

    def forward(self, X):
        num_points = X.shape[0]
        y = Value.from_nparray(np.zeros(num_points))
        for i in range(num_points):
            y[i] = self.mlp.forward(X[i])
        return y
    
    def zero_grad(self,):
        self.mlp.zero_grad()

    def backpropagation(self, X, y_true):
        self.zero_grad()
        y_pred = self.forward(X)
        self.loss = Value.MSE(y_pred, y_true)
        Value.backpropagation(self.loss)

    def get_weights(self,):
        return np.array([weight for layer in self.mlp.layers for neuron in layer.neurons for weight in neuron.weights])
    
    def get_weights_grads(self,):
        return np.array([weight.grad for weight in self.get_weights()])

    def get_biases(self,):
        return np.array([neuron.bias for layer in self.mlp.layers for neuron in layer.neurons])

    def get_biases_grads(self,):
        return np.array([bias.grad for bias in self.get_biases()])

    def get_parameters(self,):
        return np.concatenate([self.get_weights(), self.get_biases()])

    def get_parameters_grads(self,):
        return np.concatenate([self.get_weights_grads(), self.get_biases_grads()])

    def set_weights(self, new_weights):
        for weight, new_weight in zip(self.get_weights(), new_weights):
            weight.data = new_weight

    def set_biases(self, new_biases):
        for bias, new_bias in zip(self.get_biases(), new_biases):
            bias.data = new_bias

    def set_parameters(self, new_params):
        for param, new_param in zip(self.get_parameters(), new_params):
            param.data = new_param