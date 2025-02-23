import numpy as np
from NN.TinyNet import TinyNet
from NN.components import Value

if __name__ == "__main__":

    # Example usage
    X = np.array([[0.5, -1.5, 3.0], [1.0, 2.0, -1.0]])
    y_true = Value.from_nparray(np.array([1.0, 0.0]))

    model = TinyNet(shape=[3, 3, 2, 1])
    learning_rate = 0.01
    for epoch in range(100):
        model.backpropagation(X, y_true)
        print(f"Epoch {epoch + 1}, Loss:", model.loss)
        
        params = model.get_parameters()
        grads = model.get_parameters_grads()
        new_params = params - learning_rate * grads
        model.set_parameters(new_params)
