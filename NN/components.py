import numpy as np
import math

class Value:
    def __init__(self, data, prev=(), op=None):
        self.data = data
        self._prev = prev
        self._op = op
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __str__(self,):
        return f"{self.data}"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return Value(self.data + other.data, prev=(self, other), op='+')
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return Value(self.data * other.data, prev=(self, other), op='*')
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self,):
        return Value(-self.data)
    
    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return Value(self.data**other.data, (self, other), '**')

    def __truediv__(self, other):
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return self.data > other.data

    def ReLu(self,):
        return Value(max(0, self.data), prev=(self,), op='ReLu')
    
    def zero_grad(self,):
        self.grad = 0.0

    @staticmethod
    def backpropagation(y):
        
        topo = []
        visited = set()
        def topological_sorting(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sorting(child)
                topo.append(v)
        topological_sorting(y)
        
        y.grad = 1
        for node in topo[::-1]:
            Value._backward_propagate(node)

    @staticmethod
    def _backward_propagate(y):
        prev = y._prev # d, f (but we are in b! indeed, y = b !!!!)

        if len(prev) == 0:
            return
        
        if len(prev) == 1:
            p = prev[0]
            if y._op == 'ReLu':
                dp = 1 if p > 0 else 0
                p.grad += y.grad * dp
        
        if len(prev) == 2:
            p1, p2 = prev[0], prev[1]
            if y._op == '+':
                dp1 = 1
                dp2 = 1
            if y._op == '*':
                dp1 = p2.data
                dp2 = p1.data 
            if y._op == '**':
                dp1 = p2.data * (p1.data ** (p2.data-1))
                dp2 = (p1.data ** p2.data) * math.log(p1.data)
            p1.grad += y.grad * dp1
            p2.grad += y.grad * dp2

    @staticmethod
    def MSE(y_pred, y_true):
        if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
            raise TypeError("Both inputs must be NumPy arrays.")
        if not np.issubdtype(y_pred.dtype, np.object_) or not np.issubdtype(y_true.dtype, np.object_):
            raise TypeError("Both arrays must contain Value objects.")
        if y_pred.shape != y_true.shape:
            raise ValueError("Shapes of y_pred and y_true must match.")
        error = (y_pred - y_true)
        return sum(error * error) / y_true.size

    @staticmethod
    def from_nparray(data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("Numpy array must contain floats.")
        
        vectorized_conversion = np.vectorize(lambda x: Value(x), otypes=[object])
        return vectorized_conversion(data)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = [Value(weight) for weight in weights]
        self.bias = Value(bias)

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        act = sum((w*x for w, x in zip(self.weights, x)), self.bias)
        return (act).ReLu()
    
    def zero_grad(self,):
        self.bias.zero_grad()
        for w in self.weights:
            w.zero_grad()

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return np.array([n.forward(x) for n in self.neurons])

    def zero_grad(self,):
        for n in self.neurons:
            n.zero_grad()

class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        if isinstance(X, list):
            X = np.array(X)
        for l in self.layers:
            X = l.forward(X)
        return X[0] if X.size == 1 else X
    
    def zero_grad(self,):
        for l in self.layers:
            l.zero_grad()