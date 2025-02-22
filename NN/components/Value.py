class Value:
    def __init__(self, data, prev=(), op=None, label=None):
        self.data = data
        self._prev = prev
        self._op = op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

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
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return self.data > other.data

    def ReLu(self,):
        return Value(max(0, self.data), prev=(self,), op='ReLu')

    @staticmethod
    def backpropagation(y):
        y.grad = 1
        Value._propagate(y)

    @staticmethod
    def _propagate(y):
        prev = y._prev

        if prev is None:
            return
        
        if len(prev) == 1:
            p = prev[0]
            if y._op == 'ReLu':
                dp = 1 if p > 0 else 0
                p.grad += y.grad * dp
                Value._propagate(p)
        
        if len(prev) == 2:
            p1, p2 = prev[0], prev[1]
            if y._op == '+':
                dp1 = 1
                dp2 = 1
            if y._op == '*':
                dp1 = p2.data
                dp2 = p1.data 
            p1.grad += y.grad * dp1
            p2.grad += y.grad * dp2
            Value._propagate(p1)
            Value._propagate(p2)