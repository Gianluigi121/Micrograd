import math
import random
from main.draw import draw_dot

# We want to write a library class Value act as the Tensor class in PyTorch
# And we will implement the back propagation
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # We need to keep track of the children(which values generate the current value for the backpropagation)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None     
        self.label = label
        
    def __repr__(self):
        return f"Value(data: {self.data}, grad: {self.grad})"
    
    def __add__(self, other):
        # This allow us to handle the addition by the constant, such as a+3
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        # This allow us to handle the multiplication by the constant, such as a*3
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, _children=(self, other), _op='*')
        
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
        
    def __pow__(self, other):
        out = Value(self.data**other, _children=(self,), _op='pow')
        
        def _backward():
            self.grad += out.grad*(other*self.data**(other-1))
        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        # Handle the case when the constant is on the left side, such as 3+a
        # This will trigger 3.__add__(a), and the system will call a.__add__(3) instead after we define __radd__
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op='exp')
            
        def _backward():
            self.grad += out.grad*out.data
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x)-1)/(math.exp(2*x)+1), _children=(self,), _op='tanh')
        
        def _backward():
            self.grad += out.grad*(1-out.data**2)
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(max(0, self.data), _children=(self,), _op='relu')
        
        def _backward():
            self.grad += out.grad if self.data > 0 else 0
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    if child not in visited:
                        build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    


# Test section
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

o.backward()
draw_dot(o)
