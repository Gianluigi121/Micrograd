# Micrograd

# Gradient

Operation for gradients:

- **Add** will distributes the gradient to all the leaf nodes
- **Mult** will multiply the upstream gradient with the other value from the multiplication

# Value Class

### Goal:

Simplication of the Tensor class in Pytorch that can perform foward and backward propagation

### Logic:

Example: **c = a+b**

We start from having two Value object `a` and `b`

1. Forward part:
    
    As we call a+b, we internally call `a.__add__(b)`. We will first have the out value `c` as a Value object with the result value. Then, we will assign a gradient backward function to `c`.
    
    By assigning the backward function to `c`, we will be able to assign `a`'s and `b`'s gradient by calling `c._backward()`. The gradient assignment function takes the upstream gradient `out.grad` and multiply by the local gradient determined by the operation type.
    
2. Backward part:
    
    We will call `c._backward()`. This will trigger the backward function and assign gradient for its children `a` and `b`.(Details explained in part 1)
    

### Implementation:

To implement the Value class, we need to overwrite these functions:

1. **`__repr__`**: Returns a string representation of an object
2. **`__add__`**, **`__sub__`**, **`__mul__`**, **`__truediv__`**: Define the calculation of the object
3. **`__radd__`**, **`__rmul__`**: Revered calculation function. Details in appendix

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        # This allow us to handle the addition by the constant, such as a+3
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        # Implement backward function
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
        
    def __sub__(self, other):
        return self + (-other)

    def __neg__(self): # self
        return self * -1

    def __mul__(self, other):
        # This allow us to handle the multiplication by the constant, such as a*3
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # Implement backward function
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other*self.data**(other-1)*out.grad
        out._backward = _backward
        return out
    
    # other+self: 3+a
    def __radd__(self, other):
        return self + other

    # other*self: 3*a
    def __rmul__(self, other):
        return self * other
    
    # self/other
    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        # Implement backward function
        def _backward():
            self.grad += 1 - out.data**2
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

## MLP

```python
class Neuron:
    def __init__(self, nin):
        # nin: # of number that will come into one neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        out = sum(wi*xi for wi, xi in zip(self.w, x))+self.b
        out = out.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    # nin: # of numbers/neurons that will come into each neuron in this layer
    # nout: # of neurons in this layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    # nin: # of numbers that will come into each neuron in the first layer
    # nout: A list of # of neurons in each layer
    def __init__(self, nin, nouts):
        sin = [nin] + nouts
        self.layers =  [Layer(sin[i], sin[i+1]) for i in range(len(nouts))]
        
    def  __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

## Training loop:

```python
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

for k in range(50):
  
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  
  print(k, loss.data)
```

## Appendix:

### Revered Multiplication Function

The **`__rmul__`** method is a special method in Python that is used to implement the reverse multiplication operation. It is called when the multiplication operator **`*`** is used, but the left operand does not support the multiplication operation with the right operand. In this case, the right operand (if it has the **`__rmul__`** method) is given the opportunity to handle the operation.

This method is useful when you define custom classes that need to interact with built-in types (such as integers, floats) using the multiplication operator.

Here's how the process of calling the **`__rmul__`** method works:

1. When a multiplication operation **`x * y`** is executed, Python first tries to call the **`__mul__`** method of the left operand (**`x`**), with the right operand (**`y`**) as its argument. In other words, it attempts **`x.__mul__(y)`**.
2. If the left operand's **`__mul__`** method returns **`NotImplemented`** (which usually means that the left operand does not know how to handle the multiplication with the right operand), Python then checks if the right operand has the **`__rmul__`** method. If it does, Python calls this method with the left operand as its argument. In other words, it attempts **`y.__rmul__(x)`**.
3. If neither the **`__mul__`** method of the left operand nor the **`__rmul__`** method of the right operand can handle the operation, Python raises a **`TypeError`**.

For example, let's consider the case where you have a custom class **`Value`** and you want to perform multiplication with an integer:

```python
class Value:
    def __init__(self, data):
        self.data = data

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data)
        return NotImplemented

    def __rmul__(self, other):
        return Value(self.data * other)

a = Value(2)
result = 3 * a  # 3.__mul__(a) returns NotImplemented, so a.__rmul__(3) is called
```

In this example, when we execute **`3 * a`**, the integer **`3`** doesn't have a **`__mul__`** method that can handle multiplication with the **`Value`** class, so it returns **`NotImplemented`**. Python then checks if the right operand **`a`** has the **`__rmul__`** method, and since it does, the method is called with **`3`** as the argument. The result is a new **`Value`** object with the correct multiplication result.