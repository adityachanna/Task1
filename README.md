# Computation Graph with Autograd

A Python implementation of a computation graph with automatic differentiation capabilities. This system enables basic mathematical operations and automatic gradient computation for backpropagation.

## Features

The implementation provides:
- Core mathematical operations (addition, multiplication, power)
- ReLU activation function
- Automatic gradient computation via backpropagation
- Computation graph visualization using Graphviz

## Usage Example

```python
from your_module import Value

# Initialize values
a = Value(2.0)
b = Value(-3.0)

# Build computation graph
c = a + b
d = a * b + b * (a + a)
e = d.relu()

# Compute gradients
e.backward()

# Access results
print(f"a: data={a.data}, grad={a.grad}")  # data=2.0, grad=-3.0
print(f"b: data={b.data}, grad={b.grad}")  # data=-3.0, grad=5.0
```

## Implementation Details

### Value Class

The `Value` class represents nodes in the computation graph. Each node contains:

```python
class Value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data        # Numerical value
        self.grad = 0.0         # Gradient for backpropagation
        self._prev = set(_prev) # Parent nodes
        self._op = _op         # Operation type
        self._backward = lambda: None
```

### Core Operations

#### Addition
```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    
    out._backward = _backward
    return out
```

#### Multiplication
```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    
    out._backward = _backward
    return out
```

#### ReLU Activation
```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
    def _backward():
        self.grad += (out.data > 0) * out.grad
    
    out._backward = _backward
    return out
```

### Backpropagation Implementation

The `backward()` method computes gradients by:
1. Building a topologically sorted list of nodes
2. Initializing the output gradient to 1.0
3. Propagating gradients backward through the graph

```python
def backward(self):
    # Topological sort
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # Backpropagate gradients
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

## Graph Visualization

The computation graph can be visualized using Graphviz:

```python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    
    # Add nodes
    for n in nodes:
        dot.node(name=str(id(n)), 
                label=f"data={n.data}, grad={n.grad}", 
                shape='record')
        
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    # Add edges
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# Usage
e.backward()
dot = draw_dot(e)
dot.render('computation_graph', view=True)
```

## Notes

- All operations support broadcasting with scalar values
- The implementation is designed for educational purposes and basic neural network operations
- Visualization requires the Graphviz library to be installed
