import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
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
    for n in nodes:
        dot.node(name=str(id(n)), label=f"data={n.data}, grad={n.grad}", shape='record')

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
a = Value(2.0)
b = Value(-3.0)
c = a + b
d = a * b + b * (a + a)
e = d.relu()
e.backward()
dot = draw_dot(e)
dot.render('computation_graph', view=True)  
