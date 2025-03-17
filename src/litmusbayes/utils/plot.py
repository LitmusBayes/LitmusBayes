

"""Helpers for Graph Plotting"""
from typing import List

import pygraphviz as pgv


class AGraphNode:
    """The graph node.

    Parameters
    ----------
    name: str
        The node name.
    shape: str, optional
        The node shape. e.g., polygon, circle, ellipse. The default is 'polygon'.
    color: str, optional
        The node color. The default is 'black'.
    style: str, optional
        The node edge style. The default is 'bold'.

    References
    ----------
    For detailed node attributes, refer to
    pygraphviz: https://zhuanlan.zhihu.com/p/104636240
    or https://pygraphviz.github.io/

    """

    def __init__(self, name, shape='polygon', color='black', style='bold'):
        self.name, self.shape, self.color, self.style = name, shape, color, style


class AGraphEdge:
    """The graph edge.

    Parameters
    ----------
    src: str
        The source node of the edge.
    tgt: str
        The target node of the edge.
    dir: str, optional
        The edge direction. The default is 'forward'.
    """

    def __init__(self, src, tgt, label: str = "", dir: str = 'forward', color: str = "black"):
        self.src, self.tgt = src, tgt
        self.label, self.color = label, color
        self.dir = dir

    def unpack(self):
        """Unpack the edge and get the source and target nodes.

        Returns
        -------
        str, str
            The name of the source node,
            the name of the target node.

        Examples
        --------
        >>> s, t = AGraphNode('s'), AGraphNode('t')
        >>> edge = AGraphEdge('s', 't')
        >>> a, b = edge.unpack() # `a` is 's' and `b` is 't'
        """
        return self.src, self.tgt


class Cluster:
    def __init__(self, label, nodes=None):
        self.label = label
        self.nodes = nodes if nodes else []
        self.color = '#00cc66'


def plot_graph(filename, nodes, edges, clusters: List[Cluster] = None):
    """Plot a graph using the `pygraphviz` library.

    Parameters
    ----------
    filename: str
        The name of the file to save the graph to.
    nodes: list
        A list of AGraphNode objects.
    edges: list
        A list of AGraphEdge objects.
    clusters: list
        A list of Cluster (SubGraph)
    References
    ----------
    pygraphviz: https://zhuanlan.zhihu.com/p/104636240

    Examples
    --------
    Plot a graph to 'add.png'.

    >>> plot_graph('add', nodes, edges)

    """

    g = pgv.AGraph(directed=True, strict=False, rankdir="TB")

    for node in nodes:
        g.add_node(node.name, shape=node.shape, color=node.color, style=node.style)
    for edge in edges:
        g.add_edge(edge.unpack(), dir=edge.dir, label=edge.label, color=edge.color, fontcolor=edge.color)

    if clusters:
        for cluster in clusters:
            with g.add_subgraph(name='cluster_' + cluster.label, label=cluster.label) as sg:
                for n in cluster.nodes:
                    sg.add_node(n)

    g.layout()
    # g.draw(filename + '.dot', format='dot', prog='dot')
    g.draw(filename + '.png', prog='circo')
    
