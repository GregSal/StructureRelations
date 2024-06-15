'''Manage plotting of Multi-Graphs

Functions:
    draw_with_label(graph: Graph, label: str = '', pos: VertexPos = None,
                    axis: plt.Axes = None, margin_x=0.1, margin_y=0.2,
                    text_color='k'):
        Draw a graph with a title.

    count_edges(edgelist: EdgeList)->Dict[Vertex, int]:
        Count the number of identical edges in an edge list.

    edge_types(graph: Graph)->Tuple[EdgeList, EdgeList, EdgeList]:
        Identify all edge types in a given graph

    plot_multi_edge(graph: Graph, pos: VertexPos = None,
                    axis: plt.Axes = None, with_labels=True, font_weight='bold',
                    font_color='white', edge_color='k', arrow='->',
                    directed=True, rad = 0.1,  gap = 10):
        Draw a multi-graph with arcs for multiple edges and loops.

'''
#%% imports
import math
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from graph_matrix import from_adjacency_matrix


# %% Type Definitions
from typing import TypeVar, Tuple, List, Dict, Union   # pylint: disable=wrong-import-order
Graph = nx.Graph  # Networkx Graph type

Vertex = TypeVar('Vertex', str, int, float)  # Generic graph vertex (node) label
Edge = Tuple[Vertex]  # Length 2 tuple defining an edges between two vertices.
EdgeList = List[Edge]  # List of length 2 tuples as the edges of a graph.

Coordinate = np.ndarray  #  2D array of coordinates.
VertexPos = Dict[Vertex, Coordinate]  #  Plotting positions for graph vertices.

AdjacencyList = Dict[Vertex, List[Vertex]]  # An adjacency list has the form:
                                            # {VERTEX: [ADJACENT VERTICES], ...}
GraphMatrix = pd.DataFrame # Adjacency Matrix or Incident Matrix
# Composite Types
GraphDefinition = Union[EdgeList, AdjacencyList, Graph]


#%% Multi Graph
def draw_with_label(graph: Graph, label: str = '', pos: VertexPos = None,
                    axis: plt.Axes = None, margin_x=0.1, margin_y=0.2,
                    text_color='k', position='C', center_on=None):
    '''Draw a graph with a title.

    This function plots a graph and adds an optional label in the top left
    corner.

    Args:
        graph (nx.Graph): A Networkx graph
        label (str, optional): A figure heading for the table.
                               Defaults to ''.
        pos (VertexPos, optional): The vertex position coordinates to use.
                                   Defaults to None, in which case the
                                   nx.circular_layout is used.
        axis (plt.Axes, optional): The plot box in which to draw the graph.
                                   Defaults to None, in which case a new figure
                                   is generated.
        margin_x (int, optional): The x direction figure margin to use.
                                  Defaults to 0.1.
        margin_y (int, optional): The y direction figure margin to use.
                                  Defaults to 0.2.
        text_color (str, optional): The color of the text in the plot.
                                     Defaults to 'k' (black).
        position (str, optional): The placement of the graph. one of:
            ['C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            Defaults to 'C'
        center_on (Tuple[float], optional): The coordinates of the point to be
            used as the centre of the graph.

    '''
    if not pos:
        pos = nx.circular_layout(graph, scale=1.1, center=center_on, dim=2)
    if not axis:
        axis = plt.subplot()
    axis.set_aspect(aspect='equal', adjustable=None, anchor=position)

    nx.draw(graph, pos, ax=axis, with_labels=True, font_weight='bold',
            font_color=text_color)
    axis.annotate(label,
                xy=(.025, 1.0), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=20, in_layout=True, clip_on=False, color=text_color)
    axis.margins(x=margin_x, y=margin_y, tight=True)
    return axis


def count_edges(edgelist: EdgeList)->Dict[Vertex, int]:
    '''Count the number of identical edges in an edge list.

    Args:
        edgelist (EdgeList): A list of edges for a Graph or MultiGraph

    Returns:
        Dict[Vertex, int]: A dictionary with the number of edges between two
                           vertices (edge direction is ignored for the count.)
    '''
    sorted_edges = [tuple(sorted(edge)) for edge in edgelist]
    edge_count = Counter(sorted_edges)
    return edge_count


def edge_types(graph: Graph)->Tuple[EdgeList, EdgeList, EdgeList]:
    '''Identify all edge types in a given graph

    Edge types can be Single Edges, Multi-Edges or Loops.

    Args:
        graph (Graph): _description_

    Returns:
        Tuple[EdgeList, EdgeList, EdgeList]: Three different edges list:
                                             A list of Single Edges,
                                             A list of Multi-Edges,
                                             A list of Loops.
    '''
    # Drop the third value (indexer) from each edge tuple
    # The third value is an indexer for multi graphs and is not present in
    # regular graphs.
    edgelist = [(edge[0], edge[1]) for edge in graph.edges]
    edge_count = count_edges(edgelist)
    single_edges = list()
    multi_edges = list()
    loops = list()
    for edge in edgelist:
        if edge[0] == edge[1]:
            loops.append(edge)
        elif edge_count[tuple(sorted(edge))] > 1:
            multi_edges.append(edge)
        else:
            single_edges.append(edge)
    return single_edges, multi_edges, loops


def plot_multi_edge(graph: Graph, pos: VertexPos = None,
                    axis: plt.Axes = None, with_labels=True, font_weight='bold',
                    font_color='white', edge_color='k', arrow='->',
                    directed=True, rad = 0.1,  gap = 10):
    '''Draw a multi-graph with a title.

    This function plots a multi-graph with curved edges where there are
    multiple edges between the same two vertices (including loops).

    Args:
        graph (nx.Graph): A Networkx graph
        pos (VertexPos, optional): The vertex position coordinates to use.
                                   Defaults to None, in which case the
                                   nx.circular_layout is used.
        axis (plt.Axes, optional): The plot box in which to draw the graph.
                                   Defaults to None, in which case a new figure
                                   is generated.
        with_labels (bool, optional): _description_. Defaults to True.
        font_weight (str, optional): _description_. Defaults to 'bold'.
        font_color (str, optional): The color of the text in the plot.
                                     Defaults to 'white'.
        edge_color (str, optional): The color of the edges in the plot.
                                    Defaults to 'k' (black).
        arrow (str, optional): The arrow shape for directed graphs.
                               Defaults to '->'.
        directed (bool, optional): If True, arrows ar added to each edge
                                   indicating direction. Defaults to True.
        rad (float, optional): The radius increment for loops. Defaults to 0.1.
        gap (int, optional): The size of the offset from the center of the
                             vertex to teh start of the edge (accounts for the
                             size of drawn vertices). Defaults to 10.
    '''
    if not pos:
        pos = nx.circular_layout(graph, scale=1.1, center=None, dim=2)
    if not axis:
        axis = plt.subplot()
    if not directed:
        arrow='-'
        draw_arrows = False
    else:
        draw_arrows = True
    plot_params = {
        'edge_color': edge_color,
        'with_labels': with_labels,
        'font_weight': font_weight,
        'font_color': font_color
        }
    edge_props = {
        'arrowstyle': arrow,
        'color': edge_color,
        'shrinkA': 2,
        'shrinkB': gap
        }

    single_edges, multi_edges, loops = edge_types(graph)
    # Plot nodes and single edges
    nx.draw(graph, pos, ax=axis, edgelist=single_edges,
            arrows=draw_arrows, arrowstyle=arrow,
            **plot_params)

    # Draw Multi_edges
    multi_style_tpl='Arc3, rad={rad}'
    edge_count = count_edges(multi_edges)

    # Calculate curvature for the edges
    multi_index = Counter()
    for edge, count in edge_count.items():
        multi_index[edge] = 0.5 - count / 2.0

    # Plot each edge with a different curvature
    for edge in multi_edges:
        sorted_edge = tuple(sorted(edge))
        m_rad = multi_index[sorted_edge] * rad
        multi_index.update([sorted_edge])
        if edge != sorted_edge:
            m_rad = -m_rad  # Deal with different directions
        multi_style = multi_style_tpl.format(rad=m_rad)
        edge_props['connectionstyle'] = multi_style
        start = pos[edge[0]]
        end = pos[edge[1]]
        axis.annotate('', xytext=start, xy=end,
                      in_layout=True, clip_on=False,
                      arrowprops=edge_props)

    # Plot Loops
    # Position of each node in graph
    points = np.array([p for p in pos.values()])
    # Unweighted center of mass for graph
    mid_point = np.array([
        points[:,0].sum() / len(points[:,0]),
        points[:,1].sum() / len(points[:,1])
        ])

    # Count Loops
    # Calculate radius for the loops
    loop_index = Counter(loops)

    for edge in loops:
        # Loop points away from the center of mass
        e_rad = loop_index[edge] * rad
        loop_index.subtract([edge])
        vertex_pt = pos[edge[0]]
        dif = vertex_pt - mid_point
        angle = math.atan2(dif[1], dif[0])
        dx = math.cos(angle) * e_rad / 2
        dy = math.sin(angle) * e_rad / 2
        offset = np.array([dx, dy])
        loop_pt = vertex_pt + offset
        extnt = axis.get_window_extent()
        wh_ratio = extnt.width / extnt.height
        # Create loop
        loop = Arc(loop_pt, e_rad, e_rad*wh_ratio,
                   fill=False, in_layout=True, clip_on=False,
                   edgecolor=plot_params['edge_color'])
        axis.add_patch(loop)
    plt.show()

# %% Done to Here
def alternate_plot_test():
    '''Plot Edges, then Vertices, then Vertex labels'''
    adj_matrix_dict = {
        'a': [1, 2, 1],
        'b': [2, 0, 0],
        'c': [0, 2, 2]
    }
    adj_matrix = pd.DataFrame(adj_matrix_dict)
    adj_matrix.index = adj_matrix.columns
    G = from_adjacency_matrix(adj_matrix, directed=False)
    pos = nx.circular_layout(G, scale=1.1, center=None, dim=2)

    single_edges, multi_edges, loops = edge_types(G)

    #draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='#1f78b4', node_shape='o', alpha=None, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, margins=None)
    #draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True)
    #draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=300, nodelist=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0)
    single_edges = [('a', 'c', 0), ('c', 'b', 0)]

    node_params = {
        'node_shape': 'o',  # one of 'so^>v<dph8'.
        'node_color': '#1f78b4',
        'node_size': 300,
        'label': 'Test'
        }
    label_params = {
        'font_size': 12,
        'font_weight': 'bold',
        'font_color': 'white',
        'font_family': 'sans-serif',
        'clip_on': False
        }

    edge_params = {
        'arrowstyle': '-',
        #'color': 'k',
        #'shrinkA': 2,
        #'shrinkB': 10

        'width': 1.0,
        'edge_color': 'w',
        'style': 'solid',
        'alpha': None,
        'arrowsize': 10,
        'edge_cmap': None,
        'edge_vmin': None,
        'edge_vmax': None,
        'arrows': None,
        'label': None,
        'node_size': 300,
        'nodelist': None,
        'node_shape': 'o',
        'connectionstyle': 'arc3',
        'min_source_margin': 0,
        'min_target_margin': 0
        }
    axis = plt.subplot()
    nx.draw_networkx_nodes(G, pos, ax=axis, **node_params)

    nx.draw_networkx_labels(G, pos, ax=axis, **label_params)

    nx.draw_networkx_edges(G, pos, ax=axis, edgelist=single_edges, **edge_params)
    # TODO add additional plotting commands for loops and multi-edges
