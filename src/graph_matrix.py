
# %load graph_matrix.py
'''Functions related to graph matrices.

Classes:
    DictOfList(dict) A dictionary where each value is a list.
    DictOfDict(dict) A dictionary where each value is a dictionary.

Functions:
    adjacency_matrix(graph_input: GraphDefinition)->GraphMatrix:
        Build an adjacency matrix from a graph, adjacency list, or edge list.

    from_adjacency_matrix(adj_matrix: GraphMatrix, directed=True)->Graph:
        Build a graph from an adjacency matrix.

    incident_matrix(graph_input: GraphDefinition, multi=False)->GraphMatrix:
        Build an incident matrix from a graph, adjacency list, or edge list.

    complete_incidence(vertex_list: List[str] = None,
                       n: int = None)->GraphMatrix:
        Generate an incidence matrix for a complete graph.

    cycle_incidence(vertex_list: List[str] = None, n: int = None)->GraphMatrix:
        Generate an incidence matrix for a cycle graph.

    wheel_incidence(vertex_list: List[str] = None, n: int = None)->GraphMatrix:
        Generate an incidence matrix for a wheel graph.

    complete_bipartite_incidence(vertex_list: List[List[str]] = None,
                                 m: int = None,
                                 n: int = None)->GraphMatrix:
        Generate an incidence matrix for a complete bipartite graph.

    get_text_dim(text: str, fig: plt.Figure = None,
                 **font_properties)->Tuple[float, float]:
        Determine text render dimensions in a figure.

    draw_table(df: GraphMatrix, label: str = '', axis: plt.Axes = None,
               fit: bool = True, pad: float = 0.1, cell_color: str = 'c'):
        Draw an Adjacency or Incidence Matrix as a figure element.
'''
#%% Imports
from typing import List, Dict, Tuple, Union, TypeVar

from itertools import product, combinations_with_replacement
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# %% Type Definitions
Vertex = TypeVar('Vertex')  # Generic graph vertex (node) label
Edge = Tuple[Vertex]  # Length 2 tuple defining an edges between two vertices.
EdgeList = List[Edge]  # List of length 2 tuples as the edges of a graph.

Graph = nx.Graph  # networkx Graph type

AdjacencyList = Dict[Vertex, List[Vertex]]  # An adjacency list has the form:
                                            # {VERTEX: [ADJACENT VERTICES], ...}
GraphMatrix = pd.DataFrame # Adjacency Matrix or Incident Matrix

# Composite Types
GraphDefinition = Union[EdgeList, AdjacencyList, Graph]


# %% Type definitions
class DictOfList(dict):
    '''A dictionary where each value is a list.

    Requesting a non-existing key returns an empty list.

    Methods:
        add_item(idx, val): Appends *val* to the list in *idx* item.
    '''
    def __missing__(self, key):
        return list()

    def add_item(self, idx, val):
        '''Appends *val* to the list in *idx* item.

        Args:
            idx (Any): The dictionary key referencing the desired list
            val (Any): The value to be appended to the list
        '''
        lst = self[idx]
        lst.append(val)
        self[idx] = lst


class DictOfDict(dict):
    '''A dictionary where each value is a dictionary.

    Requesting a non-existing key returns an empty dictionary.

    Methods:
        add_item(idx1, idx2, val): Adds *val* as an item with key idx2 to the
                                   dictionary in *idx1* item.
    '''
    def __missing__(self, key):
        return dict()

    def add_item(self, idx1, idx2, val):
        '''Adds a value to a sub-dictionary.

        Adds *val* as an item with key *idx2* to the sub-dictionary located in
        *idx1* value of the top dictionary.

        Args:
            idx1 (Any): The top-level dictionary key.
            idx2 (Any): The sub-dictionary key.
            val (Any): The value to be appended to the sub-dictionary
        '''
        sdt = self[idx1]
        sdt[idx2] = val
        self[idx1] = sdt


#%% Adjacency
def adjacency_matrix(graph_input: GraphDefinition)->GraphMatrix:
    '''Build an adjacency matrix from a graph, adjacency list, or edge list.

    An adjacency list has the form:
        {VERTEX: [ADJACENT VERTICES], ...}
    An edge list has the form:
        [(Vertex1, Vertex2), ...]

    Args:
        graph_input (GraphDefinition): A graph definition in the form of an
                                       adjacency list, edge list, or Graph

    Returns:
        GraphMatrix: A table with the graph vertices as both column and index
                     labels. Table values indicate edges between vertices.
    '''
    def item_counts(item_list):
        items = set(item_list)
        if len(items) == len(item_list):
            item_count = {item: 1 for item in item_list}
        else:
            raise ValueError('There are repeated items in the list')
        return item_count

    if hasattr(graph_input, 'items'):
        # graph_input is an adjacency list.
        adjacency_list = graph_input
    elif hasattr(graph_input, 'edges'):
        # graph_input is a graph Convert a graph to an adjacency list.
        adj_dict = nx.to_dict_of_dicts(graph_input)
        adjacency_list = {key: list(value) for key, value in adj_dict.items()}
    else:
        # graph_input is an edge list; convert it to an adjacency list.
        adjacency_list = DictOfList()
        for vtx_pair in graph_input:
            adjacency_list.add_item(vtx_pair[0], vtx_pair[1])
    adjacents = {node: item_counts(adjacents)
                 for node, adjacents in adjacency_list.items()}
    matrix = pd.DataFrame(adjacents).T
    matrix[matrix == 0] = 1
    matrix.fillna(0,inplace=True)
    matrix.sort_index(axis=0, inplace=True)
    matrix.sort_index(axis=1, inplace=True)
    matrix = matrix.astype('Int32')
    return matrix


def from_adjacency_matrix(adj_matrix: GraphMatrix, directed=True)->Graph:
    '''Build a graph from an adjacency matrix.

    Args:
        adj_matrix (GraphMatrix): A table with the graph vertices as both column
                                  and index labels. Table values indicate edges
                                  between vertices.
        directed (bool, optional): If True return a directed graph.
                                   Defaults to True.

    Returns:
        Graph: The resulting networkx Graph.
    '''
    nodes = list(adj_matrix.columns)
    edge_list = list()
    if directed:
        edge_selection = product(nodes, repeat=2)
    else:
        edge_selection = combinations_with_replacement(nodes, 2)

    # Usi a Multi graph if multiple edges found or if loops are found
    is_multi = False
    for edge in edge_selection:
        count = adj_matrix.at[edge[0], edge[1]]
        if count > 1:  # Multiple edges
            is_multi = True
        if (count > 0) &  (edge[0] == edge[1]):  #  Loops
            is_multi = True
        edge_list.extend([edge]*count)
    # Set graph type
    if directed:
        if is_multi:
            graph_constructor = nx.MultiDiGraph
        else:
            graph_constructor = nx.DiGraph
    else:
        if is_multi:
            graph_constructor = nx.MultiGraph
        else:
            graph_constructor = nx.Graph
    graph = graph_constructor(edge_list)
    return graph


#%% Incident Matrix
def incident_matrix(graph_input: GraphDefinition, multi=False)->GraphMatrix:
    '''Build an incident matrix from a graph, adjacency list, or edge list.

    An adjacency list has the form:
        {VERTEX: [ADJACENT VERTICES], ...}
    An edge list has the form:
        [(Vertex1, Vertex2), ...]

    Args:
        graph_input (GraphDefinition): A graph definition in the form of an
                                       adjacency list, edge list, or Graph
        multi (bool): If True, the graph is a MultiGraph (multiple identical
                      edges). If False, duplicate edges will be removed.
                      Default is False.

    Returns:
        GraphMatrix: A table with the graph vertices as the index labels and
                     edges as the columns.  Table values are zeros and ones
                     indicating the two end points of each edge.
    '''
    if hasattr(graph_input, 'edges'):
        # graph_input is a Graph; generate an edge list from it.
        edge_list = list(graph_input.edges)
    elif hasattr(graph_input, 'items'):
        # graph_input is an Adjacency list; convert it to an edge list.
        # Note, the method used here will generate duplicate edges for
        # non-directional graphs. these are removed later with the
        # drop_duplicates function.
        edge_list = []
        for node, adjacents in graph_input.items():
            for adj in adjacents:
                edge_list.append((node, adj))
    else:
        # graph_input is already an edge list.
        edge_list = graph_input
    edge_dict = dict()
    for idx, edg in enumerate(edge_list):
        edge_name = 'e' + str(idx)
        edg_map = {edg[0]: 1, edg[1]: 1}
        edge_dict[edge_name] = edg_map
    matrix = pd.DataFrame(edge_dict)
    if not multi:
        matrix = matrix.T.drop_duplicates().T
    matrix.fillna(0, inplace=True)
    matrix = matrix.astype(int)
    return matrix


#%% Incident Matrix for a Complete Graph
def complete_incidence(vertex_list: List[str] = None,
                       n: int = None)->GraphMatrix:
    # pylint: disable=invalid-name
    '''Generate an incidence matrix for a complete graph.

    Generate an incidence matrix for a complete graph $K_n$. The resulting matrix
    is returned as a DataFrame with graph vertices as the index and edges as the
    columns.  Table values are either zero or one; with one indicating the two
    end points of each edge. The complete graph is defined either by its size
    *n* or by the names of its vertices *vertex_list*. If vertex_list is
    supplied, *n* will be ignored and the size of the graph will be determined
    by the number of vertices in the list.

    Args:
        vertex_list (List[str], optional): The names of the vertices for the
            complete graph. Defaults to None.
        n (int, optional): The size (number of vertices) of the complete graph.
            Ignored if *vertex_list* is given. Defaults to None.

    Returns:
        GraphMatrix: A table with the graph vertices as the index labels and
                     edges as the columns.  Table values are zeros and ones
                     indicating the two end points of each edge.
    '''
    if vertex_list:
        n = len(vertex_list)
    else:
        vertex_list = [f'$v_{{{i + 1}}}$' for i in range(n)]

    # Dictionary of edges for the complete graph.
    edge_dict = {}
    # Loop through all starting vertices. (All except last vertex).
    for j in range(n-1):
        # Loop through all ending vertices. (All except last vertex).
        for k in range(j + 1,n):
            # Calculate the edge number
            x = j * (n - 1) + k - sum(range(j + 1))
            # Define the edge
            edge_dict[f'$e_{{{x}}}$'] = {vertex_list[j]: 1, vertex_list[k]: 1}

    # Build the DataFrame table from the edge dictionary
    inc_matrix = pd.DataFrame(edge_dict)
    inc_matrix.fillna(0, inplace=True)
    inc_matrix = inc_matrix.astype(int)
    return inc_matrix


def cycle_incidence(vertex_list: List[str] = None, n: int = None)->GraphMatrix:
    # pylint: disable=invalid-name
    '''Generate an incidence matrix for a cycle graph.

    Generate an incidence matrix for a cycle graph C_n. The resulting matrix
    is returned as a DataFrame with graph vertices as the index and edges as the
    columns.  Table values are either zero or one; with one indicating the two
    end points of each edge. The cycle graph is defined either by its size
    *n* or by the names of its vertices *vertex_list*. If vertex_list is
    supplied, *n* will be ignored and the size of the graph will be determined
    by the number of vertices in the list.

    Args:
        vertex_list (List[str], optional): The names of the vertices for the
            cycle graph in the order that they are connected. Defaults to None.
        n (int, optional): The size (number of vertices) of the cycle graph.
            Ignored if *vertex_list* is given. Defaults to None.

    Returns:
        GraphMatrix: A table with the graph vertices as the index labels and
                     edges as the columns.  Table values are zeros and ones
                     indicating the two end points of each edge.
    '''
    if vertex_list:
        n = len(vertex_list)
    else:
        vertex_list = [f'$v_{{{i + 1}}}$' for i in range(n)]

    # Dictionary of edges for the complete graph.
    edge_dict = {}
    # Iterate through all vertices.
    for j in range(1, n):
        # Define the edge from the current vertex to the next vertex
        edge_dict[f'e{j}'] = {vertex_list[j - 1]: 1, vertex_list[j]: 1}
    # Add the edge from the last vertex to the first vertex
    edge_dict[f'e{n}'] = {vertex_list[n - 1]: 1, vertex_list[0]: 1}

    # Build the DataFrame table from the edge dictionary
    inc_matrix = pd.DataFrame(edge_dict)
    inc_matrix.fillna(0, inplace=True)
    inc_matrix = inc_matrix.astype(int)
    return inc_matrix


def wheel_incidence(vertex_list: List[str] = None, n: int = None)->GraphMatrix:
    # pylint: disable=invalid-name
    '''Generate an incidence matrix for a wheel graph.

    Generate an incidence matrix for a wheel graph W_n. The resulting matrix
    is returned as a DataFrame with graph vertices as the index and edges as the
    columns.  Table values are either zero or one; with one indicating the two
    end points of each edge. The wheel graph is defined either by its size
    *n* or by the names of its vertices *vertex_list*. If vertex_list is
    supplied, *n* will be ignored and the size of the graph will be determined
    by the number of vertices in the list.

    Args:
        vertex_list (List[str], optional): The names of the vertices for the
            wheel graph, with the first vertex being the centre and the
            remainder, in the order that they are connected to each other.
            Defaults to None.
        n (int, optional): The size (number of vertices on the rim) of the wheel
            graph. Ignored if *vertex_list* is given. Defaults to None.

    Returns:
        GraphMatrix: A table with the graph vertices as the index labels and
                     edges as the columns.  Table values are zeros and ones
                     indicating the two end points of each edge.
    '''
    if vertex_list:
        n = len(vertex_list) - 1
    else:
        vertex_list = [f'$v_{{{i}}}$' for i in range(n + 1)]

    # Dictionary of edges for the complete graph.
    edge_dict = {}
    for j in range(1,2*n+1):
        if j < n:
            # Define the edge from the current vertex to the next vertex
            edge_dict[f'$e_{{{j}}}$'] = {vertex_list[j]: 1, vertex_list[j+1]: 1}
        elif j == n:
            # Add the edge from the last vertex to the first vertex
            edge_dict[f'$e_{{{j}}}$'] = {vertex_list[j]: 1, vertex_list[1]: 1}
        else:
            # Define the edge from the current vertex to the centre vertex
            edge_dict[f'$e_{{{j}}}$'] = {vertex_list[j-n]: 1, vertex_list[0]: 1}

    # Build the DataFrame table from the edge dictionary
    inc_matrix = pd.DataFrame(edge_dict)
    inc_matrix.fillna(0, inplace=True)
    inc_matrix = inc_matrix.astype(int)
    return inc_matrix


def complete_bipartite_incidence(vertex_list: List[List[str]] = None,
                                 m: int = None,
                                 n: int = None)->GraphMatrix:
    # pylint: disable=invalid-name
    '''Generate an incidence matrix for a complete bipartite graph.

    Generate an incidence matrix for a complete bipartite graph $K_{m,n}$.
    The resulting matrix is returned as a DataFrame with graph vertices as the
    index and edges as the columns.  Table values are either zero or one; with
    one indicating the two end points of each edge. The wheel graph is defined
    either by its size *n* or by the names of its vertices *vertex_list*. If
    vertex_list is supplied, *n* will be ignored and the size of the graph will
    be determined by the number of vertices in the list.

    Args:
        vertex_list (List[List[str]], optional): The names of the vertices for
            the cycle graph in the order that they are connected. Defaults to None.
        m (int, optional): The number of vertices in the first partition of the
            complete bipartite graph. Ignored if *vertex_list* is given.
            Defaults to None.
        n (int, optional): The number of vertices in the second partition of the
            complete bipartite graph. Ignored if *vertex_list* is given.
            Defaults to None.
    Returns:
        GraphMatrix: A table with the graph vertices as the index labels and
                     edges as the columns.  Table values are zeros and ones
                     indicating the two end points of each edge.
    '''
    if vertex_list:
        u_vrtx_list = vertex_list[0]
        m = len(u_vrtx_list)

        v_vrtx_list = vertex_list[1]
        n = len(v_vrtx_list)
    else:
        u_vrtx_list = [f'$u_{{{i}}}$' for i in range(1, m + 1)]
        v_vrtx_list = [f'$v_{{{i}}}$' for i in range(1, n + 1)]

    # Dictionary of edges for the complete graph.
    edge_dict = {}
    # Iterate through all vertices in the first partition of the graph.
    for j in range(m):
        # Iterate through all vertices in the second partition of the graph.
        for k in range(n):
            # Calculate the edge number
            x = n * (j) + k + 1
            # Define the edge from vertex the in the first partition to the
            # vertex in the second partition.
            edge_dict[f'$e_{{{x}}}$'] = {u_vrtx_list[j]: 1, v_vrtx_list[k]: 1}

    # Build the DataFrame table from the edge dictionary
    inc_matrix = pd.DataFrame(edge_dict)
    inc_matrix.fillna(0, inplace=True)
    inc_matrix = inc_matrix.astype(int)
    return inc_matrix


#%% Display
def get_text_dim(text: str, fig: plt.Figure = None,
                 **font_properties)->Tuple[float, float]:
    '''Determine text render dimensions in a figure.

    _extended_summary_

    Args:
        text (str): The text to be measured
        fig (plt.Figure, optional): The figure. Defaults to None.
        **font_properties: Optional keyword arguments. If not supplied, figure
            defaults are used. Keyword arguments can be any of:
                family (str): The font family e.g:
                    - 'serif'
                    - 'sans-serif'
                    - 'monospace'
                font (str): The font name.
                size (float, str): The font size or relative size. If a string,
                    one of:
                        - 'xx-small'
                        - 'x-small'
                        - 'small'
                        - 'medium'
                        - 'large'
                        - 'x-large'
                        - 'xx-large'
                stretch (float, str): The font width scale. Either an number
                    between 0 and 1000, or one of:
                        - 'ultra-condensed'
                        - 'extra-condensed'
                        - 'condensed'
                        - 'semi-condensed'
                        - 'normal'
                        - 'semi-expanded'
                        - 'expanded'
                        - 'extra-expanded'
                        - 'ultra-expanded'
                style (str): One of
                        - 'normal'
                        - 'italic'
                        - 'oblique'
                variant (str): One of
                        - 'normal'
                        - 'small-caps'
                weight (float, str): The font width scale. Either an number
                    between 0 and 1000, or one of:
                        - 'ultralight'
                        - 'light'
                        - 'normal'
                        - 'regular'
                        - 'book'
                        - 'medium'
                        - 'roman'
                        - 'semibold'
                        - 'demibold'
                        - 'demi'
                        - 'bold'
                        - 'heavy'
                        - 'extra bold'
                        - 'black'
    Returns:
        Tuple(float, float): The width and height of the rendered text.
    '''
    if not fig:
        fig = plt.figure()
        close_fig=True
    else:
        close_fig=False
    renderer = fig.canvas.get_renderer()
    t_plt = plt.text(0.5, 0.5, text, **font_properties)
    text_box = t_plt.get_window_extent(renderer=renderer)
    width = text_box.width
    height = text_box.height

    if close_fig:
        plt.close(fig)
    else:
        t_plt.remove()
        fig.canvas.draw()

    return(width, height)

def draw_table(matrix: GraphMatrix, label: str = '', axis: plt.Axes = None,
               fit: bool = True, pad: float = 0.1, cell_color: str = 'c',
               header_color: str = 'b', label_color='k', cell_text_color='w',
               position='center left'):
    '''Draw an Adjacency or Incidence Matrix as a figure element.

    Allows a graph matrix to be plotted alongside its diagram.

    Args:
        matrix (GraphMatrix): The Adjacency or Incidence Matrix.
        label (str, optional): A figure heading for the table.
                               Defaults to ''.
        axis (plt.Axes, optional): The plot box in which to place the table.
                                   Defaults to None, in which case a new figure
                                   is generated.
        fit (bool, optional): Fit the column widths to the data.
                              Defaults to True.
        pad (float, optional): Padding around the table. Defaults to 0.1.
        cell_color (str, optional): The background color for the table cells.
                                    Defaults to 'c' (cyan).
        header_color (str, optional): The background color for the table cells.
                                    Defaults to 'b' (blue).
        label_color (str, optional): The text color for the label.
                                    Defaults to 'k' (black).
        cell_text_color (str, optional): The text color for the cell text.
                                    Defaults to 'w' (white).
        position (str, optional): The location of the matrix in the subplot
                            Options are:
                                'best', 'left', 'right',
                                'bottom', 'bottom left', 'bottom right',
                                'center', 'center left', 'center right',
                                'lower center', 'lower left', 'lower right',
                                'top', 'top left', 'top right',
                                'upper center', 'upper left', 'upper right'
                            Defaults to 'center left'.
    '''
    if not axis:
        axis = plt.subplot()
    axis.set_axis_off()
    axis.annotate(label,
                xy=(.025, 1.0), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=20, in_layout=True, clip_on=False, color=label_color)
    table_dict = matrix.to_dict(orient='split')
    col_size=len(table_dict['index'])
    row_size=len(table_dict['columns'])
    table = axis.table(cellText=table_dict['data'],
              rowLabels=table_dict['index'], colLabels=table_dict['columns'],
              cellColours=[[cell_color]*row_size]*col_size, cellLoc='center',
              #colWidths=[1]*size,
              rowColours=[header_color]*col_size, rowLoc='center',
              colColours=[header_color]*row_size, colLoc='center',
              loc=position,
              bbox=None, edges='closed',
              in_layout=True, clip_on=False)

    table.set_fontsize(12)
    table.AXESPAD = pad
    if fit:
        for n in range(col_size):
            table.auto_set_column_width(n)
    else:
        table.auto_set_font_size(False)

    for cell in table.get_celld().values():
        cell.set_text_props(color=cell_text_color)
    return table
