'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import List

# Standard Libraries
from itertools import combinations

# Shared Packages
import pandas as pd
import networkx as nx

# Local packages
from structure_slice import merge_contours
from types_and_classes import SliceIndexType, StructurePairType
from types_and_classes import RegionNode, RegionGraph, RegionIndexType
from utilities import calculate_new_slice_index, interpolate_polygon


def generate_slice_neighbours(slice_table: pd.DataFrame) -> pd.DataFrame:
    '''Generate a table of SliceNeighbours from the index of a slice_table.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndex as the index.

    Returns:
        pd.Series: A table with SliceIndex as the index and three columns:
            'this_slice', 'previous_slice', 'next_slice'.
    '''
    neighbours = slice_table.index.to_frame()
    starting_slice = neighbours.iat[0, 0]
    ending_slice = neighbours.iat[-1, 0]
    this_slice = neighbours['Slice Index']
    neighbours['previous'] = this_slice.shift(1, fill_value=starting_slice)
    neighbours['next'] =this_slice.shift(-1, fill_value=ending_slice)
    neighbours.columns = ['this_slice', 'previous_slice', 'next_slice']
    return neighbours


def make_slice_table(slice_data: pd.Series, ignore_errors=False)->pd.DataFrame:
    '''Merge contour data to build a table of StructureSlice data

    The table index is SliceIndex, sorted from smallest to largest. The table
    columns are ROI_Num.
    Individual structure contours with the same ROI_Num and SliceIndex are
    merged into a single StructureSlice instance

    Args:
        slice_data (pd.Series): A series of individual structure contours.
        ignore_errors (bool, optional): If True, overlapping contours are
            allowed and combined to generate a larger solid region.
            Defaults to False.

    Returns:
        pd.DataFrame: A table of StructureSlice data with an SliceIndex and
            ROI_Num as the index and columns respectively.
    '''
    sorted_data = slice_data.sort_index(level=['Slice Index', 'ROI Num']).copy()
    sorted_data['Area'] = sorted_data.map(lambda x: x.area)
    structure_group = sorted_data.groupby(level=['Slice Index', 'ROI Num'])
    structure_data = structure_group.apply(merge_contours,
                                           ignore_errors=ignore_errors)
    slice_table = structure_data.unstack('ROI Num')

    # Generate slice neighbours and set them for each StructureSlice
    slice_neighbours = generate_slice_neighbours(slice_table)
    for slice_index, row in slice_table.iterrows():
        previous_slice = slice_neighbours.at[slice_index, 'previous_slice']
        next_slice = slice_neighbours.at[slice_index, 'next_slice']
        for structure_slice in row.dropna():
            structure_slice.set_slice_neighbours(previous_slice, next_slice)
    return slice_table


def generate_interpolated_boundaries(graph: RegionGraph) -> None:
    '''Identify boundary nodes in the graph and generate interpolated boundary slices.

    Args:
        graph (RegionGraph): The graph containing the nodes.
    '''
    def identify_boundaries(graph: RegionGraph) -> List[RegionIndexType]:
        '''Identify boundary nodes in the graph.

        Args:
            graph (RegionGraph): The graph containing the nodes.

        Returns:
            List[Tuple]: A list of nodes that are boundaries.
        '''
        boundaries = [node for node, degree in graph.degree() if degree < 2]
        return boundaries

    boundaries = identify_boundaries(graph)
    for boundary_node in boundaries:
        node_data = graph.nodes[boundary_node]
        polygon = node_data['polygon']
        slice_neighbours = node_data['slice_neighbours']
        this_slice = slice_neighbours.this_slice
        if slice_neighbours.previous_slice != this_slice:
            other_slice = slice_neighbours.previous_slice
        else:
            other_slice = slice_neighbours.next_slice
        slice_pair = (this_slice, other_slice)
        new_slice = calculate_new_slice_index(slice_pair)
        distance=abs(this_slice - other_slice)
        node_data = RegionNode(roi=node_data['roi'], slice_index=new_slice,
                               is_hole=node_data['is_hole'], is_boundary=True,
                               is_interpolated=True, is_empty=False,
                               slice_neighbours=slice_neighbours)
        intp_poly = interpolate_polygon(slice_pair, polygon)
        node_data.polygon = intp_poly
        node_data.add_node(graph)
        intp_node_label = node_data.node_label()
        graph.add_edge(boundary_node, intp_node_label, weight=distance)


def generate_region_graph(slice_table: pd.DataFrame) -> RegionGraph:
    '''Generate a region graph from a slice table.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndex as the index.

    Returns:
        nx.Graph: A graph representing the regions in the slice table.
    '''
    def create_edges(graph: nx.Graph) -> None:
        '''Create edges between nodes in the graph based on specified criteria.

        Args:
            graph (nx.Graph): The graph containing the nodes.
        '''
        for node1, node2 in combinations(graph.nodes(data=True), 2):
            roi1, slice_index1, _ = node1[0]
            roi2, slice_index2, _ = node2[0]
            if roi1 != roi2:
                continue
            if not node1[1]['slice_neighbours'].is_neighbour(slice_index2):
                continue
            if node1[1]['is_hole'] != node2[1]['is_hole']:
                continue
            if node1[1]['polygon'].intersects(node2[1]['polygon']):
                weight = abs(slice_index1 - slice_index2)
                graph.add_edge(node1[0], node2[0], weight=weight)

    graph = nx.Graph()
    for _, row in slice_table.iterrows():
        for structure_slice in row.dropna():
            structure_slice.extract_regions(graph)
    create_edges(graph)
    generate_interpolated_boundaries(graph)
    return graph



#%% Select Slices and neighbours
def select_slices(slice_table: pd.DataFrame,
                  selected_roi: StructurePairType) -> pd.DataFrame:
    '''Select the slices that have either of the structures.

    Select all slices that have either of the structures.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndex as the index, ROI_Num for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Num to select.

    Returns:
        pd.DataFrame:  A subset of slice_table with the two selected_roi as the
            columns and the range of slices hat have either of the structures as
            the index.
    '''
    start = SliceIndexType(slice_table[selected_roi].first_valid_index())
    end = SliceIndexType(slice_table[selected_roi].last_valid_index())
    structure_slices = slice_table.loc[start:end, selected_roi]
    return structure_slices
