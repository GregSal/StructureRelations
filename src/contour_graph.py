'''Functions to build a contour graph from a contour table.
The contour graph is a undirected graph where each node represents a contour
and edges represent connect matched contours on the next and previous slice.
'''
from typing import List, Tuple
from collections import defaultdict

import pandas as pd
import networkx as nx

from contours import SliceSequence, Contour, ContourMatch, interpolate_polygon
from types_and_classes import ROI_Type, SliceIndexType, ContourGraph


# %% Contour Graph Construction Functions
def build_contours(contour_table, roi)-> defaultdict[SliceIndexType, List[Contour]]:
    '''Build contours for a given ROI from the contour table.

    This function filters the contour table for the specified ROI and creates
    Contour objects for each slice. It organizes the contours by slice
    in a dictionary.

    Args:
        contour_table (pd.DataFrame): The contour table containing contour data.
        roi (int): The ROI number to filter contours.

    Returns:
        dict: A dictionary where keys are slice indices and values are lists of
              Contour objects for that slice. Each list of contours is sorted
              by area in order of descending area.
    '''
    # Filter the contour table for the specified ROI
    contour_set = contour_table[contour_table.ROI == roi]
    # Create a dictionary to hold contours by slice
    contour_by_slice = defaultdict(list)
    # Set the index to 'Slice' for easier access
    contour_set.set_index('Slice', inplace=True)

    # Iterate over each slice and create Contour objects
    for slice_index, contour in contour_set.Polygon.items():
        contours_on_slice = contour_by_slice[slice_index]
        new_contour = Contour(roi, slice_index, contour, contours_on_slice)
        contour_by_slice[slice_index].append(new_contour)
    # Sort the contours on each slice by area in order of descending area
    for slice_index in contour_by_slice:
        contour_by_slice[slice_index].sort(key=lambda c: c.area(), reverse=True)
    return contour_by_slice


def build_contour_lookup(contour_graph: ContourGraph) -> pd.DataFrame:
    '''Build a lookup table for contours.

    This function creates a DataFrame that serves as a lookup table for contours.
    The hole type is categorized into 'Open', 'Closed', 'Unknown', and 'None'.
    The DataFrame is sorted by slice index and contour index.

    Args:
        contour_graph (ContourGraph): A Contour Graph object containing contour
            information. Each node in the graph should represent a contour and
            should have a 'contour' attribute that is an instance of the
            Contour class.

    Returns:
        pd.DataFrame: A DataFrame containing the contour lookup table.
            The DataFrame includes the following columns:
                - ROI,
                - SliceIndex,
                - HoleType,
                - Interpolated,
                - Boundary,
                - ContourIndex,
                - RegionIndex, and
                - Label.
    '''
    lookup_list = []
    for _, data in contour_graph.nodes(data=True):
        contour = data['contour']
        lookup_list.append({
            'ROI': contour.roi,
            'SliceIndex': contour.slice_index,
            'HoleType': contour.hole_type,
            'Interpolated': contour.is_interpolated,
            'Boundary': contour.is_boundary,
            'ContourIndex': contour.contour_index,
            'RegionIndex': contour.region_index,
            'Label': contour.index
        })
    contour_lookup = pd.DataFrame(lookup_list)
    contour_lookup.HoleType = contour_lookup.HoleType.astype('category')
    contour_lookup.HoleType.cat.set_categories(['Open', 'Closed',
                                                'Unknown', 'None'])
    contour_lookup.sort_values(by=['SliceIndex', 'ContourIndex'], inplace=True)
    return contour_lookup


def add_graph_edges(contour_graph: ContourGraph,
                    slice_sequence: SliceSequence) -> ContourGraph:
    '''Add edges to link Contour Neighbours.

    Edges are added between contours that are neighbours in the slice sequence
    and have the same hole type.  Matching contours are identified by the
    intersection of the contours' convex hulls.

    Args:
        contour_graph (ContourGraph): The graph representation of the contours.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.

    Returns:
        nx.Graph: The updated contour graph with edges added.
    '''
    # Create the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Iterate through each contour reference in the lookup table
    for contour_ref in contour_lookup.itertuples(index=False):
        # Get the contour reference
        this_slice = contour_ref.SliceIndex
        this_label = contour_ref.Label
        hole_type = contour_ref.HoleType
        this_contour = contour_graph.nodes(data=True)[this_label]['contour']
        # Identify the next slice in the sequence
        next_slice = slice_sequence.get_neighbors(this_slice).next_slice
        neighbour_idx = ((contour_lookup.SliceIndex == next_slice) &
                        (contour_lookup.HoleType == hole_type))
        # match this contour to the contours on the next slice
        for neighbour in contour_lookup.loc[neighbour_idx, 'Label']:
            neighbour_contour = contour_graph.nodes(data=True)[neighbour]['contour']
            if this_contour.hull.intersects(neighbour_contour.hull):
                contour_match = ContourMatch(this_contour, neighbour_contour)
                contour_graph.add_edge(this_label, neighbour, match=contour_match)
    return contour_graph


def add_boundary_contours(contour_graph: ContourGraph,
                          slice_sequence: SliceSequence) -> Tuple[ContourGraph,
                                                                  SliceSequence]:
    '''Add interpolated boundary contours to the graph.

    Args:
        contour_graph (ContourGraph): The graph representation of the contours.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.

    Returns:
        tuple: A tuple containing the updated contour graph and slice sequence:
            contour_graph (ContourGraph): The updated graph representation of
                the contours with interpolated boundary contours added.
            slice_sequence (SliceSequence): The updated slice sequence object
                with the interpolated slice indices added.
    '''
    # Select all nodes with only one edge (degree=1)
    boundary_nodes = [node for node, degree in contour_graph.degree()
                      if degree == 1]
    for original_boundary in boundary_nodes:
        # Get the contour and its slice index
        contour = contour_graph.nodes[original_boundary]['contour']
        this_slice = contour.slice_index
        # Determine the neighbor slice not linked with an edge
        neighbors = slice_sequence.get_neighbors(this_slice)
        # Get the slice index of the neighbouring contours.
        # This is the starting point for the neighbours of the interpolated slice
        slice_ref = {'PreviousSlice': neighbors.previous_slice,
                     'NextSlice': neighbors.next_slice}
        # Determine the slice index that is a neighbour of the original
        # boundary contour, so that the other slice index can be used to
        # interpolate the slice index of the interpolated contour.
        neighbouring_nodes = contour_graph.adj[original_boundary].keys()
        # Because degree=1, there should only be one neighbouring node.
        neighbour_slice = [nbr[1] for nbr in neighbouring_nodes][0]
        # Get slice index to use for interpolating (slice_beyond) and the
        # neighbouring slice references for the interpolated slice.
        if neighbors.previous_slice == neighbour_slice:
            # The next slice is the neighbour for interpolation
            slice_beyond = neighbors.next_slice
            # If the next slice is not set, use the gap to estimate the next
            # slice index.
            if pd.isna(slice_beyond):
                slice_beyond = this_slice + neighbors.gap(absolute=False)
            # The current boundary slice is the previous slice for the
            # interpolated slice.
            slice_ref['PreviousSlice'] = this_slice
        else:
            # The previous slice is the neighbour for interpolation
            slice_beyond = neighbors.previous_slice
            # If the next slice is not set, use the gap to estimate the previous
            # slice index.
            if pd.isna(slice_beyond):
                slice_beyond = this_slice - neighbors.gap(absolute=False)
            # The current boundary slice is the next slice for the
            # interpolated slice.
            slice_ref['NextSlice'] = this_slice
        # Calculate the interpolated slice index
        interpolated_slice = (this_slice + slice_beyond) / 2
        slice_ref['ThisSlice'] = interpolated_slice
        slice_ref['Original'] = False
        # Generate the interpolated boundary contour
        interpolated_polygon = interpolate_polygon([this_slice, slice_beyond],
                                                   contour.polygon)
        interpolated_contour = Contour(
            roi=contour.roi,
            slice_index=interpolated_slice,
            polygon=interpolated_polygon,
            contours=[]
            )
        interpolated_contour.is_interpolated = True
        interpolated_contour.is_boundary = True
        interpolated_contour.is_hole = contour.is_hole
        interpolated_contour.hole_type = contour.hole_type
        # Add the interpolated slice index to the slice sequence
        slice_sequence.add_slice(**slice_ref)
        # Add the interpolated contour to the graph
        interpolated_label = interpolated_contour.index
        contour_graph.add_node(interpolated_label, contour=interpolated_contour)
        # Add a ContourMatch edge between the original and interpolated contours
        contour_match = ContourMatch(contour, interpolated_contour)
        contour_graph.add_edge(original_boundary, interpolated_label,
                               match=contour_match)
    return contour_graph, slice_sequence


def set_enclosed_regions(contour_graph: ContourGraph) -> List[ContourGraph]:
    '''Create EnclosedRegion SubGraphs and assign RegionIndexes to the contours.

    Args:
        contour_graph (ContourGraph): The graph representation of the contours.

    Returns:
        List[ContourGraph]: A list of SubGraphs, each representing an enclosed
            region.
    '''
    region_counter = 0
    region_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Find connected contours in the contour graph.
    for region_nodes in nx.connected_components(contour_graph):
        # Create a SubGraph for the connected component.
        enclosed_region = contour_graph.subgraph(region_nodes)
        # Assign a unique RegionIndex to each contour in the SubGraph.
        # This index is composed of the ROI number, an uppercase letter and a
        # suffix number (if needed).)
        # Extract the ROI from the first node in the SubGraph.
        # The ROI is the same for all nodes in the Graph.
        roi = enclosed_region.nodes[list(region_nodes)[0]]['contour'].roi
        # Check if the region_counter exceeds the length of region_labels
        # If so, add a suffix to the label.
        if region_counter > len(region_labels):
            suffix = str(region_counter // len(region_labels) - 1)
        else:
            suffix = ''
        # Create a label for the enclosed region.
        region_label = f'{roi}{region_labels[region_counter]}{suffix}'
        # Assign the label to each contour in the SubGraph.
        for node in enclosed_region.nodes:
            contour = enclosed_region.nodes[node]['contour']
            contour.region_index = region_label
            # Add the region label to the related_regions
            # get the related_contours for the contour
            related_contours_list = contour.related_contours
            # Find all contours where the contour_index is in the
            # related_contours list.
            node_data = dict(contour_graph.nodes.data('contour'))
            for related_contour in node_data.values():
                if related_contour.contour_index in related_contours_list:
                    related_contour.related_regions.append(region_label)
        region_counter += 1
    return contour_graph


def set_hole_type(contour_graph: ContourGraph,
                  slice_sequence: SliceSequence)->ContourGraph:
    '''Determine whether the regions that are holes are 'Open' or 'Closed'.

    Args:
        contour_graph (ContourGraph): The graph representation of the contours.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.
    Returns:
        ContourGraph: The updated contour graph with hole types assigned to
            the holes in each region.
    '''
    # Create the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Select boundary contours that are holes using contour_lookup
    hole_boundaries = ((contour_lookup['Boundary']) &
                       (contour_lookup['HoleType'] == 'Unknown'))
    # Initialize the region hole type to 'Unknown'
    # Get a list of the RegionIndexes that reference holes
    hole_regions = contour_lookup.loc[hole_boundaries, 'RegionIndex']
    hole_regions = list(hole_regions.drop_duplicates())
    region_hole_type = {region: 'Unknown' for region in hole_regions}
    # Iterate through each region and determine the hole type
    boundary_contours = contour_lookup.loc[hole_boundaries, 'Label']
    for boundary_label in boundary_contours:
        # Get the contour from the graph
        boundary_contour = contour_graph.nodes[boundary_label]['contour']
        # Get the RegionIndex for the boundary contour
        region_index = boundary_contour.region_index
        # Get the slice index of the boundary contour
        this_slice = boundary_contour.slice_index
        # Determine the slice index just beyond the boundary contour.
        neighbouring_nodes = contour_graph.adj[boundary_label].keys()
        # Because it is a boundary contour, there should only be one
        # neighbouring node.
        neighbour_slice = [nbr[1] for nbr in neighbouring_nodes][0]
        # Get the slice index of the boundary contour
        this_slice = boundary_contour.slice_index
        # Get the slice indexes of neighbouring contours
        neighbors = slice_sequence.get_neighbors(this_slice).neighbour_list()
        # The slice that is not a neighbour of the boundary contour is the
        # slice beyond the boundary contour.
        slice_beyond = [nbr for nbr in neighbors if nbr != neighbour_slice][0]
        beyond = contour_lookup.SliceIndex==slice_beyond
        contour_labels = list(contour_lookup.loc[beyond].Label)
        # The boundary is open if no contour contains the boundary contour
        boundary_closed = False
        for label in contour_labels:
            # Get the contour from the graph
            contour = contour_graph.nodes[label]['contour']
            if contour.polygon.contains(boundary_contour.polygon):
                # The hole is closed by the contour
                boundary_closed = True
                break
        if boundary_closed:
            # The hole is closed by the contour
            # Region is closed if all boundaries are closed
            if region_hole_type[region_index] == 'Unknown':
                region_hole_type[region_index] = 'Closed'
        else:
            # The hole is open by the contour
            # Region is open if any boundary is open
            if region_hole_type[region_index] == 'Unknown':
                region_hole_type[region_index] = 'Open'
    # Set the hole type for the each region
    for region, hole_type in region_hole_type.items():
        # Get the contours in the region
        region_contours = contour_lookup.RegionIndex == region
        region_labels = list(contour_lookup.loc[region_contours, 'Label'])
        # Set the hole type for each contour in the region
        for label in region_labels:
            # Get the contour from the graph
            contour = contour_graph.nodes[label]['contour']
            # Set the hole type for the contour
            contour.hole_type = hole_type
    return contour_graph


def build_contour_graph(contour_table: pd.DataFrame,
                        slice_sequence: SliceSequence,
                        roi: ROI_Type) -> Tuple[ContourGraph, SliceSequence]:
    '''Build a graph of contours for the specified ROI.

    This is the primary outward facing function of this module.  It creates a
    graph representation of the contours for a given ROI. Each contour is
    represented as a node in the graph, and edges indicate matching contours on
    the previous and next slices.

    Matching contours are on neighbouring slices (based on the slice sequence),
    have the same hole type, and have intersecting convex hulls.

    Interpolated boundary contours are added to the graph, and the hole type for
    the regions is determined based on the contours in the graph.

    Args:
        contour_table (pd.DataFrame): The contour table containing contour data.
            The table must be sorted by descending area, or holes will not be
            identified properly.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.
        roi (ROI_Type): The ROI number to filter contours.

    Returns:
        tuple: A tuple containing the contour graph and a lookup table.
            contour_graph (ContourGraph): The graph representation of the
                contours.
            contour_lookup (SliceSequence): A DataFrame serving as a lookup
                table for contours, including slice index, contour index, and
                hole type.
    '''
    contour_by_slice = build_contours(contour_table, roi)
    # Create an empty graph
    contour_graph = nx.Graph()
    # Add nodes to the graph
    for contour_data in contour_by_slice.values():
        for contour in contour_data:
            contour_label = contour.index
            contour_graph.add_node(contour_label, contour=contour)
    # Add the edges to the graph
    contour_graph = add_graph_edges(contour_graph, slice_sequence)
    # Add the boundary contours to the graph
    contour_graph, slice_sequence = add_boundary_contours(contour_graph,
                                                          slice_sequence)
    # Identify the distinct regions in the graph
    # and assign the region index to each contour
    contour_graph = set_enclosed_regions(contour_graph)
    # Set the hole type for the regions
    contour_graph = set_hole_type(contour_graph, slice_sequence)
    return contour_graph, slice_sequence
