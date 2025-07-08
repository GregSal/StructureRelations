'''Functions to build a contour graph from a contour table.
The contour graph is a undirected graph where each node represents a contour
and edges represent connect matched contours on the next and previous slice.
'''
# %% Setup
from typing import List, Tuple
from collections import defaultdict

import pandas as pd
import networkx as nx
import shapely

from contours import SliceSequence, Contour, ContourMatch
from contours import interpolate_polygon
from types_and_classes import SliceIndexType, ContourIndex, RegionIndex
from types_and_classes import ContourGraph

from types_and_classes import InvalidSlice, ROI_Type, InvalidContour

# %% Contour Lookup Table Functions
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
    if len(contour_graph) == 0:
        # If the contour graph is empty, return an empty DataFrame
        return pd.DataFrame()
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


def get_region_contours(contour_graph: ContourGraph,
                        contour_reference: pd.DataFrame,
                        related_regions: List[RegionIndex])->List[Contour]:
    '''Return the contours for the specified regions.

    This function retrieves contours from the contour graph based on the
    specified RegionIndexes. It uses a lookup table to find the contours.
    The contours are returned as a list of Contour objects.  If no contours are
    found for the specified regions, an empty list is returned.

    Args:
        contour_graph (ContourGraph): The contour graph.
        contour_reference (pd.DataFrame): The contour lookup table.
            This DataFrame should contain the columns 'RegionIndex' and 'Label'.
            The 'RegionIndex' column should contain RegionIndexes, and the
            'Label' should contain ContourIndexes (The labels for ContourGraph
            nodes).
        related_regions (List[RegionIndex]): A list of RegionIndexes for which
            to retrieve contours.
    Returns:
        List[Contour]: A list of Contour objects for the specified regions.
    '''
    region_selection = contour_reference['RegionIndex'].isin(related_regions)
    region_reference = contour_reference.loc[region_selection]
    contour_labels = region_reference['Label'].tolist()
    contour_data = dict(contour_graph.nodes.data('contour'))
    region_contours = [contour_data[label] for label in contour_labels]
    return region_contours


# %% contour interpolation functions
def generate_interpolated_polygon(contour_graph: ContourGraph,
                                  slice_sequence: SliceSequence,
                                  starting_contour: ContourIndex,
                                  interpolated_slice: SliceIndexType = None) -> shapely.Polygon:
    '''Generate an interpolated polygon for a given starting contour in the
    contour graph.

    Args:
        contour_graph (nx.Graph): The contour graph.
        slice_sequence (SliceSequence): The slice sequence.
        starting_contour (ContourIndex): The index of the contour to interpolate
            from.
        interpolated_slice (SliceIndexType, optional): The slice index for the
            interpolated contour. If not provided, starting_contour must
            reference a graph node with only one neighbour (a boundary slice).
            If not, raise an InvalidSlice error.

    Returns:
        shapely.Polygon: An interpolated Polygon.

    Raises:
        InvalidContour: If starting_contour is not in the contour graph.
        InvalidSlice: If interpolated_slice is not provided and starting_contour
            does not reference a graph node with only one neighbour (a boundary
            slice).
    '''
    lookup = build_contour_lookup(contour_graph)
    if starting_contour not in set(lookup.Label):
        raise InvalidContour(f"Contour {starting_contour} was not found in the "
                             "contour graph.")
    contour = contour_graph.nodes[starting_contour]['contour']
    # Get the slice index of the contour
    this_slice = contour.slice_index
    # Get the neighboring slices of the contour.
    matched_slices = set()
    for nbr in contour_graph.adj[starting_contour]:
        matched_slices.add(contour_graph.nodes[nbr]['contour'].slice_index)
    # Get the neighbours of the slice in the slice sequence
    # (idx == idx) excludes NaN values
    neighbors = slice_sequence.get_neighbors(this_slice)
    if interpolated_slice:
        # Check that the interpolated slice is between the current slice and
        # a neighbour slice.  If not assume that it is a boundary contour.
        if neighbors.is_neighbour(interpolated_slice):
            # Get the neighbouring contour for interpolation
            nearest_neighbour = neighbors.nearest(interpolated_slice)
            interpolation_slices = [this_slice, nearest_neighbour]
            # Get the neighbouring contours for interpolation.  Only the contours
            # on the nearest_neighbour that are matched to the starting
            # contour are used for interpolation.
            matched_contours = []
            for nbr in contour_graph.adj[starting_contour]:
                matched_slice = contour_graph.nodes[nbr]['contour'].slice_index
                if matched_slice == nearest_neighbour:
                    matched_contours.append(contour_graph.nodes[nbr]['contour'])
            interpolated_polygons = []
            # If matched contours are found use them to generate the
            # interpolated polygon. If not assume that it is a boundary contour.
            if matched_contours:
                for matched_contour in matched_contours:
                    # Get the polygon for the matched contour
                    poly2 = matched_contour.polygon
                    # Interpolate the polygon using the matched contour
                    interpolated_polygons.append(interpolate_polygon(
                        interpolation_slices,
                        contour.polygon, poly2))
                # Combine the interpolated polygons into a single polygon
                interpolated_polygon = shapely.union_all(interpolated_polygons)
                return interpolated_polygon

    # if no interpolated slice is provided, or if the interpolated slice is not
    # a neighbour of the starting contour (e.g. first or last slice), or the
    # contour does not have any matched contours on the relevant neighbouring
    # slice, the contour should be a boundary contour.  Generate the
    # interpolated slice index using the slice index of the neighbouring slice
    # that is not matched to the boundary contour.  Generate the interpolated
    # contour using only the boundary contour.
    if len(matched_slices) > 1:
        raise InvalidSlice("Without interpolated_slice, starting_contour "
                            "must reference a boundary slice.")
    # Determine the slice index that is not matched to the boundary contour.
    # (This is the slice index that will be used to set the interpolated
    # slice index.)
    neighbour_slices = set(neighbors.neighbour_list())
    non_neighbour_slices = neighbour_slices - matched_slices
    if non_neighbour_slices:
        non_neighbour_slice = non_neighbour_slices.pop()
    else:
        # Use gap to create an estimated slice index
        gap = neighbors.gap(absolute=False)
        non_neighbour_slice = this_slice - gap
    interpolation_slices = [this_slice, non_neighbour_slice]
    # Calculate the interpolated contour using only the boundary contour.
    interpolated_polygon = interpolate_polygon(interpolation_slices,
                                                contour.polygon)
    return interpolated_polygon


def generate_interpolated_contours(contour_graph: ContourGraph,
                                   slice_sequence: SliceSequence,
                                   starting_contour: ContourIndex,
                                   interpolated_slice: SliceIndexType = None,
                                   **contour_parameters
                                   ) -> Tuple[ContourGraph, SliceSequence, SliceSequence]:
    '''Generate an interpolated contour for a given starting contour in the
    contour graph.

    Args:
        contour_graph (nx.Graph): The contour graph.
        slice_sequence (SliceSequence): The slice sequence.
        starting_contour (ContourIndex): The index of the contour to interpolate
            from.
        interpolated_slice (SliceIndexType, optional): The slice index for the
            interpolated contour. If not provided, starting_contour must
            reference a graph node with only one neighbour (a boundary slice).
            If not, raise an InvalidSlice error.
        contour_parameters (dict): Additional parameters for the contour
            generation. These parameters are passed to the Contour constructor
            and can include properties like is_boundary, related_contours,
            existing_contours etc.  The parameters  roi, slice_index, and
            is_interpolated will be set automatically based on the interpolated
            contour and will be ignored if included in contour_parameters.

    Raises:
        InvalidContour: If starting_contour is not in the contour graph.
        InvalidSlice: If interpolated_slice is not provided and starting_contour
            does not reference a graph node with only one neighbour (a boundary
            slice).

    Returns:
        tuple:  A tuple containing the updated contour graph and slice sequence:
            contour_graph (ContourGraph): The updated graph with the
                interpolated contour added.
            SliceSequence: The updated slice sequence with the interpolated
                slice added.
            ContourIndex: The index of the newly created interpolated contour.

    '''
    # Generate the interpolated polygon
    interpolated_polygon = generate_interpolated_polygon(contour_graph,
                                                   slice_sequence,
                                                   starting_contour,
                                                   interpolated_slice)
    # Get the interpolated slice index
    interpolated_slice_index = interpolated_polygon.boundary.coords[0][2]
    # set the starting contour parameters
    contour = contour_graph.nodes[starting_contour]['contour']
    contour_parameters['roi'] = contour.roi
    contour_parameters['slice_index'] = interpolated_slice_index
    contour_parameters['polygon'] = interpolated_polygon
    contour_parameters['is_interpolated'] = True
    # If existing_contours is not provided, initialize it as an empty list
    if 'existing_contours' not in contour_parameters:
        contour_parameters['existing_contours'] = []
    if 'is_boundary' not in contour_parameters:
        contour_parameters['is_boundary'] = contour.is_boundary
    if 'is_hole' not in contour_parameters:
        contour_parameters['is_hole'] = contour.is_hole
    if 'hole_type' not in contour_parameters:
        contour_parameters['hole_type'] = contour.hole_type
    if 'region_index' not in contour_parameters:
        contour_parameters['region_index'] = contour.region_index
    # Create the new Contour object
    new_contour = Contour(**contour_parameters)

    # Add the interpolated contour to the graph
    interpolated_label = new_contour.index
    contour_graph.add_node(interpolated_label, contour=new_contour)
    # Add edge between original and interpolated
    contour_match = ContourMatch(contour, new_contour)
    contour_graph.add_edge(starting_contour, interpolated_label,
                           match=contour_match)

    # Update the slice sequence with the interpolated slice
    original_slice = contour.slice_index
    # Determine the neighbor slice not linked with an edge
    neighbors = slice_sequence.get_neighbors(original_slice)
    # Get the slice index of the neighbouring contours.
    # This is the starting point for the neighbours of the interpolated slice
    slice_ref = {'PreviousSlice': neighbors.previous_slice,
                  'NextSlice': neighbors.next_slice,
                  'ThisSlice': interpolated_slice_index,
                  'Original': False}
    if original_slice < interpolated_slice_index:
        # The interpolated slice is after the original slice
        slice_ref['PreviousSlice'] = original_slice
    else:
        # The interpolated slice is before the original slice
        slice_ref['NextSlice'] = original_slice
    slice_sequence.add_slice(**slice_ref)
    # Return the updated contour graph and the interpolated label
    return contour_graph, slice_sequence, interpolated_label


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
    if contour_set.empty:
        return None
    # sort the contour set by slice_index and area in descending order
    contour_set = contour_set.sort_values(by=['Slice', 'Area'],
                                          ascending=[True, False])
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
        contour_by_slice[slice_index].sort(key=lambda c: c.area, reverse=True)
    return contour_by_slice


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
            if this_contour.polygon_with_holes.intersects(neighbour_contour.polygon_with_holes):
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
    boundary_nodes = {node for node, degree in contour_graph.degree()
                      if degree == 1}

    while boundary_nodes:
        all_related_contours = set()
        original_boundary = boundary_nodes.pop()
        contour_parameters = {'contour_graph': contour_graph,
                              'slice_sequence': slice_sequence,
                              'starting_contour': original_boundary,
                              'is_interpolated': True, 'is_boundary': True}
        # Generate the interpolated contour
        interpolation = generate_interpolated_contours(**contour_parameters)
        # Unpack the interpolation result
        contour_graph, slice_sequence, new_idx = interpolation
        # generate interpolated contours for related contours
        all_related_contours.add(new_idx)
        contour = contour_graph.nodes[original_boundary]['contour']
        new_contour = contour_graph.nodes[new_idx]['contour']
        related_contours = contour.related_contours
        for related_contour in related_contours:
            if related_contour in boundary_nodes:
                boundary_nodes.remove(related_contour)
                is_boundary = True
            else:
                is_boundary = False
            # Generate the interpolated contour for the related contour
            related_parameters = {
                'contour_graph': contour_graph,
                'slice_sequence': slice_sequence,
                'starting_contour': related_contour,
                'interpolated_slice': new_contour.slice_index,
                'is_interpolated': True,
                'is_boundary': is_boundary
            }
            try:  # This is a temporary fix to handle incorrect boundary identification
                # Generate the interpolated contour for the related contour
                interpolation = generate_interpolated_contours(**related_parameters)
            except InvalidSlice:
                # If the contour is not found, skip it
                continue
            # Unpack the interpolation result
            contour_graph, slice_sequence, related_idx = interpolation
            # Add the new index to the list of interpolated contours
            all_related_contours.add(related_idx)
        # Update the related_contours for all new contours
        for idx in all_related_contours:
            contour = contour_graph.nodes[idx]['contour']
            contour.related_contours = list(all_related_contours - {idx})
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
                    related_contour.related_contours.append(region_label)
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
    # Hole Type identification process:
    # 1. Select boundary contours that are holes.
    # 2. Get the RegionIndexes of those holes.
    # 3. initialize a dictionary to hold the hole type for each region.
    # 4. Iterate through each boundary contour and update the dictionary based
    #    on the status to the hole boundary:
    #     1. Determine the slice index that is just beyond the boundary contour.
    #         1. Get the slice index of the boundary contour.
    #         2. Get the slice index of the neighbouring contour.
    #            (Because it is a boundary contour, there should only be one
    #            neighbouring node.)
    #         3. Use slice_sequence to get the neighbouring slice indexes and
    #            select the one that is not a neighbour of the boundary contour.
    #     2. Get the contours in the slice beyond the boundary contour.
    #     3. Check if any of those contours can contain the boundary contour.
    #     4. If so, that boundary is "closed".  If that region is not already
    #        marked as "open", set the status to "closed"
    #     5. If none of the contours contain the boundary contour, that
    #        boundary is "open".  Set the status of that region to "open".
    #        ("open" always overrides "closed".)
    # 5. Use the region hole type dictionary to set the hole type for each
    #    region.

    ### 1. Select boundary contours that are holes. ###
    # Create the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Select boundary contours that are holes using contour_lookup
    hole_boundaries = ((contour_lookup['Boundary']) &
                       (contour_lookup['HoleType'] == 'Unknown'))
    boundary_contours = contour_lookup.loc[hole_boundaries, 'Label']

    ### 2. Get the RegionIndexes of those holes. ###
    hole_regions = contour_lookup.loc[hole_boundaries, 'RegionIndex']
    # Make a list of the RegionIndexes that reference holes
    hole_regions = list(hole_regions.drop_duplicates())

    ### 3. Initialize the region hole type to 'Unknown' ###
    # Get a list of the RegionIndexes that reference holes
    region_hole_type = {region: 'Unknown' for region in hole_regions}

    ### 4. Iterate through each region and determine the hole type ###
    boundary_contours = contour_lookup.loc[hole_boundaries, 'Label']
    for boundary_label in boundary_contours:
        # 4.1. Determine the slice index that is just beyond the boundary contour. #
        # Get the contour from the graph
        boundary_contour = contour_graph.nodes[boundary_label]['contour']
        # 4.1.1 Get the slice index of the boundary contour
        this_slice = boundary_contour.slice_index
        # 4.1.2 Get the slice index of the neighbouring contour.
        neighbouring_nodes = contour_graph.adj[boundary_label].keys()
        # Because it is a boundary contour, there should only be one
        # neighbouring node.
        neighbour_slice = [nbr[1] for nbr in neighbouring_nodes][0]
        # 4.1.3. Use slice_sequence to get the neighbouring slice indexes and
        #        select the one that is not a neighbour of the boundary contour.
        neighbors = slice_sequence.get_neighbors(this_slice).neighbour_list()
        # The slice that is not a neighbour of the boundary contour is the
        # slice beyond the boundary contour.
        non_neighbour_slice = [nbr for nbr in neighbors
                               if nbr != neighbour_slice]
        if non_neighbour_slice:
            # Use the non-neighbour slice as the slice beyond the boundary contour
            slice_beyond = non_neighbour_slice[0]
            beyond = contour_lookup.SliceIndex==slice_beyond
            # 4.2. Get the contours in the slice beyond the boundary contour.
            contour_labels = list(contour_lookup.loc[beyond].Label)
        else:
            # There is no non-neighbour slice, the boundary must be open.
            contour_labels = []
        # 4.3. Check if any of those contours can contain the boundary contour.
        # The boundary is closed if any contour can contain the boundary contour.
        boundary_closed = False
        for label in contour_labels:
            # Get the contour from the graph
            contour = contour_graph.nodes[label]['contour']
            if contour.polygon.contains(boundary_contour.polygon):
                # The hole is closed by the contour
                boundary_closed = True
                break
        # Get the RegionIndex for the boundary contour
        region_index = boundary_contour.region_index
        if boundary_closed:
            # The hole is closed by the contour
            # Region is closed if all boundaries are closed
            if not region_hole_type[region_index] == 'Open':
                # 4.4 Set the Region status to "closed"
                region_hole_type[region_index] = 'Closed'
        else:
            # The hole is open by the contour
            # Region is open if any boundary is open
            # 4.5 Set the Region status to "open"
            region_hole_type[region_index] = 'Open'

    ### 5. Use Set the hole type for each contour in the region. ###
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


def set_thickness(contour_graph: ContourGraph) -> ContourGraph:
    '''Assign thickness to each contour in the contour graph.

    The thickness for each contour is set to twice the average absolute distance
    between its slice_index and the slice_index of its neighboring nodes. If a
    node does not have any neighbours, use the Contour.default_thickness value.

    Args:
        contour_graph (ContourGraph): The graph representation of the contours.

    Returns:
        ContourGraph: The updated contour graph with thickness set for each contour.
    '''
    for node, data in contour_graph.nodes(data=True):
        contour = data['contour']
        # get the slice index of the contour
        this_slice = contour.slice_index
        # Get the slice indexes of the neighbouring slices.
        neighbour_slices = [contour_graph.nodes[n]['contour'].slice_index
                            for n in contour_graph.adj[node]]
        if neighbour_slices:
            total_dist = sum(abs(this_slice - n_slice)
                             for n_slice in neighbour_slices)
            avg_dist = total_dist / len(neighbour_slices)
            thickness = 2 * avg_dist
        else:
            thickness = getattr(Contour, 'default_thickness', 0)
        contour.thickness = thickness
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
    # Create an empty graph
    contour_graph = nx.Graph()
    # Get the relevant contours
    contour_by_slice = build_contours(contour_table, roi)
    if not contour_by_slice:
        # If not contours are found for the given ROI, return an empty graph.
        return contour_graph, slice_sequence
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
    # Set the thickness for the contours
    contour_graph = set_thickness(contour_graph)
    return contour_graph, slice_sequence
    return contour_graph, slice_sequence
