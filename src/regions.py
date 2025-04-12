'''Contour Classes and related Functions
'''

import re
from typing import List, Tuple
import pandas as pd
import networkx as nx
from shapely.geometry import MultiPolygon

from contours import Contour
from types_and_classes import SliceIndexType


class RegionSlice:
    '''Class representing a slice of an enclosed region.

    Attributes:
        RegionIndex (str): The index of the enclosed region.
        SliceIndex (SliceIndexType): The slice index of the region.
        Polygon (shapely.MultiPolygon): The combined polygon for the region slice.
        ExternalHoles (shapely.MultiPolygon): The combined external holes.
        Boundaries (shapely.MultiPolygon): The combined boundary polygons.
        Thickness (float): The thickness of the slice.
        Contours (List[ContourIndex]): A list of associated contour indexes.
    '''
    def is_empty(self) -> bool:
        '''Check if the RegionSlice is empty.

        Returns:
            bool: True if the RegionSlice is empty, False otherwise.
        '''
        return not self.Contours

    def __init__(self, contour_graph: nx.Graph, contour_lookup: pd.DataFrame,
                 region_index: str, slice_index: SliceIndexType) -> None:
        '''Initialize the RegionSlice.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.
            contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table for contours.
            region_index (str): The index of the enclosed region.
            slice_index (SliceIndexType): The slice index of the region.
        '''
        # Filter contours by RegionIndex and SliceIndex
        selected_contours = ((contour_lookup['RegionIndex'] == region_index) &
                             (contour_lookup['SliceIndex'] == slice_index))
        contour_labels = list(contour_lookup.loc[selected_contours, 'Label'])
        self.RegionIndex = region_index
        self.SliceIndex = slice_index
        self.Contours = contour_labels
        if not contour_labels:
            return
        non_hole_polygons = []
        hole_polygons = []
        boundary_holes = []
        boundary_polygons = []
        external_hole_polygons = []
        for label in contour_labels:
            contour = contour_graph.nodes[label]['contour']
            if contour.is_boundary:
                if contour.is_hole:
                    boundary_holes.append(contour.polygon)
                else:
                    boundary_polygons.append(contour.polygon)
            elif contour.is_hole:
                hole_polygons.append(contour.polygon)
                if contour.hole_type == 'Open':
                    external_hole_polygons.append(contour.polygon)
            else:
                non_hole_polygons.append(contour.polygon)
        # Combine polygons
        combined_polygon = MultiPolygon(non_hole_polygons)
        combined_holes = MultiPolygon(hole_polygons)
        combined_boundaries = MultiPolygon(boundary_polygons)
        combined_external_holes = MultiPolygon(external_hole_polygons)
        # Subtract holes from the main polygon
        combined_polygon = combined_polygon - combined_holes
        # Subtract hole boundaries from the boundary polygons
        boundary_polygons = combined_boundaries - boundary_holes
        # Assign attributes
        self.polygon = combined_polygon
        self.boundaries = combined_boundaries
        self.external_holes = combined_external_holes


def build_region_table(contour_graph: nx.Graph, contour_lookup: pd.DataFrame) -> pd.DataFrame:
    '''Build a DataFrame of RegionSlices for each RegionIndex and SliceIndex.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table for contours.

    Returns:
        pd.DataFrame: A DataFrame containing RegionSlices with the following columns:
            - RegionIndex
            - SliceIndex
            - RegionSlice
    '''
    enclosed_region_data = []

    # Iterate through each unique combination of RegionIndex and SliceIndex
    for region_index in contour_lookup['RegionIndex'].unique():
        region_slices = contour_lookup[contour_lookup['RegionIndex'] == region_index]['SliceIndex'].unique()
        for slice_index in region_slices:
            # Create a RegionSlice for the given RegionIndex and SliceIndex
            region_slice = RegionSlice(contour_graph, contour_lookup, region_index, slice_index)
            # Add the RegionSlice to the DataFrame
            enclosed_region_data.append({
                'RegionIndex': region_index,
                'SliceIndex': slice_index,
                'RegionSlice': region_slice
            })

    # Create the enclosed_region DataFrame
    enclosed_region_table = pd.DataFrame(enclosed_region_data)
    return enclosed_region_table


def calculate_node_volume(node: Contour, edges: List[Tuple]) -> float:
    '''Calculate a volume for a node based on its edges.

    Args:
        node (Contour): The contour node for which the volume is calculated.
        edges (List[Tuple]): A list of edges connected to the node. Each edge
            contains the edge data with a 'match' attribute of type
            ContourMatch.

    Returns:
        float: The calculated volume for the node.
    '''
    # Separate edges by direction
    positive_edges = []
    negative_edges = []
    for edge in edges:
        if edge[2]['match'].direction(node) > 0:
            positive_edges.append(edge)
        else:
            negative_edges.append(edge)

    def calculate_volume(edges):
        if not edges:
            return 0.0
        num_edges = len(edges)
        node_area = node.polygon.area
        area_ratio = sum(edge[2]['match'].combined_area / node_area
                         for edge in edges) - num_edges
        area_factor = (3 / area_ratio + 1)
        pseudo_volume = 0.0
        for edge in edges:
            neighbour_area = edge[2]['match'].combined_area - node_area
            thickness = edge[2]['match'].thickness
            pseudo_volume += (area_factor * neighbour_area * thickness) / 4
        return pseudo_volume

    node_volume = (calculate_volume(positive_edges) +
                   calculate_volume(negative_edges))
    return node_volume


def calculate_physical_volume(contour_graph: nx.Graph) -> float:
    '''Calculate the physical volume of a ContourGraph.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.

    Returns:
        float: The total physical volume of the ContourGraph.
    '''
    total_volume = 0.0

    for node, data in contour_graph.nodes(data=True):
        contour = data['contour']
        node_volume = calculate_node_volume(contour, list(contour_graph.edges(node, data=True)))
        if contour.is_hole:
            node_volume = -node_volume
        total_volume += node_volume

    return total_volume
