'''Contains the structure class.
'''

from typing import List, Tuple

import pandas as pd
import networkx as nx

from contours import Contour, ContourMatch, build_contour_graph
from contours import build_contour_lookup, interpolate_polygon
from types_and_classes import ROI_Type, ContourIndex, SliceIndexType
from contours import SliceSequence


def calculate_node_volume(node: Contour, edges: List[Tuple],
                          use_hull=False) -> float:
    '''Calculate a volume for a node based on its edges.

    Args:
        node (Contour): The contour node for which the volume is calculated.
        edges (List[Tuple]): A list of edges connected to the node. Each edge
            contains the edge data with a 'match' attribute of type
            ContourMatch.
        use_hull (bool): If True, use the convex hull area for calculations.
            If False, use the actual area of the contour.

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

    def calculate_area_factor(node: Contour, edges: List[Tuple],
                              use_hull=False):
        node_index = node.index
        total_neighbour_area = 0.0
        for edge in edges:
            contour_match = edge[2]['match']
            index1 = contour_match.contour1.index
            index2 = contour_match.contour2.index
            if index2 == node_index:
                neighbor_contour = contour_match.contour1
            elif index1 == node_index:
                neighbor_contour = contour_match.contour2
            else:
                raise ValueError("Edge does not connect to the node.")
            if use_hull:
                neighbour_area = neighbor_contour.polygon.convex_hull.area
            else:
                neighbour_area = neighbor_contour.polygon.area
            total_neighbour_area += neighbour_area
        if total_neighbour_area == 0:
            return 0.0
        if use_hull:
            node_area = node.polygon.convex_hull.area
        else:
            node_area = node.polygon.area

        area_factor = (3 * node_area / total_neighbour_area + 1) / 4
        return area_factor

    def calculate_volume(node: Contour, edges: List[Tuple],
                         use_hull=False):
        if not edges:
            return 0.0
        node_index = node.index
        area_factor = calculate_area_factor(node, edges, use_hull)
        pseudo_volume = 0.0
        for edge in edges:
            contour_match = edge[2]['match']
            thickness = edge[2]['match'].thickness
            index1 = contour_match.contour1.index
            index2 = contour_match.contour2.index
            if index2 == node_index:
                neighbor_contour = contour_match.contour1
            elif index1 == node_index:
                neighbor_contour = contour_match.contour2
            else:
                raise ValueError("Edge does not connect to the node.")
            if use_hull:
                neighbour_area = neighbor_contour.polygon.convex_hull.area
            else:
                neighbour_area = neighbor_contour.polygon.area
            pseudo_volume += (area_factor * neighbour_area * thickness)
        return pseudo_volume

    node_volume = (calculate_volume(node, positive_edges, use_hull) +
                   calculate_volume(node, negative_edges, use_hull))
    return node_volume




class StructureShape():
    '''Class containing the data for the shape of a structure.

    Attributes:
        name (str): Name of the structure.
        roi_number (ROI_Type): ROI number of the structure.
        contour_graph (nx.Graph): Graph representation of the contours that
            make up the structure.
        contour_lookup (pd.DataFrame): A table used to reference contours in
            the contour_graph. The table contains the following columns:
                - ROI,
                - SliceIndex,
                - HoleType,
                - Interpolated,
                - Boundary,
                - ContourIndex,
                - RegionIndex, and
                - Label.
        region_table (SliceIndexType): A table of StructureSlice.

        physical_volume (float): The physical volume of the structure in cm^3.
        exterior_volume (float): The exterior volume of the structure in cm^3.
        hull_volume (float): The volume of the convex hull of the structure in cm^3.
    '''

    def __init__(self, roi: ROI_Type, name: str = None):
        if name is None:
            self.name = f"Structure {roi}"
        else:
            self.name = name
        self.roi = roi
        self.contour_graph = nx.Graph()
        self.contour_lookup = pd.DataFrame()
        self.region_table = []
        self.physical_volume = 0.0
        self.exterior_volume = 0.0
        self.hull_volume = 0.0

    def build_contour_graph(self, contour_table: pd.DataFrame,
                        slice_sequence: SliceSequence) -> None:
        '''Builds the contour graph and the contour lookup table.

        Args:
            contour_table (pd.DataFrame): The contour table.
            slice_sequence (SliceSequence): The slice sequence.
        '''
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            self.roi)
        self.contour_graph = contour_graph
        self.contour_lookup = build_contour_lookup(contour_graph)
        self.calculate_physical_volume()
        self.calculate_exterior_volume()
        self.calculate_hull_volume()
        return slice_sequence


    def calculate_physical_volume(self):
        '''Calculate the physical volume of a ContourGraph.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.

        Returns:
            float: The total physical volume of the ContourGraph.
        '''
        total_volume = 0.0
        for node, data in self.contour_graph.nodes(data=True):
            contour = data['contour']
            edges = list(self.contour_graph.edges(node, data=True))
            node_volume = calculate_node_volume(contour, edges)
            if contour.is_hole:
                node_volume = -node_volume
            total_volume += node_volume
        self.physical_volume = total_volume

    def calculate_exterior_volume(self):
        '''Calculate the exterior volume of a ContourGraph.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.

        Returns:
            float: The total exterior volume of the ContourGraph.
        '''
        total_volume = 0.0
        for node, data in self.contour_graph.nodes(data=True):
            contour = data['contour']
            edges = list(self.contour_graph.edges(node, data=True))
            node_volume = calculate_node_volume(contour, edges)
            if contour.is_hole:
                if contour.hole_type == 'Open':
                    node_volume = -node_volume
                elif contour.hole_type == 'Closed':
                    node_volume = 0.0
            total_volume += node_volume
        self.exterior_volume = total_volume


    def calculate_hull_volume(self):
        '''Calculate the hull volume of an EnclosedRegionTable.

        Note: this is not a true convex hull volume, but rather a volume based on
        the convex hulls in the xy plane.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.

        Returns:
            float: The total hull volume of the EnclosedRegionTable.
        '''
        total_volume = 0.0
        for node, data in self.contour_graph.nodes(data=True):
            contour = data['contour']
            if not contour.is_hole:
                edges = list(self.contour_graph.edges(node, data=True))
                node_volume = calculate_node_volume(contour, edges, use_hull=True)
                total_volume += node_volume
        self.hull_volume = total_volume

    def get_contour(self, label: ContourIndex) -> Contour:
        '''Retrieve the contour representation of the structure.

        Returns:
            Contour: The selected contour from the structure.
        '''
        contour = self.contour_graph.nodes(data=True)[label]['contour']
        return contour

    def generate_interpolated_contours(self, slice_sequence: SliceSequence,
                                       interpolated_slice_indexes: List[SliceIndexType]) -> None:
        '''Generate interpolated contours for the structure.

        Args:
            slice_sequence (SliceSequence): The table of all slices used in the
                structure set.
            interpolated_slice_indexes (List[SliceIndexType]): A list of slice
                indexes containing interpolated contours.
        '''
        nbr_pairs = []
        match_index = ['ROI', 'HoleType', 'Interpolated', 'Boundary',
               'RegionIndex', 'ThisSlice']
        selected_columns = ['ThisSlice', 'SliceIndex_prv', 'SliceIndex_nxt',
                    'Label_prv', 'Label_nxt']
        nbr_pairs = []
        for interpolated_slice in interpolated_slice_indexes:
            nbr = slice_sequence.get_neighbors(interpolated_slice)

            is_previous = self.contour_lookup.SliceIndex == nbr.previous_slice
            prv_contours = self.contour_lookup.loc[is_previous, :].copy()
            prv_contours['ThisSlice'] = interpolated_slice

            is_next = self.contour_lookup.SliceIndex == nbr.next_slice
            nxt_contours = self.contour_lookup.loc[is_next, :].copy()
            nxt_contours['ThisSlice'] = interpolated_slice
            prv_nxt = prv_contours.merge(nxt_contours, how='outer',
                                         on=match_index,
                                        suffixes=('_prv', '_nxt'))
            prv_nxt = prv_nxt[selected_columns]
            nbr_pairs.append(prv_nxt)
        nbr_pairs = pd.concat(nbr_pairs, ignore_index=True).dropna(axis=0,
                                                                   how='any')
        nbr_pairs.reset_index(drop=True, inplace=True)
        for intp_param in nbr_pairs.itertuples():
            slices = [intp_param.SliceIndex_prv, intp_param.SliceIndex_nxt]
            contour_prv = self.get_contour(intp_param.Label_prv)
            contour_nxt = self.get_contour(intp_param.Label_nxt)
            interpolated_polygon = interpolate_polygon(slices,
                                                       contour_prv.polygon,
                                                       contour_nxt.polygon)
            interpolated_contour = Contour(
                roi=self.roi,
                slice_index=intp_param.ThisSlice,
                polygon=interpolated_polygon,
                contours=[]
                )
            interpolated_contour.is_interpolated = True
            interpolated_contour.is_boundary = False
            interpolated_contour.is_hole = contour_prv.is_hole
            interpolated_contour.hole_type = contour_prv.hole_type
            interpolated_contour.region_index = contour_prv.region_index
            # Add the interpolated contour to the graph
            interpolated_label = interpolated_contour.index
            self.contour_graph.add_node(interpolated_label,
                                            contour=interpolated_contour)
            # Add ContourMatch edges between the original contours and the
            # interpolated contour.
            contour_match = ContourMatch(contour_prv, interpolated_contour)
            self.contour_graph.add_edge(contour_prv.index, interpolated_label,
                                        match=contour_match)
            contour_match = ContourMatch(contour_nxt, interpolated_contour)
            self.contour_graph.add_edge(contour_nxt.index, interpolated_label,
                                        match=contour_match)
            self.contour_lookup = build_contour_lookup(self.contour_graph)
