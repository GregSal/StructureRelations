'''Contains the structure class.
'''

from typing import List

import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type, SliceIndexType
from types_and_classes import ContourIndex
from contours import SliceSequence, Contour, ContourMatch
from contours import interpolate_polygon
from contour_graph import build_contour_graph, build_contour_lookup
from region_slice import build_region_table


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

    def add_contour_graph(self, contour_table: pd.DataFrame,
                          slice_sequence: SliceSequence) -> SliceSequence:
        '''Builds the contour graph and the contour lookup table.

        Args:
            contour_table (pd.DataFrame): The contour table.
            slice_sequence (SliceSequence): The slice sequence.

        Returns:
            SliceSequence: The updated slice sequence with the contour graph
        '''
        contour_graph, slice_sequence = build_contour_graph(contour_table,
                                                            slice_sequence,
                                                            self.roi)
        self.contour_graph = contour_graph
        self.contour_lookup = build_contour_lookup(contour_graph)
        not_original = slice_sequence.sequence.Original is False
        intp_idx = list(slice_sequence.sequence.loc[not_original, 'ThisSlice'])
        self.generate_interpolated_contours(slice_sequence, intp_idx)
        self.calculate_physical_volume()
        self.calculate_exterior_volume()
        self.calculate_hull_volume()
        self.region_table = build_region_table(self.contour_graph,
                                               slice_sequence)
        return slice_sequence

    def calculate_physical_volume(self):
        '''Calculate the physical volume of a ContourGraph.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.

        Returns:
            float: The total physical volume of the ContourGraph.
        '''
        total_volume = 0.0
        # Iterate over all edges in the contour graph
        for _, _, data in self.contour_graph.edges(data=True):
            match = data['match']
            contour1 = match.contour1
            volume = match.volume()
            # If contour1 is a hole, subtract the volume; otherwise, add it
            if contour1.is_hole:
                total_volume -= volume
            else:
                total_volume += volume
        self.physical_volume = total_volume

    def calculate_exterior_volume(self):
        '''Calculate the exterior volume of a ContourGraph.

        Args:
            contour_graph (nx.Graph): The graph representation of the contours.

        Returns:
            float: The total exterior volume of the ContourGraph.
        '''
        total_volume = 0.0
        for _, _, data in self.contour_graph.edges(data=True):
            match = data['match']
            contour1 = match.contour1
            volume = match.volume()
            if contour1.is_hole:
                if contour1.hole_type == 'Open':
                    total_volume -= volume
                elif contour1.hole_type == 'Closed':
                    volume = 0.0
                    # Do not add/subtract closed hole volume
                else:
                    total_volume -= volume
            else:
                total_volume += volume
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
        for _, _, data in self.contour_graph.edges(data=True):
            match = data['match']
            contour1 = match.contour1
            volume = match.volume(use_hull=True)
            if contour1.is_hole:
                volume = 0.0
            total_volume += volume
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
        # check for empty slice_sequence
        if not slice_sequence:
            return
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
                existing_contours=[]
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
            self.contour_lookup = build_contour_lookup(self.contour_graph)
            self.contour_graph.add_edge(contour_nxt.index, interpolated_label,
                                        match=contour_match)
            self.contour_lookup = build_contour_lookup(self.contour_graph)
            self.contour_lookup = build_contour_lookup(self.contour_graph)

    def relate(self, other: 'StructureShape') -> None:
        '''Relate this structure to another structure.

        Args:
            other (StructureShape): The other structure to relate to.
        '''
        # This method is a placeholder for future implementation
        # It can be used to establish relationships between structures
        #1. Create an initial blank DE27IM relationship object.
        #1. Identify the slices where both structures have non-empty RegionSlice objects.
        #2. For each of these slices
        #    1. Get the DE27IM relationship for the two RegionSlice objects.
        #    3. Merge the DE27IM relationship into the composite DE27IM relationship object.
        pass
