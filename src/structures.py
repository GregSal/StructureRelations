'''Contains the structure class.
'''
import re
import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type, SliceIndexType
from types_and_classes import ContourIndex
from contours import SliceSequence, Contour, ContourMatch
from contours import interpolate_polygon
from contour_graph import build_contour_graph, build_contour_lookup
from region_slice import RegionSlice
from relations import DE27IM


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
        self.region_table = pd.DataFrame()
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
        return slice_sequence

    def add_interpolated_contours(self, slice_sequence: SliceSequence) -> None:
        '''Generate interpolated contours for the structure.

        Generates interpolated contours to match with boundary contours
        generated for other structures.  It creates interpolated contours for
        all slice indexes from the SliceSequence that fall between edges in the
        contour graph, but do not have contours in the contour graph.

        Args:
            slice_sequence (SliceSequence): The table of all slices used in the
                structure set.
        '''
        # Check for empty slice_sequence
        if not slice_sequence:
            return

        # Get the min and max slice indexes in the contour graph
        if len(self.contour_graph) == 0:
            return

        contour_slices = [
            self.contour_graph.nodes[node]['contour'].slice_index
            for node in self.contour_graph.nodes
        ]
        min_slice = min(contour_slices)
        max_slice = max(contour_slices)

        # For all slice indexes in the SliceSequence
        for slice_index in slice_sequence.slices:
            # If the slice index is not between the maximum and minimum slice
            # indexes in the contour graph, continue
            if slice_index < min_slice or slice_index > max_slice:
                continue

            # If the slice index is already in the contour graph, continue
            slice_exists = any(
                self.contour_graph.nodes[node]['contour'].slice_index == slice_index
                for node in self.contour_graph.nodes
            )
            if slice_exists:
                continue

            # Find all edges where the slice index is between the node slices
            edges_to_interpolate = []
            for edge in self.contour_graph.edges():
                node1, node2 = edge
                slice1 = self.contour_graph.nodes[node1]['contour'].slice_index
                slice2 = self.contour_graph.nodes[node2]['contour'].slice_index

                # Check if slice_index is between slice1 and slice2
                if min(slice1, slice2) < slice_index < max(slice1, slice2):
                    edges_to_interpolate.append((node1, node2))

            # For each edge that spans this slice index
            for node1, node2 in edges_to_interpolate:
                contour1 = self.contour_graph.nodes[node1]['contour']
                contour2 = self.contour_graph.nodes[node2]['contour']

                # Create interpolated polygon
                slices = [contour1.slice_index, contour2.slice_index]
                interpolated_polygon = interpolate_polygon(
                    slices, contour1.polygon, contour2.polygon
                )

                # Create interpolated contour with parameters from first node
                interpolated_contour = Contour(
                    roi=contour1.roi,
                    slice_index=slice_index,
                    polygon=interpolated_polygon,
                    existing_contours=[],
                    is_interpolated=True,
                    is_boundary=contour1.is_boundary,
                    is_hole=contour1.is_hole,
                    hole_type=contour1.hole_type,
                    region_index=contour1.region_index
                )

                # Add the interpolated contour to the graph
                interpolated_label = interpolated_contour.index
                self.contour_graph.add_node(interpolated_label,
                                          contour=interpolated_contour)

                # Add edges from interpolated contour to both original nodes
                contour_match1 = ContourMatch(contour1, interpolated_contour)
                self.contour_graph.add_edge(node1, interpolated_label,
                                          match=contour_match1)

                contour_match2 = ContourMatch(contour2, interpolated_contour)
                self.contour_graph.add_edge(node2, interpolated_label,
                                          match=contour_match2)

        # Update the contour lookup table
        self.contour_lookup = build_contour_lookup(self.contour_graph)

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

    def build_region_table(self):
        '''Build a DataFrame of RegionSlices for each RegionIndex and SliceIndex.

        The region table is a DataFrame containing RegionSlices with the
        following columns:
            - SliceIndex
            - RegionSlice
            - Empty
            - Interpolated
        Args:
            structure (StructureShape): The structure containing the contour graph
                and slice sequence.
        '''
        enclosed_region_data = []

        # Get slice indexes from contour_lookup instead of slice_sequence
        contour_lookup = build_contour_lookup(self.contour_graph)
        slice_indexes = contour_lookup['SliceIndex'].unique()

        # Iterate through each unique SliceIndex
        for slice_index in slice_indexes:
            # Create a RegionSlice for the given SliceIndex
            region_slice = RegionSlice(self.contour_graph, slice_index)
            # Add the RegionSlice to the DataFrame
            enclosed_region_data.append({
                'SliceIndex': slice_index,
                'RegionSlice': region_slice,
                'Empty': region_slice.is_empty,
                'Interpolated': region_slice.is_interpolated
            })

        # Create the enclosed_region DataFrame
        enclosed_region_table = pd.DataFrame(enclosed_region_data)
        self.region_table = enclosed_region_table

    def finalize(self, slice_sequence: SliceSequence):
        '''Add interpolated contours, calculate volumes and build the region table.

        This method adds interpolated contours, based on the input slice
        sequence. It then calculates the physical, exterior, and hull volumes
        of the structure and builds the region table.

        Args:
            slice_sequence (SliceSequence): The slice sequence to use for
                interpolated contours.
        '''
        self.add_interpolated_contours(slice_sequence)
        # Calculate volumes
        self.calculate_physical_volume()
        self.calculate_exterior_volume()
        self.calculate_hull_volume()
        self.build_region_table()

    def get_contour(self, label: ContourIndex) -> Contour:
        '''Retrieve the contour representation of the structure.

        Returns:
            Contour: The selected contour from the structure.
        '''
        contour = self.contour_graph.nodes(data=True)[label]['contour']
        return contour

    def get_slice(self, slice_index: SliceIndexType)->RegionSlice:
        '''Get the RegionSlice for a given slice index.

        Args:
            slice_index (SliceIndexType): The slice index of the desired
                RegionSlice.

        Returns:
            RegionSlice: The RegionSlice object for the given slice index.
        '''
        idx = self.region_table['SliceIndex'] == slice_index
        if not any(idx):
            return None
        return self.region_table.loc[idx, 'RegionSlice'].values[0]

    def relate(self, other: 'StructureShape') -> 'DE27IM':
        '''Relate this structure to another structure.

        This method identifies common slices between the two structures and
        creates a DE27IM relationship for each common slice and merges the slice
        relations to get the overall relationship between the two structures.

        Args:
            other (StructureShape): The other structure to relate to.

        Returns:
            DE27IM: A DE27IM relationship object containing the relationship
            between the two structures.
        '''
        # 1. Create an initial blank DE27IM relationship object.
        composite_relation = DE27IM()
        # 2. Identify slices where either structures have non-empty RegionSlice objects.
        slices_self = set(self.region_table['SliceIndex'])
        slices_other = set(other.region_table['SliceIndex'])
        used_slices = slices_self | slices_other
        # 3. Find the common slices for the two structures.
        this_slice_mask = self.region_table.SliceIndex.isin(used_slices)
        this_mask = this_slice_mask & ~self.region_table.Empty
        regions_self = self.region_table.loc[this_mask,
                                             ['SliceIndex', 'RegionSlice']]
        regions_self.set_index('SliceIndex', inplace=True)

        other_slice_mask = other.region_table.SliceIndex.isin(used_slices)
        other_mask = other_slice_mask & ~other.region_table.Empty
        regions_other = other.region_table.loc[other_mask,
                                             ['SliceIndex', 'RegionSlice']]
        regions_other.set_index('SliceIndex', inplace=True)
        regions = regions_self.join(regions_other, how='outer',
                                    lsuffix='_self', rsuffix='_other')

        # 4. For each common slice, get and merge the DE27IM relationship.
        for slice_index, row in regions.iterrows():
            region_self = row['RegionSlice_self']
            region_other = row['RegionSlice_other']
            relation = DE27IM(region_self, region_other)
            composite_relation.merge(relation)
        return composite_relation
