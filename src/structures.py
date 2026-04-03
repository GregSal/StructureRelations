'''Contains the structure class.
'''
from typing import Callable, List, Optional
from itertools import combinations
from dataclasses import dataclass
import logging
import math

import pandas as pd
import networkx as nx
from shapely.errors import GEOSException

from types_and_classes import ROI_Type, SliceIndexType
from types_and_classes import ContourIndex
from contours import SliceSequence, Contour, ContourMatch
from contours import interpolate_polygon
from contour_graph import build_contour_graph, build_contour_lookup
from region_slice import RegionSlice
from relations import DE27IM


# %% Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolumeMetrics:
    '''Container for structure volume metrics in cm^3.'''

    physical: float = 0.0
    exterior: float = 0.0
    hull: float = 0.0

# %% StructureShape class
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

        volume_metrics (VolumeMetrics): Physical, exterior, and hull volumes
            in cm^3.
    '''

    def __init__(self, roi: ROI_Type, name: str = None):
        if name is None:
            self.name = f"Structure {roi}"
        else:
            self.name = name
        self.roi = roi
        self.contour_graph = nx.DiGraph()
        self.contour_lookup = pd.DataFrame()
        self.region_table = pd.DataFrame()
        self.volume_metrics = VolumeMetrics()

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

        related_contour_ref = {}
        daughter_contour_ref = {}

        # For all slice indexes in the SliceSequence
        for slice_index in slice_sequence.slices:
            # If the slice index is not between the maximum and minimum slice
            # indexes in the contour graph, continue
            if slice_index < min_slice or slice_index > max_slice:
                continue
            # Find all edges where the slice index is between the node slices
            edges_to_interpolate = []
            for edge in self.contour_graph.edges():
                source, target = edge
                slice_source = self.contour_graph.nodes[source]['contour'].slice_index
                slice_target = self.contour_graph.nodes[target]['contour'].slice_index
                # For directed graph, source should always be < target
                # Check if slice_index is between them
                if slice_source < slice_index < slice_target:
                    edges_to_interpolate.append((source, target))

            # For each edge that spans this slice index
            for source, target in edges_to_interpolate:
                contour_source = self.contour_graph.nodes[source]['contour']
                contour_target = self.contour_graph.nodes[target]['contour']

                # Create interpolated polygon
                slices = [contour_source.slice_index, contour_target.slice_index]
                interpolated_polygon = interpolate_polygon(
                    slices, contour_source.polygon, contour_target.polygon
                )

                # Skip only true duplicates. Multiple contours on the same
                # slice can legitimately share region/role metadata.
                def _is_same_geometry(existing_polygon, new_polygon) -> bool:
                    # equals_exact avoids topology operations that can fail on
                    # near-degenerate interpolated polygons.
                    try:
                        if existing_polygon.equals_exact(
                            new_polygon,
                            tolerance=1e-3,
                        ):
                            return True
                    except GEOSException:
                        pass

                    # Fallback: coarse invariants prevent duplicate inserts
                    # without triggering topology predicates.
                    area_close = math.isclose(
                        existing_polygon.area,
                        new_polygon.area,
                        rel_tol=1e-6,
                        abs_tol=1e-4,
                    )
                    bounds_close = all(
                        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-3)
                        for a, b in zip(
                            existing_polygon.bounds,
                            new_polygon.bounds,
                        )
                    )
                    return area_close and bounds_close

                duplicate_exists = any(
                    existing.slice_index == slice_index
                    and existing.region_index == contour_source.region_index
                    and existing.is_boundary == contour_source.is_boundary
                    and existing.is_hole == contour_source.is_hole
                    and existing.hole_type == contour_source.hole_type
                    and _is_same_geometry(existing.polygon, interpolated_polygon)
                    for existing in (
                        self.contour_graph.nodes[node]['contour']
                        for node in self.contour_graph.nodes
                    )
                )
                if duplicate_exists:
                    continue

                # Create interpolated contour with parameters from source node
                interpolated_contour = Contour(
                    roi=contour_source.roi,
                    slice_index=slice_index,
                    polygon=interpolated_polygon,
                    existing_contours=[],
                    is_interpolated=True,
                    is_boundary=contour_source.is_boundary,
                    is_hole=contour_source.is_hole,
                    hole_type=contour_source.hole_type,
                    region_index=contour_source.region_index
                )

                # Add the interpolated contour to the graph
                interpolated_label = interpolated_contour.index
                self.contour_graph.add_node(interpolated_label,
                                          contour=interpolated_contour)

                # Add directed edges: source -> interpolated -> target
                # Remove the original edge from source to target
                self.contour_graph.remove_edge(source, target)

                # Add two new directed edges
                contour_match1 = ContourMatch(contour_source, interpolated_contour)
                self.contour_graph.add_edge(source, interpolated_label,
                                          match=contour_match1)

                contour_match2 = ContourMatch(interpolated_contour, contour_target)
                self.contour_graph.add_edge(interpolated_label, target,
                                          match=contour_match2)

                # Combine unique related_contours from source and target
                related_contours = (set(contour_source.related_contours) |
                                  set(contour_target.related_contours))
                related_contour_ref[interpolated_label] = list(related_contours)

                # Add daughter contour references
                if source not in daughter_contour_ref:
                    daughter_contour_ref[source] = []
                daughter_contour_ref[source].append(interpolated_label)
                if target not in daughter_contour_ref:
                    daughter_contour_ref[target] = []
                daughter_contour_ref[target].append(interpolated_label)

        for key, related_list in related_contour_ref.items():
            # Find all keys in daughter_contour_ref that match items in related_list
            matching_keys = [k for k in daughter_contour_ref if k in related_list]
            # Iterate over all unique pairs of matching_keys
            for k1, k2 in combinations(matching_keys, 2):
                daughters1 = set(daughter_contour_ref[k1])
                daughters2 = set(daughter_contour_ref[k2])
                common_daughters = daughters1 & daughters2
                if common_daughters:
                    contour = self.contour_graph.nodes[key]['contour']
                    contour.related_contours.extend(list(common_daughters))
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
        self.volume_metrics.physical = total_volume

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
        self.volume_metrics.exterior = total_volume

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
        self.volume_metrics.hull = total_volume

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

    def relate_to(
        self,
        other: 'StructureShape',
        tolerance=0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> 'DE27IM':
        '''Relate this structure to another structure.

        This method identifies common slices between the two structures and
        creates a DE27IM relationship for each common slice and merges the slice
        relations to get the overall relationship between the two structures.

        Args:
            other (StructureShape): The other structure to relate to.
            tolerance (float): The tolerance value to use for the boundaries of
                the relation.
            progress_callback (Optional[Callable[[int, int], None]]): Optional
                callback receiving (current_slice_index, total_slices).

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
        total_slices = len(regions.index)
        for current_slice_index, (slice_index, row) in enumerate(
            regions.iterrows(),
            start=1,
        ):
            region_self = row['RegionSlice_self']
            region_other = row['RegionSlice_other']
            relation = DE27IM(region_self, region_other, tolerance=tolerance)

            # Log slice_index and relation at debug level using lazy formatting
            logger.debug('SliceIndex: %s,\nRelationType: %s\nRelation:\n%s\n',
                         slice_index, relation.identify_relation(), relation)

            composite_relation.merge(relation)
            if progress_callback is not None:
                progress_callback(current_slice_index, total_slices)
        return composite_relation

    def relate(self, other: 'StructureShape', tolerance=0.0) -> 'DE27IM':
        '''Backward-compatible wrapper for relate_to.'''
        return self.relate_to(other=other, tolerance=tolerance)

    def get_region_indexes(self, include_boundaries=True,
                           include_holes=True,
                           include_interpolated=True)->List[str]:
        '''Get the list of RegionIndexes in the structure.

        Returns:
            List[str]: The list of RegionIndexes in the structure.
        '''
        region_indexes = set()
        if not include_interpolated:
            region_slices = self.region_table.loc[
                ~self.region_table.Interpolated, 'RegionSlice']
        else:
            region_slices = self.region_table['RegionSlice']
        for region_slice in list(region_slices):
            region_labels = region_slice.get_region_indexes(include_boundaries,
                                                            include_holes)
            region_indexes.update(region_labels)
        return list(region_indexes)
