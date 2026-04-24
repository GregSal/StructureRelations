'''StructureSet class for managing multiple structures and their relationships.
'''
# %% Imports
from typing import Callable, Dict, List, Optional
import logging
import re
from pathlib import Path

import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type
from contours import build_contour_table
from structures import StructureShape
from relations import RELATIONSHIP_TYPES, RelationshipType
from relationships import StructureRelationship
from dicom import DicomStructureFile
from utilities import round_value
from metrics import get_config as get_metrics_config, MetricCalculatorRegistry
from metrics.data_structures import (
    RelationshipMetrics,
    MarginMetrics,
    DistanceMetrics,
    VolumeMetrics,
    SurfaceMetrics,
    GeometryMetrics,
)


# %% Configure logging if not already configured
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


# %% Class Definition
class StructureSet:
    '''Class for managing multiple StructureShape objects and their relationships.

    Attributes:
        structures (Dict[ROI_Type, StructureShape]): Dictionary of structures
            keyed by ROI.
        slice_sequence (SliceSequence): The slice sequence used across all
            structures.
        relationship_graph (nx.Graph): Graph where nodes are structures and
            edges are relationships.
        dicom_structure_file (DicomStructureFile): Optional reference to the
            source DICOM file.
        tolerance (float): Tolerance value for structure relationships.
        unit (str): Unit of measurement for coordinates and distances.
            Defaults to 'cm'.
    '''

    def __init__(self,
                 slice_data: Optional[List] = None,
                 dicom_structure_file: Optional[DicomStructureFile] = None,
                 tolerance=0.0,
                 unit: str = 'cm',
                 auto_calculate_relationships: bool = True,
                 auto_calculate_logical_flags: bool = True,
                 include_structures: Optional[List[str]] = None,
                 exclude_structures: Optional[List[str]] = None):
        '''Initialize the StructureSet.

        Args:
            slice_data (List, optional): List of ContourPoints for building structures.
                If None, creates an empty StructureSet.
                Deprecated - use dicom_structure_file instead.
            dicom_structure_file (DicomStructureFile, optional): DICOM structure
                file object containing contour points. Takes precedence over
                slice_data if both are provided. When using DICOM data, the
                unit is automatically set to 'cm' (DICOM coordinates are
                converted from mm to cm).
            tolerance (float, optional): Tolerance value for structure
                relationships. Defaults to 0.0.
            unit (str, optional): Unit of measurement for coordinates and
                distances. Defaults to 'cm'. When building from DICOM data,
                this is automatically set to 'cm'.
            auto_calculate_relationships (bool, optional): Whether to
                automatically calculate relationships as part of finalize.
                Defaults to True.
            auto_calculate_logical_flags (bool, optional): Whether to
                automatically calculate logical flags during finalize.
                Defaults to True.
            include_structures (List[str], optional): List of structure name
                patterns to include. A structure is included if its name
                exactly matches or matches via regex any pattern in this list.
                If None, all structures are included (subject to exclusion).
            exclude_structures (List[str], optional): List of structure name
                patterns to exclude. A structure is excluded if its name
                exactly matches or matches via regex any pattern in this list.
                Exclusion takes precedence over inclusion.
        '''
        self.structures = {}
        self.slice_sequence = None
        self.relationship_graph = nx.DiGraph()
        self.dicom_structure_file = dicom_structure_file
        self.tolerance = tolerance
        self.unit = unit
        self.auto_calculate_relationships = auto_calculate_relationships
        self.auto_calculate_logical_flags = auto_calculate_logical_flags
        self.include_structures = include_structures
        self.exclude_structures = exclude_structures
        self.relationship_progress: Dict[str, float | int | str] = {
            'current_pair': 0,
            'total_pairs': 0,
            'current_slice': 0,
            'total_slices': 0,
            'status': 'idle',
            'percent_complete': 0.0,
        }
        self._relationship_matrix_cache: Optional[pd.DataFrame] = None
        self._summary_cache: Optional[pd.DataFrame] = None
        self.slice_relationship_records: Dict[str, List[dict]] = {}

        # Prioritize dicom_structure_file over slice_data
        if dicom_structure_file is not None:
            self.build_from_dicom_file()
        elif slice_data is not None:
            self.build_from_slice_data(slice_data)

    def _should_include_structure(self, structure_name: str) -> bool:
        '''Check if a structure should be included based on filter patterns.

        Args:
            structure_name (str): Name of the structure to check.

        Returns:
            bool: True if structure should be included, False otherwise.

        Notes:
            - Exclusion patterns take precedence over inclusion patterns.
            - Patterns are matched using both exact string matching and regex.
            - Invalid regex patterns are silently skipped.
        '''
        # Check exclusion patterns first (takes precedence)
        if self.exclude_structures:
            for pattern in self.exclude_structures:
                # Try exact match first
                if structure_name == pattern:
                    logger.debug('Structure %s excluded by exact match: %s',
                                 structure_name, pattern)
                    return False
                # Try regex match
                try:
                    if re.search(pattern, structure_name):
                        logger.debug('Structure %s excluded by regex match: %s',
                                     structure_name, pattern)
                        return False
                except re.error:
                    # Invalid regex, skip
                    logger.debug('Invalid regex pattern in exclude_structures: %s',
                                 pattern)

        # Check inclusion patterns
        if self.include_structures:
            for pattern in self.include_structures:
                # Try exact match first
                if structure_name == pattern:
                    logger.debug('Structure %s included by exact match: %s',
                                 structure_name, pattern)
                    return True
                # Try regex match
                try:
                    if re.search(pattern, structure_name):
                        logger.debug('Structure %s included by regex match: %s',
                                     structure_name, pattern)
                        return True
                except re.error:
                    # Invalid regex, skip
                    logger.debug('Invalid regex pattern in include_structures: %s',
                                 pattern)
            # No match found in include list, exclude
            logger.debug('Structure %s not matched by any include pattern',
                         structure_name)
            return False

        # No include_structures filter specified, include by default
        # (structure already passed exclusion checks if we got here)
        return True

    def build_from_dicom_file(self) -> None:
        '''Build structures from DicomStructureFile following the StructureSet
        process.

        When building from DICOM, the unit is set to 'cm' because DICOM
        coordinates (in mm) are converted to cm during parsing.
        '''
        if not self.dicom_structure_file.contour_points:
            logger.warning("No contour points found in DicomStructureFile")
            return
        self.tolerance = self.dicom_structure_file.resolution
        # DICOM coordinates are converted from mm to cm, so unit is always 'cm'
        self.unit = 'cm'
        logger.info("Building StructureSet from %d contour points (unit: %s)",
                    len(self.dicom_structure_file.contour_points), self.unit)
        self.build_from_slice_data(self.dicom_structure_file.contour_points)

    def build_from_slice_data(self, slice_data: List) -> None:
        '''Build structures from slice data following the StructureSet process.

        Args:
            slice_data (List): List of ContourPoints objects.
        '''
        # 1. Create a contour_table
        contour_table, self.slice_sequence = build_contour_table(slice_data)

        # Get unique ROIs
        unique_rois = contour_table['ROI'].unique()

        # Get structure names from DicomStructureFile if available
        structure_names_dict = {}
        if self.dicom_structure_file and hasattr(self.dicom_structure_file, 'structure_names'):
            structure_names_dict = self.dicom_structure_file.structure_names
            logger.debug("Using structure names from DicomStructureFile: %s", structure_names_dict)

        # 2. For each ROI in the contour_table:
        for roi in unique_rois:
            # 2.1. Create a StructureShape object from the contour table
            logger.debug('Building structure for ROI: %s', roi)

            # Use meaningful name from DICOM file if available, otherwise use generic name
            if roi in structure_names_dict:
                structure_name = structure_names_dict[roi]
                logger.debug('Using DICOM structure name for ROI %s: %s', roi, structure_name)
            else:
                structure_name = f'Structure_{roi}'
                logger.debug('Using generic name for ROI %s: %s', roi, structure_name)

            # Check if structure should be included based on filters
            if not self._should_include_structure(structure_name):
                logger.info('Skipping structure %s (%s) due to filters', structure_name, roi)
                continue

            logger.info('Adding structure %s (%s)', structure_name, roi)
            structure = StructureShape(roi=roi, name=structure_name)
            self.slice_sequence = structure.add_contour_graph(
                contour_table,
                self.slice_sequence
                )
            #logger.debug('Slice sequence after ROI %s:\n%s', roi,
            #             self.slice_sequence.sequence)

            # 2.2. Add the StructureShape object to dictionary with the ROI as the key
            self.structures[roi] = structure

            # Add structure to relationship graph
            self.relationship_graph.add_node(roi, structure=structure)

        # 2.3 & 2.4. Use the SliceSequence to add interpolated contours and generate RegionSlices
        for structure in self.structures.values():
            logger.debug('Finalizing structure %s (%s)', structure.name, structure.roi)
            structure.finalize(self.slice_sequence)
        self.finalize(
            calculate_relationships=self.auto_calculate_relationships,
            calculate_logical_flags=self.auto_calculate_logical_flags,
        )

    def finalize(
        self,
        calculate_relationships: bool = True,
        calculate_logical_flags: bool = True,
    ) -> None:
        '''Complete the StructureSet process by calculating relationships.

        This method:
        3. Determines the relationships between the StructureShape objects.
        4. Constructs a graph where the nodes are the StructureShape objects
           and the edges are the relationships between them.
        5. Calculates logical relationship flags based on graph topology.

        Args:
            calculate_relationships (bool): Whether to calculate geometric
                relationships.
            calculate_logical_flags (bool): Whether to calculate logical flags.
        '''
        if calculate_relationships:
            self.calculate_relationships()
        if calculate_logical_flags:
            self.calculate_logical_flags()

    def _invalidate_caches(self) -> None:
        '''Invalidate cached summary and matrix data.'''
        self._relationship_matrix_cache = None
        self._summary_cache = None

    def add_structure(self, structure: StructureShape) -> None:
        '''Add a structure to the set.

        Args:
            structure (StructureShape): The structure to add.
        '''
        self.structures[structure.roi] = structure
        self._invalidate_caches()

        # Add structure to relationship graph
        self.relationship_graph.add_node(structure.roi, structure=structure)

    def calculate_relationships(
            self,
            force=False,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        '''Calculate relationships between all structure pairs.

        This method calculates the DE27IM spatial relationship between every
        pair of structures and stores the results in the relationship_graph
        as edge attributes.

        When supplied, progress_callback is called after each structure pair
        completes with (completed_pairs, total_pairs).

        If relationships have already been calculated, this method does nothing.
        '''
        sorted_structures = self.get_structures_by_volume()
        structure_rois = [s.roi for s in sorted_structures]

        # Check if relationships have already been calculated
        # If the graph has the expected number of edges, skip recalculation
        expected_edges = len(structure_rois) * (len(structure_rois) - 1) // 2
        self.relationship_progress.update({
            'current_pair': 0,
            'total_pairs': expected_edges,
            'current_slice': 0,
            'total_slices': 0,
            'status': 'starting',
            'percent_complete': 0.0,
        })
        if self.relationship_graph.number_of_edges() >= expected_edges:
            if not force:
                logger.debug('Relationships already calculated, skipping recalculation')
                self.relationship_progress.update({
                    'status': 'already calculated',
                    'percent_complete': 100.0 if expected_edges else 0.0,
                })
                return
            logger.debug('Force recalculation enabled, recalculating relationships')
            # Clear existing edges but keep nodes for recalculation
            self.relationship_graph.clear_edges()
        self._invalidate_caches()
        self.slice_relationship_records = {}

        logger.info('Calculating relationships for %d structures',
                    len(structure_rois))
        completed_pairs = 0
        total_pairs = expected_edges

        # Calculate relationships for all pairs
        for i, roi_a in enumerate(structure_rois):
            for roi_b in structure_rois[i+1:]:
                structure_a = self.structures[roi_a]
                structure_b = self.structures[roi_b]

                current_pair = completed_pairs + 1
                pair_status = (
                    f'pair {current_pair}/{total_pairs}: '
                    f'{structure_a.name} vs {structure_b.name}'
                )
                self.relationship_progress.update({
                    'current_pair': current_pair,
                    'status': pair_status,
                })

                def update_slice_progress(
                    current_slice_index: int,
                    total_slices: int,
                ) -> None:
                    self.relationship_progress['current_slice'] = current_slice_index
                    self.relationship_progress['total_slices'] = total_slices
                    if total_pairs <= 0:
                        self.relationship_progress['percent_complete'] = 100.0
                        return
                    pair_fraction = (
                        current_slice_index / total_slices
                        if total_slices > 0
                        else 1.0
                    )
                    self.relationship_progress['percent_complete'] = (
                        (completed_pairs + pair_fraction) * 100.0 / total_pairs
                    )

                pair_key = f'{min(roi_a, roi_b)}|{max(roi_a, roi_b)}'
                pair_slice_records: List[dict] = []

                def make_capture_slice_relation(
                    pair_records: List[dict],
                ) -> Callable[[float, object, object, object], None]:
                    def capture_slice_relation(
                        slice_index: float,
                        relation: object,
                        region_self: object,
                        region_other: object,
                    ) -> None:
                        relation_type = relation.identify_relation()

                        def _slice_is_interpolated(region: object) -> bool:
                            return bool(getattr(region, 'is_interpolated', False))

                        def _slice_has_boundary(region: object) -> bool:
                            if region is None or not hasattr(region, 'merged_boundary'):
                                return False
                            try:
                                boundary = region.merged_boundary()
                            except TypeError:
                                return False
                            return bool(boundary) and not boundary.is_empty

                        pair_records.append({
                            'slice_index': float(slice_index),
                            'relation_type': relation_type.label,
                            'relation_symbol': relation_type.symbol,
                            'is_interpolated': (
                                _slice_is_interpolated(region_self)
                                or _slice_is_interpolated(region_other)
                            ),
                            'has_boundary': (
                                _slice_has_boundary(region_self)
                                or _slice_has_boundary(region_other)
                            ),
                        })

                    return capture_slice_relation

                capture_slice_relation = make_capture_slice_relation(pair_slice_records)

                # Calculate the DE27IM relationship
                de27im_relationship = structure_a.relate_to(
                    structure_b,
                    tolerance=self.tolerance,
                    progress_callback=update_slice_progress,
                    slice_relation_callback=capture_slice_relation,
                )
                relation_type = de27im_relationship.identify_relation()
                status_message = (
                    f'Calculated relationships between {structure_a.name} '
                    f'(ROI {structure_a.roi}) and {structure_b.name} '
                    f'(ROI {structure_b.roi}) as: {relation_type.label}'
                )
                logger.info(status_message)
                # Create StructureRelationship object
                relationship = StructureRelationship(
                    de27im=de27im_relationship,
                    is_identical=False
                )

                # Compute per-region relationships and attach to the result.
                per_region = structure_a.relate_regions(
                    structure_b, tolerance=self.tolerance
                )
                relationship.per_region_relations = per_region if per_region else None

                # Add edge to graph with relationship data
                self.relationship_graph.add_edge(
                    roi_a, roi_b,
                    relationship=relationship
                )
                self.slice_relationship_records[pair_key] = pair_slice_records

                completed_pairs += 1
                self.relationship_progress.update({
                    'current_pair': completed_pairs,
                    'current_slice': self.relationship_progress.get('total_slices', 0),
                    'status': status_message,
                    'current_structure': (
                        f'{structure_a.name} (ROI {structure_a.roi}) vs '
                        f'{structure_b.name} (ROI {structure_b.roi})'
                    ),
                    'percent_complete': (
                        (completed_pairs * 100.0 / total_pairs)
                        if total_pairs > 0
                        else 100.0
                    ),
                })
                if progress_callback is not None:
                    progress_callback(completed_pairs, total_pairs)

                try:
                    rel_type = relationship.relationship_type
                    logger.debug('Calculated relationship between ROI %s and ROI %s: %s',
                                 structure_a.name, structure_b.name, rel_type)
                except (AttributeError, KeyError) as e:
                    logger.debug('Calculated relationship between ROI %s and ROI %s (error accessing type: %s)',
                                 structure_a.name, structure_b.name, str(e))

        self.relationship_progress.update({
            'status': 'complete',
            'percent_complete': 100.0,
        })
        self._invalidate_caches()

    def calculate_relationships_with_progress(self, force=False) -> None:
        '''Calculate relationships with optional tqdm terminal progress output.

        Uses tqdm when available. If tqdm is not installed, logs progress at
        info level after each completed pair.
        '''
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except ImportError:
            logger.info('tqdm not available; using logging-based progress output')

            def log_progress(completed_pairs: int, total_pairs: int) -> None:
                if total_pairs <= 0:
                    percent = 100.0
                else:
                    percent = completed_pairs * 100.0 / total_pairs
                logger.info(
                    'Relationship progress: %d/%d pairs (%.1f%%)',
                    completed_pairs,
                    total_pairs,
                    percent,
                )

            self.calculate_relationships(
                force=force,
                progress_callback=log_progress,
            )
            return

        total_pairs = len(self.structures) * (len(self.structures) - 1) // 2
        with tqdm(total=total_pairs, desc='Relationships', unit='pair') as pbar:
            def tqdm_progress(completed_pairs: int, total_pairs: int) -> None:
                pbar.total = total_pairs
                pbar.n = completed_pairs
                pbar.refresh()

            self.calculate_relationships(
                force=force,
                progress_callback=tqdm_progress,
            )

    def _build_transitive_subgraph(self) -> nx.DiGraph:
        '''Build a directed graph containing only transitive relationships.

        This subgraph includes all edges where the relationship type is
        transitive (e.g., CONTAINS, SHELTERS, SURROUNDS, EQUAL).

        Returns:
            nx.DiGraph: Subgraph with same nodes as relationship_graph but
                containing only transitive relationship edges.
        '''
        transitive_graph = nx.DiGraph()
        # Add all nodes
        transitive_graph.add_nodes_from(self.relationship_graph.nodes())

        # Add transitive edges
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(
                data=True):
            relationship = edge_data['relationship']
            # Skip self-relationships
            if relationship.is_identical:
                continue
            rel_type = relationship.relationship_type
            if not rel_type:
                continue

            # Add edge if transitive
            if rel_type.is_transitive:
                transitive_graph.add_edge(roi_a, roi_b,
                                          relationship=relationship)
                if rel_type.is_symmetric:
                    transitive_graph.add_edge(roi_b, roi_a,
                                              relationship=relationship)

        return transitive_graph

    def _build_implied_subgraph(
        self,
        target_relation: 'RelationshipType'
    ) -> nx.DiGraph:
        '''Build a directed graph of implied relationships for a target type.

        Args:
            target_relation (RelationshipType): The relationship to infer
                from implied edges (e.g., CONTAINS implied by PARTITIONED).
                Direct edges of the target relation are also included so
                mixed paths can be evaluated.

        Returns:
            nx.DiGraph: Subgraph with direct target edges and edges that
                imply the target relation.
        '''
        implied_graph = nx.DiGraph()
        implied_graph.add_nodes_from(self.relationship_graph.nodes())

        for roi_a, roi_b, edge_data in self.relationship_graph.edges(
                data=True):
            relationship = edge_data['relationship']
            if relationship.is_identical:
                continue

            rel_type = relationship.relationship_type
            if not rel_type:
                continue

            is_direct_target_edge = rel_type == target_relation
            is_implied_target_edge = target_relation in rel_type.implied
            if is_direct_target_edge or is_implied_target_edge:
                implied_graph.add_edge(roi_a, roi_b,
                                       relationship=relationship,
                                       is_direct_target_edge=
                                       is_direct_target_edge)
                if rel_type.is_symmetric or target_relation.is_symmetric:
                    implied_graph.add_edge(roi_b, roi_a,
                                           relationship=relationship,
                                           is_direct_target_edge=
                                           is_direct_target_edge)

        return implied_graph

    def _is_valid_implied_path(
        self,
        implied_graph: nx.DiGraph,
        path: list[ROI_Type],
        target_relation: 'RelationshipType'
    ) -> bool:
        '''Check whether an alternate implied path can justify a relation.

        For CONTAINS/WITHIN-style inference, reject paths made entirely of
        implied PARTITIONED/PARTITIONS edges because those paths remain
        ambiguous. Mixed paths are valid, including when the final segment is
        the only direct target edge.

        Args:
            implied_graph (nx.DiGraph): Graph containing direct and implied
                edges for the target relation.
            path (list[ROI_Type]): Candidate path from source to target.
            target_relation (RelationshipType): Direct relation being tested.

        Returns:
            bool: True when the path is a valid alternate explanation for the
                target relation.
        '''
        if len(path) < 3:
            return False

        path_edge_data = [
            implied_graph[path[index]][path[index + 1]]
            for index in range(len(path) - 1)
        ]

        if target_relation.relation_type in {'CONTAINS', 'WITHIN'}:
            return any(
                edge_data.get('is_direct_target_edge', False)
                for edge_data in path_edge_data
            )

        return True

    def _iter_simple_paths(
        self,
        graph: nx.DiGraph,
        source: ROI_Type,
        target: ROI_Type,
    ):
        '''Yield simple paths from source to target using a generator.

        Wraps nx.shortest_simple_paths and suppresses no-path exceptions.

        Args:
            graph (nx.DiGraph): Graph to search.
            source (ROI_Type): Source node.
            target (ROI_Type): Target node.

        Yields:
            list: Simple paths from source to target.
        '''
        try:
            yield from nx.shortest_simple_paths(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return

    def _find_longest_alternate_transitive_path(
        self,
        transitive_graph: nx.DiGraph,
        source: ROI_Type,
        target: ROI_Type,
    ) -> Optional[list]:
        '''Return the longest simple path when more than one path exists.

        Returns None when only one path (the direct edge) exists, meaning
        the relationship is not logically derived.

        Args:
            transitive_graph (nx.DiGraph): Graph of transitive relationships.
            source (ROI_Type): Source structure ROI.
            target (ROI_Type): Target structure ROI.

        Returns:
            Optional[list]: Longest path as node list, or None if fewer than
                two paths exist.
        '''
        longest: Optional[list] = None
        count = 0
        for path in self._iter_simple_paths(transitive_graph, source, target):
            count += 1
            if longest is None or len(path) > len(longest):
                longest = path
        return longest if count > 1 else None

    def _find_longest_valid_implied_path(
        self,
        implied_graph: nx.DiGraph,
        source: ROI_Type,
        target: ROI_Type,
        target_relation: 'RelationshipType',
    ) -> Optional[list]:
        '''Return the longest valid implied path from source to target.

        Streams paths on demand, validates each with _is_valid_implied_path,
        and tracks the longest valid candidate without building a full list.

        Args:
            implied_graph (nx.DiGraph): Graph of direct and implied edges.
            source (ROI_Type): Source structure ROI.
            target (ROI_Type): Target structure ROI.
            target_relation (RelationshipType): Relation type for validation.

        Returns:
            Optional[list]: Longest valid path as node list, or None.
        '''
        longest: Optional[list] = None
        for path in self._iter_simple_paths(implied_graph, source, target):
            if self._is_valid_implied_path(implied_graph, path,
                                           target_relation):
                if longest is None or len(path) > len(longest):
                    longest = path
        return longest

    def calculate_logical_flags(self) -> None:
        '''Calculate logical relationship flags based on graph topology.

        This method analyzes the relationship graph structure to identify
        relationships that are logically derived from the graph topology
        rather than being direct geometric relationships. For example,
        relationships implied by transitivity or relationships that exist
        only through intermediate structures.

        The is_logical flag in StructureRelationship objects will be set
        to True for relationships identified as logical. The
        intermediate_structures field will contain ROI numbers of structures
        forming the logical path.

        Algorithm:
        1. Build a subgraph of transitive relationships
        2. For each edge in the original graph:
           - If multiple paths exist in transitive subgraph, mark as logical
           - Extract ROIs from longest path as intermediate structures
        3. Handle EQUAL relationships: mark downstream edges as logical
        '''
        logger.info('Calculating logical flags for relationships')

        # Build transitive subgraph
        transitive_graph = self._build_transitive_subgraph()

        # Find logical relationships via transitivity
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(
                data=True):
            relationship = edge_data['relationship']

            # Skip self-relationships and already marked logical
            if relationship.is_identical or relationship.is_logical:
                continue

            # Check if both ROIs exist in transitive subgraph
            if (roi_a not in transitive_graph or
                    roi_b not in transitive_graph):
                continue

            longest_path = self._find_longest_alternate_transitive_path(
                transitive_graph, roi_a, roi_b
            )
            if longest_path is not None:
                intermediate_rois = longest_path[1:-1]
                intermediate_structures = [
                    ROI_Type(roi) for roi in intermediate_rois
                ]
                relationship.is_logical = True
                relationship.intermediate_structures = intermediate_structures
                logger.debug(
                    'Identified logical relationship: ROI %d -> ROI %d '
                    '(intermediates: %s)',
                    roi_a, roi_b, intermediate_structures
                )

        # Find logical relationships via implied and mixed alternate paths
        implied_graph_cache = {}
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(
                data=True):
            relationship = edge_data['relationship']

            # Skip self-relationships and already marked logical
            if relationship.is_identical or relationship.is_logical:
                continue

            rel_type = relationship.relationship_type
            if not rel_type:
                continue

            if rel_type.relation_type not in implied_graph_cache:
                implied_graph_cache[rel_type.relation_type] = (
                    self._build_implied_subgraph(rel_type)
                )

            implied_graph = implied_graph_cache[rel_type.relation_type]
            if implied_graph.number_of_edges() == 0:
                continue
            if not any(
                not edge_data.get('is_direct_target_edge', False)
                for _, _, edge_data in implied_graph.edges(data=True)
            ):
                continue

            longest_path = self._find_longest_valid_implied_path(
                implied_graph, roi_a, roi_b, rel_type
            )
            if longest_path is not None:
                intermediate_rois = longest_path[1:-1]
                intermediate_structures = [
                    ROI_Type(roi) for roi in intermediate_rois
                ]
                relationship.is_logical = True
                relationship.intermediate_structures = intermediate_structures
                logger.debug(
                    'Identified implied logical relationship: ROI %d -> '
                    'ROI %d (intermediates: %s)',
                    roi_a, roi_b, intermediate_structures
                )

        # Handle EQUAL relationships
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(
                data=True):
            relationship = edge_data['relationship']
            if relationship.is_identical:
                continue

            rel_type = relationship.relationship_type
            if not rel_type or rel_type.relation_type != 'EQUAL':
                continue

            # Determine downstream (higher ROI number)
            upstream_roi = min(roi_a, roi_b)
            downstream_roi = max(roi_a, roi_b)

            # Mark all outgoing edges from downstream as logical
            for next_roi in self.relationship_graph.successors(
                    downstream_roi):
                if next_roi != upstream_roi:
                    next_relationship = self.relationship_graph[
                        downstream_roi][next_roi]['relationship']
                    if not next_relationship.is_identical:
                        next_relationship.is_logical = True
                        next_relationship.intermediate_structures = [
                            ROI_Type(upstream_roi)]
                        logger.debug(
                            'Identified EQUAL-derived logical '
                            'relationship: ROI %d -> ROI %d '
                            '(via EQUAL with ROI %d)',
                            downstream_roi, next_roi, upstream_roi
                        )

        logger.info('Logical flag calculation complete')

    def get_relationship(self, roi_a: ROI_Type, roi_b: ROI_Type) -> Optional[StructureRelationship]:
        '''Get the relationship between two structures.

        Args:
            roi_a (ROI_Type): First structure ROI (subject).
            roi_b (ROI_Type): Second structure ROI (object).

        Returns:
            Optional[StructureRelationship]: The complete relationship object
                containing DE27IM, flags, and metrics. Returns None if no
                relationship exists in that specific direction.
        '''
        if self.relationship_graph.has_edge(roi_a, roi_b):
            return self.relationship_graph[roi_a][roi_b]['relationship']
        else:
            return None

    def describe_relationship(
        self,
        roi_a: ROI_Type,
        roi_b: ROI_Type,
        show_multi: bool = True,
        drop_disjoint: bool = False,
        drop_overlaps: bool = False,
    ) -> Optional[dict]:
        '''Return a rich description of the relationship between two structures.

        Args:
            roi_a: First structure ROI (subject).
            roi_b: Second structure ROI (object).
            show_multi: When True, apply multi-region display-label rules.
            drop_disjoint: Exclude DISJOINT region pairs from label / summary.
            drop_overlaps: Exclude OVERLAPS region pairs from label / summary.

        Returns:
            Dict with keys:
              - ``consolidated_label`` (str): display label from
                StructureRelationship.display_label().
              - ``consolidated_type`` (str): relation_type string for the
                consolidated relationship (e.g. ``'CONTAINS'``).
              - ``has_multiple_regions`` (bool): True when multiple region
                pairs are present.
              - ``per_region`` (list[dict]): One entry per region pair with
                keys ``region_a``, ``region_b``, ``label``, ``type``.
              - ``summary`` (list[dict]): Unique relationship types ranked by
                priority, each with keys ``label``, ``type``, ``count``.
            Returns None when no relationship edge exists.
        '''
        from relations import count_relations_by_rank
        relationship = self.get_relationship(roi_a, roi_b)
        if relationship is None:
            return None

        per_types = relationship.region_relationship_types
        consolidated_type = relationship.relationship_type

        per_region_list = []
        for (idx_a, idx_b), rel_type in per_types.items():
            if drop_disjoint and rel_type.relation_type == 'DISJOINT':
                continue
            if drop_overlaps and rel_type.relation_type == 'OVERLAPS':
                continue
            per_region_list.append({
                'region_a': idx_a,
                'region_b': idx_b,
                'label': rel_type.label,
                'type': rel_type.relation_type,
            })

        ranked = count_relations_by_rank(
            per_types,
            drop_disjoint=drop_disjoint,
            drop_overlaps=drop_overlaps,
        )
        summary = [
            {'label': rt.label, 'type': rt.relation_type, 'count': cnt}
            for rt, cnt in ranked
        ]

        return {
            'consolidated_label': relationship.display_label(
                show_multi=show_multi,
                drop_disjoint=drop_disjoint,
                drop_overlaps=drop_overlaps,
            ),
            'consolidated_type': (
                consolidated_type.relation_type if consolidated_type else 'UNKNOWN'
            ),
            'has_multiple_regions': relationship.has_multiple_regions(),
            'per_region': per_region_list,
            'summary': summary,
        }

    def calculate_metric(
        self,
        roi_a: ROI_Type,
        roi_b: ROI_Type,
        metric_name: str
    ):
        '''Calculate a specific metric between two structures.

        This method provides a convenient interface for calculating metrics
        between structure pairs. It retrieves the relationship, gets the
        appropriate calculator, and computes the requested metric. The
        calculated metric is stored in a RelationshipMetrics object and
        persisted in the relationship graph.

        Args:
            roi_a (ROI_Type): First structure ROI.
            roi_b (ROI_Type): Second structure ROI.
            metric_name (str): Name of the metric calculator to use.
                Available metrics:
                - 'orthogonal_margins': Clearance in 6 directions (±X, ±Y, ±Z)
                - 'minimum_margin': Worst-case clearance distance
                - 'maximum_margin': Largest clearance (Hausdorff)
                - 'minimum_distance': Gap between disjoint structures

        Returns:
            Metric result dataclass (MarginMetrics, DistanceMetrics, etc.)
            The result is also stored in the relationship graph at:
            relationship_graph[roi_a][roi_b]['relationship'].metrics.<category>
            where <category> is 'margin', 'distance', 'volume', 'surface', or 'geometry'.

        Raises:
            ValueError: If metric_name is not recognized.
            KeyError: If ROI numbers don't exist in the structure set.

        Example:
            >>> structure_set = StructureSet(slice_data)
            >>> structure_set.finalize()
            >>> distance = structure_set.calculate_metric(1, 2, 'minimum_distance')
            >>> print(f"Distance: {distance.minimum_distance:.2f} cm")
            >>>
            >>> # Access via RelationshipMetrics object
            >>> rel = structure_set.get_relationship(1, 2)
            >>> print(f"Distance: {rel.metrics.distance.minimum_distance:.2f} cm")
            >>> print(f"Structures: {rel.metrics.structure_a_name} vs {rel.metrics.structure_b_name}")
        '''
        # Validate ROI numbers exist
        if roi_a not in self.structures:
            raise KeyError(f'Structure with ROI {roi_a} not found in structure set')
        if roi_b not in self.structures:
            raise KeyError(f'Structure with ROI {roi_b} not found in structure set')

        # Get structures
        structure_a = self.structures[roi_a]
        structure_b = self.structures[roi_b]

        # Get or create relationship
        relationship = self.get_relationship(roi_a, roi_b)
        if relationship is None:
            # Need to calculate relationship first
            logger.debug(
                'Relationship between %d and %d not found, calculating...',
                roi_a, roi_b
            )
            # Calculate DE27IM relationship
            de27im_relationship = structure_a.relate(structure_b, tolerance=self.tolerance)
            # Wrap in StructureRelationship object
            relationship = StructureRelationship(
                de27im=de27im_relationship,
                is_identical=False
            )
            # Add to graph
            self.relationship_graph.add_edge(
                roi_a, roi_b,
                relationship=relationship
            )
            logger.debug(
                'Created and stored new relationship between %d and %d in graph',
                roi_a, roi_b
            )

        # Get metric calculator
        registry = MetricCalculatorRegistry()
        calculator = registry.get_calculator(metric_name, get_metrics_config())

        if calculator is None:
            available = list(registry.get_all_calculators().keys())
            raise ValueError(
                f'Metric calculator "{metric_name}" not found. '
                f'Available calculators: {available}'
            )

        # Check if metric is applicable
        if not calculator.is_applicable(relationship):
            logger.warning(
                '%s metric is not applicable to %s relationship between '
                'structures %d and %d',
                metric_name,
                relationship.relationship_type.relation_type,
                roi_a,
                roi_b
            )
            # Still call calculate() to get N/A result
            return calculator.calculate(structure_a, structure_b, relationship)

        # Calculate metric
        logger.debug(
            'Calculating %s for structures %d (%s) and %d (%s)',
            metric_name,
            roi_a,
            structure_a.name,
            roi_b,
            structure_b.name
        )

        result = calculator.calculate(structure_a, structure_b, relationship)

        # Initialize RelationshipMetrics if it doesn't exist
        if relationship.metrics is None:
            relationship.metrics = RelationshipMetrics(
                structure_a_name=structure_a.name,
                structure_b_name=structure_b.name,
                structure_a_roi=roi_a,
                structure_b_roi=roi_b,
                region_count_a=structure_a.region_count,
                region_count_b=structure_b.region_count,
                unit=self.unit,
            )

        # Store result in appropriate field based on type
        if isinstance(result, MarginMetrics):
            relationship.metrics.margin = result
        elif isinstance(result, DistanceMetrics):
            relationship.metrics.distance = result
        elif isinstance(result, VolumeMetrics):
            relationship.metrics.volume = result
        elif isinstance(result, SurfaceMetrics):
            relationship.metrics.surface = result
        elif isinstance(result, GeometryMetrics):
            relationship.metrics.geometry = result
        else:
            logger.warning(
                'Unknown metric type %s for metric %s',
                type(result).__name__,
                metric_name
            )

        # Note: No need to update graph - relationship object is stored by reference
        # Changes to relationship.metrics are automatically reflected in the graph
        logger.debug(
            'Stored %s (%s) in relationship for structures %d and %d',
            metric_name,
            type(result).__name__,
            roi_a,
            roi_b
        )

        return result

    def get_structures_by_volume(
            self,
            volume_type: str | List[str] | None = None
    ) -> List[StructureShape]:
        '''Get structures sorted by volume.

        Args:
            volume_type (str | List[str] | None): Volume type(s) to sort by.
                - If str, sort by that single type.
                - If list, sort by the list order as priority.
                - If None, defaults to ['hull', 'exterior', 'physical'].
                Allowed values are 'physical', 'exterior', and 'hull'.

        Returns:
            List[StructureShape]: Structures sorted by volume (descending).
        '''
        if volume_type is None:
            volume_types = ['hull', 'exterior', 'physical']
        elif isinstance(volume_type, str):
            volume_types = [volume_type]
        else:
            volume_types = list(volume_type)

        if not volume_types:
            raise ValueError('At least one volume type must be provided.')

        # Define functions that return the relevant volume type for a structure.
        volume_getters = {
            'physical': lambda structure: structure.volume_metrics.physical,
            'exterior': lambda structure: structure.volume_metrics.exterior,
            'hull': lambda structure: structure.volume_metrics.hull,
        }
        # Verify that all provided volume types are one of the defined types.
        invalid_types = [
            vol_type for vol_type in volume_types
            if vol_type not in volume_getters
        ]
        if invalid_types:
            raise ValueError(
                f"Unknown volume type(s): {', '.join(invalid_types)}"
            )

        return sorted(
            self.structures.values(),
            key=lambda structure: tuple(
                volume_getters[vol_type](structure)
                for vol_type in volume_types
            ),
            reverse=True
        )

    def summary(self) -> pd.DataFrame:
        '''Get a summary of all structures in the set.

        Returns:
            pd.DataFrame: Summary table with structure information.
        '''
        if self._summary_cache is not None:
            return self._summary_cache.copy()

        summary_data = []
        for roi, structure in self.structures.items():
            interpolated_regions = structure.region_table.Interpolated
            original_regions = structure.region_table[~interpolated_regions]
            summary_data.append({
                'ROI': roi,
                'Name': structure.name,
                'Physical_Volume': round_value(structure.volume_metrics.physical,
                                               self.tolerance),
                'Exterior_Volume': round_value(structure.volume_metrics.exterior,
                                              self.tolerance),
                'Hull_Volume': round_value(structure.volume_metrics.hull,
                                          self.tolerance),
                'Num_Contours': len(structure.contour_graph),
                'Num_Regions': len(structure.get_region_indexes(include_holes=False)),
                'Num_Slices': len(original_regions),
                'Slice_Range': (f"{original_regions['SliceIndex'].min():.2f} to "
                                f"{original_regions['SliceIndex'].max():.2f}")
            })
        summary_df = pd.DataFrame(summary_data)
        if self.dicom_structure_file:
            summary_df = summary_df.join(
                self.dicom_structure_file.get_roi_labels(), on='ROI')

        self._summary_cache = summary_df
        return summary_df.copy()

    @property
    def relationship_matrix(self) -> pd.DataFrame:
        '''Get a summary of all relationships between structures as a matrix.

        Returns:
            pd.DataFrame: Matrix with Structure_A as index, Structure_B as columns,
                and StructureRelationship objects as values. The matrix is symmetric for
                symmetric relationships.
        '''
        if self._relationship_matrix_cache is not None:
            return self._relationship_matrix_cache.copy()

        # Return empty DataFrame if no structures
        if not self.structures:
            return pd.DataFrame()

        # Get all unique ROIs and their corresponding names for matrix dimensions
        sorted_structures = self.get_structures_by_volume()
        all_rois = [s.roi for s in sorted_structures]
        all_names = [self.structures[roi].name for roi in all_rois]

        # Create ROI to name mapping for lookups
        roi_to_name = {roi: self.structures[roi].name for roi in all_rois}

        # Create empty dict of dicts to hold relationship data
        # Initialize with StructureRelationship objects for unknown relationships
        relationship_data = {name_a: {name_b: StructureRelationship(
                                         de27im=None,
                                         is_identical=False
                                     )
                                      for name_b in all_names}
                                      for name_a in all_names}

        # Fill diagonal with self-relationships (identical structures)
        for name in all_names:
            relationship_data[name][name] = StructureRelationship(
                de27im=None,
                is_identical=True
            )

        # Fill matrix with calculated relationships
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(data=True):
            relationship_obj = edge_data['relationship']

            # Get structure names for the ROIs
            name_a = roi_to_name[roi_a]
            name_b = roi_to_name[roi_b]

            # Store the StructureRelationship object
            relationship_data[name_a][name_b] = relationship_obj

            # For symmetric relationships, also set the transpose
            if relationship_obj.relationship_type.is_symmetric:
                relationship_data[name_b][name_a] = relationship_obj
            else:
                # For non-symmetric relationships, set the complementary relationship
                complementary_type = relationship_obj.relationship_type.complementary
                if complementary_type:
                    complementary_obj = StructureRelationship(
                        de27im=None,
                        is_identical=False,
                        _override_type=complementary_type
                    )
                    relationship_data[name_b][name_a] = complementary_obj
        relationship_matrix = pd.DataFrame(relationship_data)

        # Set index and columns names for clarity
        relationship_matrix.index.name = 'Structure_B'
        relationship_matrix.columns.name = 'Structure_A'

        self._relationship_matrix_cache = relationship_matrix
        return relationship_matrix.copy()

    def filter_relationships_by_logical_mode(
            self,
            mode: str = 'limited',
            visible_rois: list = None
    ) -> tuple[pd.DataFrame, dict]:
        '''Filter relationships based on logical display mode.

        Filters relationships in the relationship matrix based on the
        specified logical display mode and currently visible ROIs.

        Args:
            mode (str): Display mode for logical relationships.
                - 'hide': No logical relationships are displayed
                - 'limited' (default): Logical relationships not shown if all
                  intermediates are visible; shown when any intermediate is hidden
                - 'show': All relationships shown normally
                - 'faded': Logical relationships shown with reduced opacity

            visible_rois (list, optional): List of ROI numbers currently visible
                in the diagram. Used for 'limited' mode to determine whether to
                show logical relationships dynamically. If None, all ROIs are
                assumed visible. Defaults to None.

        Returns:
            tuple: (filtered_matrix, logical_info_dict)
                - filtered_matrix: pd.DataFrame with relationships filtered
                  according to mode. For 'hide' mode, logical relationships
                  are replaced with None.
                - logical_info_dict: Dict mapping (roi_a, roi_b) tuples to
                  'should_fade' boolean. Used by frontend to apply styling.
        '''
        if not visible_rois:
            # Default to all ROIs visible
            visible_rois = list(self.structures.keys())

        # Get base relationship matrix
        matrix = self.relationship_matrix.copy()

        # Dict to track which relationships should be faded
        fade_info = {}

        for roi_a in self.structures:
            for roi_b in self.structures:
                if roi_a == roi_b:
                    continue

                # Get the relationship object
                name_a = self.structures[roi_a].name
                name_b = self.structures[roi_b].name

                try:
                    rel_obj = matrix.loc[name_b, name_a]
                except KeyError:
                    continue

                if not rel_obj or not hasattr(rel_obj, 'is_logical'):
                    continue

                # Skip non-logical relationships
                if not rel_obj.is_logical:
                    fade_info[(roi_a, roi_b)] = False
                    continue

                # Handle logical relationship based on mode
                if mode == 'hide':
                    # Hide all logical relationships
                    matrix.loc[name_b, name_a] = None
                    fade_info[(roi_a, roi_b)] = False

                elif mode == 'limited':
                    # Check if all intermediate structures are visible
                    intermediates = rel_obj.intermediate_structures
                    all_intermediates_visible = all(
                        roi in visible_rois for roi in intermediates
                    )

                    if all_intermediates_visible:
                        # Hide because all intermediates are shown
                        matrix.loc[name_b, name_a] = None
                        fade_info[(roi_a, roi_b)] = False
                    else:
                        # Show because at least one intermediate is hidden
                        fade_info[(roi_a, roi_b)] = False

                elif mode == 'faded':
                    # Mark for fading (frontend will apply opacity)
                    fade_info[(roi_a, roi_b)] = True

                elif mode == 'show':
                    # Show all relationships normally
                    fade_info[(roi_a, roi_b)] = False

        return matrix, fade_info

    def relationship_summary(self, symbol_map=None) -> pd.DataFrame:
        '''Get a summary of all relationships between structures.

        Args:
            symbol_map (dict, optional): Custom mapping from RelationshipType to
            symbols. Overrides default symbols from relationship_definitions.json.
            If provided, returns symbols instead of labels.
        Returns:
            pd.DataFrame: Adjacency matrix of a graph where nodes are structures
                and edges represent relationships. The values of the matrix are
                labels by default, or symbols if symbol_map is provided. The index
                and columns are structure names.
        '''
        def to_label_or_symbol(struct_relationship: StructureRelationship) -> str:
            if struct_relationship:
                try:
                    relationship_type = struct_relationship.relationship_type
                    # Check custom mapping first (if provided, use symbols)
                    if symbol_map and relationship_type in symbol_map:
                        return symbol_map[relationship_type]
                    # If no symbol_map, return label; otherwise return symbol
                    if symbol_map is None:
                        return relationship_type.label if relationship_type else 'Unknown'
                    return relationship_type.symbol if relationship_type else ''
                except (AttributeError, KeyError):
                    # Fallback for any error accessing relationship_type
                    return 'Unknown' if symbol_map is None else '?'
            # For empty/unknown relationships
            unknown_type = RELATIONSHIP_TYPES.get('UNKNOWN')
            if symbol_map and unknown_type in symbol_map:
                return symbol_map[unknown_type]
            if symbol_map is None:
                return unknown_type.label if unknown_type else 'Unknown'
            return unknown_type.symbol if unknown_type else ''

        relationship_matrix = self.relationship_matrix
        if relationship_matrix.empty:
            return pd.DataFrame()
        # Convert matrix to labels or symbols
        labeled_matrix = relationship_matrix.map(to_label_or_symbol)
        return labeled_matrix

    def get_relationship_matrix(self, row_rois=None, col_rois=None,
                                use_symbols=True) -> pd.DataFrame:
        '''Get a filtered relationship matrix with optional symbol notation.

        Args:
            row_rois (List[ROI_Type], optional): List of ROI numbers for matrix rows.
                If None, uses all structures.
            col_rois (List[ROI_Type], optional): List of ROI numbers for matrix columns.
                If None, uses all structures.
            use_symbols (bool, optional): If True, use symbolic notation instead of labels.
                Defaults to True.

        Returns:
            pd.DataFrame: Filtered relationship matrix with structure names as index/columns.
        '''
        # Get the base relationship matrix
        relationship_matrix = self.relationship_matrix
        if relationship_matrix.empty:
            return pd.DataFrame()

        # Filter by row ROIs if specified
        if row_rois is not None:
            # Convert ROI numbers to structure names
            row_names = [self.structures[roi].name for roi in row_rois if roi in self.structures]
            # Filter to only requested rows that exist in the matrix
            row_names = [name for name in row_names if name in relationship_matrix.index]
            relationship_matrix = relationship_matrix.loc[row_names, :]

        # Filter by column ROIs if specified
        if col_rois is not None:
            # Convert ROI numbers to structure names
            col_names = [self.structures[roi].name for roi in col_rois if roi in self.structures]
            # Filter to only requested columns that exist in the matrix
            col_names = [name for name in col_names if name in relationship_matrix.columns]
            relationship_matrix = relationship_matrix.loc[:, col_names]

        # Apply symbol mapping if requested
        if use_symbols:
            def get_symbol(sr):
                if not sr:
                    return '?'
                try:
                    rel_type = sr.relationship_type
                    if not rel_type:
                        return '?'
                    # Use reversed arrow indicator if applicable
                    symbol = rel_type.symbol
                    if rel_type.reversed_arrow:
                        # Add indicator that this is reversed (e.g., ⊃ for WITHIN)
                        # For now, we'll flip common symbols
                        symbol_flip = {'⊂': '⊃', '⊏': '⊐'}
                        symbol = symbol_flip.get(symbol, symbol)
                    # Add brackets around logical relationships
                    if getattr(sr, 'is_logical', False):
                        symbol = f'[{symbol}]'
                    return symbol
                except (AttributeError, KeyError):
                    return '?'
            relationship_matrix = relationship_matrix.map(get_symbol)
        else:
            # Use labels
            def get_label(sr):
                if not sr:
                    return 'Unknown'
                try:
                    rel_type = sr.relationship_type
                    label = rel_type.label if rel_type else 'Unknown'
                    # Add brackets around logical relationships
                    if getattr(sr, 'is_logical', False):
                        label = f'[{label}]'
                    return label
                except (AttributeError, KeyError):
                    return 'Unknown'
            relationship_matrix = relationship_matrix.map(get_label)

        return relationship_matrix

    def _get_default_symbol_map(self) -> dict:
        '''Get the default symbol map for relationship types.

        Returns:
            dict: Mapping from RelationshipType to unicode symbols loaded from
                  relationship_definitions.json.
        '''
        return {rel: rel.symbol for rel in RELATIONSHIP_TYPES.values()}

    @staticmethod
    def _make_pair_key(roi_a: ROI_Type, roi_b: ROI_Type) -> str:
        '''Return a stable key for a pair of ROI values.'''
        return f'{min(roi_a, roi_b)}|{max(roi_a, roi_b)}'

    def _serialize_slice_relationships(self, all_rois: List[int]) -> dict:
        '''Build slice-indexed relationship summaries for frontend use.'''
        if not self.slice_relationship_records:
            return {}

        allowed_rois = set(all_rois)
        slice_relationships: Dict[str, List[dict]] = {}

        for pair_key, records in self.slice_relationship_records.items():
            roi_a_str, roi_b_str = pair_key.split('|')
            roi_a = int(roi_a_str)
            roi_b = int(roi_b_str)
            if roi_a not in allowed_rois or roi_b not in allowed_rois:
                continue

            structure_a = self.structures.get(roi_a)
            structure_b = self.structures.get(roi_b)
            if structure_a is None or structure_b is None:
                continue

            for record in records:
                slice_key = f"{float(record['slice_index']):.4f}"
                slice_relationships.setdefault(slice_key, []).append({
                    'rois': [roi_a, roi_b],
                    'pair_key': pair_key,
                    'label': (
                        f'{structure_a.name} / {structure_b.name}: '
                        f"{record['relation_type']}"
                    ),
                    'relation_type': record['relation_type'],
                    'relation_symbol': record['relation_symbol'],
                    'is_interpolated': record['is_interpolated'],
                    'has_boundary': record['has_boundary'],
                })

        return slice_relationships

    def to_dict(self, row_rois=None, col_rois=None, use_symbols=True,
                logical_relations_mode='limited', visible_rois=None) -> dict:
        '''Convert relationship matrix to dictionary for JSON serialization.

        Args:
            row_rois (List[ROI_Type], optional): List of ROI numbers for matrix rows.
            col_rois (List[ROI_Type], optional): List of ROI numbers for matrix columns.
            use_symbols (bool, optional): If True, use symbolic notation.
            logical_relations_mode (str, optional): How to display logical relationships.
                - 'hide': Hide all logical relationships
                - 'limited' (default): Hide logical if all intermediates visible
                - 'show': Show all relationships
                - 'faded': Show logical relationships with reduced opacity
            visible_rois (List[ROI_Type], optional): Currently visible ROI numbers
                for 'limited' mode. Defaults to None (all visible).

        Returns:
            dict: Dictionary with structure including faded_relationships field
                when logical_relations_mode is not 'show'.
                {
                    'rows': [roi_numbers],
                    'columns': [roi_numbers],
                    'data': [[relationship_values]],
                    'row_names': [structure_names],
                    'col_names': [structure_names],
                    'colors': {roi: [r, g, b]},
                    'faded_relationships': {matrix_position: bool},
                    'unit': str (unit of measurement, e.g., 'cm')
                }
        '''
        # Extract ROI numbers from structure names early
        name_to_roi = {struct.name: roi for roi, struct in self.structures.items()}

        # Get filtered matrix
        matrix = self.get_relationship_matrix(row_rois, col_rois, use_symbols)

        if matrix.empty:
            result = {
                'rows': [],
                'columns': [],
                'data': [],
                'row_names': [],
                'col_names': [],
                'colors': {}
            }
            if logical_relations_mode == 'faded':
                result['faded_relationships'] = {}
            return result

        # Apply logical relationships filtering if not in 'show' mode
        faded_relationships = {}
        if logical_relations_mode != 'show':
            # Build visible_rois list from row and col rois
            if visible_rois is None:
                visible_rois = list(self.structures.keys())

            # Filter relationships based on logical mode
            _, fade_info = (
                self.filter_relationships_by_logical_mode(
                    mode=logical_relations_mode,
                    visible_rois=visible_rois
                )
            )

            # Create fade info keyed by matrix position (row_index, col_index)
            if logical_relations_mode == 'faded':
                faded_relationships = {}
                for idx, row_name in enumerate(matrix.index):
                    for jdx, col_name in enumerate(matrix.columns):
                        row_roi = name_to_roi.get(row_name)
                        col_roi = name_to_roi.get(col_name)
                        if row_roi and col_roi and (col_roi, row_roi) in fade_info:
                            key = f"{idx}_{jdx}"
                            faded_relationships[key] = fade_info[(col_roi, row_roi)]

        # Convert ROI numbers list for results (convert to int for JSON serialization)
        row_rois_list = [int(name_to_roi[name]) for name in matrix.index if name in name_to_roi]
        col_rois_list = [int(name_to_roi[name]) for name in matrix.columns if name in name_to_roi]

        # Get all unique ROIs (both rows and columns)
        all_rois = list(set(row_rois_list + col_rois_list))
        all_rois.sort()

        # Get summary data for all structures
        summary_df = self.summary()

        # Extract colors from DICOM file if available
        colors = {}
        if self.dicom_structure_file and hasattr(self.dicom_structure_file, 'dataset'):
            try:
                for roi_contour in self.dicom_structure_file.dataset.ROIContourSequence:
                    # Convert to int for JSON
                    roi_num = int(roi_contour.ReferencedROINumber)
                    if hasattr(roi_contour, 'ROIDisplayColor'):
                        # Convert color values to int
                        colors[roi_num] = [int(c) for c in roi_contour.ROIDisplayColor]
            except AttributeError:
                pass

        # Build comprehensive data dictionaries from summary DataFrame
        dicom_types_dict = {}
        code_meanings_dict = {}
        volumes_dict = {}
        num_regions_dict = {}
        slice_ranges_dict = {}
        structure_slices_dict = {}
        structure_slices_original_dict = {}
        structure_slices_interpolated_dict = {}
        original_slice_pool = set()

        for roi in all_rois:
            # Get row from summary DataFrame for this ROI
            roi_data = summary_df[summary_df['ROI'] == roi]

            if not roi_data.empty:
                row = roi_data.iloc[0]
                # Handle NaN values from DataFrame
                if 'DICOM_Type' in row.index and pd.notna(row['DICOM_Type']):
                    dicom_type = row['DICOM_Type']
                else:
                    dicom_type = ''
                if 'CodeMeaning' in row.index and pd.notna(row['CodeMeaning']):
                    code_meaning = row['CodeMeaning']
                else:
                    code_meaning = ''
                if pd.notna(row['Physical_Volume']):
                    volume = row['Physical_Volume']
                else:
                    volume = 0.0
                if pd.notna(row['Num_Regions']):
                    num_regions = int(row['Num_Regions'])
                else:
                    num_regions = 0
                if pd.notna(row['Slice_Range']):
                    slice_range = row['Slice_Range']
                else:
                    slice_range = ''

                # Convert numpy types to Python native types for JSON serialization
                roi_key = int(roi)
                dicom_types_dict[roi_key] = dicom_type
                code_meanings_dict[roi_key] = code_meaning
                volumes_dict[roi_key] = float(volume)
                num_regions_dict[roi_key] = int(num_regions)
                slice_ranges_dict[roi_key] = slice_range

                # Get slice indices for this structure
                if roi in self.structures:
                    structure = self.structures[roi]
                    if not structure.region_table.empty:
                        region_table = structure.region_table
                        non_empty_mask = ~region_table.Empty
                        non_empty_rows = region_table.loc[non_empty_mask]

                        if non_empty_rows.empty:
                            structure_slices_dict[roi_key] = []
                            structure_slices_original_dict[roi_key] = []
                            structure_slices_interpolated_dict[roi_key] = []
                        else:
                            slice_index_list = sorted(
                                non_empty_rows['SliceIndex'].unique().tolist()
                            )
                            structure_slices_dict[roi_key] = slice_index_list

                            if 'Interpolated' in non_empty_rows.columns:
                                original_rows = non_empty_rows.loc[
                                    ~non_empty_rows.Interpolated
                                ]
                                interpolated_rows = non_empty_rows.loc[
                                    non_empty_rows.Interpolated
                                ]
                                original_slices = sorted(
                                    original_rows['SliceIndex'].unique().tolist()
                                )
                                interpolated_slices = sorted(
                                    interpolated_rows['SliceIndex'].unique().tolist()
                                )
                            else:
                                original_slices = slice_index_list
                                interpolated_slices = []

                            structure_slices_original_dict[roi_key] = original_slices
                            structure_slices_interpolated_dict[
                                roi_key
                            ] = interpolated_slices
                            original_slice_pool.update(original_slices)
                    else:
                        structure_slices_dict[roi_key] = []
                        structure_slices_original_dict[roi_key] = []
                        structure_slices_interpolated_dict[roi_key] = []
                else:
                    structure_slices_dict[roi_key] = []
                    structure_slices_original_dict[roi_key] = []
                    structure_slices_interpolated_dict[roi_key] = []
            else:
                roi_key = int(roi)
                dicom_types_dict[roi_key] = ''
                code_meanings_dict[roi_key] = ''
                volumes_dict[roi_key] = 0.0
                num_regions_dict[roi_key] = 0
                slice_ranges_dict[roi_key] = ''
                structure_slices_dict[roi_key] = []
                structure_slices_original_dict[roi_key] = []
                structure_slices_interpolated_dict[roi_key] = []

        return {
            'rows': row_rois_list,
            'columns': col_rois_list,
            'data': matrix.values.tolist(),
            'row_names': matrix.index.tolist(),
            'col_names': matrix.columns.tolist(),
            'colors': colors,
            'dicom_types': dicom_types_dict,
            'code_meanings': code_meanings_dict,
            'volumes': volumes_dict,
            'num_regions': num_regions_dict,
            'slice_ranges': slice_ranges_dict,
            'slice_indices': self.slice_sequence.slices if self.slice_sequence else [],
            'slice_indices_original': sorted(original_slice_pool),
            'structure_slices': structure_slices_dict,
            'structure_slices_original': structure_slices_original_dict,
            'structure_slices_interpolated': structure_slices_interpolated_dict,
            'slice_relationships': self._serialize_slice_relationships(all_rois),
            'faded_relationships': faded_relationships if logical_relations_mode == 'faded' else None,
            'unit': self.unit
        }


class StructureSetBuilder:
    '''Fluent builder for StructureSet creation from DICOM inputs.'''

    def __init__(self):
        self._dicom_file: Optional[DicomStructureFile] = None
        self._tolerance: float = 0.0

    def with_dicom_file(self, path: str | Path) -> 'StructureSetBuilder':
        '''Set the DICOM RTSTRUCT file to use for StructureSet creation.'''
        file_path = Path(path)
        self._dicom_file = DicomStructureFile(
            top_dir=file_path.parent,
            file_path=file_path,
        )
        return self

    def with_tolerance(self, tol: float) -> 'StructureSetBuilder':
        '''Set relationship tolerance used by the constructed StructureSet.'''
        self._tolerance = tol
        return self

    def build(self) -> StructureSet:
        '''Build and return the configured StructureSet instance.'''
        if self._dicom_file is None:
            raise ValueError('A DICOM file must be provided before calling build().')
        return StructureSet(
            dicom_structure_file=self._dicom_file,
            tolerance=self._tolerance,
        )
