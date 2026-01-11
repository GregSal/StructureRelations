'''StructureSet class for managing multiple structures and their relationships.
'''
# %% Imports
from typing import List, Optional
import logging

import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type
from contours import build_contour_table
from structures import StructureShape
from relations import RELATIONSHIP_TYPES
from relationships import StructureRelationship
from dicom import DicomStructureFile
from utilities import round_value


# %% Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
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
    '''

    def __init__(self,
                 slice_data: Optional[List] = None,
                 dicom_structure_file: Optional[DicomStructureFile] = None,
                 tolerance=0.0):
        '''Initialize the StructureSet.

        Args:
            slice_data (List, optional): List of ContourPoints for building structures.
                If None, creates an empty StructureSet.
                Deprecated - use dicom_structure_file instead.
            dicom_structure_file (DicomStructureFile, optional): DICOM structure
                file object containing contour points. Takes precedence over
                slice_data if both are provided.
            tolerance (float, optional): Tolerance value for structure
                relationships. Defaults to 0.0.
        '''
        self.structures = {}
        self.slice_sequence = None
        self.relationship_graph = nx.DiGraph()
        self.dicom_structure_file = dicom_structure_file
        self.tolerance = tolerance

        # Prioritize dicom_structure_file over slice_data
        if dicom_structure_file is not None:
            self.build_from_dicom_file()
        elif slice_data is not None:
            self.build_from_slice_data(slice_data)

    def build_from_dicom_file(self) -> None:
        '''Build structures from DicomStructureFile following the StructureSet
        process.
        '''
        if not self.dicom_structure_file.contour_points:
            logger.warning("No contour points found in DicomStructureFile")
            return
        self.tolerance = self.dicom_structure_file.resolution
        logger.info("Building StructureSet from %d contour points",
                    len(self.dicom_structure_file.contour_points))
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
            logger.debug('Finalizing structure for ROI %s', structure.name)
            structure.finalize(self.slice_sequence)
        self.finalize()

    def finalize(self) -> None:
        '''Complete the StructureSet process by calculating relationships.

        This method:
        3. Determines the relationships between the StructureShape objects.
        4. Constructs a graph where the nodes are the StructureShape objects
           and the edges are the relationships between them.
        5. Calculates logical relationship flags based on graph topology.
        '''
        self.calculate_relationships()
        self.calculate_logical_flags()

    def add_structure(self, structure: StructureShape) -> None:
        '''Add a structure to the set.

        Args:
            structure (StructureShape): The structure to add.
        '''
        self.structures[structure.roi] = structure

        # Add structure to relationship graph
        self.relationship_graph.add_node(structure.roi, structure=structure)

    def calculate_relationships(self) -> None:
        '''Calculate relationships between all structure pairs.

        This method calculates the DE27IM spatial relationship between every
        pair of structures and stores the results in the relationship_graph
        as edge attributes.

        If relationships have already been calculated, this method does nothing.
        '''
        sorted_structures = self.get_structures_by_volume()
        structure_rois = [s.roi for s in sorted_structures]

        # Check if relationships have already been calculated
        # If the graph has the expected number of edges, skip recalculation
        expected_edges = len(structure_rois) * (len(structure_rois) - 1) // 2
        if self.relationship_graph.number_of_edges() >= expected_edges:
            logger.debug('Relationships already calculated, skipping recalculation')
            return

        logger.info('Calculating relationships for %d structures',
                    len(structure_rois))
        # Calculate relationships for all pairs
        for i, roi_a in enumerate(structure_rois):
            for roi_b in structure_rois[i+1:]:
                structure_a = self.structures[roi_a]
                structure_b = self.structures[roi_b]

                # Calculate the DE27IM relationship
                de27im_relationship = structure_a.relate(structure_b,
                                                         tolerance=self.tolerance)

                # Create StructureRelationship object
                relationship = StructureRelationship(
                    de27im=de27im_relationship,
                    is_identical=False
                )

                # Add edge to graph with relationship data
                self.relationship_graph.add_edge(
                    roi_a, roi_b,
                    relationship=relationship
                )

                try:
                    rel_type = relationship.relationship_type
                    logger.debug('Calculated relationship between ROI %s and ROI %s: %s',
                                 structure_a.name, structure_b.name, rel_type)
                except (AttributeError, KeyError) as e:
                    logger.debug('Calculated relationship between ROI %s and ROI %s (error accessing type: %s)',
                                 structure_a.name, structure_b.name, str(e))

    def calculate_logical_flags(self) -> None:
        '''Calculate logical relationship flags based on graph topology.

        This method analyzes the relationship graph structure to identify
        relationships that are logically derived from the graph topology
        rather than being direct geometric relationships. For example,
        relationships implied by transitivity or relationships that exist
        only through intermediate structures.

        The is_logical flag in StructureRelationship objects will be set
        to True for relationships identified as logical.

        Note:
            This is a placeholder for future implementation. The specific
            algorithm for identifying logical relationships will be developed
            based on the semantic requirements of the relationship graph.
        '''
        logger.info('Calculating logical flags for relationships')
        logger.debug('Logical flag calculation not yet implemented')

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

    def get_structures_by_volume(self, volume_type: str = 'hull') -> List[StructureShape]:
        '''Get structures sorted by volume.

        Args:
            volume_type (str): Type of volume to sort by ('physical', 'exterior', 'hull').

        Returns:
            List[StructureShape]: Structures sorted by volume (descending).
        '''
        if volume_type == 'physical':
            return sorted(self.structures.values(), key=lambda s: s.physical_volume, reverse=True)
        elif volume_type == 'exterior':
            return sorted(self.structures.values(), key=lambda s: s.exterior_volume, reverse=True)
        elif volume_type == 'hull':
            return sorted(self.structures.values(), key=lambda s: s.hull_volume, reverse=True)
        else:
            raise ValueError(f"Unknown volume type: {volume_type}")

    def summary(self) -> pd.DataFrame:
        '''Get a summary of all structures in the set.

        Returns:
            pd.DataFrame: Summary table with structure information.
        '''
        summary_data = []
        for roi, structure in self.structures.items():
            interpolated_regions = structure.region_table.Interpolated
            original_regions = structure.region_table[~interpolated_regions]
            summary_data.append({
                'ROI': roi,
                'Name': structure.name,
                'Physical_Volume': round_value(structure.physical_volume,
                                               self.tolerance),
                'Exterior_Volume': round_value(structure.exterior_volume,
                                              self.tolerance),
                'Hull_Volume': round_value(structure.hull_volume,
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

        return summary_df

    @property
    def relationship_matrix(self) -> pd.DataFrame:
        '''Get a summary of all relationships between structures as a matrix.

        Returns:
            pd.DataFrame: Matrix with Structure_A as index, Structure_B as columns,
                and StructureRelationship objects as values. The matrix is symmetric for
                symmetric relationships.
        '''
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

        return relationship_matrix

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
                    return rel_type.label if rel_type else 'Unknown'
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

    def to_dict(self, row_rois=None, col_rois=None, use_symbols=True) -> dict:
        '''Convert relationship matrix to dictionary for JSON serialization.

        Args:
            row_rois (List[ROI_Type], optional): List of ROI numbers for matrix rows.
            col_rois (List[ROI_Type], optional): List of ROI numbers for matrix columns.
            use_symbols (bool, optional): If True, use symbolic notation.

        Returns:
            dict: Dictionary with structure:
                {
                    'rows': [roi_numbers],
                    'columns': [roi_numbers],
                    'data': [[relationship_values]],
                    'row_names': [structure_names],
                    'col_names': [structure_names],
                    'colors': {roi: [r, g, b]}
                }
        '''
        # Get filtered matrix
        matrix = self.get_relationship_matrix(row_rois, col_rois, use_symbols)

        if matrix.empty:
            return {
                'rows': [],
                'columns': [],
                'data': [],
                'row_names': [],
                'col_names': [],
                'colors': {}
            }

        # Extract ROI numbers from structure names (convert to int for JSON serialization)
        name_to_roi = {struct.name: roi for roi, struct in self.structures.items()}
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
                        slice_indexes = structure.region_table['SliceIndex']
                        slice_index_list = slice_indexes.unique().tolist()
                        structure_slices_dict[roi_key] = sorted(slice_index_list)
                    else:
                        structure_slices_dict[roi_key] = []
                else:
                    structure_slices_dict[roi_key] = []
            else:
                roi_key = int(roi)
                dicom_types_dict[roi_key] = ''
                code_meanings_dict[roi_key] = ''
                volumes_dict[roi_key] = 0.0
                num_regions_dict[roi_key] = 0
                slice_ranges_dict[roi_key] = ''
                structure_slices_dict[roi_key] = []

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
            'structure_slices': structure_slices_dict
        }
