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
from relations import DE27IM, RelationshipType
from dicom import DicomStructureFile


# %% Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.relationship_graph = nx.Graph()
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

    def finalize(self) -> None:
        '''Complete the StructureSet process by calculating relationships.

        This method:
        3. Determines the relationships between the StructureShape objects.
        4. Constructs a graph where the nodes are the StructureShape objects
           and the edges are the relationships between them.
        '''
        self.calculate_relationships()

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
        '''
        structure_rois = list(self.structures.keys())

        # Calculate relationships for all pairs
        for i, roi_a in enumerate(structure_rois):
            for roi_b in structure_rois[i+1:]:
                structure_a = self.structures[roi_a]
                structure_b = self.structures[roi_b]

                # Calculate the relationship
                relationship = structure_a.relate(structure_b,
                                                  tolerance=self.tolerance)

                # Add edge to graph with relationship data
                self.relationship_graph.add_edge(
                    roi_a, roi_b,
                    relationship=relationship,
                    relationship_type=str(relationship)
                )

                logger.debug('Calculated relationship between ROI %s and ROI %s:\n%s',
                             roi_a, roi_b, relationship)

    def get_relationship(self, roi_a: ROI_Type, roi_b: ROI_Type) -> DE27IM:
        '''Get the relationship between two structures.

        Args:
            roi_a (ROI_Type): First structure ROI.
            roi_b (ROI_Type): Second structure ROI.

        Returns:
            DE27IM: The spatial relationship between the structures.
        '''
        if self.relationship_graph.has_edge(roi_a, roi_b):
            return self.relationship_graph[roi_a][roi_b]['relationship']
        elif self.relationship_graph.has_edge(roi_b, roi_a):
            return self.relationship_graph[roi_b][roi_a]['relationship']
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
            summary_data.append({
                'ROI': roi,
                'Name': structure.name,
                'Physical_Volume': structure.physical_volume,
                'Exterior_Volume': structure.exterior_volume,
                'Hull_Volume': structure.hull_volume,
                'Num_Contours': len(structure.contour_graph),
                'Num_Slices': len(structure.region_table)
            })

        return pd.DataFrame(summary_data)

    @property
    def relationship_matrix(self) -> pd.DataFrame:
        '''Get a summary of all relationships between structures as a matrix.

        Returns:
            pd.DataFrame: Matrix with Structure_A as index, Structure_B as columns,
                and Relationship_Type as values. The matrix is symmetric for
                symmetric relationships.
        '''
        if not self.relationship_graph.edges():
            return pd.DataFrame()

        # Get all unique ROIs and their corresponding names for matrix dimensions
        all_rois = sorted(list(self.structures.keys()))
        all_names = [self.structures[roi].name for roi in all_rois]

        # Create empty matrix filled with None using structure names
        relationship_matrix = pd.DataFrame(
            index=all_names,
            columns=all_names,
            dtype='object'
        )
        # Fill with None values explicitly
        relationship_matrix[:] = RelationshipType.UNKNOWN

        # Create ROI to name mapping for lookups
        roi_to_name = {roi: self.structures[roi].name for roi in all_rois}

        # Fill diagonal with self-relationships (typically "Equals")
        for name in all_names:
            relationship_matrix.loc[name, name] = RelationshipType.EQUALS

        # Fill matrix with calculated relationships
        for roi_a, roi_b, edge_data in self.relationship_graph.edges(data=True):
            #relationship_type = edge_data['relationship_type']
            relationship_obj = edge_data['relationship']
            relationship_type = relationship_obj.identify_relation()

            # Get structure names for the ROIs
            name_a = roi_to_name[roi_a]
            name_b = roi_to_name[roi_b]

            # Set the relationship in the matrix using names
            relationship_matrix.loc[name_a, name_b] = relationship_type

            # For symmetric relationships, also set the transpose
            if relationship_type.is_symmetric:
                relationship_matrix.loc[name_b, name_a] = relationship_type

        # Transpose the matrix so that Structure_A is rows and Structure_B is
        # columns.
        relationship_matrix = relationship_matrix.T
        # Set index and columns names for clarity
        relationship_matrix.index.name = 'Structure_A'
        relationship_matrix.columns.name = 'Structure_B'

        return relationship_matrix

    def relationship_summary(self, symbol_map=None) -> pd.DataFrame:
        '''Get a summary of all relationships between structures with optional
        symbolic notation.

        Args:
            symbol_map (dict, optional): Custom mapping from RelationshipType to
            symbols.
        Returns:
            pd.DataFrame: Adjacency matrix of a graph where nodes are structures
                and edges represent relationships. The values of teh matrix are either
                labels or symbols representing the relationship types. The index and columns
                are structure names.
        '''
        def to_symbol(relationship_type: RelationshipType) -> str:
            if relationship_type:
                return default_symbol_map[relationship_type]
            return default_symbol_map[RelationshipType.UNKNOWN]

        default_symbol_map = {
            RelationshipType.UNKNOWN: RelationshipType.UNKNOWN.label,
            RelationshipType.DISJOINT: RelationshipType.DISJOINT.label,
            RelationshipType.EQUALS: RelationshipType.EQUALS.label,
            RelationshipType.OVERLAPS: RelationshipType.OVERLAPS.label,
            RelationshipType.CONTAINS: RelationshipType.CONTAINS.label,
            RelationshipType.SURROUNDS: RelationshipType.SURROUNDS.label,
            RelationshipType.SHELTERS: RelationshipType.SHELTERS.label,
            RelationshipType.BORDERS: RelationshipType.BORDERS.label,
            RelationshipType.CONFINES: RelationshipType.CONFINES.label,
            RelationshipType.PARTITION: RelationshipType.PARTITION.label
        }
        if symbol_map:
            default_symbol_map.update(symbol_map)

        relationship_matrix = self.relationship_matrix
        if relationship_matrix.empty:
            return pd.DataFrame()
        # Convert matrix to labels for better readability
        labeled_matrix = relationship_matrix.map(to_symbol)
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
            symbol_map = self._get_default_symbol_map()
            relationship_matrix = relationship_matrix.map(lambda rt: symbol_map.get(rt, '?'))
        else:
            # Use labels
            relationship_matrix = relationship_matrix.map(lambda rt: rt.label if rt else 'Unknown')

        return relationship_matrix

    def _get_default_symbol_map(self) -> dict:
        '''Get the default symbol map for relationship types.

        Returns:
            dict: Mapping from RelationshipType to unicode symbols.
        '''
        return {
            RelationshipType.UNKNOWN: '?',
            RelationshipType.EQUALS: '=',
            RelationshipType.CONTAINS: '⊂',
            RelationshipType.OVERLAPS: '∩',
            RelationshipType.PARTITION: '⊕',
            RelationshipType.BORDERS: '|',
            RelationshipType.SURROUNDS: '○',
            RelationshipType.SHELTERS: '△',
            RelationshipType.DISJOINT: '∅',
            RelationshipType.CONFINES: '⊏'
        }

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

        # Extract ROI numbers from structure names
        name_to_roi = {struct.name: roi for roi, struct in self.structures.items()}
        row_rois_list = [name_to_roi[name] for name in matrix.index if name in name_to_roi]
        col_rois_list = [name_to_roi[name] for name in matrix.columns if name in name_to_roi]

        # Get all unique ROIs (both rows and columns)
        all_rois = list(set(row_rois_list + col_rois_list))
        all_rois.sort()

        # Extract colors from DICOM file if available
        colors = {}
        roi_labels = None

        if self.dicom_structure_file and hasattr(self.dicom_structure_file, 'dataset'):
            try:
                for roi_contour in self.dicom_structure_file.dataset.ROIContourSequence:
                    roi_num = roi_contour.ReferencedROINumber
                    if hasattr(roi_contour, 'ROIDisplayColor'):
                        colors[roi_num] = list(roi_contour.ROIDisplayColor)
            except AttributeError:
                pass

            # Get ROI labels for DICOM Type and Code Meaning
            roi_labels = self.dicom_structure_file.get_roi_labels()

        # Build comprehensive data dictionaries for ALL structures
        dicom_types_dict = {}
        code_meanings_dict = {}
        volumes_dict = {}
        num_regions_dict = {}
        slice_ranges_dict = {}

        for roi in all_rois:
            # Get DICOM Type and Code Meaning
            dicom_type = ''
            code_meaning = ''
            if roi_labels is not None and not roi_labels.empty and roi in roi_labels.index:
                dicom_type = roi_labels.loc[roi].get('DICOM_Type', '')
                code_meaning = roi_labels.loc[roi].get('CodeMeaning', '')
            dicom_types_dict[roi] = dicom_type
            code_meanings_dict[roi] = code_meaning

            # Get structure data
            if roi in self.structures:
                struct = self.structures[roi]
                volumes_dict[roi] = round(struct.physical_volume, 2)

                # Number of regions: count unique RegionSlice objects
                num_regions_dict[roi] = len(struct.region_table)

                # Slice range
                if not struct.region_table.empty:
                    min_slice = struct.region_table['SliceIndex'].min()
                    max_slice = struct.region_table['SliceIndex'].max()
                    slice_ranges_dict[roi] = f'{min_slice:.2f} to {max_slice:.2f}'
                else:
                    slice_ranges_dict[roi] = ''
            else:
                volumes_dict[roi] = 0.0
                num_regions_dict[roi] = 0
                slice_ranges_dict[roi] = ''

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
            'slice_ranges': slice_ranges_dict
        }
