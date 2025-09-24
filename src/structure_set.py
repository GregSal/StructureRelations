'''StructureSet class for managing multiple structures and their relationships.
'''
# %% Imports
from typing import List, Optional
import logging

# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type
from contours import build_contour_table
from structures import StructureShape
from relations import DE27IM, RelationshipType
from dicom import DicomStructureFile


class StructureSet:
    '''Class for managing multiple StructureShape objects and their relationships.

    Attributes:
        structures (Dict[ROI_Type, StructureShape]): Dictionary of structures keyed by ROI.
        slice_sequence (SliceSequence): The slice sequence used across all structures.
        relationship_graph (nx.Graph): Graph where nodes are structures and edges are relationships.
        dicom_structure_file (DicomStructureFile): Optional reference to the source DICOM file.
    '''

    def __init__(self, 
                 slice_data: Optional[List] = None,
                 dicom_structure_file: Optional[DicomStructureFile] = None):
        '''Initialize the StructureSet.

        Args:
            slice_data (List, optional): List of ContourPoints for building structures.
                If None, creates an empty StructureSet. Deprecated - use dicom_structure_file instead.
            dicom_structure_file (DicomStructureFile, optional): DICOM structure file object
                containing contour points. Takes precedence over slice_data if both are provided.
        '''
        self.structures = {}
        self.slice_sequence = None
        self.relationship_graph = nx.Graph()
        self.dicom_structure_file = dicom_structure_file

        # Prioritize dicom_structure_file over slice_data
        if dicom_structure_file is not None:
            self.build_from_dicom_file(dicom_structure_file)
        elif slice_data is not None:
            self.build_from_slice_data(slice_data)

    def build_from_dicom_file(self, dicom_file: DicomStructureFile) -> None:
        '''Build structures from DicomStructureFile following the StructureSet process.

        Args:
            dicom_file (DicomStructureFile): DICOM structure file object containing contour points.
        '''
        if not dicom_file.contour_points:
            logger.warning("No contour points found in DicomStructureFile")
            return
            
        logger.info("Building StructureSet from %d contour points", len(dicom_file.contour_points))
        self.build_from_slice_data(dicom_file.contour_points)

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
            logger.debug('Slice sequence after ROI %s:\n%s', roi,
                         self.slice_sequence.sequence)

            # 2.2. Add the StructureShape object to dictionary with the ROI as the key
            self.structures[roi] = structure

            # Add structure to relationship graph
            self.relationship_graph.add_node(roi, structure=structure)

        # 2.3 & 2.4. Use the SliceSequence to add interpolated contours and generate RegionSlices
        for structure in self.structures.values():
            structure.finalize(self.slice_sequence)

    def apply_exclusions(self, exclusion_patterns: Optional[List[str]] = None, 
                        exclude_default: bool = True) -> None:
        '''Apply exclusions to filter out unwanted structures and rebuild.
        
        Args:
            exclusion_patterns (List[str], optional): List of patterns to exclude.
                If None, uses default patterns ['x', 'z'] if exclude_default is True.
            exclude_default (bool): Whether to apply default exclusion patterns.
        '''
        if self.dicom_structure_file is None:
            logger.warning("No DicomStructureFile available for applying exclusions")
            return
            
        # Apply exclusions to the dicom structure file
        excluded_contours = self.dicom_structure_file.filter_exclusions(
            exclude_prefixes=exclusion_patterns,
            exclude_empty=True
        )
        
        if excluded_contours:
            logger.info("Applied exclusions, removed %d contour sets", len(excluded_contours))
            # Rebuild structures with filtered contour points
            self.structures.clear()
            self.relationship_graph.clear()
            self.build_from_slice_data(self.dicom_structure_file.contour_points)
        else:
            logger.info("No structures excluded")

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
                relationship = structure_a.relate(structure_b)
                
                # Add edge to graph with relationship data
                self.relationship_graph.add_edge(
                    roi_a, roi_b, 
                    relationship=relationship,
                    relationship_type=str(relationship)
                )
                
                logger.debug(f"Calculated relationship between ROI {roi_a} and ROI {roi_b}: {relationship}")

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
        
        # Set index and columns names for clarity
        relationship_matrix.index.name = 'Structure_A'
        relationship_matrix.columns.name = 'Structure_B'
        
        return relationship_matrix
    def relationship_summary(self, symbol_map=None) -> pd.DataFrame:
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
