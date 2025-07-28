'''StructureSet class for managing multiple structures and their relationships.
'''
# %% Imports
from typing import Dict, List
import pandas as pd
import networkx as nx

from types_and_classes import ROI_Type
from contours import SliceSequence, build_contour_table
from structures import StructureShape
from relations import DE27IM


class StructureSet:
    '''Class for managing multiple StructureShape objects and their relationships.

    Attributes:
        structures (Dict[ROI_Type, StructureShape]): Dictionary of structures keyed by ROI.
        slice_sequence (SliceSequence): The slice sequence used across all structures.
        relationship_graph (nx.Graph): Graph where nodes are structures and edges are relationships.
    '''

    def __init__(self, slice_data: List = None):
        '''Initialize the StructureSet.

        Args:
            slice_data (List, optional): List of ContourPoints for building structures.
                If None, creates an empty StructureSet.
        '''
        self.structures = {}
        self.slice_sequence = None
        self.relationship_graph = nx.Graph()

        if slice_data:
            self.build_from_slice_data(slice_data)

    def build_from_slice_data(self, slice_data: List) -> None:
        '''Build structures from slice data following the StructureSet process.

        Args:
            slice_data (List): List of ContourPoints objects.
        '''
        # 1. Create a contour_table
        contour_table, self.slice_sequence = build_contour_table(slice_data)

        # Get unique ROIs
        unique_rois = contour_table['ROI'].unique()

        # 2. For each ROI in the contour_table:
        for roi in unique_rois:
            # 2.1. Create a StructureShape object from the contour table
            structure = StructureShape(roi=roi, name=f'Structure_{roi}')
            self.slice_sequence = structure.add_contour_graph(contour_table, self.slice_sequence)

            # 2.2. Add the StructureShape object to dictionary with the ROI as the key
            self.structures[roi] = structure

            # Add structure to relationship graph
            self.relationship_graph.add_node(roi, structure=structure)

        # 2.3 & 2.4. Use the SliceSequence to add interpolated contours and generate RegionSlices
        for structure in self.structures.values():
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
        '''Calculate relationships between all structures in the set.'''
        roi_list = list(self.structures.keys())

        # Calculate relationships between all pairs of structures
        for i, roi_a in enumerate(roi_list):
            for roi_b in roi_list[i+1:]:
                structure_a = self.structures[roi_a]
                structure_b = self.structures[roi_b]

                # Calculate relationship
                relationship = structure_a.relate(structure_b)

                # Add edge to relationship graph
                self.relationship_graph.add_edge(
                    roi_a, roi_b,
                    relationship=relationship,
                    relation_type=relationship.identify_relation()
                )

    def get_relationship(self, roi_a: ROI_Type, roi_b: ROI_Type) -> DE27IM:
        '''Get the relationship between two structures.

        Args:
            roi_a (ROI_Type): First structure ROI.
            roi_b (ROI_Type): Second structure ROI.

        Returns:
            DE27IM: The relationship between the structures.
        '''
        if self.relationship_graph.has_edge(roi_a, roi_b):
            return self.relationship_graph[roi_a][roi_b]['relationship']
        elif self.relationship_graph.has_edge(roi_b, roi_a):
            # Return transposed relationship
            relationship = self.relationship_graph[roi_b][roi_a]['relationship']
            return relationship.transpose()
        else:
            return None

    def get_structures_by_volume(self, volume_type: str = 'hull') -> List[StructureShape]:
        '''Get structures sorted by volume.

        Args:
            volume_type (str): Type of volume to sort by ('physical', 'exterior', 'hull').

        Returns:
            List[StructureShape]: Structures sorted by volume (ascending).
        '''
        volume_attr = f'{volume_type}_volume'
        return sorted(self.structures.values(),
                     key=lambda s: getattr(s, volume_attr))

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

    def relationship_summary(self) -> pd.DataFrame:
        '''Get a summary of all relationships in the set.

        Returns:
            pd.DataFrame: Summary table with relationship information.
        '''
        relationship_data = []
        for roi_a, roi_b, data in self.relationship_graph.edges(data=True):
            relationship_data.append({
                'Structure_A': roi_a,
                'Structure_B': roi_b,
                'Relationship_Type': data['relation_type'],
                'DE27IM': str(data['relationship'])
            })

        return pd.DataFrame(relationship_data)
