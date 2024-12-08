'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import LiteralString

# Standard Libraries
from enum import Enum, auto
from dataclasses import dataclass

# Shared Packages
import numpy as np
import pandas as pd
import shapely

# Local packages
from types_and_classes import StructurePairType
from types_and_classes import DE9IM_Type, DE27IM_Type
from structure_slice import StructureSlice, find_boundary_slices
from structure_slice import empty_structure
from structure_slice import select_slices


# Global Settings
PRECISION = 3


# %% Relationship Type Definitions
class RelationshipType(Enum):
    '''The names for defined relationship types.

    The relationship types are based on the DE-9IM relationship between two
    structures.  The relationship types are defined as follows:
    DISJOINT: The two structures have no regions in common.
    SURROUNDS: One structure resides completely within a hole in another
                structure.
    SHELTERS: One structure is within the convex outline of the other.
    BORDERS: The two structures share a common exterior boundary.
    CONFINES: The two structures share a common boundary and one is
                        within the other.
    OVERLAPS: The two structures share a common region.
    PARTITION: The two structures share a common region and one is within the
                other.
    CONTAINS: One structure contains the other.
    EQUALS: The two structures are identical.
    LOGICAL: The relationship is based on a logical combination of other
                relationships.
    UNKNOWN: The relationship has not been identified.

    The class also includes properties to check if the relationship is
    symmetric, and / or transitive.

    Text formatting is provided by the label and __str__ properties.
    '''
    DISJOINT = auto()
    SURROUNDS = auto()
    SHELTERS = auto()
    BORDERS = auto()
    BORDERS_INTERIOR = auto()
    OVERLAPS = auto()
    PARTITION = auto()
    CONTAINS = auto()
    EQUALS = auto()
    LOGICAL = auto()
    UNKNOWN = 999  # Used for initialization

    def __bool__(self):
        if self == self.UNKNOWN:
            return False
        return True

    @property
    def is_symmetric(self) -> bool:
        '''Check if the relationship is symmetric.'''
        symmetric_relations = [
            self.DISJOINT,
            self.OVERLAPS,
            self.BORDERS,
            self.EQUALS,
            self.UNKNOWN
            ]
        return self in symmetric_relations

    @property
    def is_transitive(self) -> bool:
        '''Check if the relationship is transitive.'''
        transitive_relations = [
            self.EQUALS,
            self.SHELTERS,
            self.SURROUNDS,
            self.CONTAINS,
            ]
        return self in transitive_relations

    @property
    def label(self) -> LiteralString:
        '''Get the capitalized name of the relationship type.'''
        return self.name.capitalize()

    def __str__(self) -> LiteralString:
        return f'Relationship: {self.label}'


@dataclass()
class RelationshipTest:
    '''The test binaries used to identify a relationship type.

    Each test definitions consists of 2 27-bit binaries, a mask and a value.
    Each of the 27-bit binaries contain 3 9-bit parts associated with DE-9IM
    relationships. The left-most 9 bits are associated with the relationship
    between one structure's convex hull and another structure's contour. The
    middle 9 bits are associated with the relationship between the first
    structure's exterior polygon (i.e. with any holes filled) and the second
    structure's contour. The right-most 9 bits are associated with the
    relationship between first structure's contour and the second structure's
    contour.

    Named relationships are identified by logical patterns such as: T*T*F*FF*
        The 'T' indicates the bit must be True.
        The 'F' indicates the bit must be False.
        The '*' indicates the bit can be either True or False.
    Ane example of a complete relationship logic is:
    Surrounds (One structure resides completely within a hole in another
               structure):
        Region Test =   FF*FF****  - The contours of the two structures have no
                                     regions in common.
        Exterior Test = T***F*F**  - With holes filled, one structure is within
                                     the other.
        Hull Test =     *********  - Together, the Region and Exterior Tests
                                     sufficiently identifies the relationship,
                                     so the Hull Test is not necessary.
    The mask binary is a sequence of 0s and 1s with every '*' as a '0' and every
    'T' or 'F' bit as a '1'.  The operation: relationship_integer & mask will
    set all of the bit that are allowed to be either True or False to 0.

    The value binary is a sequence of 0s and 1s with every 'T' as a '1' and
    every '*' or 'F' bit as a '0'. The relationship is identified when value
    binary is equal to the result of the `relationship_integer & mask`
    operation.
    '''
    relation_type: RelationshipType = RelationshipType.UNKNOWN
    mask: int = 0b000000000000000000000000000
    value: int = 0b000000000000000000000000000

    def __repr__(self) -> str:
        rep_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            ' ' * 4,
            f'mask =  0b{self.mask:0>27b}\n',
            ' ' * 4,
            f'value = 0b{self.value:0>27b}'
            ])
        return rep_str

    def test(self, relation: int)->RelationshipType:
        '''Apply the defined test to the supplied relation binary.

        Args:
            relation (int): The number corresponding to a 27-bit binary of
                relationship values.

        Returns:
            RelationshipType: The RelationshipType if the test passes,
                otherwise None.
        '''
        masked_relation = relation & self.mask
        if masked_relation == self.value:
            return self.relation_type
        return None


def identify_relation(relation_binary) -> RelationshipType:
    '''Applies a collection of definitions for named relationships to a supplied
    relationship binary.

    The defined relationships are:
        Relationship      Region Test   Exterior Test   Hull Test
        Disjoint          FF*FF****     FF*FF****       FF*FF****
        Shelters          FF*FF****     FF*FF****       TTT***F**
        Surrounds         FF*FF****     T***F*FF*
        Borders_Interior  FF*FT****     T***T****
        Borders           FF*FT****     T*T*F*FF*
        Contains          T*T*F*FF*
        Partition         T*T*T*FF*
        Equals	          T*F**FFF*
        Overlaps          T*T***T**

    Args:
        relation_binary (int): An integer generated from the combined DE-9IM
            tests.

    Returns:
        RelationshipType: The identified RelationshipType if one of the tests
            passes, otherwise RelationshipType.UNKNOWN.
    '''
    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.SURROUNDS,
                         0b000000000100010110110110000,
                         0b000000000100000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,
                         0b111000100110110000110110000,
                         0b111000000000000000000000000),
        RelationshipTest(RelationshipType.DISJOINT,
                         0b110110000110110000110110000,
                         0b000000000000000000000000000),
        RelationshipTest(RelationshipType.BORDERS,
                         0b000000000001001110110110000,
                         0b000000000001001110000010000),
        RelationshipTest(RelationshipType.BORDERS_INTERIOR,
                         0b000000000101010110110110000,
                         0b000000000101000000000010000),
        RelationshipTest(RelationshipType.OVERLAPS,
                         0b000000000000000000101000100,
                         0b000000000000000000101000100),
        RelationshipTest(RelationshipType.PARTITION,
                         0b000000000000000000101010110,
                         0b000000000000000000101010000),
        RelationshipTest(RelationshipType.CONTAINS,
                         0b000000000000000000101010110,
                         0b000000000000000000101000000),
        RelationshipTest(RelationshipType.EQUALS,
                         0b000000000000000000101001110,
                         0b000000000000000000100000000)
        ]
    for rel_def in test_binaries:
        result = rel_def.test(relation_binary)
        if result:
            return result
    return RelationshipType.UNKNOWN


# %% Relationship Identification Functions
def relate_contours(contour1: StructureSlice,
                    contour2: StructureSlice)->DE27IM_Type:
    '''Get the 27 bit relationship integer for two polygons,

    When written in binary, the 27 bit relationship contains 3 9-bit
    parts corresponding to DE-9IM relationships. The left-most 9 bits
    are the relationship between the second structure's contour and the
    first structure's convex hull polygon. The middle 9 bits are the
    relationship between the second structure's contour and the first
    structure's exterior polygon (i.e. with any holes filled). The
    right-most 9 bits are the relationship between the second
    structure's contour and the first structure's contour.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

    Returns:
        DE9IM_Type: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between contour2 and
            contour1's convex hull, exterior and polygon.
    '''
    def compare(mpoly1: shapely.MultiPolygon,
                mpoly2: shapely.MultiPolygon)->DE9IM_Type:
        '''Get the DE-9IM relationship string for two contours

        The relationship string is converted to binary format, where 'F'
        is '0' and '1' or '2' is '1'.

        Args:
            mpoly1 (shapely.MultiPolygon): All contours for a structure on
                a single slice.
            mpoly2 (shapely.MultiPolygon): All contours for a second
                structure on the same slice.

        Returns:
            DE9IM_Type: A length 9 string of '1's and '0's reflecting the DE-9IM
                relationship between the supplied contours.
        '''
        relation_str = shapely.relate(mpoly1, mpoly2)
        # Convert relationship string in the form '212FF1FF2' into a
        # boolean string.
        relation_bool = relation_str.replace('F','0').replace('2','1')
        return relation_bool

    primary_relation = compare(contour1.contour, contour2.contour)
    external_relation = compare(contour1.exterior, contour2.contour)
    convex_hull_relation = compare(contour1.hull, contour2.contour)
    full_relation = ''.join([convex_hull_relation,
                                external_relation,
                                primary_relation])
    binary_relation = int(full_relation, base=2)
    return binary_relation


def relate_structures(slice_structures: pd.DataFrame,
                      structures: StructurePairType)->DE27IM_Type:
    '''Get the 27 bit relationship integer for two structures on a given slice.

    This is a convenience function that allows the relate_contours function to
    be used with a DataFrame.  The DataFrame should contain contours for both
    structures on a single slice.  The contours are selected based on the
    supplied ROI numbers.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

        structures (StructurePairType): A tuple of ROI numbers which index
            columns in slice_structures.
    Returns:
        DE9IM_Type: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between the
            second contour and the first contour convex hull, exterior and
            contour. If either contour is empty, np.nan is returned.
    '''
    structure = slice_structures[structures[0]]
    if empty_structure(structure):
        return np.nan
    other_contour = slice_structures[structures[1]]
    if empty_structure(other_contour):
        return np.nan
    binary_relation = relate_contours(structure, other_contour)
    return binary_relation


def adjust_boundary_relation(relation: DE27IM_Type,
                             shift_type: str = 'both')->DE27IM_Type:
    '''Adjust the DE-9IM relationship metrics of a boundary slice.

    For the beginning and ending slices of a structure the entire contour must
    be treated as a boundary.  The structure does not have an interior on these
    slices. In this case the “Interior” relations become “Boundary” relations.
    For the `b` structure the, first three values of the DE-9IM relationship
    metric are shifted to become the second three.  For the `a` structure,
    every third value of the DE-9IM relationship metric is shifted by 1.

    Args:
        relation (DE27IM_Type): A triplet numeric DE-9IM relationship metric.
        shift_type (str, optional): The polygon(s) that are boundaries:
            'a' indicates that the first (primary) polygon is at a boundary.
            'b' indicates that the second (secondary) polygon is at a boundary.
            'both' (The default) indicates that both polygons are at a boundary.

    Returns:
        DE27IM_Type: The supplied relationship metric with the interior portion
            of each of the three DE-9IM relationships shifted to the appropriate
            border metric.
    '''

    def shift_value(value: DE27IM_Type, mask: DE27IM_Type,
                     shift: int)->DE27IM_Type:
        # Select the DE27IM relationship values related to the interior.
        interior_relations = value & mask
        # Select the DE27IM relationship values related to the exterior.
        # (Not interior and not Boundary.)
        value_mask = (mask >> shift) + mask
        other_relations = value & ~value_mask
        # Convert the interior relations into corresponding boundary relations.
        boundary_relations = interior_relations >> shift
        # Combine the interior (zeros), boundary and exterior values to form
        # the adjusted DE27IM relationship metric.
        relations_bin = boundary_relations + other_relations
        return relations_bin

    try:
        relation = int(relation)
    except ValueError:
        raise ValueError(f"Invalid DE27IM value: {relation}")
    b_mask = 0b111000000111000000111000000
    a_mask = 0b100100100100100100100100100
    if shift_type == 'a':
        relations_bin = shift_value(relation, a_mask, 1)
    elif shift_type == 'b':
        relations_bin = shift_value(relation, b_mask, 3)
    elif shift_type == 'both':
        relations_bin = shift_value(relation, b_mask, 3)
        relations_bin = shift_value(relations_bin, a_mask, 1)
    else:
        raise ValueError(f"Invalid shift type: {shift_type}")
    return relations_bin


def merge_rel(relation_seq: pd.Series)->int:
    '''Aggregate all the relationship values from each slice to obtain
        one relationship value for the two structures.

    Args:
        relation_seq (pd.Series): The relationship values between the
            contours from each slice.

    Returns:
        int: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between struct2
            and the struct1 convex hulls, exteriors and contours.
    '''
    relation_seq.drop_duplicates(inplace=True)
    relation_seq.dropna(inplace=True)
    relation_seq = relation_seq.astype(int)
    merged_rel = 0
    for rel in list(relation_seq):
        merged_rel = merged_rel | DE27IM_Type(rel)
    return merged_rel


def match_neighbour_slices(slice_table, selected_roi):
    # For each boundary slice of the Primary ROI identify the neighbouring
    # slice(s) that do not have a primary.

    def get_neighbour_slices(boundary_index, slice_index,
                             shift_dir)->pd.DataFrame:
        # Find the boundary neighbours
        # Get the index of the previous slice
        neighbour_slice = slice_index.shift(shift_dir)
        # Select only the slices that are boundary slices of the primary ROI
        neighbour_boundary_slices = neighbour_slice[boundary_index]
        # Drop the neighbour slices that contain a primary contour
        neighbour_boundary_slices.dropna(inplace=True)
        # Reset the index to get the boundary slice number
        neighbour_boundary_slices = neighbour_boundary_slices.reset_index()
        neighbour_boundary_slices.columns = ['Boundary', 'Neighbour']
        return neighbour_boundary_slices

    def no_structure_idx(slice_table, roi_num):
        # Create a series containing the slice index for slices that
        # do NOT have a primary contour
        no_contour_idx = slice_table.index.to_series(name='ROI_Index')
        # Select all slices that do not contain a contour for the Primary ROI
        missing_contour = slice_table[roi_num].apply(empty_structure)
        # Remove the slice indexes that have a primary contour
        no_contour_idx[~missing_contour] = np.nan
        return no_contour_idx

    roi_a, _ = selected_roi
    primary_boundaries = find_boundary_slices(slice_table[roi_a])
    # Get the slice index for slices that do NOT have a primary contour
    no_primary_slice_index = no_structure_idx(slice_table, roi_a)
    # Identify the previous and next slice for each boundary slice that do
    # not have a primary contour
    previous_slice = get_neighbour_slices(primary_boundaries,
                                          no_primary_slice_index, shift_dir=-1)
    next_slice = get_neighbour_slices(primary_boundaries,
                                      no_primary_slice_index, shift_dir=1)
    # Combine the previous and next slices
    neighbouring_slices = pd.concat([previous_slice, next_slice],
                                    ignore_index=True)
    return neighbouring_slices


def boundary_match(slice_table, selected_roi):
    #For each boundary slice of the Primary ROI identify the neighbouring
    # slice(s) that do not have a primary.
    #For each of these neighbouring slices select a Secondary slice
    # for boundary tests
    roi_a, roi_b = selected_roi
    # For each boundary slice of the Primary ROI identify the neighbouring
    # slice(s) that do not have a primary.
    matched_slices = match_neighbour_slices(slice_table, selected_roi)
    # If the slice has a Secondary contour, select that Secondary slice.
    matched_slices = matched_slices.merge(slice_table[roi_b],
                                          left_on='Neighbour',
                                          right_index=True, how='left')
    # If the slice does not have a Secondary contour, but there is a Secondary
    # contour on the same slice as the Primary boundary, select that
    # Secondary slice.
    same_slices = slice_table.loc[matched_slices.Boundary, roi_b]
    same_slices.index = matched_slices.index
    no_nbr = matched_slices[roi_b].isnull()
    matched_slices.loc[no_nbr, roi_b] = same_slices[no_nbr]
    # If neither the neighbouring slice nor the same slice as the Primary
    # boundary have a Secondary contour, do not select a Secondary slice.
    matched_slices.dropna(subset=[roi_b], inplace=True)
    # Select the Primary slice for each pair of Primary and Secondary slices
    matched_slices = matched_slices.merge(slice_table[roi_a],
                                          left_on='Boundary',
                                          right_index=True, how='left')
    # Generate a slice index for the Secondary ROI
    matched_slices['IdxB'] = matched_slices.Neighbour.copy()
    matched_slices.loc[no_nbr, 'IdxB'] = matched_slices.Boundary[no_nbr]
    matched_slices.set_index('IdxB', inplace=True)
    matched_slices.drop(columns=['Boundary', 'Neighbour'], inplace=True)
    return matched_slices


def find_relationship(slice_table: pd.DataFrame,
                      selected_roi: StructurePairType) -> RelationshipType:
    '''Get the relationship between two structures.

    The relationship is based on the DE-9IM relationships between individual
    slices for the two structures' contours, hulls and exteriors.  For slices
    that are not at the boundary of the first structure, the DE-9IM relationship
    is obtained by comparing the contours of the two structures.  For boundary

    The relationship is determined by comparing the DE-9IM relationships between
    the two structures.  The relationships are calculated for all slices that
    contain contours for both structures.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndex as the index, ROI_Num for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Num to select.


    Returns:
        RelationshipType: The relationship between the two structures.
    '''
    _, secondary = selected_roi
    # Slice range = Min(starting slice) to Max(ending slice)
    selected_slices = select_slices(slice_table, selected_roi)
    # Send all slices with both Primary and Secondary contours for standard
    # relation testing
    mid_relations = selected_slices.agg(relate_structures,
                                        structures=selected_roi,
                                        axis='columns')
    mid_relations.name = 'DE27IM'
    # Test the relation between the boundary Primary and the selected Secondary.
    matched_slices = boundary_match(slice_table, selected_roi)
    bdry_rel = matched_slices.agg(relate_structures, structures=selected_roi,
                                  axis='columns')
    bdry_rel.name = 'DE27IM'
    # Apply a Primary boundary shift to the relation results.
    bdry_rel = bdry_rel.apply(adjust_boundary_relation, shift_type='a')
    # If the selected Secondary is also a Secondary boundary, apply a Secondary
    #   boundary shift as well.
    secondary_boundaries = find_boundary_slices(slice_table[secondary])
    bdry_b = [idx for idx in secondary_boundaries
                                if idx in bdry_rel.index]
    bdry_rel.loc[bdry_b] = bdry_rel[bdry_b].apply(adjust_boundary_relation,
                                                  shift_type='b')
    # Merge all results and reduce to single relation
    mid_relations = pd.concat([mid_relations, bdry_rel], axis='index',
                              ignore_index=True)
    relation_binary = merge_rel(mid_relations)
    return identify_relation(relation_binary)
