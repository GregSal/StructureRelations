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
from structure_slice import StructureSlice
from structure_slice import empty_structure
from structure_slice import identify_boundary_slices, select_slices


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
    BORDERS_INTERIOR: The two structures share a common boundary and one is
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
                      structures: StructurePairType)->DE9IM_Type:
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
            contour.
    '''
    structure = slice_structures[structures[0]]
    other_contour = slice_structures[structures[1]]
    binary_relation = relate_contours(structure, other_contour)
    return binary_relation


def adjust_slice_boundary_relations(relation_seq: pd.Series,
                                    structures: StructurePairType,
                                    boundary_slices: pd.DataFrame=None)->pd.Series:
    '''Adjust the DE-9IM relationship metric for the boundary slices of both
    structures.

    For the beginning and ending slices of a structure the entire contour must
    be treated as a boundary.  The structure does not have an interior on these
    slices. In this case the “Interior” relations become “Boundary” relations.
    For the `b` structure the, first three values of the DE-9IM relationship
    metric are shifted to become the second three.  For the `a` structure,
    every third value of the DE-9IM relationship metric is shifted by 1.

    Args:
        relation_seq (pd.Series): A series with SliceIndexType as the index and
            DE-9IM relationship metrics as the values.
        structures (StructurePairType): A tuple of ROI numbers which index
            columns in boundary_slices.
        boundary_slices (pd.DataFrame, optional): Table with ROI_Type as columns
            and SliceIndexType as values. Every value in a given column indicates a
            boundary slice for that ROI.  If not supplied, every row
            in relation_seq is assumed to be a boundary slice for both ROIs.

    Returns:
        pd.Series: The supplied relation_seq with adjusted DE-9IM relationship
        metrics for the boundary slices of both structures.
    '''
    def shift_values(value: DE9IM_Type, mask: DE9IM_Type,
                     shift: int)->DE9IM_Type:
        # Select the DE-9IM relationship values related to the interior.
        interior_relations = value & mask
        # Select the DE-9IM relationship values related to the exterior.
        # (Not interior and not Boundary.)
        value_mask = (mask >> shift) + mask
        other_relations = value & ~value_mask
        # Convert the interior relations into corresponding boundary relations.
        boundary_relations = interior_relations >> shift
        # Combine the interior (zeros), boundary and exterior values to form
        # the adjusted DE-9IM relationship metric.
        relations_bin = boundary_relations + other_relations
        return relations_bin

    roi_a, roi_b = structures
    if boundary_slices is None:
        b_boundary_slices = list(relation_seq.index)
        a_boundary_slices = b_boundary_slices
    else:
        b_boundary_slices = list(boundary_slices[roi_b])
        a_boundary_slices = list(boundary_slices[roi_a])
    # Boundaries of b
    # Select the first three values of the DE-9IM relationship metric.
    b_mask = 0b111000000111000000111000000
    for boundary_slice in b_boundary_slices:
        value = relation_seq[boundary_slice]
        relations_bin = shift_values(value, b_mask, 3)
        relation_seq[boundary_slice] = relations_bin
    # Boundaries of a
    a_mask = 0b100100100100100100100100100
    for boundary_slice in a_boundary_slices:
        value = relation_seq[boundary_slice]
        relations_bin = shift_values(value, a_mask, 1)
        relation_seq[boundary_slice] = relations_bin
    return relation_seq


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
    merged_rel = 0
    for rel in list(relation_seq):
        merged_rel = merged_rel | rel
    return merged_rel


def get_non_boundary_relations(slice_table: pd.DataFrame,
                           selected_roi: StructurePairType) -> pd.Series:
    '''Determine the relation for contours that are not on boundary slices.

    The 27 bit relationship integers are calculated for the slices that contain
    primary contours and are not boundary slices of the primary structure.
    Boundary slices for the secondary structure are not considered unless they
    are also boundary slices of the primary structure.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndexType as the index, ROI_Type for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Type to select.

    Returns:
        pd.Series: The relationship values between the contours from each
            non-boundary slice, with SliceIndexType as the index.
    '''
    # select the slices spanned by both structures
    structure_slices = select_slices(slice_table, selected_roi)
    roi_a, roi_b = selected_roi
    # Select the slices that are not boundary slices of the primary structure.
    is_boundary_slice = identify_boundary_slices(structure_slices[roi_a])
    mid_slices = structure_slices.loc[~is_boundary_slice, :].copy()
    # Select only the slices that have the primary structure
    #empty_primary_slices = mid_slices[roi_a].apply(empty_structure)
    #mid_slices = mid_slices[~empty_primary_slices]
    # Replace the nan values with empty polygons for duck typing.
    mid_slices.fillna(StructureSlice([]), inplace=True)
    # Calculate the DE-9IM relations for these slices.
    relation_seq = mid_slices.agg(relate_structures, structures=selected_roi,
                                  axis=1)
    relation_seq.name = 'DE9IM'
    return relation_seq


def get_matched_bdy_rel(slice_table: pd.DataFrame,
                         selected_roi: StructurePairType) -> pd.Series:
    '''Determine the relation for contours on matched boundary slices.

    Matched boundary slices are those that are boundary slices of both
    structures.  The 27 bit relationship integer is calculated for these slices.
    If there are no matched boundary slices, an empty Series is returned.
    The boundary slices are selected based on the following conditions:
        1. The slice is at a boundary of the primary structure.
        2. The slice has both a primary and secondary contour.
        3. The one of the neighbouring slices has neither primary nor secondary
            contour.
    The relations are adjusted to account for the fact that these are boundary
    slices of the 3D structure.  As a result the interior relations are shifted
    to become boundary relations.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndexType as the index, ROI_Type for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Type to select.

    Returns:
        pd.Series: The relationship values between the contours from each
            matched boundary slice, with SliceIndexType as the index.  If there are
            no matched boundary slices, an empty Series is returned.
    '''
    structure_slices = slice_table[selected_roi]
    # Select rows where both columns contain non-empty StructureSlice objects
    empty_rows = structure_slices.isna().any(axis=1)
    # Check if neighboring rows are NaN or contain empty StructureSlice objects.
    prev_row = structure_slices.shift(1)
    next_row = structure_slices.shift(-1)
    missing_neighbour = (
        (prev_row.map(empty_structure).all(axis=1)) |
        (next_row.map(empty_structure).all(axis=1))
        )
    # Select rows based on the condition
    selected_rows = ~empty_rows & missing_neighbour
    if not selected_rows.any():
        return pd.Series(name='DE9IM')
    boundary_slices = structure_slices[selected_rows].copy()
    # Replace the nan values with empty polygons for duck typing.
    boundary_slices.fillna(StructureSlice([]), inplace=True)
    # Calculate the DE-9IM relations for these slices.
    relation_seq = boundary_slices.agg(relate_structures,
                                       structures=selected_roi, axis=1)
    relation_seq.name = 'DE9IM'
    # Adjust the DE-9IM relations to account for the fact that these are
    # boundary slices of the 3D structure
    relation_seq = adjust_slice_boundary_relations(relation_seq, selected_roi)
    return relation_seq


def get_offset_bdy_rel(slice_table: pd.DataFrame,
                         selected_roi: StructurePairType) -> pd.Series:
    '''Determine the relation for contours on offset boundary slices.

    Offset boundary slices are those where the boundary of the primary structure
    is a neighbour of the boundary of the secondary structure.  The 27 bit
    relationship integer is calculated for these slices. If there are no
    offset boundary slices, an empty Series is returned.  The boundary slices
    are selected on the following basis:
        1. The slice is at a boundary of the primary structure.
        2. The slice does not have the secondary structure.
        3. The one of the neighbouring slices has only the second structure.
    Relations are calculated between the primary slice and the neighbouring
    secondary slice and adjusted to account for the fact that these are boundary
    slices of the 3D structure.  As a result the interior relations are shifted
    to become boundary relations.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndexType as the index, ROI_Type for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Type to select.

    Returns:
        pd.Series: The relationship values between the contours from each
            offset boundary slice, with SliceIndexType as the index.  If there are
            no offset boundary slices, an empty Series is returned.
    '''
    def select_boundary_slices(slice_table: pd.DataFrame,
                               selected_roi: StructurePairType)->pd.Series:
        '''Select boundary slices where the boundaries of the two structures meet.

        The boundary slices are selected based on the following conditions:
            1. The slice is at a boundary of the primary structure.
            2. The slice does not have the secondary structure.
            3. The one of the neighbouring slices has only the second structure.

        Args:
            slice_table (pd.DataFrame): A table of StructureSlice data with
                SliceIndexType as the index, ROI_Type for columns and StructureSlice
                or NaN as the values.
            selected_roi (StructurePairType): A tuple of two ROI_Type to select.

        Returns:
            pd.Series: A boolean series with SliceIndexType as the index. Values
            are True for slices where the boundaries of the two structures meet.
        '''
        # Select the slices spanned by both structures
        roi_a, roi_b = selected_roi
        # Select the slices where roi_b is NaN or is an empty StructureSlice
        # object.
        empty_or_nan_roi_b = slice_table[roi_b].apply(empty_structure)
        # Select only the boundary slices of the primary structure.
        boundary_slices = identify_boundary_slices(slice_table[roi_a])
        # Check if neighboring rows are NaN or contain empty StructureSlice
        # objects.
        prev_row = slice_table.shift(1)
        next_row = slice_table.shift(-1)
        neighbour_empty_or_nan = (
            prev_row.map(empty_structure).any(axis=1) |
            next_row.map(empty_structure).any(axis=1)
        )
        # Exclude rows where neighbouring rows are both empty.
        neighbour_both_empty = (
            prev_row.map(empty_structure).all(axis=1) |
            next_row.map(empty_structure).all(axis=1)
        )
        # Select rows based on the above conditions.
        selected_rows = (empty_or_nan_roi_b & boundary_slices &
                         neighbour_empty_or_nan & ~neighbour_both_empty)
        return selected_rows

    def pair_neighbouring_slices(slice_table: pd.DataFrame,
                                        selected_roi: StructurePairType,
                                        selected_rows: pd.Series)->pd.DataFrame:
        '''Combine primary slices from the selected rows with neighbouring
        secondary slices.

        Join the primary slice with a neighbouring secondary slice. If a
        primary slice has two neighbouring secondary slices, the one with the
        lower SliceIndexType is selected.

        Args:
            slice_table (pd.DataFrame): A table of StructureSlice data with
                SliceIndexType as the index, ROI_Type for columns and StructureSlice
                or NaN as the values.
            selected_roi (StructurePairType): A tuple of two ROI_Type to select.
            selected_rows (pd.Series): A boolean series with SliceIndexType as the
                index. Values are True for slices where the boundaries of the
                two structures meet.

        Returns:
            pd.Series: A table of StructureSlice pairs with SliceIndexType as
                the index, ROI_Type for columns and StructureSlice as the values.
                The table contains the StructureSlice for primary slice and the
                neighbouring StructureSlice for the secondary slice.
        '''
        structure_slices = slice_table[selected_roi]
        roi_a, roi_b = selected_roi
        # Get neighbouring secondary slices
        prev_row = structure_slices[roi_b].shift(1)
        next_row = structure_slices[roi_b].shift(-1)
        # Convert empty StructureSlice to NaN.
        prev_row[prev_row.apply(empty_structure)] = np.NaN
        next_row[next_row.apply(empty_structure)] = np.NaN
        # Combine the neighbouring secondary slices
        neighbouring_secondary = prev_row.combine_first(next_row)
        # Include only rows where roi_a has a value
        has_primary = structure_slices[roi_a].notna()
        neighbouring_secondary = neighbouring_secondary[has_primary]
        # Pair the neighbouring secondary slices with the primary slices from
        # the selected rows
        paired_slices = pd.DataFrame({
            roi_a: structure_slices.loc[selected_rows, roi_a],
            roi_b: neighbouring_secondary.loc[selected_rows]
            }).dropna()
        return paired_slices

    # Select the boundary slices where the boundaries of the two structures meet.
    boundary_slices = select_boundary_slices(slice_table, selected_roi)
    if not boundary_slices.any():
        return pd.Series(name='DE9IM')
    # Pair the primary slices with neighbouring secondary slices.
    paired_slices = pair_neighbouring_slices(slice_table, selected_roi,
                                             boundary_slices)
    # If there are no paired slices, return an empty DataFrame.
    if paired_slices.empty:
        return pd.Series(name='DE9IM')
    # Calculate the DE-9IM relations for the slices that are not boundary slices.
    relation_seq = paired_slices.agg(relate_structures, structures=selected_roi,
                                     axis=1)
    relation_seq.name = 'DE9IM'
    # Adjust the DE-9IM relations to account for the fact that these are
    # boundary slices of the 3D structure
    relation_seq = adjust_slice_boundary_relations(relation_seq, selected_roi)
    return relation_seq


def find_relationship(slice_table: pd.DataFrame,
                      selected_roi: StructurePairType) -> RelationshipType:
    '''Get the relationship between two structures.

    The relationship is based on the DE-9IM relationships between individual
    slices for the two structures' contours, hulls and exteriors.  For slices
    that are not at the boundary of the first structure, the DE-9IM relationship
    is obtained by comparing the contours of the two structures.  For boundary

    The relationship is identified by comparing the DE-9IM

    determined by comparing the DE-9IM relationships between
    the two structures.  The relationships are calculated for all slices that
    contain contours for both structures.  The relationship is identified by
    comparing the DE-9IM relationships for the two

    Args:
        slice_table (pd.DataFrame): _description_
        selected_roi (StructurePairType): _description_

    Returns:
        RelationshipType: _description_
    '''
    all_relations = []
    mid_relations = get_non_boundary_relations(slice_table, selected_roi)
    if not mid_relations.empty:
        all_relations.append(mid_relations)
    matching_boundary_relations = get_matched_bdy_rel(slice_table, selected_roi)
    if not matching_boundary_relations.empty:
        all_relations.append(matching_boundary_relations)
    offset_boundary_relations = get_offset_bdy_rel(slice_table, selected_roi)
    if not offset_boundary_relations.empty:
        all_relations.append(offset_boundary_relations)
    relation_binary = merge_rel(pd.concat(all_relations))
    return identify_relation(relation_binary)
