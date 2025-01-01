'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import List, LiteralString, Tuple, Union

# Standard Libraries
from enum import Enum, auto
from dataclasses import dataclass

# Shared Packages
import numpy as np
import pandas as pd
import shapely

# Local packages
from types_and_classes import StructurePairType, RegionIndexType
from structure_slice import StructureSlice, empty_structure
from utilities import interpolate_polygon


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
    CONFINES = auto()
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


class DE9IM():
    '''The DE-9IM relationship string for two polygons.
    '''
    def __init__(self,
                 poly_a: shapely.MultiPolygon = None,
                 poly_b: shapely.MultiPolygon = None,
                 relation_str: str = None):
        if (poly_a is not None) & (poly_a is not None):
            self.relation_str = shapely.relate(poly_a, poly_b)
        elif relation_str is not None:
            self.relation_str = relation_str

        else:
            raise ValueError(''.join([
                'Must supply either polygons or a relationship string to ',
                'create a DE9IM object.'
                ]))
        # Convert relationship string in the form '212FF1FF2' into a
        # boolean string.
        self.relation = self.to_bool()
        self.int = self.to_int()

    def to_bool(self):
        relation = self.relation_str.replace('F','0').replace('2','1')
        return relation

    def to_int(self, shift=0):
        shift_factor = 2**shift
        binary_relation = int(self.to_bool(), base=2) * shift_factor
        return binary_relation

    def boundary_adjustment(self, boundary_type: str)->'DE9IM':
        '''Adjust the DE-9IM relationship matrix of a boundary slice.
        '''
        new_str_list = []
        if boundary_type == 'a':
            interiors = self.relation_str[0:3]
            boundaries = self.relation_str[3:6]
            exteriors = self.relation_str[6:9]
            new_str_list.extend(['F', 'F', 'F'])
            for i, b in zip(interiors, boundaries):
                if i == 'F':
                    new_str_list.append(b)
                else:
                    new_str_list.append(i)
            new_str_list.extend(exteriors)
        elif boundary_type == 'b':
            interiors = self.relation_str[0:9:3]
            boundaries = self.relation_str[1:9:3]
            exteriors = self.relation_str[2:9:3]
            for i, b, e in zip(interiors, boundaries, exteriors):
                new_str_list.append('F')
                if i == 'F':
                    new_str_list.append(b)
                else:
                    new_str_list.append(i)
                new_str_list.append(e)
        else:
            raise ValueError(f'Invalid boundary type: {boundary_type}')
        new_str = ''.join(new_str_list)
        return self.__class__(relation_str=new_str)

    def hole_adjustment(self, hole: str)->'DE9IM':
        '''Adjust the DE-9IM relationship matrix of a hole (negative space).
        The “*Interior*” bits of the DE-9IM relationship metric become the
        “*Exterior*” bits and the new “*Interior*” bits become 'F'.
        '''
        if hole == 'a':
            # Get the Interior, Boundary and Exterior relations for the "a" polygon.
            # The interior of a hole is the exterior of the structure.
            # the exterior of a hole is undefined. It may or may not be within
            # the interior of the structure.
            interiors = self.relation_str[0:3]
            boundaries = self.relation_str[3:6]
            exteriors = 'F' * 3
            # The Interior relations become Exterior relations
            new_str_list = exteriors + boundaries + interiors
        elif hole == 'b':
            # The Interior relations become Exterior relations
            interiors = self.relation_str[0:9:3]
            boundaries = self.relation_str[1:9:3]
            exteriors = 'F' * 3
            # Swap the Interior and Exterior relations
            new_str_list = []
            for i, b, e in zip(interiors, boundaries, exteriors):
                new_str_list.extend([e, b, i])
        else:
            raise ValueError(f'Invalid hole type: {hole}')
        new_str = ''.join(new_str_list)
        return self.__class__(relation_str=new_str)

    def transpose(self)->'DE9IM':
        '''Transpose the DE-9IM relationship matrix.
        '''
        # Select every third character from the string.
        interiors = self.relation_str[0:9:3]
        boundaries = self.relation_str[1:9:3]
        exteriors = self.relation_str[2:9:3]
        new_str_list = interiors + boundaries + exteriors
        new_str = ''.join(new_str_list)
        return self.__class__(relation_str=new_str)

    def test_relation(self, mask: int, value: int)->RelationshipType:
        '''Apply the defined test to the supplied relation binary.

        Args:
            relation (int): The number corresponding to a 9-bit binary of
                relationship values.
            mask (int): A sequence of 0s and 1s with every '*' as a '0' and every
                'T' or 'F' bit as a '1'.  The operation:
                `relationship_integer & mask` will set all of the bits that are
                allowed to be either True or False to 0.
            value (int): The value binary is a sequence of 0s and 1s with every 'T'
                as a '1' and every '*' or 'F' bit as a '0'. The relationship is
                identified when value binary is equal to the result of the
                `relationship_integer & mask` operation.
        Returns:
            bool: True if the test passes, False otherwise.
        '''
        relation_int = self.to_int()
        masked_relation = relation_int & mask
        if masked_relation == value:
            return True
        return False

    def merge(self, relations: List["DE9IM"]) -> "DE9IM":
        def to_str(relation_int: int)->str:
            size=9
            str_size = size + 2  # Accounts for '0b' prefix.
            bin_str = bin(relation_int)
            if len(bin_str) < str_size:
                zero_pad = str_size - len(bin_str)
                bin_str = '0' * zero_pad + bin_str[2:]
            elif len(bin_str) > str_size:
                raise ValueError(''.join([
                    'The input integer must be {size} bits long. The input integer ',
                    'was: ', f'{len(bin_str) - 2}'
                    ]))
            else:
                bin_str = bin_str[2:]
            return bin_str

        num = self.to_int()
        for relation in relations:
            num = num | relation.to_int()
        num_str = to_str(num)
        matrix = DE9IM(relation_str=num_str)
        return matrix

    def __repr__(self):
        return f'<DE9IM>: {self.relation_str}'

    def __str__(self):
        bin_str = self.relation
        if len(bin_str) < 9:
            zero_pad = 9 - len(bin_str)
            bin_str = '0' * zero_pad + bin_str[2:]
        bin_fmt = '|{bin1}|\n|{bin2}|\n|{bin3}|'
        bin_dict = {'bin1': bin_str[0:3],
                    'bin2': bin_str[3:6],
                    'bin3': bin_str[6:9]}
        return bin_fmt.format(**bin_dict)


class DE27IM():
    '''The DE-9IM relationships string for two contours, their exteriors, and
    the corresponding convex hull.

        The defined relationships are:
        Relationship  Region Test  Exterior Test  Hull Test
        Disjoint      FF*FF****    F***F****      F***F****
        Shelters      FF*FF****    F***F****      T***F****
        Surrounds     FF*FF****    T***F****      *********
        Borders       F***T****    F***T****      *********
        Confines      F***T****    T***T****      *********
        Partitions    T*T*T*F**    T*T*T*F**      T*T***F**
        Contains      TF*FF****    T***F****      T********
        Overlaps      T*T*T*T**    T*T*T*T**      T*T***T**
        Equals        T*F*T****    T***T****      T********


    '''
    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.DISJOINT,
                         0b110110000100000000100000000,
                         0b000000000000000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,
                         0b111011000010000000010000000,
                         0b000000000000000000100000000),
        RelationshipTest(RelationshipType.SURROUNDS,
                         0b110110000100000000000000000,
                         0b000000000100000000000000000),
        RelationshipTest(RelationshipType.BORDERS,
                         0b100010000100000000000000000,
                         0b000010000000000000000000000),
        RelationshipTest(RelationshipType.CONFINES,
                         0b100010000100000000000000000,
                         0b000010000100000000000000000),
        RelationshipTest(RelationshipType.OVERLAPS,
                         0b100100000000000000000000000,
                         0b100100000000000000000000000),
        RelationshipTest(RelationshipType.CONTAINS,
                         0b100110000000000000000000000,
                         0b100000000000000000000000000),
        RelationshipTest(RelationshipType.EQUALS,
                         0b101110000000000000000000000,
                         0b100010000000000000000000000),
        RelationshipTest(RelationshipType.PARTITION,
                         0b101110000000000000000000000,
                         0b101010000000000000000000000),
        ]

    def __init__(self, contour_a: Union[StructureSlice, shapely.Polygon] = None,
                 contour_b: StructureSlice = None,
                 relation_str: str = None,
                 relation_int: int = None,
                 adjustments: List[str] = None):
        if not empty_structure(contour_a):
            if not empty_structure(contour_b):
                # If both contours are supplied, the relationship is calculated.
                self.relation = self.relate_contours(contour_a, contour_b, adjustments)
                self.int = self.to_int(self.relation)
            else:
                # If only the A contour is supplied, the relationship is
                # Then A is exterior to B
                self.int = 0b001001001001001001001001001
                self.relation = self.to_str(self.int)
        elif contour_b is not None:
            # If only the B contour is supplied, the relationship is
            # Then B is exterior to A
            self.int = 0b000000111000000111000000111
            self.relation = self.to_str(self.int)
        elif relation_str is not None:
            # If contours are not supplied, but a relationship string is
            # supplied, the relationship is set.
            self.relation = relation_str
            self.int = self.to_int(relation_str)
        elif relation_int is not None:
            # If contours are not supplied, but a relationship integer is
            # supplied, the relationship is set.
            self.int = relation_int
            self.relation = self.to_str(relation_int)
        else:
            raise ValueError(''.join([
                'Must supply either StructureSlices or a relationship string ',
                'to create a DE27IM object.'
                ]))

    @property
    def is_null(self)->bool:
        '''Check if the relationship is null.
        '''
        return self.int == 0

    @staticmethod
    def to_str(relation_int: int)->str:
        bin_str = bin(relation_int)
        if len(bin_str) < 29:
            zero_pad = 29 - len(bin_str)
            bin_str = '0' * zero_pad + bin_str[2:]
        elif len(bin_str) > 29:
            raise ValueError(''.join([
                'The input integer must be 27 bits long. The input integer ',
                'was: ', str(relation_int)
                ]))
        else:
            bin_str = bin_str[2:]
        return bin_str

    @staticmethod
    def to_int(relation_str: str)->int:
        try:
            relation_int = int(relation_str, base=2)
        except ValueError as err:
            raise ValueError(''.join([
                'The input string must be a 27 bit binary string. The input ',
                'string was: ', relation_str
                ])) from err
        return relation_int

    @staticmethod
    def relate_contours(contour_a: StructureSlice,
                        contour_b: StructureSlice,
                        adjustments: List[str] = None)->str:
        '''Get the 27 bit relationship for two structures on a given slice.
        Possible adjustments are:
            'transpose': Transpose the relationship matrix.
            'boundary_a': Adjust the relationship matrix for the boundary slice
                of contour_a.
            'boundary_b': Adjust the relationship matrix for the boundary slice
                of contour_b.
            'hole_a': Adjust the relationship matrix for the hole (negative space)
                of contour_a.
            'hole_b': Adjust the relationship matrix for the hole (negative space)
                of contour_b.
        '''
        # If contour_a and contour_b are both StructureSlices, then get the
        # full 27 bit relationship.
        # If contour_a and contour_b are both Shapley Polygons, then get the
        # 9 bit DE9IM relationship and pad the other 18 bits with 0s.
        padding = 'F' * 9
        if isinstance(contour_a, StructureSlice):
            if isinstance(contour_b, StructureSlice):
                contour = DE9IM(contour_a.contour, contour_b.contour)
                external = DE9IM(contour_a.exterior, contour_b.contour)
                convex_hull = DE9IM(contour_a.hull, contour_b.contour)
            else:
                raise ValueError(''.join([
                    'Both contours must either be StructureSlice objects or ',
                    'shapely Polygon objects. contour_b input was: ',
                    f'{str(type(contour_b))}'
                    ]))
        elif isinstance(contour_a, shapely.Polygon):
            if isinstance(contour_b, shapely.Polygon):
                contour = DE9IM(contour_a, contour_b)
                external = DE9IM(relation_str=padding)
                convex_hull = DE9IM(relation_str=padding)
            else:
                raise ValueError(''.join([
                    'Both contours must either be StructureSlice objects or ',
                    'shapely Polygon objects. contour_b input was: ',
                    f'{str(type(contour_b))}'
                    ]))
        else:
            raise ValueError(''.join([
                'Both contours must either be StructureSlice objects or ',
                'shapely Polygon objects. contour_a input was: ',
                f'{str(type(contour_a))}'
                ]))
        # Apply adjustments to the relationship matrix.
        # Note: The order of the adjustments is important.
        # When hole adjustments are applied, only the "contour" bits are relevant,
        # the external and hull bits are set to 'FFFFFFFFF'.
        if adjustments:
            # Apply Boundary Adjustments
            if 'boundary_a' in adjustments:
                contour = contour.boundary_adjustment('a')
                external = external.boundary_adjustment('a')
                convex_hull = convex_hull.boundary_adjustment('a')
            if 'boundary_b' in adjustments:
                contour = contour.boundary_adjustment('b')
                external = external.boundary_adjustment('b')
                convex_hull = convex_hull.boundary_adjustment('b')
            # Apply Hole Adjustments
            if 'hole_a' in adjustments:
                contour = contour.hole_adjustment('a')
                external = DE9IM(relation_str='F' * 9)
                convex_hull = DE9IM(relation_str='F' * 9)
            if 'hole_b' in adjustments:
                contour = contour.hole_adjustment('b')
                external = DE9IM(relation_str='F' * 9)
                convex_hull = DE9IM(relation_str='F' * 9)
            # Apply Transpose Adjustment
            if 'transpose' in adjustments:
                contour = contour.transpose()
                external = external.transpose()
                convex_hull = convex_hull.transpose()
        # Convert the DE-9IM relationships into a DE-27IM relationship string.
        full_relation = ''.join([contour.relation,
                                 external.relation,
                                 convex_hull.relation])
        relation_str = full_relation.replace('F','0').replace('2','1')
        return relation_str

    def merge(self, other: Union['DE27IM', int])->'DE27IM':
        '''Combine two DE27IM relationships.

        Returns:
            int: An integer corresponding to a 27 bit binary value
                reflecting the combined relationships.
        '''
        if isinstance(other, DE27IM):
            other_rel = other.int
        elif isinstance(other, int):
            other_rel = other
        else:
            raise ValueError(''.join([
                'Must supply either a DE27IM object or an integer to merge ',
                'relationships.'
                ]))
        merged_rel = self.int | other_rel
        self.__class__(relation_int = merged_rel)
        return self.__class__(relation_int = merged_rel)

    def identify_relation(self) -> RelationshipType:
        '''Applies a collection of definitions for named relationships to a supplied
        relationship binary.

        Returns:
            RelationshipType: The identified RelationshipType if one of the tests
                passes, otherwise RelationshipType.UNKNOWN.
        '''
        relation_binary = self.int
        for rel_def in self.test_binaries:
            result = rel_def.test(relation_binary)
            if result:
                return result
        return RelationshipType.UNKNOWN

    def __str__(self):
        bin_str = self.relation
        if len(bin_str) < 27:
            zero_pad = 27- len(bin_str)
            bin_str = '0' * zero_pad + bin_str[2:]
        bin_dict = {}
        bin_fmt = '|{bin#}|_'
        bin_list = []
        for idx in range(9):
            row_num = idx % 3
            col_num = idx // 3
            index = row_num * 3 + col_num
            bin_dict[f'bin{index}'] = bin_str[idx*3:(idx+1)*3]

            bin_ref = bin_fmt.replace('#', str(idx))
            if idx % 3 == 2:
                bin_ref = bin_ref.replace('_', '\n')
            else:
                bin_ref = bin_ref.replace('_', '\t')
            bin_list.append(bin_ref)
        return ''.join(bin_list).format(**bin_dict)

    def __repr__(self):
        return f'<DE27IM>: {self.relation}'


def relate_structures(slice_structures: pd.DataFrame,
                      structures: StructurePairType)->DE27IM:
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
        DE27IM: The 27 bit relationship integer  reflecting the combined DE-9IM
            relationship between the second contour and the first contour
            convex hull, exterior and contour. If either contour is empty,
            DE27IM(relation_int=0) is returned.
    '''
    structure = slice_structures[structures[0]]
    if empty_structure(structure):
        structure = None
    other_contour = slice_structures[structures[1]]
    if empty_structure(other_contour):
        other_contour = None
    relation = DE27IM(structure, other_contour)
    return relation


def merged_relations(relations):
    merged = DE27IM(relation_int=0)
    for relation in list(relations):
        merged = merged.merge(relation)
    return merged


    has_boundary = region_table.map(is_boundary)
    boundaries = has_boundary.fillna(0).astype(bool)
    if group_regions:
        boundaries = has_boundary.stack('ROI', future_stack=True)
        boundaries = boundaries.apply(any, axis='columns')
        boundaries = boundaries.unstack('ROI')
    return boundaries


def identify_neighbour_slices(region_table: pd.DataFrame):
    def has_neighbour(region_table: pd.DataFrame, offset: int):
        not_empty = region_table.map(empty_structure, invert=True)
        neighbour = not_empty.shift(offset)
        neighbour['neighbour'] = neighbour.index.to_series()
        neighbour['neighbour'] = neighbour['neighbour'].shift(offset)
        neighbour = neighbour.dropna()
        neighbour.set_index('neighbour', append=True, inplace=True)
        return neighbour

    next_slice = has_neighbour(region_table, 1)
    prev_slice = has_neighbour(region_table, -1)
    neighbour_match = pd.concat([next_slice, prev_slice], axis='index')
    neighbour_match.sort_index(inplace=True)
    return neighbour_match


def find_boundary_pairs(neighbour, region_boundaries, region_idx, is_hole):
    # X If the region is a hole, select the neighbour slice that has a contour.
    # X Otherwise, select the neighbour slice that has no contour.
    # Changed my mind, always select the neighbour slice that has no contour.
    #if is_hole:
    #    nbr_match = neighbour.T
    #else:
    nbr_match = neighbour.map(lambda x: not x).T
    boundary_neighbour_index = nbr_match.join(region_boundaries[region_idx],
                                              rsuffix='bdr')
    boundary_neighbour_index = boundary_neighbour_index.T.apply(all)
    index_list = list(boundary_neighbour_index[boundary_neighbour_index].index)
    return index_list


def interpolate_region(region_num, regions: pd.DataFrame,
                       slice_pair: Tuple[int, int]):
    # Matched Interpolated Secondary Slice
    region_1 = regions.at[slice_pair[0], region_num]
    region_2 = regions.at[slice_pair[1], region_num]
    if empty_structure(region_1):
        if empty_structure(region_2):
            interpolated_region = np.nan
        else:
            interpolated_region = interpolate_polygon(slice_pair,
                                                      region_2.polygon)
    else:
        if empty_structure(region_2):
            interpolated_region = interpolate_polygon(slice_pair,
                                                      region_1.polygon)
        else:
            interpolated_region = interpolate_polygon(slice_pair,
                                                      region_1.polygon,
                                                      region_2.polygon)
    return interpolated_region


def set_adjustments(region_table: pd.DataFrame, region_boundaries,
                    selected_roi: StructurePairType,
                    region1: RegionIndexType, region2: RegionIndexType,
                    slices: List[float]):
    # The first region is always a boundary.
    adjustments = ['boundary_a']
    # If the region is a hole, then adjust interior and exterior parts
    # of the relation need to be swapped.
    is_hole = region_table[region1].dropna().iat[1].is_hole
    if is_hole:
        adjustments.append('hole_a')
    # If the "Secondary" ROI is the primary ROI, then the relation needs to be
    # transposed.
    is_secondary_roi = selected_roi.index(region1[0]) == 1
    if is_secondary_roi:
        adjustments.append('transpose')
    # Check whether the secondary slices are also at a boundary.
    if any(region_boundaries.loc[list(slices), region2]):
        adjustments.append('boundary_b')
    is_hole2 = region_table[region2].dropna().iat[1].is_hole
    if is_hole2:
        adjustments.append('hole_b')
    return adjustments


def get_boundary_relations(region_table, neighbour_match, region_boundaries, selected_roi):
    boundary_relations = {}
    nbr_grp = neighbour_match[selected_roi].T.groupby(level=['ROI', 'Label'])
    for region_idx, neighbour in nbr_grp:
        is_hole = region_table[region_idx].dropna().iat[1].is_hole
        index_list = find_boundary_pairs(neighbour, region_boundaries, region_idx, is_hole)
        other_roi_num = [r for r in selected_roi if not r == region_idx[0]][0]
        for slice_pair in index_list:
            # Interpolated Boundary Slice
            boundary_region = interpolate_region(region_idx, region_table,
                                                slice_pair)
            # Loop through the regions of the other roi
            for other_region in region_table[other_roi_num].columns:
                other_region_idx = (other_roi_num, other_region)
                # Interpolate the other roi
                other_region = interpolate_region(other_region_idx, region_table,
                                                slice_pair)
                adjustments = set_adjustments(region_table, region_boundaries,
                                              selected_roi, region_idx,
                                              other_region_idx, slice_pair)
                # Get boundary relation as a DE27IM object.
                boundary_relation = DE27IM(boundary_region, other_region,
                                        adjustments=adjustments)
                region_pair = tuple([region_idx, other_region_idx, slice_pair])
                boundary_relations[region_pair] = boundary_relation
    return boundary_relations


def find_relations(slice_table, selected_roi):
    # Split each Structure into distinct regions for boundary tests.
    region_table = make_region_table(slice_table)
    # Identify the boundary slices of each region.
    region_boundaries = find_boundary_slices(region_table)
    # For each slice of each region identify the whether the region is present on
    # the neighbouring slice.
    neighbour_match = identify_neighbour_slices(region_table)

    boundary_relations = get_boundary_relations(region_table, neighbour_match, region_boundaries, selected_roi)
    # Slice range = Min(starting slice) to Max(ending slice)
    selected_slices = select_slices(slice_table, selected_roi)
    # Send all slices with both Primary and Secondary contours for standard
    # relation testing
    mid_relations = list(selected_slices.agg(relate_structures,
                                             structures=selected_roi,
                                             axis='columns'))
    mid_relations.extend(boundary_relations.values())
    relation =  merged_relations(mid_relations)
    return relation
