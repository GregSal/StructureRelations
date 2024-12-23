'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import List, LiteralString, Union

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
        The “*Interior*” bits of the DE-9IM relationship metric are swapped
        with the “*Exterior*”.
        '''
        if hole == 'a':
            # Get the Interior, Boundary and Exterior relations for the "a" polygon.
            interiors = self.relation_str[0:3]
            boundaries = self.relation_str[3:6]
            exteriors = self.relation_str[6:9]
            # Swap the Interior and Exterior relations
            new_str_list = exteriors + boundaries + interiors
        elif hole == 'b':
            # Get the Interior, Boundary and Exterior relations for the "b" polygon.
            interiors = self.relation_str[0:9:3]
            boundaries = self.relation_str[1:9:3]
            exteriors = self.relation_str[2:9:3]
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

    def merge(self, relations: List[DE9IM]):
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
        num_str = to_str(num, size=9)
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
        RelationshipTest(RelationshipType.SURROUNDS,
                         0b110110000100010000000000000,
                         0b000000000100000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,
                         0b110110000100010000100010000,
                         0b000000000000000000100000000),
        RelationshipTest(RelationshipType.DISJOINT,
                         0b110110000100010000100010000,
                         0b000000000000000000000000000),
        RelationshipTest(RelationshipType.BORDERS,
                         0b100010000100010000000000000,
                         0b000010000000010000000000000),
        RelationshipTest(RelationshipType.BORDERS_INTERIOR,
                         0b100010000100010000000000000,
                         0b000010000100010000000000000),
        RelationshipTest(RelationshipType.OVERLAPS,
                         0b101010100101010100101000100,
                         0b101010100101010100101000100),
        RelationshipTest(RelationshipType.PARTITION,
                         0b101010100101010100101000100,
                         0b101010000101010000101000000),
        RelationshipTest(RelationshipType.CONTAINS,
                         0b110110000100010000100000000,
                         0b110000000100000000100000000),
        RelationshipTest(RelationshipType.EQUALS,
                         0b101010000100010000100000000,
                         0b100010000100010000100000000),
        ]

    def __init__(self, contour_a: StructureSlice = None,
                 contour_b: StructureSlice = None,
                 relation_str: str = None,
                 relation_int: int = None,
                 adjustments: List[str] = None):
        if contour_a is not None:
            if contour_b is not None:
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

    def relate_contours(self,
                        contour_a: StructureSlice,
                        contour_b: StructureSlice,
                        adjustments: List[str] = None)->DE27IM_Type:
        '''Get the 27 bit relationship for two structures on a given slice.
        '''
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
                external = DE9IM(relation_str='FFFFFFFFF')
                convex_hull = DE9IM(relation_str='FFFFFFFFF')
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
        if adjustments:
            for adj in adjustments:
                if adj == 'transpose':
                    contour = contour.transpose()
                    external = external.transpose()
                    convex_hull = convex_hull.transpose()
                elif adj == 'boundary_a':
                    contour = contour.boundary_adjustment('a')
                    external = external.boundary_adjustment('a')
                    convex_hull = convex_hull.boundary_adjustment('a')
                elif adj == 'boundary_b':
                    contour = contour.boundary_adjustment('b')
                    external = external.boundary_adjustment('b')
                    convex_hull = convex_hull.boundary_adjustment('b')
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
