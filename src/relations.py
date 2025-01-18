'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import List, LiteralString, Tuple, Union

# Standard Libraries
from enum import Enum, auto
from dataclasses import dataclass, asdict
from functools import partial

# Shared Packages
import pandas as pd
import shapely
import networkx as nx

# Local packages
from types_and_classes import ROI_Type, RegionNode, SliceIndexType, SliceNeighbours, StructurePairType
from types_and_classes import RegionGraph, RegionNodeType, RegionIndexType
from utilities import calculate_new_slice_index, interpolate_polygon
from structure_slice import ContourType, StructureSlice, empty_structure
from structure_set import generate_region_graph, select_slices


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
    # A length 9 string of '1's and '0's representing a DE-9IM relationship.
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

    def __eq__(self, value):
        if isinstance(value, self.__class__):
            return self.to_int() == value.to_int()
        if isinstance(value, int):
            return self.to_int() == value
        if isinstance(value, str):
            value_str = value.replace('F','0').replace('2','1')
            return self.to_bool() == value_str

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
    # A 27 bit binary value composed of three DE9IM_Values concatenated.
    # The right-most 9 binary digits represent the DE-9IM relationship between
    #    polygon b and polygon a.
    # The middle 9 binary digits represent the DE-9IM relationship between
    #    polygon b and the *exterior* of polygon a.
    # The left 9 binary digits represent the DE-9IM relationship between
    #    polygon b and the *convex hull* of polygon a.

    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.DISJOINT,
                         0b110110000100010000100010000,
                         0b000000000000000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,
                         0b110110000100010000100010000,
                         0b000000000000000000100000000),
        RelationshipTest(RelationshipType.SURROUNDS,
                         0b110110000100010000000000000,
                         0b000000000100000000000000000),
        RelationshipTest(RelationshipType.BORDERS,
                         0b100010000100000000000000000,
                         0b000010000000000000000000000),
        RelationshipTest(RelationshipType.CONFINES,
                         0b100010000100000000000000000,
                         0b000010000100000000000000000),
        RelationshipTest(RelationshipType.CONTAINS,
                         0b110110100100000000100000000,
                         0b110000000100000000100000000),
        RelationshipTest(RelationshipType.EQUALS,
                         0b101010100000000000000000000,
                         0b100010000000000000000000000),
        RelationshipTest(RelationshipType.PARTITION,
                         0b101010100000000000000000000,
                         0b101010000000000000000000000),
        RelationshipTest(RelationshipType.OVERLAPS,
                         0b100000100100000000100000000,
                         0b100000100100000000100000000)
        ]
    # padding is a string of 'FFFFFFFFF', which becomes 9 zeros when converted
    # to binary.  Padding is used in cases where Exterior and Hull relationships
    # are not relevant.
    padding = 'F' * 9  # 'FFFFFFFFF'
    # If only the A contour is supplied, then A is exterior to B
    exterior_a = 'FF1' * 3  # 'FF1FF1FF1'
    # If only the B contour is supplied, then B is exterior to A
    exterior_b = 'F' * 3 + 'F' * 3 + '1' * 3  # 'FFFFFF111'

    def __init__(self, contour_a: ContourType = None,
                 contour_b: ContourType = None,
                 relation_str: str = None,
                 relation_int: int = None,
                 adjustments: List[str] = None):
        if not empty_structure(contour_a):
            if not empty_structure(contour_b):
                # If both contours are supplied, the relationship is calculated.
                self.relation = self.relate_contours(contour_a, contour_b,
                                                     adjustments)
                self.int = self.to_int(self.relation)
            else:
                # If only the A contour is supplied, the relationship is
                #   A is exterior to B
                relation_group = tuple([DE9IM(relation_str=self.exterior_a)] * 3)
                self.relation = self.combine_groups(relation_group, adjustments)
                self.int = self.to_int(self.relation)
        elif contour_b is not None:
            # If only the B contour is supplied, the relationship is
            #   B is exterior to A
            relation_group = tuple([DE9IM(relation_str=self.exterior_b)] * 3)
            self.relation = self.combine_groups(relation_group, adjustments)
            self.int = self.to_int(self.relation)
        elif relation_str is not None:
            # If contours are not supplied, but a relationship string is
            # supplied, the relationship is set.
            # Note: adjustments are not applied to the relationship string.
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

    def relate_contours(self, contour_a: StructureSlice,
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
        if isinstance(contour_a, StructureSlice):
            if isinstance(contour_b, StructureSlice):
                contour = DE9IM(contour_a.contour, contour_b.contour)
                external = DE9IM(contour_a.exterior, contour_b.contour)
                convex_hull = DE9IM(contour_a.hull, contour_b.contour)
                relation_group = (contour, external, convex_hull)
            else:
                raise ValueError(''.join([
                    'Both contours must either be StructureSlice objects or ',
                    'shapely Polygon objects. contour_b input was: ',
                    f'{str(type(contour_b))}'
                    ]))
        else:
            # If contour_a and contour_b are shapely Polygons or RegionNodeType
            # objects, then get the 9 bit DE9IM relationship and pad the other
            # 18 bits with 0s.
            if isinstance(contour_a, shapely.Polygon):
                poly_a = contour_a
            else:
                try:
                    poly_a = contour_a['polygon']
                except AttributeError as err:
                    raise ValueError(''.join([
                        'Both contours must either be StructureSlice objects ',
                        'or a combination of shapely Polygon objects and ',
                        'RegionNodeType objects. contour_a input was: ',
                    f'{str(type(contour_a))}'
                    ])) from err
            if isinstance(contour_b, shapely.Polygon):
                poly_b = contour_b
            else:
                try:
                    poly_b = contour_b['polygon']
                except AttributeError as err:
                    raise ValueError(''.join([
                        'Both contours must either be StructureSlice objects ',
                        'or a combination of shapely Polygon objects and ',
                        'RegionNodeType objects. contour_b input was: ',
                    f'{str(type(contour_b))}'
                    ])) from err
            contour = DE9IM(poly_a, poly_b)
            external = DE9IM(relation_str=self.padding)
            convex_hull = DE9IM(relation_str=self.padding)
            relation_group = (contour, external, convex_hull)
        relation_str = self.combine_groups(relation_group, adjustments)
        return relation_str

    def apply_adjustments(self, relation_group: Tuple[DE9IM, DE9IM, DE9IM],
                          adjustments: List[str])-> tuple[DE9IM, DE9IM, DE9IM]:
        # Apply adjustments to the relationship matrix.
        # Note: The order of the adjustments is important.
        # When hole adjustments are applied, only the "contour" bits are relevant,
        # the external and hull bits are set to 'FFFFFFFFF'.
        # Apply Boundary Adjustments
        if 'boundary_a' in adjustments:
            relation_group = tuple(de9im.boundary_adjustment('a')
                                   for de9im in relation_group)
        if 'boundary_b' in adjustments:
            relation_group = tuple(de9im.boundary_adjustment('b')
                                   for de9im in relation_group)
        # Apply Hole Adjustments
        if 'hole_a' in adjustments:
            contour, external, convex_hull = relation_group
            contour = contour.hole_adjustment('a')
            external = DE9IM(relation_str=self.padding)
            convex_hull = DE9IM(relation_str=self.padding)
            relation_group = (contour, external, convex_hull)
        if 'hole_b' in adjustments:
            contour, external, convex_hull = relation_group
            contour = contour.hole_adjustment('b')
            external = DE9IM(relation_str=self.padding)
            convex_hull = DE9IM(relation_str=self.padding)
            relation_group = (contour, external, convex_hull)
        # Apply Transpose Adjustment
        if 'transpose' in adjustments:
            relation_group = tuple(de9im.transpose()
                                   for de9im in relation_group)
        return relation_group

    def combine_groups(self, relation_group: Tuple[DE9IM, DE9IM, DE9IM],
                       adjustments: List[str])-> str:
        if adjustments:
            relation_group = self.apply_adjustments(relation_group, adjustments)
        # Convert the DE-9IM relationships into a DE-27IM relationship string.
        full_relation = ''.join(de9im.relation for de9im in relation_group)
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


def set_adjustments(region1: RegionNodeType,
                    region2: Union[RegionNodeType, None],
                    selected_roi: StructurePairType):
    # The first region is always a boundary.
    adjustments = ['boundary_a']
    # If either regions is a hole, then the interior and exterior parts of the
    # relation need to be adjusted.
    if region1['is_hole']:
        adjustments.append('hole_a')
    if region2 is not None and region2['is_hole']:
        adjustments.append('hole_b')
    # If the "Secondary" ROI is the primary ROI, then the relation needs to be
    # transposed.
    is_secondary_roi = selected_roi.index(region1['roi']) == 1
    if is_secondary_roi:
        adjustments.append('transpose')
    # Check whether the secondary slices are also at a boundary.
    if region2 is not None and region2['is_boundary']:
        adjustments.append('boundary_b')
    return adjustments


# %% Functions for boundary relations
def node_selector(region_graph: RegionGraph, region: RegionNodeType,
                  selected_roi: StructurePairType) -> List[RegionIndexType]:
    # Select regions from the other ROI that are between region's
    # prev_slice and next_slice.

    # get the other roi
    roi = region['roi']
    if roi == selected_roi[0]:
        other_roi = selected_roi[1]
    else:
        other_roi = selected_roi[0]
    # get the slice neighbours
    slice_neighbours = region['slice_neighbours']
    prev_slice = slice_neighbours.previous_slice
    next_slice = slice_neighbours.next_slice
    # Select regions from the other ROI that are between region's
    # prev_slice and next_slice.
    selected_nodes = []
    for label, node in region_graph.nodes.items():
        if node['roi'] != other_roi:
            continue
        if node['slice_index'] < prev_slice:
            continue
        if node['slice_index'] > next_slice:
            continue
        selected_nodes.append(label)

    selection = region_graph.subgraph(selected_nodes)
    return selection


def get_boundaries(graph: RegionGraph,
                    selected_roi: StructurePairType)->List[RegionIndexType]:
    boundaries = []
    for node in graph.nodes:
        region = graph.nodes[node]
        if region['is_boundary'] and region['roi'] in selected_roi:
            boundaries.append(node)
    return boundaries


def drop_nodes(graph: nx.Graph, node):
    # drop the node and its neighbours from the graph
    neighbours = [neighbour for neighbour in graph.neighbors(node)]
    if neighbours:
        graph.remove_node(neighbours[0])
        drop_nodes(graph, node)
    else:
        graph.remove_node(node)


def get_relation(region1: RegionNodeType,
                  region2: Union[RegionNodeType, None],
                  selected_roi: StructurePairType):
    # Get the necessary adjustments for the relationship.
    adjustments = set_adjustments(region1, region2, selected_roi)
    relation = DE27IM(region1, region2, adjustments=adjustments)
    return relation


def get_matching_region(sub_graph: RegionGraph,
                        region1: RegionNodeType)->Union[RegionNodeType, None]:
    # if a region has the same slice index as the boundary, then return it.
    slice_index = region1['slice_index']
    sub_graph_slices = dict(sub_graph.nodes.data('slice_index'))
    other_indexes = {idx: node for node, idx in sub_graph_slices.items()}
    if slice_index in other_indexes:
        other_node = other_indexes[slice_index]
        region2 = sub_graph.nodes[other_node]
        # remove the node and it's neighbours from the sub-graph
        drop_nodes(sub_graph, other_node)
        return region2
    return None


def get_interpolated_region(sub_graph):
    # Select the first node in sub_graph.
    # Select that node's neighbour.
    # build an interpolated region from these two regions.

    # Select the first node in the subgraph
    first_node_label = list(sub_graph.nodes)[0]
    first_node = sub_graph.nodes[first_node_label]
    # Select the first node's neighbour as the second node,
    neighbour_node = next(sub_graph.neighbors(first_node_label))
    second_node = sub_graph.nodes[neighbour_node]
    # Get the slice indexes for the two nodes
    slices = (first_node['slice_index'], second_node['slice_index'])
    new_slice = calculate_new_slice_index(slices)
    new_neighbours = SliceNeighbours(new_slice, *slices)
    # Define a new node to store the interpolated region
    intp_node = RegionNode(**second_node)
    intp_node.is_interpolated = True
    intp_node.slice_index = new_slice
    intp_node.slice_neighbours = new_neighbours
    # If either node is a boundary, then the interpolated node is considered
    # a boundary.
    if second_node['is_boundary']:
        intp_node.is_boundary = True
    # Interpolate the region to match the boundary slice
    intp_poly = interpolate_polygon(slices, first_node['polygon'],
                                    second_node['polygon'])
    # Update the node with the interpolated polygon
    intp_node.polygon = intp_poly
    # Remove the original nodes from the sub-graph
    drop_nodes(sub_graph, first_node_label)
    return asdict(intp_node)


def get_boundary_relations(region_graph: RegionGraph,
                           selected_roi: Tuple[int, int]) -> List[DE27IM]:
    '''Get boundary relations between regions in the graph for the selected ROIs.

    Args:
        region_graph (nx.Graph): The region graph.
        selected_roi (Tuple[int, int]): A tuple of two ROI numbers which refer
            to the structures to be compared.

    Returns:
        List[DE27IM]: A list of the boundary relationships for the two selected
            ROIs.
    '''
    boundary_relations = []
    boundaries = get_boundaries(region_graph, selected_roi)
    for boundary_node in boundaries:
        region1 = region_graph.nodes[boundary_node]
        # Select for neighbouring regions in the other ROI.
        sub_graph = node_selector(region_graph, region1, selected_roi).copy()
        if len(sub_graph) == 0:
            region2 = None  # No regions in the other ROI.
            relation = get_relation(region1, region2, selected_roi)
            boundary_relations.append(relation)
            continue

        # if a region has the same slice index as the boundary, then use
        # it to determine a relationship.
        region2 = get_matching_region(sub_graph, region1)
        if region2 is not None:
            relation = get_relation(region1, region2, selected_roi)
            boundary_relations.append(relation)

        # Interpolate the remaining regions to match the boundary slice.
        done = len(sub_graph) == 0
        while not done:
            # interpolate each relevant region to match the boundary slice.
            region2 = get_interpolated_region(sub_graph)
            relation = get_relation(region1, region2, selected_roi)
            boundary_relations.append(relation)
            done = len(sub_graph) == 0
    return boundary_relations


# %% Functions for finding relations
def merged_relations(relations):
    merged = DE27IM(relation_int=0)
    for relation in list(relations):
        merged = merged.merge(relation)
    return merged


def find_relations(slice_table, regions, selected_roi):
    selected_slices = select_slices(slice_table, selected_roi)
    # Send all slices with both Primary and Secondary contours for standard
    # relation testing
    mid_relations = list(selected_slices.agg(relate_structures,
                                             structures=selected_roi,
                                             axis='columns'))
    boundary_relations = get_boundary_relations(regions, selected_roi)
    mid_relations.extend(boundary_relations)
    relation =  merged_relations(mid_relations)
    return relation
