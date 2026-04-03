'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import List, Tuple, Union, Dict, Optional

# Standard Libraries
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging

# Shared Packages
import shapely

# Local packages
from contours import Contour, ContourMargin
from region_slice import RegionSlice, empty_structure
from types_and_classes import PolygonType, HoleType
from utilities import make_solid, int2matrix


RELATION_SCHEMA_VERSION = 2


class AmbiguousRelationshipError(ValueError):
    '''Raised when a relationship matches multiple relationship definitions.'''


BOUNDARY_TOKEN = HoleType.BOUNDARY.value.lower()


class AdjustmentType(str, Enum):
    '''Internal adjustment tokens applied to DE-9IM bands before merge.'''

    BOUNDARY_A = f'{BOUNDARY_TOKEN}_a'
    BOUNDARY_B = f'{BOUNDARY_TOKEN}_b'
    HOLE_A = 'hole_a'
    HOLE_B = 'hole_b'
    TRANSPOSE = 'transpose'

# An object that can be used for obtaining DE9IM relationships.
# Either a polygon or and object that contains a polygon attribute.
SinglePolygonType = Union[shapely.Polygon, shapely.MultiPolygon,
                          Contour]
# An object that can be used for obtaining DE27IM relationships.
# Either a RegionSlice object or a SinglePolygonType.
# if SinglePolygonType is used, then the DE27IM relationships is created by
# padding the DE9IM relationship.
AllPolygonType = Union[RegionSlice, SinglePolygonType]


# %% Relationship Type Definitions
@dataclass
class RelationshipType:
    '''Relationship type definition loaded from relationship_definitions.json.

    Each relationship type contains metadata and pattern-matching definitions
    for identifying spatial relationships between structures using DE-27IM.

    Attributes:
        relation_type: Unique identifier (e.g., 'CONTAINS', 'OVERLAPS')
        label: Human-readable label (e.g., 'Contains', 'Overlaps')
        symbol: Unicode symbol for relationship (e.g., '⊃', '∩')
        color: Hex color code for visualization (e.g., '#FF0000')
        description: Detailed description of relationship
        complementary_relation: Name of complementary relationship or empty string
        implied_relation: List of implied relationship names
        symmetric: Whether relationship is symmetric (A rel B ⇒ B rel A)
        transitive: Whether relationship is transitive (A rel B and B rel C ⇒ A rel C)
        reversed_arrow: Whether this is a complementary stub (True) or primary (False/None)
        pattern: DE-27IM pattern string (29 chars: T/F/* with tabs at pos 9, 19)
        mask: Binary mask as string (e.g., '0b111000111000111000111000111')
        value: Binary value as string (e.g., '0b111000000000111000000000000')
        examples: List of example scenarios
    '''
    relation_type: str
    label: str
    symbol: str
    color: str
    description: str
    complementary_relation: str
    implied_relation: List[str]
    symmetric: bool
    transitive: bool
    reversed_arrow: bool
    pattern: str
    mask: str
    value: str
    examples: List[str]

    def __bool__(self) -> bool:
        '''Return False for UNKNOWN, True for all others.'''
        return self.relation_type != 'UNKNOWN'

    @property
    def is_symmetric(self) -> bool:
        '''Check if the relationship is symmetric.'''
        return self.symmetric

    @property
    def is_transitive(self) -> bool:
        '''Check if the relationship is transitive.'''
        return self.transitive

    @property
    def mask_decimal(self) -> int:
        '''Return the integer value of mask.'''
        if not self.mask:
            return 0
        return int(self.mask, 2)

    @property
    def value_decimal(self) -> int:
        '''Return the integer value of value.'''
        if not self.value:
            return 0
        return int(self.value, 2)

    @property
    def complementary(self) -> Optional['RelationshipType']:
        '''Get the complementary relationship.

        Returns:
            RelationshipType object for complementary relationship, or None
        '''
        if not self.complementary_relation:
            return None
        return RELATIONSHIP_TYPES.get(self.complementary_relation)

    @property
    def implied(self) -> List['RelationshipType']:
        '''Get list of implied relationships.

        Returns:
            List of RelationshipType objects that are implied by this relationship
        '''
        if not self.implied_relation:
            return []

        if isinstance(self.implied_relation, str):
            implied_names = [self.implied_relation]
        else:
            implied_names = list(self.implied_relation)

        result = []
        for rel_name in implied_names:
            rel_type = RELATIONSHIP_TYPES.get(rel_name)
            if rel_type:
                result.append(rel_type)
        return result

    def __str__(self) -> str:
        return f'Relationship: {self.label}'

    def __repr__(self) -> str:
        return f'RelationshipType({self.relation_type})'

    def __eq__(self, other) -> bool:
        '''Compare relationships by relation_type.'''
        if isinstance(other, RelationshipType):
            return self.relation_type == other.relation_type
        elif isinstance(other, str):
            return self.relation_type == other
        return False

    def __hash__(self) -> int:
        '''Hash by relation_type for use in sets and dicts.'''
        return hash(self.relation_type)


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
    relation_type: RelationshipType
    mask: int = 0b000000000000000000000000000
    value: int = 0b000000000000000000000000000

    def __repr__(self) -> str:
        rep_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            '\t',
            f'mask =  0b{self.mask:0>27b}\n',
            '\t',
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

    def __str__(self) -> str:
        test_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            '  Mask:\n',
            int2matrix(self.mask, indent='\t'),
            '  Value\n',
            int2matrix(self.value, indent='\t'),
            '\n'
            ])
        return test_str

    @staticmethod
    def _transpose_segment(bits: str) -> str:
        '''Transpose a 9-bit DE-9IM segment by swapping A/B positions.'''
        if len(bits) != 9:
            raise ValueError(f'Expected 9 bits, got {len(bits)} bits')
        bit_list = list(bits)
        for idx_a, idx_b in ((1, 3), (2, 6), (5, 7)):
            bit_list[idx_a], bit_list[idx_b] = bit_list[idx_b], bit_list[idx_a]
        return ''.join(bit_list)

    @classmethod
    def _transpose_27bit(cls, value: int) -> int:
        '''Transpose each 9-bit DE-9IM segment in a 27-bit integer.'''
        full_bits = f'{value:027b}'
        segments = [full_bits[0:9], full_bits[9:18], full_bits[18:27]]
        transposed = ''.join(cls._transpose_segment(segment)
                             for segment in segments)
        return int(transposed, 2)

    def transpose(self, relation_type: RelationshipType) -> 'RelationshipTest':
        '''Create a transposed test definition for reciprocal relation lookup.

        Args:
            relation_type (RelationshipType): Relationship type for the
                transposed test (usually the complementary relationship).

        Returns:
            RelationshipTest: A new relationship test with transposed mask
                and value binaries.
        '''
        return RelationshipTest(
            relation_type=relation_type,
            mask=self._transpose_27bit(self.mask),
            value=self._transpose_27bit(self.value),
        )


class DE9IM():
    '''The DE-9IM relationship string for two polygons.
    '''
    # A length 9 string of '1's and '0's representing a DE-9IM relationship.
    def __init__(self,
                 poly_a: SinglePolygonType = None,
                 poly_b: SinglePolygonType = None,
                 relation_str: str = None,
                 tolerance=0.0):
        if (poly_a is not None) & (poly_a is not None):
            if not isinstance(poly_a, (shapely.Polygon, shapely.MultiPolygon)):
                poly_a = getattr(poly_a, 'polygon', None)
                if poly_a is None:
                    raise ValueError(''.join([
                        'poly_a must be a shapely Polygon, MultiPolygon, or ',
                        'Contour object.'
                        ]))
            if not isinstance(poly_b, (shapely.Polygon, shapely.MultiPolygon)):
                poly_b = getattr(poly_b, 'polygon', None)
                if poly_b is None:
                    raise ValueError(''.join([
                        'poly_b must be a shapely Polygon, MultiPolygon, or ',
                        'Contour object.'
                        ]))
            if tolerance > 0.0:
                self.relation_str = self.relate_with_margins(poly_a, poly_b, tolerance)
            else:
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
        '''Return the DE-9IM relationship as a boolean string.'''
        relation = self.relation_str.replace('F','0').replace('2','1')
        return relation

    def to_int(self, shift=0):
        '''Convert the DE-9IM relationship to an integer.'''
        shift_factor = 2**shift
        binary_relation = int(self.to_bool(), base=2) * shift_factor
        return binary_relation

    def boundary_adjustment(self, boundary_type: str)->'DE9IM':
        '''Adjust the DE-9IM relationship matrix of a boundary slice.

        The “*Interior*” bits of the DE-9IM relationship metric become the
        “*Boundary*” bits of the new relationship metric.
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

    @staticmethod
    def transpose_a(relation_str)->str:
        '''Transpose the DE-9IM relationship matrix along the horizontal axis.
        '''
        # Select every third character from the string.
        exteriors = relation_str[0:3]
        boundaries = relation_str[3:6]
        interiors = relation_str[6:9]
        new_str_list = interiors + boundaries + exteriors
        new_str = ''.join(new_str_list)
        return new_str

    @staticmethod
    def transpose_b(relation_str)->str:
        '''Transpose the DE-9IM relationship matrix along the vertical axis.
        '''
        # Select every third character from the string.
        interiors = relation_str[2::-1]
        boundaries = relation_str[5:2:-1]
        exteriors = relation_str[8:5:-1]
        new_str_list = interiors + boundaries + exteriors
        new_str = ''.join(new_str_list)
        return new_str

    def relate_with_margins(self, poly_a, poly_b, margin: float)->str:
        '''Get the DE-9IM relationship with margins applied to the polygons.

        Args:
            margin (float): The margin to apply to the polygons.
        Returns:
            DE9IM: The DE-9IM relationship with margins applied.
        '''
        buffered_polygon_a = ContourMargin(poly_a, margin)
        buffered_polygon_b = ContourMargin(poly_b, margin)
        r_matrix = [
            shapely.relate(buffered_polygon_a.true_interior,
                           buffered_polygon_b.true_interior)[0],
            shapely.relate(buffered_polygon_a.true_interior,
                           buffered_polygon_b.boundary)[0],
            self.transpose_b(shapely.relate(buffered_polygon_a.true_interior,
                                            buffered_polygon_b.full_interior))[0],
            shapely.relate(buffered_polygon_a.boundary,
                           buffered_polygon_b.true_interior)[0],
            shapely.relate(buffered_polygon_a.boundary,
                           buffered_polygon_b.boundary)[0],
            self.transpose_b(shapely.relate(buffered_polygon_a.boundary,
                                            buffered_polygon_b.full_interior))[0],
            self.transpose_a(shapely.relate(buffered_polygon_a.full_interior,
                                            buffered_polygon_b.true_interior))[0],
            self.transpose_a(shapely.relate(buffered_polygon_a.full_interior,
                                            buffered_polygon_b.boundary))[0],
            '2'  # Exteriors always intersect
            ]
        relation_str = ''.join(r_matrix)
        return relation_str

    def test_relation(self, mask: int, value: int)->bool:
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
        '''Merge multiple DE-9IM relationships into a single DE-9IM relationship.

        Merging is done by performing a bitwise OR operation on the integer
        representations of the DE-9IM relationships.

        Args:
            relations (List[DE9IM]): A list of DE-9IM relationships to merge
        '''
        def to_str(relation_int: int)->str:
            size=9
            str_size = size + 2  # Accounts for '0b' prefix.
            bin_str = bin(relation_int)
            if len(bin_str) < str_size:
                zero_pad = str_size - len(bin_str)
                bin_str = '0' * zero_pad + bin_str[2:]
            elif len(bin_str) > str_size:
                raise ValueError(''.join([
                    'The input integer must be {size} bits long. ',
                    'The input integer was: ',
                    f'{len(bin_str) - 2}'
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
    '''Three DE-9IM relationship strings concatenated.

    The DE-27IM relationship is derived from three DE-9IM relationships.
    It provides a more comprehensive relationship between two polygons by including
    the relationships between the polygons, their exteriors, and their convex
    hulls.  For example a polygon with a hole can be compared to another polygon
    residing within the hole. A DE-9IM relationship would classify the two
    polygons as *disjoint*, but a DE-27IM relationship would capture the fact
    that one polygon is contained within the other.

    The DE-27IM relationship string is composed of three DE-9IM relationship
    strings concatenated. The left-most 9-bit string represents the DE-9IM
    relationship between the two polygons.  The middle 9-bit string represents
    the DE-9IM relationship between the second polygon and the *exterior* of
    the first polygon. The right-most 9-bit string represents the DE-9IM
    relationship between the second polygon and the *convex hull* of the first
    polygon.

    A number of these relationships also have a reciprocal relationship, e.g.
    *Contains* has a reciprocal relationship with *B is fully within A*.
    However, by requiring that the primary polygon (**A**) is larger than the
    secondary polygon (**B**), we can ensure that the relationship is properly
    defined without the need to include the reciprocal relationships.

    This size requirement is not explicitly checked in the DE-27IM class because
    DE-27IM relationships are usually obtained in the context of a 3D volume
    and it is possible for individual contours from the two volumes to have the
    opposite size difference to their respective volumes.

    One of the value of the binary DE-27IM relationship is that the individual
    relationships between contours that define two 3D volumes can be merged
    using a logical OR to obtain the relationship between the 3D volumes.

    Args:
        region_a (AllPolygonType): The first polygon structure.
        region_b (AllPolygonType): The second polygon structure.
        relation_str (str): A DE-27IM relationship string.
        relation_int (int): An integer representation of the DE-27IM
            relationship.
        adjustments (List[str]): A list of strings that define the adjustments
            to be applied when region_a and region_b are not RegionSlices.
            Possible adjustments are:
            - 'transpose': Transpose the relationship matrix.
            - 'boundary_a': Adjust the relationship matrix for the boundary
                slice of region_a.
            - 'boundary_b': Adjust the relationship matrix for the boundary
                slice of region_b.
            - 'hole_a': Adjust the relationship matrix for the hole (negative
                space) of region_a.
            - 'hole_b': Adjust the relationship matrix for the hole (negative
                space) of region_b.
    Raises:
        ValueError: If the relationship cannot be determined.
        ValueError: If neither contours nor a relationship string is supplied.

    Attributes:
        relation (str): The DE-27IM relationship string.
        int (int): The integer representation of the DE-27IM relationship.
        is_null (bool): A boolean property that indicates if the relationship
            is null.

    Class Attributes:
        padding (DE9IM): A DE-9IM null object.
        exterior_a (DE9IM): The DE-9IM relationship to use when only the
            A contour is supplied.
        exterior_b (DE9IM): The DE-9IM relationship to use when only the
            B contour is supplied.
        test_binaries (List[RelationshipTest]): The RelationshipTest objects
            used to identify named relationships.

    Methods:
        to_str(relation_int: int)->str: A static method that converts a 27-bit
            integer into a string.
        to_int(relation_str: str)->int: A static method that converts a 27-bit
            string into an integer.
        relate_contours(structure_a: AllPolygonType, structure_b: AllPolygonType,
                        adjustments: List[str] = None): A method that calculates
            the 27-bit relationship for two structures on a given slice.

    '''
    # These will be populated when relationships are loaded from JSON
    # A dictionary of relationship type definitions loaded from JSON,
    # keyed by relation_type.
    relationship_definitions: Dict[str, RelationshipType] = {}
    # Relationship Test Definitions - loaded from JSON
    test_binaries: List[RelationshipTest] = []
    # padding is a DE9IM object built from a string of 'FFFFFFFFF', which
    # becomes 9 zeros when converted to binary.  Padding is used in cases where
    # Exterior and Hull relationships are not relevant.
    padding = DE9IM(relation_str='FFFFFFFFF')  # 'F' * 9
    # If only the A contour is supplied, then A is exterior to B
    exterior_a = DE9IM(relation_str='FF1FF1FF1')  # 'FF1' * 3
    # If only the B contour is supplied, then B is exterior to A
    exterior_b = DE9IM(relation_str='FFFFFF111')  # 'F'*3 + 'F'*3 + '1'*3

    def __init__(self, region_a: AllPolygonType = None,
                 region_b: AllPolygonType = None,
                 relation_str: str = None,
                 relation_int: int = None,
                 adjustments: List[str] = None,
                 tolerance=0.0):
        if not empty_structure(region_a):
            if not empty_structure(region_b):
                # If both contours are supplied, the relationship is calculated.
                # Initialize the relationship with all padding.
                relation_group = tuple([self.padding] * 3)
                self.relation = self.combine_groups(relation_group)
                # calculate the 27 bit relationships and merge them with the
                # initial relationship.
                self.relate_contours(region_a, region_b, adjustments, tolerance)
                self.int = self.to_int(self.relation)
            else:
                # If only the A contour is supplied, the relationship is
                #   A is exterior to B
                relation_group = tuple([self.exterior_a] * 3)
                self.relation = self.combine_groups(relation_group, adjustments)
                self.int = self.to_int(self.relation)
        elif region_b is not None:
            # If only the B contour is supplied, the relationship is
            #   B is exterior to A
            relation_group = tuple([self.exterior_b] * 3)
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
            # If neither contours nor a relationship string is supplied, the
            # relationship is set to null.
            self.int = 0
            # Initialize the relationship with all padding.
            relation_group = tuple([self.padding] * 3)
            self.relation = self.combine_groups(relation_group)

    @classmethod
    def _initialize_relationships(cls):
        '''Initialize relationship types and tests from JSON definitions.

        This function is called automatically on module import.
        Future work includes generating complementary relationship tests from
        primary relationship tests, using the RelationshipTest.transpose method,
        (which currently is not implemented).
        '''
        def _load_relationship_definitions() -> List[Dict]:
            '''Load relationship definitions from JSON file.

            Returns:
                List of relationship definitions from JSON

            Raises:
                ImportError: If relationship_definitions.json is not found or
                    cannot be loaded
            '''
            json_path = Path(__file__).parent / 'relationship_definitions.json'
            if not json_path.exists():
                raise ImportError(
                    f'relationship_definitions.json not found at {json_path}. '
                    'This file is required for the relations module to function.'
                )
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('Relationships', [])
            except (json.JSONDecodeError, KeyError) as err:
                raise ImportError(
                    f'Error loading relationship_definitions.json: {err}'
                ) from err

        cls.relationship_definitions.clear()
        cls.test_binaries.clear()
        definitions = _load_relationship_definitions()
        # Create RelationshipType objects for all relationships
        for defn in definitions:
            rel_type = RelationshipType(
                relation_type=defn['relation_type'],
                label=defn['label'],
                symbol=defn['symbol'],
                color=defn.get('color', '#000000'),
                description=defn.get('description', ''),
                complementary_relation=defn.get('complementary_relation', ''),
                implied_relation=defn.get('implied_relation', []),
                symmetric=defn.get('symmetric', False),
                transitive=defn.get('transitive', False),
                reversed_arrow=defn.get('reversed_arrow', False),
                pattern=defn.get('pattern', ''),
                mask=defn.get('mask', ''),
                value=defn.get('value', ''),
                examples=defn.get('examples', [])
            )
            cls.relationship_definitions[rel_type.relation_type] = rel_type
            # Create RelationshipTest for relationships with patterns
            if rel_type.mask and rel_type.value and not rel_type.reversed_arrow:
                cls.test_binaries.append(
                    RelationshipTest(
                        relation_type=rel_type,
                        mask=rel_type.mask_decimal,
                        value=rel_type.value_decimal
                    )
                )

        primary_tests_by_name = {
            test.relation_type.relation_type: test
            for test in cls.test_binaries
        }
        for rel_type in cls.relationship_definitions.values():
            if not rel_type.reversed_arrow:
                continue
            complementary_name = rel_type.complementary_relation
            primary_test = primary_tests_by_name.get(complementary_name)
            if primary_test is None:
                continue
            cls.test_binaries.append(primary_test.transpose(rel_type))


    @property
    def is_null(self)->bool:
        '''Check if the relationship is null.
        '''
        return self.int == 0

    @staticmethod
    def to_str(relation_int: int)->str:
        '''Convert the 27 bit binary integer into a formatted string.'''
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
        '''Convert the 27 bit binary string into an integer.'''
        value_str = relation_str.replace('F','0').replace('2','1')
        try:
            relation_int = int(value_str, base=2)
        except ValueError as err:
            raise ValueError(''.join([
                'The input string must be a 27 bit binary string. The input ',
                'string was: ', relation_str
                ])) from err
        return relation_int

    def relate_contours(self, region_a: AllPolygonType,
                        region_b: AllPolygonType,
                        adjustments: List[str] = None,
                        tolerance=0.0):
        '''Get the 27 bit relationship for two structures on a given slice.
        The supplied adjustments are only applied if the structures are not
        RegionSlices.  If the structures are RegionSlices, then the adjustments
        are determined from the RegionSlice objects.
        Possible adjustments are:
            'transpose': Transpose the relationship matrix.
            'boundary_a': Adjust the relationship matrix for the boundary slice
                of structure_a.
            'boundary_b': Adjust the relationship matrix for the boundary slice
                of structure_b.
            'hole_a': Adjust the relationship matrix for the hole (negative space)
                of structure_a.
            'hole_b': Adjust the relationship matrix for the hole (negative space)
                of structure_b.
        '''
        logger = logging.getLogger(__name__)
        logger.debug(
            'relate_contours start schema=%s type_a=%s type_b=%s tolerance=%s',
            RELATION_SCHEMA_VERSION,
            type(region_a).__name__,
            type(region_b).__name__,
            tolerance,
        )
        # If structure_a and structure_b are both RegionSlices, then get the
        # full 27 bit relationship.
        if isinstance(region_a, RegionSlice):
            if isinstance(region_b, RegionSlice):
                # All regions in region_a are compared with all regions and
                # boundaries in region_b.
                # All boundaries in region_a are compared with all regions
                # and boundaries in region_b.
                # All of these comparisons are merged into a single DE27IM object.
                #
                # 1. Merge the MultiPolygon in RegionSlice.regions of region_a:
                #    a. Get a 27 bit relation with the combined MultiPolygons in
                #        RegionSlice.regions and in RegionSlice.boundaries of region_b.
                #          - MultiPolygon vs other MultiPolygon
                #          - RegionSlice.exterior vs other MultiPolygon
                #          - RegionSlice.hull vs other MultiPolygon
                #          *Note: External and hull comparisons are not required for boundaries*
                #    b. Apply appropriate corrections for holes and boundaries.
                # 2. for each MultiPolygon in RegionSlice.boundaries of region_a:
                #    a. Get a 27 bit relation with all MultiPolygons in
                #       RegionSlice.regions and in RegionSlice.boundaries of region_b
                #       that are on the same slice.
                #           - Boundary MultiPolygon vs other MultiPolygon
                #           *Note: External and hull comparisons are not required for boundaries*
                #    b. Apply appropriate corrections for holes and boundaries.
                # 3. Combine all relations with OR.
                adjustments = []
                if region_a.has_regions():
                    # Get the 27 bit relationship for the combined regions in
                    # region_a with all regions and boundaries in region_b.
                    poly_a = region_a.merged_region
                    exterior = region_a.merged_exterior
                    hull = region_a.merged_hull
                    if region_b.has_regions():
                        poly_b = region_b.merged_region
                        exterior_b = region_b.merged_exterior
                        hull_b = region_b.merged_hull
                        self.relate_poly(
                            poly_a,
                            poly_b,
                            external_polygon_a=exterior,
                            hull_polygon_a=hull,
                            external_polygon_b=exterior_b,
                            hull_polygon_b=hull_b,
                            adjustments=adjustments,
                            tolerance=tolerance,
                        )
                    if region_b.has_boundaries():
                        adjustments.append(AdjustmentType.BOUNDARY_B)
                        boundary_b = region_b.merged_boundary
                        self.relate_poly(
                            poly_a,
                            boundary_b,
                            adjustments=adjustments,
                            tolerance=tolerance,
                        )
                adjustments = []
                if region_a.has_boundaries():
                    # Get the 27 bit relationship for the combined boundaries in
                    # region_a with all regions and boundaries in region_b.
                    adjustments.append(AdjustmentType.BOUNDARY_A)
                    boundary_a = region_a.merged_boundary
                    if region_b.has_regions():
                        poly_b = region_b.merged_region
                        self.relate_poly(
                            boundary_a,
                            poly_b,
                            adjustments=adjustments,
                            tolerance=tolerance,
                        )
                    if region_b.has_boundaries():
                        # Add the boundary_b adjustment to the existing
                        # 'boundary_a' adjustment.
                        adjustments.append(AdjustmentType.BOUNDARY_B)
                        boundary_b = region_b.merged_boundary
                        self.relate_poly(
                            boundary_a,
                            boundary_b,
                            adjustments=adjustments,
                            tolerance=tolerance,
                        )
            else:
                raise ValueError(''.join([
                    'Region_a and region_b must both be RegionSlice, ',
                    'Contour, shapely Polygon, or shapely MultiPolygon objects. ',
                    f'Region_a input was: {str(type(region_a))}\t',
                    f'Region_b input was: {str(type(region_b))}'
                    ]))
        else:
            # If contour_a and contour_b are Contour objects, shapely Polygons
            # or MultiPolygons then get the 9 bit DE9IM relationships for the
            # contour, external and hull.
            # Note: If contour_a is a Contour object, the external will
            # match the contour relation, because a Contour polygon does not
            # contain holes.
            if isinstance(region_a, Contour):
                poly_a = region_a.polygon
                external_polygon = make_solid(region_a.polygon)
                hull_polygon = region_a.hull
            elif isinstance(region_a, (shapely.Polygon,
                                       shapely.MultiPolygon)):
                poly_a = region_a
                external_polygon = make_solid(region_a)
                hull_polygon = region_a.convex_hull
            else:
                raise ValueError(''.join([
                    'Region_a and region_b must both be RegionSlice, ',
                    'Contour, shapely Polygon, or shapely MultiPolygon objects. ',
                    f'Region_a input was: {str(type(region_a))}\t',
                    f'Region_b input was: {str(type(region_b))}'
                    ]))
            if isinstance(region_b, Contour):
                poly_b = region_b.polygon
                external_polygon_b = make_solid(region_b.polygon)
                hull_polygon_b = region_b.hull
            elif isinstance(region_b, (shapely.Polygon,
                                          shapely.MultiPolygon)):
                poly_b = region_b
                external_polygon_b = make_solid(region_b)
                hull_polygon_b = region_b.convex_hull
            else:
                raise ValueError(''.join([
                    'Region_a and region_b must both be RegionSlice, ',
                    'Contour, shapely Polygon, or shapely MultiPolygon objects. ',
                    f'Region_a input was: {str(type(region_a))}\t',
                    f'Region_b input was: {str(type(region_b))}'
                    ]))
            self.relate_poly(
                poly_a,
                poly_b,
                external_polygon_a=external_polygon,
                hull_polygon_a=hull_polygon,
                external_polygon_b=external_polygon_b,
                hull_polygon_b=hull_polygon_b,
                adjustments=adjustments,
                tolerance=tolerance,
            )

    def relate_poly(self, poly_a: PolygonType, poly_b: PolygonType,
                    external_polygon_a: PolygonType = None,
                    hull_polygon_a: PolygonType = None,
                    external_polygon_b: PolygonType = None,
                    hull_polygon_b: PolygonType = None,
                    adjustments: List[str] = None,
                    tolerance=0.0):
        '''Calculate the DE-27IM relationship for two polygons.

          The DE-27IM relationship is calculated by creating three DE-9IM
          relationships:
            1. The DE-9IM relationship between the two polygons (contour).
            2. The DE-9IM relationship between the *exterior* of the first
                polygon (external) and the *exterior* of the second polygon.
            3. The DE-9IM relationship between the *convex hull* of the first
                polygon (convex_hull) and the *convex hull* of the second
                polygon.

        Args:
            poly_a (PolygonType): The first polygon.
            poly_b (PolygonType): The second polygon.
            external_polygon_a (PolygonType): The external polygon of the first
                polygon. If not supplied, self.padding is used.
            hull_polygon_a (PolygonType): The convex hull polygon of the first
                polygon. If not supplied, self.padding is used.
            external_polygon_b (PolygonType): The external polygon of the
                second polygon. If not supplied, poly_b is used.
            hull_polygon_b (PolygonType): The convex hull polygon of the
                second polygon. If not supplied, poly_b is used.
            adjustments (List[str]): A list of strings that define the
                   adjustments to be applied to the DE-9IM relationships before
                   combining them into a DE-27IM relationship string.
                   Possible adjustments are:
                    - 'boundary_a': Apply boundary adjustments to the first
                       DE-9IM.
                    - 'boundary_b': Apply boundary adjustments to the second
                          DE-9IM.
                    - 'hole_a': Apply hole adjustments to the first DE-9IM.
                    - 'hole_b': Apply hole adjustments to the second DE-9IM.
                    - 'transpose': Apply transpose adjustments to both DE-9IMs.
        '''
        contour = DE9IM(poly_a, poly_b, tolerance=tolerance)
        # If external polygons are supplied, create a DE9IM relationship for
        # the exterior band.
        if external_polygon_a is None:
            external = self.padding
        else:
            poly_b_exterior = (
                poly_b if external_polygon_b is None else external_polygon_b
            )
            external = DE9IM(external_polygon_a, poly_b_exterior,
                             tolerance=tolerance)
        # If hull polygons are supplied, create a DE9IM relationship for the
        # hull band.
        if hull_polygon_a is None:
            convex_hull = self.padding
        else:
            poly_b_hull = poly_b if hull_polygon_b is None else hull_polygon_b
            convex_hull = DE9IM(hull_polygon_a, poly_b_hull,
                                tolerance=tolerance)
        # Create a tuple of the three DE-9IM relationships.
        # The order is important: (contour, external, convex_hull).
        relation_group = (contour, external, convex_hull)
        # If adjustments are supplied, apply them to the relationship.
        relation_str = self.combine_groups(relation_group, adjustments)
        # Merge the relationship into the DE27IM object.
        self.merge(self.to_int(relation_str))

    def apply_adjustments(
        self,
        relation_group: Tuple[DE9IM, DE9IM, DE9IM],
        adjustments: List[Union[str, AdjustmentType]],
    ) -> tuple[DE9IM, DE9IM, DE9IM]:
        '''Apply adjustments to the DE-9IM relationship matrix.

        Three types of adjustments can be applied to the DE-9IM relationship
        matrix in the following order:
            1. Boundary Adjustments: Adjust the relationship matrix when one of
               the contours is a boundary. Because the contour is a boundary,
               the “*Interior*” bits of the DE-9IM relationship metric become
               the “*Boundary*” bits of the new relationship metric.

        2. Hole Adjustments: Adjust the relationship matrix when one of
               the contours is a hole.  Because the contour is a hole, the
               “*Interior*” bits of the DE-9IM relationship metric become the
               “*Exterior*” bits and the new “*Interior*” bits become 'F'.

        3. Transpose Adjustment: Transpose the relationship matrix. This
           converts a relationship between polygon A and polygon B into a
           relationship between polygon B and polygon A. This is useful because
           the order of the polygons is important.

        Note: The order of the adjustments is important.
        When hole adjustments are applied, only the "contour" bits are relevant,
        the external and hull bits are set to 'FFFFFFFFF'.
        '''
        normalized = {AdjustmentType(item) if isinstance(item, str) else item
                      for item in adjustments}
        # Apply Boundary Adjustments
        if AdjustmentType.BOUNDARY_A in normalized:
            relation_group = tuple(de9im.boundary_adjustment('a')
                                   for de9im in relation_group)
        if AdjustmentType.BOUNDARY_B in normalized:
            relation_group = tuple(de9im.boundary_adjustment('b')
                                   for de9im in relation_group)
        # Apply Hole Adjustments
        if AdjustmentType.HOLE_A in normalized:
            contour, external, convex_hull = relation_group
            contour = contour.hole_adjustment('a')
            external = self.padding
            convex_hull = self.padding
            relation_group = (contour, external, convex_hull)
            self.to_int(self.relation)
        if AdjustmentType.HOLE_B in normalized:
            contour, external, convex_hull = relation_group
            contour = contour.hole_adjustment('b')
            external = self.padding
            convex_hull = self.padding
            relation_group = (contour, external, convex_hull)
        # Apply Transpose Adjustment
        if AdjustmentType.TRANSPOSE in normalized:
            relation_group = tuple(de9im.transpose()
                                   for de9im in relation_group)
        return relation_group

    def combine_groups(self, relation_group: Tuple[DE9IM, DE9IM, DE9IM],
                       adjustments: List[str] = None)-> str:
        '''Combine the DE-9IM relationships into a DE-27IM relationship string.

        The three DE-9IM relationships (contour, external, and convex_hull) are
        combined into a DE-27IM relationship string. If adjustments are
        supplied, they are applied to the DE-9IM relationships before combining
        them into a DE-27IM relationship string.

        Args:
            relation_group (Tuple[DE9IM, DE9IM, DE9IM]): A tuple of the three
                DE-9IM relationships in the following order:
                    (contour, external, convex hull).
            adjustments (List[str]): A list of strings that define the
                adjustments to be applied to the DE-9IM relationships before
                combining them. Possible adjustments are:
                - 'boundary_a': Apply boundary adjustments to the first DE-9IM.
                - 'boundary_b': Apply boundary adjustments to the second DE-9IM.
                - 'hole_a': Apply hole adjustments to the first DE-9IM.
                - 'hole_b': Apply hole adjustments to the second DE-9IM.
                - 'transpose': Apply transpose adjustments to both DE-9IMs.
        Returns:
            str: A DE-27IM relationship string.
        '''
        if adjustments:
            relation_group = self.apply_adjustments(relation_group, adjustments)
        # Convert the DE-9IM relationships into a DE-27IM relationship string.
        full_relation = ''.join(de9im.relation for de9im in relation_group)
        relation_str = full_relation.replace('F','0').replace('2','1')
        return relation_str

    def merge(self, other: Union['DE27IM', int]):
        '''Combine two DE27IM relationships.
        The other relationship can be either a DE27IM object or an integer.
        The relationship is merged by performing a bitwise OR operation on the
        integer representations of the relationships.
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
        # Use the self.to_int() rather than self.int, because self.int may not
        # be up to date.
        merged_rel = self.to_int(self.relation) | other_rel
        self.int = merged_rel
        self.relation = self.to_str(merged_rel)

    def identify_relation(self) -> RelationshipType:
        '''Applies a collection of definitions for named relationships to a supplied
        relationship binary.

        Returns:
            RelationshipType: The identified RelationshipType if one of the tests
                passes, otherwise RelationshipType.UNKNOWN.
        '''
        relation_binary = self.int
        matches: Dict[str, RelationshipType] = {}
        for rel_def in self.test_binaries:
            result = rel_def.test(relation_binary)
            if result:
                matches[result.relation_type] = result
        unique_matches = list(matches.values())
        if len(unique_matches) == 1:
            return unique_matches[0]
        if len(unique_matches) > 1:
            match_names = ', '.join(
                match.relation_type for match in unique_matches
            )
            raise AmbiguousRelationshipError(
                f'Multiple relationships matched: {match_names}'
            )
        # Return UNKNOWN relationship type
        return self.relationship_definitions.get('UNKNOWN', None)

    def __eq__(self, value):
        if isinstance(value, self.__class__):
            return self.int() == value.int()
        if isinstance(value, int):
            return self.int() == value
        if isinstance(value, str):
            value_str = value.replace('F','0').replace('2','1')
            return self.int() == self.to_int(value_str)

    def __str__(self):
        return int2matrix(self.int)

    def __repr__(self):
        return f'<DE27IM>: {self.relation}'


# === Module-level Registry ===
# Create module-level constants for backward compatibility
# These allow: from relations import CONTAINS, OVERLAPS, etc.
#for _name, _rel_type in RELATIONSHIP_TYPES.items():
#    globals()[_name] = _rel_type

# === Module Initialization ===
# Initialize relationship types and tests on module import
DE27IM._initialize_relationships()

# Make RELATIONSHIP_TESTS available at module level
RELATIONSHIP_TYPES = DE27IM.relationship_definitions
