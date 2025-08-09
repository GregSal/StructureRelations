'''Relationships between structures.
'''
# %% Imports
# Type imports
from typing import List, LiteralString, Tuple, Union

# Standard Libraries
from enum import Enum, auto
from dataclasses import dataclass

# Shared Packages
import shapely

# Local packages
from contours import Contour
from region_slice import RegionSlice, empty_structure
from types_and_classes import PolygonType
from utilities import make_multi, make_solid

# An object that can be used for obtaining DE9IM relationships.
# Either a polygon or and object that contains a polygon attribute.
SinglePolygonType = Union[shapely.Polygon, shapely.MultiPolygon,
                          Contour]
# An object that can be used for obtaining DE27IM relationships.
# Either a RegionSlice object or a SinglePolygonType.
# if SinglePolygonType is used, then the DE27IM relationships is created by
# padding the DE9IM relationship.
AllPolygonType = Union[RegionSlice, SinglePolygonType]

# Global Settings
PRECISION = 3

# %% Relationship display functions
def int2str(relation_int: int, length=27)->str:
    '''Convert a 9 or 27 bit binary integer into a formatted string.

    Converts a 9 or 27 bit binary integer into a formatted string. The string
    is formatted as a binary number with leading zeros to make the string the
    specified length.

    Args:
        relation_int (int): The integer representation of the 9 or 27 bit
            relationship.
        length (int, optional): The expected length of the string.
            (Generally should be 9 or 27.) Defaults to 27.

    Raises:
        ValueError: If the input integer is longer than the specified length.
    Returns:
        str: The integer converted into a zero-padded binary integer.
    '''
    str_len = length + 2  # Accounts for '0b' prefix.
    bin_str = bin(relation_int)
    if len(bin_str) < str_len:
        zero_pad = str_len - len(bin_str)
        bin_str = '0' * zero_pad + bin_str[2:]
    elif len(bin_str) > str_len:
        raise ValueError(''.join([
            f'The input integer must be {length} bits long. The input integer ',
            'was: ', str(relation_int)
            ]))
    else:
        bin_str = bin_str[2:]
    return bin_str


def int2matrix(relation_int: int, indent: str = '') -> str:
    '''Convert a 27 bit binary integer into a formatted matrix.

    The display matrix is formatted as follows:
        |001|	|111|	|111|
        |001|	|001|	|001|
        |111|	|001|	|001|

    Args:
        relation_int (int): The integer representation of the 27 bit
            relationship.
        indent (str, optional): The string to prefix each row of the 3-line
            matrix display. Usually this will be a sequence of spaces to indent
            the display text.  Defaults to ''.

    Returns:
        str: A multi-line string displaying the 27 formatted bit relationship
            matrix.
    '''
    bin_str = int2str(relation_int, length=27)
    # This is the template for one row of the matrix.
    # *bin#* is replaced with the binary string for the row.
    # *#* is replaced with the row and matrix index.
    bin_fmt = '|{bin#}|_'
    bin_list = []
    # Generate the 3-line matrix template.
    for row_num in range(3):
        # The template for the 3-line matrix
        for matrix_num in range(3):
            # index represents where the 3-bit sequence should be placed in the
            # formatted string.  The first row of the string is the first row of
            # each matrix. The second row is the second row of each matrix.
            index = row_num * 3 + matrix_num
            bin_text = bin_fmt.replace('#', str(index))
            if matrix_num == 0:
                # The first matrix has an indent before the binary string and a
                # tab after the binary string.
                bin_text = indent + bin_text.replace('_', '\t')
            elif matrix_num == 1:
                # The second matrix has a tab after the binary string.
                bin_text = bin_text.replace('_', '\t')
            elif matrix_num == 2:
                # The third matrix has a newline after the binary string.
                bin_text = bin_text.replace('_', '\n')
            bin_list.append(bin_text)
    bin_template = ''.join(bin_list)
    # Split the 27 bit binary string into 9 3-bit sections; rows in the 3
    # matrices.
    bin_dict = {}
    for idx in range(9):
        # Calculate the row and column (matrix) number for the current 3-bit
        # sequence
        row_num = idx % 3  # The row number in the 3-line matrix
        matrix_num = idx // 3  # The matrix number
        # index represents where the 3-bit sequence should be placed in the
        # formatted string.  The first row of the string is the first row of
        # each matrix (every third sequence in the binary string). The second
        # row is the second row of each matrix (every third sequence plus 1).
        index = row_num * 3 + matrix_num
        # Add the 3-bit sequence to the dictionary so that it van be inserted
        # into the template at the appropriate spot.
        bin_dict[f'bin{index}'] = bin_str[idx*3:(idx+1)*3]
    return bin_template.format(**bin_dict)


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

    def __str__(self) -> str:
        test_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            '  Mask:\n',
            int2matrix(self.mask, indent=' ' * 4),
            '  Value\n',
            int2matrix(self.value, indent=' ' * 4),
            '\n'
            ])
        return test_str


class DE9IM():
    '''The DE-9IM relationship string for two polygons.
    '''
    # A length 9 string of '1's and '0's representing a DE-9IM relationship.
    def __init__(self,
                 poly_a: SinglePolygonType = None,
                 poly_b: SinglePolygonType = None,
                 relation_str: str = None):
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

    From this 27-bit string, a number of physical relationship can be derived.
    The relationships are defined as follows:

        Relationship  Region Test  Exterior Test  Hull Test   Description
        ----------------------------------------------------  -----------
        Disjoint      FF*FF****    F***F****      F***F****   No there is no overlap between ***A*** and ***B***.
        Shelters      FF*FF****    F***F****      T***F****   The Convex Hull of ***A*** contains ***B***.
        Surrounds     FF*FF****    T***F****      *********   The Exterior of ***A*** contains ***B***.
        Borders       F***T****    F***T****      *********   Part of the *exterior* boundary of ***A***
                                                              touches the exterior boundary of ***B***.
        Confines      F***T****    T***T****      *********   Part of the *interior* boundary of ***A***
                                                              touches the exterior boundary of ***B***.
        Partitions    T*T*T*F**    T*T*T*F**      T*T***F**   ***A*** contains ***B*** and
                                                              part of the *boundary* of ***B*** touches
                                                              part of the *boundary* of ***A***.
        Contains      TF*FF****    T***F****      T********   ***B*** is fully within ***A***.
        Overlaps      T*T*T*T**    T*T*T*T**      T*T***T**   ***A*** and ***B*** intersect, but neither contains the other.
        Equals        T*F*T****    T***T****      T********   ***A*** and ***B*** enclose the identical space.

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


    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.DISJOINT,
                         0b110110000000010100000010100,
                         0b000000000000000100000000100),
        RelationshipTest(RelationshipType.SHELTERS,
                         0b110110000000010100101010100,
                         0b000000000000000100101000000),
        RelationshipTest(RelationshipType.SURROUNDS,
                         0b110110000101010100000000000,
                         0b000000000101000000000000000),
        RelationshipTest(RelationshipType.BORDERS,
                         0b100010000000000100000000000,
                         0b000010000000000100000000000),
        RelationshipTest(RelationshipType.CONFINES,
                         0b100010000101000100000000000,
                         0b000010000101000000000000000),
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
                 adjustments: List[str] = None):
        if not empty_structure(region_a):
            if not empty_structure(region_b):
                # If both contours are supplied, the relationship is calculated.
                # Initialize the relationship with all padding.
                relation_group = tuple([self.padding] * 3)
                self.relation = self.combine_groups(relation_group)
                # calculate the 27 bit relationships and merge them with the
                # initial relationship.
                self.relate_contours(region_a, region_b, adjustments)
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
        try:
            relation_int = int(relation_str, base=2)
        except ValueError as err:
            raise ValueError(''.join([
                'The input string must be a 27 bit binary string. The input ',
                'string was: ', relation_str
                ])) from err
        return relation_int

    def relate_contours(self, region_a: AllPolygonType,
                        region_b: AllPolygonType,
                        adjustments: List[str] = None):
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
                # 1. for each MultiPolygon in RegionSlice.regions of region_a:
                #    a. Get a 27 bit relation with all MultiPolygons in
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
                # 3. Combine all relations with OR
                exteriors = region_a.exterior
                hulls = region_a.hull
                adjustments = []
                if region_a.has_regions():
                    for region_index, poly_a in region_a.regions.items():
                        # Get the 27 bit relationship for each region in region_a
                        # with all regions and boundaries in region_b.
                        if region_b.has_regions():
                            for poly_b in region_b.regions.values():
                                self.relate_poly(poly_a, poly_b,
                                                exteriors[region_index],
                                                hulls[region_index], adjustments)
                        if region_b.has_boundaries():
                            adjustments.append('boundary_b')
                            for boundary_b in region_b.boundaries.values():
                                self.relate_poly(poly_a, boundary_b,
                                                 adjustments=adjustments)
                adjustments = []
                if region_a.has_boundaries():
                    # Get the 27 bit relationship for each boundary in region_a
                    # with all regions and boundaries in region_b.
                    adjustments.append('boundary_a')
                    for boundary_a in region_a.boundaries.values():
                        # Get the 27 bit relationship for each boundary in region_a
                        # with all regions and boundaries in region_b.
                        if region_b.has_regions():
                            for poly_b in region_b.regions.values():
                                self.relate_poly(boundary_a, poly_b,
                                                 adjustments=adjustments)
                        if region_b.has_boundaries():
                            # If region_b has boundaries, then get the 27 bit
                            # relationship for each boundary in region_a with
                            # all boundaries in region_b.
                            # Add the boundary_b adjustment to the existing
                            # 'boundary_a' adjustment.
                            adjustments.append('boundary_b')
                            for boundary_b in region_b.boundaries.values():
                                # G``et the 27 bit relationship for each boundary
                                # in region_a with all boundaries in region_b.
                                self.relate_poly(boundary_a, boundary_b,
                                                 adjustments=adjustments)
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
                poly_a = make_multi(region_a.polygon)
                external_polygon = make_solid(poly_a.polygon)
                hull_polygon = region_a.hull
            elif isinstance(region_a, (shapely.Polygon,
                                          shapely.MultiPolygon)):
                poly_a = make_multi(region_a)
                external_polygon = make_solid(poly_a)
                hull_polygon = poly_a.convex_hull
            else:
                raise ValueError(''.join([
                    'Region_a and region_b must both be RegionSlice, ',
                    'Contour, shapely Polygon, or shapely MultiPolygon objects. ',
                    f'Region_a input was: {str(type(region_a))}\t',
                    f'Region_b input was: {str(type(region_b))}'
                    ]))
            if isinstance(region_b, Contour):
                poly_b = region_b.polygon
            elif isinstance(region_b, (shapely.Polygon,
                                          shapely.MultiPolygon)):
                poly_b = region_b
            else:
                raise ValueError(''.join([
                    'Region_a and region_b must both be RegionSlice, ',
                    'Contour, shapely Polygon, or shapely MultiPolygon objects. ',
                    f'Region_a input was: {str(type(region_a))}\t',
                    f'Region_b input was: {str(type(region_b))}'
                    ]))
            self.relate_poly(poly_a, poly_b, external_polygon, hull_polygon,
                             adjustments=adjustments)


    def relate_poly(self, poly_a: PolygonType, poly_b: PolygonType,
                    external_polygon: PolygonType = None,
                    hull_polygon: PolygonType = None,
                    adjustments: List[str] = None):
        '''Calculate the DE-27IM relationship for two polygons.

          The DE-27IM relationship is calculated by creating three DE-9IM
          relationships:
            1. The DE-9IM relationship between the two polygons (contour).
            2. The DE-9IM relationship between the *exterior* of the first
                polygon (external) and the second polygon.
            3. The DE-9IM relationship between the *convex hull* of the first
                polygon (convex_hull) and the second polygon.

        Args:
            poly_a (PolygonType): The first polygon.
            poly_b (PolygonType): The second polygon.
            external_polygon (PolygonType): The external polygon of the first
                polygon. If not supplied, self.padding is used.
            hull_polygon (PolygonType): The convex hull polygon of the first
                polygon. If not supplied, self.padding is used.
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
        contour = DE9IM(poly_a, poly_b)
        # If an external polygon is supplied, create a DE9IM relationship for
        # the external polygon.
        if external_polygon is None:
            external = self.padding
        else:
            external = DE9IM(external_polygon, poly_b)
        # If a hull polygon is supplied, create a DE9IM relationship for the
        # hull polygon.
        if hull_polygon is None:
            convex_hull = self.padding
        else:
            convex_hull = DE9IM(hull_polygon, poly_b)
        # Create a tuple of the three DE-9IM relationships.
        # The order is important: (contour, external, convex_hull).
        relation_group = (contour, external, convex_hull)
        # If adjustments are supplied, apply them to the relationship.
        relation_str = self.combine_groups(relation_group, adjustments)
        # Merge the relationship into the DE27IM object.
        self.merge(self.to_int(relation_str))


    def apply_adjustments(self, relation_group: Tuple[DE9IM, DE9IM, DE9IM],
                          adjustments: List[str])-> tuple[DE9IM, DE9IM, DE9IM]:
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
            external = self.padding
            convex_hull = self.padding
            relation_group = (contour, external, convex_hull)
            self.to_int(self.relation)
        if 'hole_b' in adjustments:
            contour, external, convex_hull = relation_group
            contour = contour.hole_adjustment('b')
            external = self.padding
            convex_hull = self.padding
            relation_group = (contour, external, convex_hull)
        # Apply Transpose Adjustment
        if 'transpose' in adjustments:
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
        for rel_def in self.test_binaries:
            result = rel_def.test(relation_binary)
            if result:
                return result
        return RelationshipType.UNKNOWN

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
