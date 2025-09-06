'''Utility Functions'''
# %% Imports
# Type imports
from typing import List, Tuple

# Shared Packages
import shapely
from shapely.geometry import Polygon

from types_and_classes import PRECISION, PolygonType
from types_and_classes import InvalidContour


# %% Rounding Functions
def point_round(point: shapely.Point, precision: int = PRECISION)->List[float]:
    '''Round the coordinates of a shapely point to the specified precision.

    Args:
        point (shapely.Point): A shapely point.

        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION value.

    Returns:
        shapely.Point: A shapely point with coordinates rounded to the
            specified precision.
    '''
    dim = shapely.get_coordinate_dimension(point)
    if dim == 2:
        x, y = shapely.get_coordinates(point)[0]
        clean_coords = [round(x,precision), round(y,precision)]
    elif dim == 3:
        x, y, z = shapely.get_coordinates(point, include_z=True)[0]
        clean_coords = [round(x,precision), round(y,precision),
                        round(z,precision)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_coords = (round(x,precision), round(y,precision))
    return shapely.Point(clean_coords)


def poly_round(polygon: shapely.Polygon,
               precision: int = PRECISION)->shapely.Polygon:
    '''Round the coordinates of a polygon to the specified precision.

    Note: this function does not work for polygons with holes.

    Args:
        polygon (shapely.Polygon): The polygon to clean.

        precision (int, optional): The number of decimal points sto round to.
            Defaults to global PRECISION constant.

    Returns:
        shapely.Polygon: The supplied polygon with all coordinate points
            rounded to the supplied precision.
    '''
    if len(polygon.interiors) > 0:
        raise ValueError("Cannot round polygons with holes using this function.")
    dim = shapely.get_coordinate_dimension(polygon)
    if dim == 2:
        polygon_points = [(round(x,precision), round(y,precision))
                          for x,y in shapely.get_coordinates(polygon)]
    elif dim == 3:
        polygon_points = [(round(x,precision), round(y,precision),
                           round(z,precision))
                          for x,y,z in shapely.get_coordinates(polygon,
                                                               include_z=True)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_poly = shapely.Polygon(polygon_points)
    return clean_poly

# %% Polygon Functions
def make_multi(poly: PolygonType) -> shapely.MultiPolygon:
    '''Convert a polygon to a multipolygon.

    Args:
        poly (PolygonType): The polygon to convert.

    Returns:
        shapely.MultiPolygon: The converted multipolygon.
    '''
    if isinstance(poly, shapely.MultiPolygon):
        multi_poly = shapely.MultiPolygon(poly)
    else:
        multi_poly = shapely.MultiPolygon([poly])
    return multi_poly


def make_solid(polygon: PolygonType, external_holes: PolygonType = None,
               ) -> shapely.MultiPolygon:
    '''Create a solid Polygon from polygon.

    All holes in the supplied polygon are filled in to create a solid
    polygon.  If an external_holes polygon is supplied, it is subtracted
    from the final solid polygon.

    Args:
        polygon (PolygonType): The polygon to convert to a solid.
        external_holes (PolygonType, optional): A polygon or multipolygon
            representing external holes to be subtracted from the final solid.
            Defaults to None.

    Returns:
        shapely.Polygon: The resulting solid polygon.
    '''
    polygon = make_multi(polygon)
    solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                for poly in polygon.geoms]
    external_polygon = shapely.unary_union(solids)
    external_polygon = make_multi(external_polygon)
    if external_holes is not None:
        external_polygon = external_polygon.difference(external_holes)
    return external_polygon


def points_to_polygon(points: List[Tuple[float, float]]) -> Polygon:
    '''Convert a list of points to a Shapely polygon and validate it.

    Args:
        points (List[Tuple[float, float]]): A list of tuples containing 2D or
            3D points.

    Raises:
        InvalidContour: If the points cannot form a valid polygon.

    Returns:
        Polygon: A valid Shapely polygon.
    '''
    if not points:
        return shapely.Polygon()
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise InvalidContour("Invalid polygon created from points.")
    return polygon


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
