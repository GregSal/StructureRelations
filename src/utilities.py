'''Utility Functions'''
# %% Imports
# Type imports
from typing import List, Tuple

import math
import logging

# Shared Packages
import shapely
from shapely.geometry import Polygon
import numpy as np

# Local Packages
from types_and_classes import DEFAULT_TRANSVERSE_TOLERANCE
from types_and_classes import ContourPointsType, PolygonType
from types_and_classes import InvalidContour

# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% Rounding Functions
def round_value(number: float, tolerance: float) -> float:
    '''Round a single value to the specified tolerance.

    Rounds the value to the nearest tolerance increment.
    Args:
        number (float): The value to round.
        tolerance (float): The tolerance increment to round to.
            If tolerance is 0.0 no rounding will be performed.
    Returns:
        float: The rounded value.

    '''
    if tolerance < 0:
        raise ValueError("Resolution must be a positive number")
    if tolerance == 0:
        return number

    # Determine the number of decimal places for rounding based on resolution
    decimal_places = -int(np.log10(tolerance)) + 1

    # Round the value to the specified resolution
    rounded_value = round(round(number / tolerance, decimal_places) * tolerance, decimal_places)

    return rounded_value


def round_one_up(x: float) -> float:
    '''Rounds a number up to one significant digit.

    Args:
        x: Input floating-point value.

    Returns:
        The value rounded upward to one significant digit.
    '''
    if x == 0.0:
        return 0.0

    sign = 1.0 if x > 0 else -1.0
    ax = abs(x)

    exp = math.floor(math.log10(ax))
    scale = 10.0 ** exp

    return sign * math.ceil(ax / scale) * scale


def round_contour_points(contour_points: ContourPointsType,
                         tolerance=DEFAULT_TRANSVERSE_TOLERANCE) -> ContourPointsType:
    '''Round contour points to the specified tolerance.

    Rounds the x and y coordinates of all contour points to the nearest
    tolerance increment. Z coordinates are left unchanged as they typically
    represent slice positions that should remain precise.

    Args:
        contour_points (ContourPointsType): A list of length 2 or three tuples
            of float containing the (x, y) or (x, y, z) coordinates that define
            a contour.
        tolerance (float, optional): The tolerance increment to round to in
            cm. Defaults to DEFAULT_TRANSVERSE_TOLERANCE.

    Returns:
        ContourPointsType: A new list of contour points with x and y
            coordinates rounded to the specified tolerance.

    '''
    if not contour_points:
        logger.debug("No contour points to round")
        return []

    if tolerance <= 0:
        raise ValueError("Tolerance must be a positive number")
    # Determine the number of decimal places for rounding based on tolerance
    decimal_places = -int(np.log10(tolerance)) + 1
    logger.debug('Rounding to tolerance: %s cm, decimal places: %d',
                 tolerance, decimal_places)

    # convert contour_points to numpy array for easier manipulation
    points = np.array(contour_points)
    original_points = points.copy()

    # convert x and y coordinates to multiple of resolution
    points[:, 0] = (points[:, 0] // tolerance) * tolerance
    points[:, 1] = (points[:, 1] // tolerance) * tolerance
    # Round x and y coordinates to avoid floating point issues
    points[:, 0] = np.round(points[:, 0], decimals=decimal_places)
    points[:, 1] = np.round(points[:, 1], decimals=decimal_places)

    # Z coordinates ([:, 2]) are left unchanged

    # convert back to list of tuples
    rounded_points = points.tolist()

    # Difference between the original and rounded points
    differences = np.abs(np.array(rounded_points) - original_points)
    logger.debug('Max difference (x, y): %s',
                 np.max(differences[:, :2], axis=0))
    logger.debug('Average difference (x, y): %s',
                 np.mean(differences[:, :2], axis=0))

    return rounded_points


def point_round(point: shapely.Point,
                tolerance: float = DEFAULT_TRANSVERSE_TOLERANCE)->List[float]:
    '''Round the coordinates of a shapely point to the specified tolerance.

    Args:
        point (shapely.Point): A shapely point.

        tolerance (float, optional): The tolerance to round to.
            Defaults to global DEFAULT_TRANSVERSE_TOLERANCE value.

    Returns:
        shapely.Point: A shapely point with coordinates rounded to the
            specified tolerance.
    '''
    dim = shapely.get_coordinate_dimension(point)
    if dim == 2:
        x, y = shapely.get_coordinates(point)[0]
        clean_coords = [round_value(x,tolerance), round_value(y,tolerance)]
    elif dim == 3:
        x, y, z = shapely.get_coordinates(point, include_z=True)[0]
        clean_coords = [round_value(x,tolerance), round_value(y,tolerance),
                        round_value(z,tolerance)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    return shapely.Point(clean_coords)


def poly_round(polygon: shapely.Polygon,
               tolerance: float = DEFAULT_TRANSVERSE_TOLERANCE)->shapely.Polygon:
    '''Round the coordinates of a polygon to the specified tolerance.

    Note: this function does not work for polygons with holes.

    Args:
        polygon (shapely.Polygon): The polygon to clean.

        tolerance (float, optional): The tolerance to round to.
            Defaults to global DEFAULT_TRANSVERSE_TOLERANCE value.

    Returns:
        shapely.Polygon: The supplied polygon with all coordinate points
            rounded to the supplied tolerance.
    '''
    if len(polygon.interiors) > 0:
        raise ValueError("Cannot round polygons with holes using this function.")
    dim = shapely.get_coordinate_dimension(polygon)
    if dim == 2:
        polygon_points = [(round_value(x,tolerance), round_value(y,tolerance))
                          for x,y in shapely.get_coordinates(polygon)]
    elif dim == 3:
        polygon_points = [(round_value(x,tolerance), round_value(y,tolerance),
                           round_value(z,tolerance))
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
        raise InvalidContour("Invalid polygon created from points.") #from e
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
                # space after the binary string.
                bin_text = indent + bin_text.replace('_', '\t')
            elif matrix_num == 1:
                # The second matrix has a space after the binary string.
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
    final_string = bin_template.format(**bin_dict)
    return final_string
