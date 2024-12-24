'''Utility Functions'''
# %% Imports
# Type imports
from dataclasses import asdict, dataclass
from typing import Dict, List, Union

# Shared Packages
import numpy as np
import shapely

from types_and_classes import PRECISION, SliceIndexType


# %% Rounding Functions
def point_round(point: shapely.Point, precision: int = PRECISION)->List[float]:
    '''Round the coordinates of a shapley point to the specified precision.

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
        clean_coords = [round(x,precision), round(y,precision), round(z,precision)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_coords = (round(x,precision), round(y,precision))
    return shapely.Point(clean_coords)


def poly_round(polygon: shapely.Polygon, precision: int = PRECISION)->shapely.Polygon:
    '''Round the coordinates of a polygon to the specified precision.

    Args:
        polygon (shapely.Polygon): The polygon to clean.

        precision (int, optional): The number of decimal points sto round to.
            Defaults to global PRECISION constant.

    Returns:
        shapely.Polygon: The supplied polygon with all coordinate points
            rounded to the supplied precision.
    '''
    dim = shapely.get_coordinate_dimension(polygon)
    if dim == 2:
        polygon_points = [(round(x,precision), round(y,precision))
                          for x,y in shapely.get_coordinates(polygon)]
    elif dim == 3:
        polygon_points = [(round(x,precision), round(y,precision), round(z,precision))
                          for x,y,z in shapely.get_coordinates(polygon, include_z=True)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_poly = shapely.Polygon(polygon_points)
    return clean_poly


#%% Interpolation Functions
def interpolate_polygon(slices: Union[List[SliceIndexType], SliceIndexType],
                        p1: shapely.Polygon,
                        p2: shapely.Polygon = None) -> shapely.Polygon:
    # If multiple slices given, calculate the new z value.
    if isinstance(slices, (list, tuple)):
        new_z = np.mean(slices)
    else:
        new_z = slices
    # If no second polygon given, use the centroid of the first polygon as the
    # boundary.
    if p2 is None:
        boundary2 = p1.centroid
    else:
        boundary2 = p2.boundary
    # Interpolate the new polygon coordinates as half way between the p1
    # boundary and boundary 2.
    new_cords = []
    for crd in p1.boundary.coords:
        ln = shapely.shortest_line(shapely.Point(crd), boundary2)
        ptn = ln.interpolate(0.5, normalized=True)
        new_cords.append(ptn)
    # Build the new polygon from the interpolated coordinates.
    itp_poly = shapely.Polygon(new_cords)
    # Add the z value to the polygon.
    itp_poly = shapely.force_3d(itp_poly, new_z)
    return itp_poly


# %% Extent Functions
@dataclass(eq=True, order=True)
class Extent():
    '''The extents of a polygon, or a set of orthogonal distances.

    The parameters are in the order given by the shapley.bounds property.
    '''
    x_neg: float
    y_neg: float
    x_pos: float
    y_pos: float

    def get_limit(self, other: "Extent")->"Extent":
        '''Determine the maximum extent of two polygons.

        The extent is determined by the minimum x and y values of the two
        polygons.

        Args:
            other (Extent): The other extent to compare to.

        Returns:
            Extent: The maximum extent of the two polygons.
        '''
        # The method to use for each limit. The min is used for the negative
        # limits and the max is used for the positive limits.
        label_methods = {'x_neg': min,
                         'y_neg': min,
                         'y_pos': max,
                         'x_pos': max}
        limit_dict = {}
        # For each x and y limit, compare the two values and store the result.
        for limit, compare in label_methods.items():
            this = self.__getattribute__(limit)
            that = other.__getattribute__(limit)
            # The compare method is either the min() or max() function.
            new = compare([this, that])
            limit_dict[limit] = new
        return self.__class__(**limit_dict)

    def get_min(self, other: "Extent")->"Extent":
        '''Determine the minimum of two extents.

        This function is used to determine the minimums of two sets of
        orthogonal distances.

        Args:
            other (Extent): The other extent to compare to.

        Returns:
            Extent: The minimums of two sets of orthogonal distances.
        '''
        limit_dict = {}
        for limit, val in asdict(self).items():
            other_val = other.__getattribute__(limit)
            new = min([val, other_val])
            limit_dict[limit] = new
        return self.__class__(**limit_dict)


def get_extent(poly_a: shapely.Polygon,
               poly_b: shapely.Polygon) -> Dict[str, float]:
    '''Get the extent of two polygons.

    Args:
        poly_a (shapely.Polygon): The first polygon.
        poly_b (shapely.Polygon): The second polygon.

    Returns:
        Dict[str, float]: The maximum extent of the two polygons.
            The dictionary contains the keys 'x_neg', 'y_neg', 'x_pos', and
            'y_pos'.
    '''
    extent_a = Extent(*poly_a.bounds)
    extent_b = Extent(*poly_b.bounds)
    extent = extent_a.get_limit(extent_b)
    return asdict(extent)

# %% Center Functions
def get_center(poly_a: shapely.Polygon, poly_b: shapely.Polygon,
                use_centre: str = 'secondary') -> np.array:
    '''Get the centre of the selected polygon.

    Return the center of the selected polygon. The polygon selection is
    determined by the 'use_centre' parameter. If 'use_centre' is 'primary',
    the center of 'poly_a' is returned. If 'use_centre' is 'secondary' (the
    default), the center of 'poly_b' is returned.

    Args:
        poly_a (shapely.Polygon): The primary polygon.
        poly_b (shapely.Polygon): The secondary polygon.
        use_centre (str): The polygon to select. Must be either 'primary' or
            'secondary'. Defaults to 'secondary'.

    Raises:
        ValueError: If 'use_centre' is not 'primary' or 'secondary'.

    Returns:
        np.array: A size 2 array containing the x and y coordinates of the
            selected polygon's center.
    '''
    # Obtain the coordinates of the center of the selected structure.
    if use_centre == 'primary':
        center = shapely.get_coordinates(poly_a.centroid)[0]
    elif use_centre == 'secondary':
        center = shapely.get_coordinates(poly_b.centroid)[0]
    else:
        raise ValueError('use_centre must be either "primary" or "secondary".')
    return center


def generate_orthogonal_lines(poly_a: shapely.Polygon, poly_b: shapely.Polygon,
                              use_centre: str) -> Dict[str, shapely.LineString]:
    '''Generate orthogonal lines from the center of two polygons.

    Args:
        poly_a (shapely.Polygon): The primary polygon.
        poly_b (shapely.Polygon): The secondary polygon.
        use_centre (str): The polygon to center the orthogonal lines on. Must be
        either 'primary' or 'secondary'.

    Returns:
        Dict[str, shapely.LineString]: A dictionary containing the orthogonal
            lines. The keys are 'x_neg', 'y_neg', 'x_pos', and 'y_pos'.
    '''

    center = get_center(poly_a, poly_b, use_centre)
    extent = get_extent(poly_a, poly_b)

    orthogonal_lines = {}
    for label, limit in extent.items():
        if 'x' in label:
            end_point = (limit, center[1])
        else:
            end_point = (center[0], limit)
        line = shapely.linestrings([center, end_point])
        orthogonal_lines[label] = line
    return orthogonal_lines


# %% Intersection Functions
def find_intersection(line: shapely.LineString,
                      poly: shapely.Polygon) -> shapely.MultiPoint:
    '''Find the intersection of a line with a polygon.

    Args:
        line (shapely.LineString): The line to intersect with the polygon.
        poly (shapely.Polygon): The polygon to intersect with the line.

    Returns:
        shapely.MultiPoint: The intersection points of the line with the
            polygon.
    '''
    pnt = shapely.intersection(line, poly.boundary)
    if isinstance(pnt, shapely.Point):
        pnt = shapely.MultiPoint([pnt])
    return pnt
