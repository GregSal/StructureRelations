'''Utility Functions'''
# %% Imports
# Type imports
from dataclasses import asdict, dataclass
from typing import Dict, List, Union

# Shared Packages
import numpy as np
import shapely
from shapely.geometry import Polygon

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
        clean_coords = [round(x,precision), round(y,precision),
                        round(z,precision)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_coords = (round(x,precision), round(y,precision))
    return shapely.Point(clean_coords)


def poly_round(polygon: shapely.Polygon,
               precision: int = PRECISION)->shapely.Polygon:
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
        polygon_points = [(round(x,precision), round(y,precision),
                           round(z,precision))
                          for x,y,z in shapely.get_coordinates(polygon,
                                                               include_z=True)]
    else:
        raise ValueError(f"Invalid coordinate dimension: {dim}")
    clean_poly = shapely.Polygon(polygon_points)
    return clean_poly


#%% Interpolation Functions
def calculate_new_slice_index(slices: Union[List[SliceIndexType], SliceIndexType]) -> float:
    '''Calculate the new z value based on the given slices.

    Args:
        slices (Union[List[SliceIndexType], SliceIndexType]): The slices to calculate the new z value from.

    Returns:
        float: The calculated new z value.
    '''
    if isinstance(slices, (list, tuple)):
        return np.mean(slices)
    else:
        return slices


def interpolate_polygon(slices: Union[List[SliceIndexType], SliceIndexType],
                        p1: shapely.Polygon,
                        p2: shapely.Polygon = None) -> shapely.Polygon:
    def match_boundaries(p1, p2):
        if p1.is_empty:
            boundary1 = None
        else:
            boundary1 = p1.exterior
        if p2 is None:
            if boundary1:
                boundary2 = p1.centroid
            else:
                raise ValueError('No second polygon given and first polygon is empty.')
        else:
            boundary2 = p2.exterior
            if not boundary1:
                boundary1 = p2.centroid
        return boundary1, boundary2

    def match_holes(p1, p2):
        if p1.is_empty:
            holes1 = []
        else:
            holes1 = list(p1.interiors)
        # If no second polygon given, use the centroid of each first hole as the
        # matching second hole boundary.
        if p2 is None:
            if holes1:
                matched_holes = [(hole, hole.centroid) for hole in holes1]
            else:
                matched_holes = []
            return matched_holes
        # If the first polygon does not have any holes, and the second polygon
        # does, use the centroid of the second hole as the matching first hole
        # boundary.
        holes2 =  list(p2.interiors)
        if not holes1:
            matched_holes = [(hole, hole.centroid) for hole in holes2]
            return matched_holes
        # If both polygons have holes, match the holes match the holes of the
        # second polygon to the first polygon.
        matched_holes = []
        # set each second hole as not matched.
        hole2_matched = {i: False for i in range(len(holes2))}
        for hole1 in holes1:
            matched1 = False  # set the first hole as not matched.
            for idx, hole2 in enumerate(holes2):
                if hole1.overlaps(hole2):
                    matched_holes.append((hole1, hole2))
                    matched1 = True # set the first hole as matched.
                    hole2_matched[idx] = True  # set the second hole as matched.
            # If the first hole is not matched, use the centroid of the first
            # hole as the matching second hole boundary.
            if not matched1:
                matched_holes.append((hole1, hole1.centroid))
        # Add any unmatched holes from the second polygon, using the centroid of
        # the second hole as the matching first hole boundary.
        for idx, hole2 in enumerate(holes2):
            if not hole2_matched[idx]:
                matched_holes.append((hole2, hole2.centroid))
        return matched_holes

    def interpolate_boundaries(boundary1, boundary2):
        new_cords = []
        for crd in boundary1.coords:
            ln = shapely.shortest_line(shapely.Point(crd), boundary2)
            ptn = ln.interpolate(0.5, normalized=True)
            new_cords.append(ptn)
        return new_cords
    # Use the new function
    new_z = calculate_new_slice_index(slices)
    # If either of the polygons are multi-polygons, raise an error.
    if isinstance(p1, shapely.MultiPolygon):
        raise ValueError('Only single polygons are supported.')
    if isinstance(p2, shapely.MultiPolygon):
        raise ValueError('Only single polygons are supported.')

    boundary1, boundary2 = match_boundaries(p1, p2)
    # Interpolate the new polygon coordinates as half way between the p1
    # boundary and boundary 2.
    new_cords = interpolate_boundaries(boundary1, boundary2)
    # Add the holes to the new polygon.
    new_holes = []
    matched_holes = match_holes(p1, p2)
    for hole1, hole2 in matched_holes:
        new_hole = interpolate_boundaries(hole1, hole2)
        new_holes.append(new_hole)
    # Build the new polygon from the interpolated coordinates.
    itp_poly = shapely.Polygon(new_cords, holes=new_holes)
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
