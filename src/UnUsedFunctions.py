from dataclasses import asdict
from typing import Dict, Union
import numpy as np
import pandas as pd
import shapely

from structure_slice import ContourType, StructureSlice, empty_structure


def build_slice_spacing_table(slice_table, shift_direction=-1)->pd.DataFrame:
    def slice_spacing(contour):
        # Index is the slice position of all slices in the image set
        # Columns are structure IDs
        # Values are the distance (INF) to the next contour
        inf = contour.dropna().index.min()
        sup = contour.dropna().index.max()
        contour_range = (contour.index <= sup) & (contour.index >= inf)
        slices = contour.loc[contour_range].dropna().index.to_series()
        gaps = slices.shift(shift_direction) - slices
        return gaps
    # Find distance between slices with contours
    def get_slices(structure: pd.Series):
        used_slices = structure.dropna().index.to_series()
        return used_slices

    contour_slices = slice_table.apply(get_slices)
    slice_spacing_data = contour_slices.apply(slice_spacing)
    return slice_spacing_data



def has_area(poly: ContourType)->bool:
    '''Check if the structure has area.

    Tests whether the structure has an area greater than zero or is empty.

    Args:
        poly (ContourType): A StructureSlice, RegionNode, Polygon, or NaN object.

    Returns:
        bool: True if the structure has an area greater than zero, False
            otherwise.
    '''
    if empty_structure(poly):
        area = False
    else:
        try:
             # check for a StructureSlice or shapely.Polygon object.
            area = poly.area
        except AttributeError:
            # check for a RegionNodeType object.
            try:
                area = poly['polygon'].area
            except (TypeError, KeyError):
                # Anything else is considered to have no area.
                area = False
    if area:
        return area > 0
    return area


def contains_point(poly: ContourType, point: shapely.Point)->bool:
    '''Check if the structure contains the given point.

    Tests whether the structure contains the given point or is empty.
    This is a convenience function that wraps the shapely Polygon.contains
    method, allowing it to be applied to a series of StructureSlice objects.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.
        point (shapely.Point): A shapely Point object.

    Returns:
        bool: True if the structure contains the point, False otherwise.
    '''
    contains = False
    # check for an empty poly.
    if empty_structure(poly):
        contains = False
    else:
        try:
            # check for a StructureSlice object.
            contains = poly.contour.contains(point)
        except AttributeError:
            try:
                # check for a shapely.Polygon object.
                contains = poly.contains(point)
            except AttributeError:
                try:
                    # check for a RegionNodeType object.
                    contains = poly['polygon'].contains(point)
                except (TypeError, KeyError):
                    contains = False
    return contains


def get_centroid(poly: Union[StructureSlice, float])->shapely.Point:
    '''Get the centroid of the structure.

    Returns the centroid of the structure or an empty Point object the
    structure is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice, shapely.Polygon
            or object containing a 'polygon' attribute.

    Returns:
        shapely.Point: The centroid of the structure.
    '''
    if empty_structure(poly):
        centroid = shapely.Point()
    else:
        try:
            # check for a StructureSlice object.
            centroid = poly.contour.centroid
        except AttributeError:
            try:
                # check for a shapely.Polygon object.
                centroid = poly.centroid
            except AttributeError:
                try:
                    # check for a RegionNodeType object.
                    centroid = poly['polygon'].centroid
                except (TypeError, KeyError):
                    # Anything else is considered to have no centroid.
                    centroid = shapely.Point()
    return centroid






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
