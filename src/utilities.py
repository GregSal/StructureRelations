'''Utility Functions'''
# %% Imports
# Type imports
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List, Union

# Shared Packages
import numpy as np
import shapely
from shapely.geometry import Polygon

from types_and_classes import PRECISION, SliceIndexType, InvalidContour

slice_sequence = Union[List[SliceIndexType],
                       Tuple[SliceIndexType, SliceIndexType],
                       SliceIndexType]
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


#%% Interpolation Functions
def calculate_new_slice_index(slices: slice_sequence,
                              precision=PRECISION) -> float:
    '''Calculate the new z value based on the given slices.

    Args:
        slices (Union[List[SliceIndexType], SliceIndexType]): The slices to calculate the new z value from.

    Returns:
        float: The calculated new z value.
    '''
    if isinstance(slices, (list, tuple)):
        new_slice = round(np.mean(slices), precision)
        return new_slice
    else:
        return slices


def interpolate_polygon(slices: slice_sequence, p1: shapely.Polygon,
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

def points_to_polygon(points: List[Tuple[float, float]]) -> Polygon:
    '''Convert a list of points to a Shapely polygon and validate it.

    Args:
        points (List[Tuple[float, float]]): A list of tuples containing 2D or 3D points.

    Raises:
        InvalidContour: If the points cannot form a valid polygon.

    Returns:
        Polygon: A valid Shapely polygon.
    '''
    if not points:
        return Polygon()
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise InvalidContour("Invalid polygon created from points.")
    return polygon
