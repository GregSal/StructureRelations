'''Utility Functions'''
# %% Imports
# Type imports
from typing import List, Union

# Shared Packages
import shapely

from types_and_classes import PRECISION


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

# %% Polygon Functions
def make_multi(poly: Union[shapely.Polygon, shapely.MultiPolygon]
               ) -> shapely.MultiPolygon:
    '''Convert a polygon to a multipolygon.
    Args:
        poly (Union[shapely.Polygon, shapely.MultiPolygon]): The polygon to
            convert.
    Returns:
        shapely.MultiPolygon: The converted multipolygon.
    '''
    if isinstance(poly, shapely.MultiPolygon):
        multi_poly = shapely.MultiPolygon(poly)
    else:
        multi_poly = shapely.MultiPolygon([poly])
    return multi_poly
