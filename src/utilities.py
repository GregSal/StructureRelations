'''Utility Functions'''
# %% Imports
# Type imports

from itertools import chain
from typing import List, Dict

# Standard Libraries
from math import ceil, sin, cos, radians, sqrt


# Shared Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line

from types_and_classes import PRECISION, ROI_Num, SliceIndex, Contour, StructurePair



# %% Rounding Functions
def point_round(point: shapely.Point, precision: int = PRECISION)->List[float]:
    '''Round the coordinates of a shapley point to the specified precision.

    Args:
        point (shapely.Point): A shapely point.

        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION value.

    Returns:
        List[float]: A list of rounded point coordinates.
    '''
    x, y = shapely.get_coordinates(point)[0]
    clean_coords = (round(x,precision), round(y,precision))
    return clean_coords


def poly_round(polygon: shapely.Polygon, precision: int = PRECISION)->shapely.Polygon:
    '''Round the coordinates of a polygon to the specified precision.

    Args:
        polygon (shapely.Polygon): The polygon to clean.

        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION constant.

    Returns:
        shapely.Polygon: The supplied polygon with all coordinate points
            rounded to the supplied precision.
    '''
    polygon_points = [(round(x,precision), round(y,precision))
                      for x,y in shapely.get_coordinates(polygon)]
    clean_poly = shapely.Polygon(polygon_points)
    return clean_poly
