'''StructureSlice Tests.'''
# To add a new markdown cell, type '# %% [markdown]'
# %% Imports
# Type imports
from typing import Any, Dict, Tuple, List

# Standard Libraries
from enum import Enum
from pathlib import Path
from math import sqrt, pi, sin, cos, tan, radians
from statistics import mean
from itertools import zip_longest

# Shared Packages
import numpy as np
import matplotlib.pyplot as plt
import shapely
#from shapely.plotting import plot_polygon, plot_points
# %% Global Settings
PRECISION = 3

# %% Contour Creation Functions
def circle_points(radius: float, offset_x: float = 0, offset_y: float = 0,
                  num_points: int = 16, precision=3)->list[tuple[float, float]]:
    deg_step = radians(360/num_points)
    degree_points = np.arange(stop=radians(360), step=deg_step)
    x_coord = np.array([round(radius*sin(d), precision) for d in degree_points])
    y_coord = np.array([round(radius*cos(d), precision) for d in degree_points])

    x_coord = x_coord + offset_x
    y_coord = y_coord + offset_y
    coords = [(x,y) for x,y in zip(x_coord,y_coord)]
    return coords


def box_points(width:float, height: float = None, offset_x: float = 0,
               offset_y: float = 0) -> list[tuple[float, float]]:
    x1_unit = width / 2
    if not height:
        y1_unit = x1_unit
    else:
        y1_unit = height / 2
    coords = [
        ( x1_unit + offset_x,  y1_unit + offset_y),
        ( x1_unit + offset_x, -y1_unit + offset_y),
        (-x1_unit + offset_x, -y1_unit + offset_y),
        (-x1_unit + offset_x,  y1_unit + offset_y)
        ]
    return coords

# %% StructureSlice Class
class StructureSlice():
    '''Assemble a shapely.MultiPolygon.

    Iteratively create a shapely MultiPolygon from a list of shapely Polygons.
    polygons that are contained within the already formed MultiPolygon are
    treated as holes and subtracted from the MultiPolygon.  Polygons
    overlapping with the already formed MultiPolygon are rejected. Polygons that
    are disjoint with the already formed MultiPolygon are combined with a union.

    Two custom properties exterior and hull are defined. Exterior returns the
    equivalent with all holes filled in.  Hull returns a MultiPolygon that is
    the convex hull surrounding the entire MultiPolygon.

    Args:
        contours (List[shapely.Polygon]): A list of polygons to be merged
        into a single MultiPolygon.

    Attributes:
        contour (shapely.MultiPolygon): The MultiPolygon created by combining
            the supplied list of polygons.
        exterior (shapely.MultiPolygon): The contour MultiPolygon with all
            holes filled in.
        hull (shapely.MultiPolygon): The MultiPolygon that is the convex hull
            surrounding the contour MultiPolygon.
    '''
    def __init__(self, contours: List[shapely.Polygon]) -> None:
        '''Iteratively create a shapely MultiPolygon from a list of shapely
        Polygons.

        Polygons that are contained within the already formed MultiPolygon are
        treated as holes and subtracted from the MultiPolygon.  Polygons
        overlapping with the already formed MultiPolygon are rejected. Polygons
        that are disjoint with the already formed MultiPolygon are combined.

        Args:
            contours (List[shapely.Polygon]): A list of polygons to be merged
            into a single MultiPolygon.
        '''
        self.contour = shapely.MultiPolygon()
        for contour in contours:
            self.add_contour(contour)

    def add_contour(self, contour: shapely.Polygon) -> None:
        '''Add a shapely Polygon to the current MultiPolygon from a list of shapely
        Polygons.

        Polygons that are contained within the already formed MultiPolygon are
        treated as holes and subtracted from the MultiPolygon.  Polygons
        overlapping with the already formed MultiPolygon are rejected. Polygons
        that are disjoint with the already formed MultiPolygon are combined.

        Args:
            contour (shapely.Polygon): The shapely Polygon to be added.
                The shapely Polygon must either be contained in or be disjoint
                with the existing MultiPolygon.

        Raises:
            ValueError: When the supplied shapely Polygon overlaps with the
                existing MultiPolygon.
        '''
        # Check for non-overlapping structures
        if self.contour.disjoint(contour):
            # Combine non-overlapping structures
            new_contours = self.contour.union(contour)
        # Check for hole contour
        elif self.contour.contains(contour):
            # Subtract hole contour
            new_contours = self.contour.difference(contour)
        else:
            raise ValueError('Cannot merge overlapping contours.')
        # Enforce the MultiPolygon type for self.contour
        if isinstance(new_contours, shapely.MultiPolygon):
            self.contour = new_contours
        else:
            self.contour = shapely.MultiPolygon([new_contours])

    @property
    def exterior(self)-> shapely.MultiPolygon:
        '''The solid exterior contour MultiPolygon.

        Returns:
            shapely.MultiPolygon: The contour MultiPolygon with all holes
                filled in.
        '''
        solid = [shapely.Polygon(shapely.get_exterior_ring(poly))
                 for poly in self.contour.geoms]
        return shapely.MultiPolygon(solid)

    @property
    def hull(self)-> shapely.MultiPolygon:
        '''A bounding contour generated from the entire contour MultiPolygon.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        Returns:
            shapely.MultiPolygon: The bounding contour for the entire contour
                MultiPolygon.
        '''
        hull = shapely.convex_hull(self.contour)
        return shapely.MultiPolygon([hull])

# %% Testing the StructureSlice Class
box6 = shapely.Polygon(box_points(6))
StructureSlice([box6])


#box6 = Polygon(box_points(6))
#box4 = Polygon(box_points(4))
#StructureSlice([box6, box4])
#
#
#box6 = Polygon(box_points(6))
#offset_box6 = Polygon(box_points(6,offset_x=3))
#StructureSlice([box6, offset_box6])
#
#
#box2a = shapely.Polygon(box_points(2, offset_x=-3))
#box2b = shapely.Polygon(box_points(2, offset_x=3))
#StructureSlice([box2a, box2b])
#
#
#a = StructureSlice([box4, box6])
#b = [p for p in a.geoms]
#b
