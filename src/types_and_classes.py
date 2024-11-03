'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports

from typing import Any, Dict, List, LiteralString, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Standard Libraries
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from math import sqrt, pi, sin, cos, tan, radians
from statistics import mean
from itertools import zip_longest
from itertools import product

# Shared Packages
import numpy as np
import pandas as pd
#import xlwings as xw
import pydicom
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line
import pygraphviz as pgv
import networkx as nx


# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Num = int

# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.
SliceIndex = float

# An enclosed region representing either a structure area, or a hole within .
# that structure.
Contour = shapely.Polygon

# The reference numbers for two different structures to be compared.
StructurePair =  Tuple[ROI_Num, ROI_Num]

# A length 9 string of '1's and '0's representing a DE-9IM relationship.
DE9IM_Value = str

# A 27 bit binary value composed of three DE9IM_Values concatenated.
# The right-most 9 binary digits represent the DE-9IM relationship between
#    polygon b and polygon a.
# The middle 9 binary digits represent the DE-9IM relationship between
#    polygon b and the *exterior* of polygon a.
# The left 9 binary digits represent the DE-9IM relationship between
#    polygon b and the *convex hull* of polygon a.
DE27IM_Value = int


# Global Settings
PRECISION = 3

# Exception Types
class StructuresException(Exception): pass
class InvalidContour(ValueError, StructuresException): pass
class InvalidContourRelation(ValueError, StructuresException): pass


# %% Utility functions
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
    def __init__(self, contours: List[shapely.Polygon], **kwargs) -> None:
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
        if 'precision' in kwargs:
            self.precision = kwargs['precision']
        else:
            self.precision = PRECISION
        if 'ignore_errors' in kwargs:
            ignore_errors = kwargs['ignore_errors']
        else:
            ignore_errors = False
        self.contour = shapely.MultiPolygon()
        for contour in contours:
            self.add_contour(contour, ignore_errors=ignore_errors)

    def add_contour(self, contour: shapely.Polygon, ignore_errors=False) -> None:
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
        # Apply requisite rounding to polygon
        contour_round = poly_round(contour, self.precision)
        # Check for valid contour
        if not shapely.is_valid(contour_round):
            if ignore_errors:
                # TODO Add optional text stream / function to receive warning
                # messages when contours are skipped.
                return
            error_str = shapely.is_valid_reason(contour_round)
            raise InvalidContour(error_str)
        # Check for non-overlapping structures
        if self.contour.disjoint(contour_round):
            # Combine non-overlapping structures
            new_contours = self.contour.union(contour_round)
        # Check for hole contour
        elif self.contour.contains(contour_round):
            # Subtract hole contour
            new_contours = self.contour.difference(contour_round)
        else:
            if ignore_errors:
                return
            raise InvalidContourRelation('Cannot merge overlapping contours.')
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
        solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                  for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        if isinstance(solid, shapely.MultiPolygon):
            ext_poly = shapely.MultiPolygon(solid)
        else:
            ext_poly = shapely.MultiPolygon([solid])
        return ext_poly

    @property
    def hull(self)-> shapely.MultiPolygon:
        '''A bounding contour generated from the entire contour MultiPolygon.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        If contour contains more than one distinct region the hull will be the
        combination of the convex_hulls for each distinct region.  It will not
        contain the area between the regions.  in other words, the convex hull
        will consist of multiple elastic bands stretched around each external
        contour rather that one elastic band stretched around all external
        contours.

        Returns:
            shapely.MultiPolygon: The bounding contour for the entire contour
                MultiPolygon.
        '''
        solids = [shapely.convex_hull(poly) for poly in self.contour.geoms]
        combined = shapely.unary_union(solids)
        if isinstance(combined, shapely.MultiPolygon):
            hull = combined
        else:
            hull = shapely.MultiPolygon([combined])
        return hull

    @property
    def area(self)-> float:
        '''The area encompassed by solid exterior contour MultiPolygon.

        Returns:
            float: The area encompassed by each polygon on the slice.
        '''
        area_p = sum(poly.area for poly in self.contour.geoms)
        return area_p

    @property
    def external_area(self)-> float:
        '''The area encompassed by the convex hulls of each polygon on the
        slice.

        Returns:
            float: The area encompassed by the convex hulls of each polygon on
                the slice.
        '''
        solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                  for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        return solid.area

    @property
    def hull_area(self)-> float:
        '''The area encompassed by the convex hulls of each polygon on the
        slice.

        Returns:
            float: The area encompassed by the convex hulls of each polygon on
                the slice.
        '''
        solids = [shapely.convex_hull(poly) for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        return solid.area

    @property
    def is_empty(self)-> bool:
        '''Check if the slice is empty.

        Returns:
            bool: True if the slice is empty, False otherwise.
        '''
        count = len(self.contour.geoms)
        return count == 0
