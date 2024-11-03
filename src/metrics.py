'''Calculate metrics for relationships between structures.
'''
# %% Imports
# Type imports

from typing import Any, Dict, List, Tuple, Union
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
import networkx as nx

from types_and_classes import InvalidContourRelation
from types_and_classes import ROI_Num, SliceIndex, Contour, StructurePair
from structure_slice import StructureSlice
from relations import RelationshipType
from structure_slice import structure_neighbours

# Global Settings
PRECISION = 3

# %% Metric Functions
def broadcast_coords(center: np.array, limits: np.array) -> list[np.array]:
    '''Create points at each of the 4 limits, aligned with center_coords.

    Each limit value in limits is placed into an xy pair along with the
    appropriate x or y values from center_coords.

    Args:
        center_coords (np.array): length 2 array of float with center
            coordinates.

    limits (np.array): length 4 array of float with x and y limits.

    precision (int, optional): The number of decimal points to round to.
        Defaults to global PRECISION constant.

Returns:
    list[np.array]: A list of xy coordinate pairs at the specified limits,
        which can form orthogonal lines crossing through the center point.
    '''
    xy_pairs = [None] * 4
    for i in range(2):
        # Start with center coordinates as the xy pairs.
        xy_pairs[i * 2] = center.copy()
        xy_pairs[i * 2 + 1] = center.copy()
        for j in range(2):
            idx = i * 2 + j
            # replace the appropriate x or y value with one of the limits.
            xy_pairs[idx][j] = limits[i][j]
    return xy_pairs


def length_between(line: shapely.LineString,
                   poly_a: Contour, poly_b: Contour)->float:
    '''Calculate the length of the line between poly_a and poly_b.

    Args:
        line (shapely.LineString): A line passing through both poly_a and
            poly_b.
        poly_a (Contour): The outer polygon.
        poly_b (Contour): A polygon contained within poly_a

    Returns:
        float: The length of the line segment that lies between the outside
            of poly_b and the outside of poly_a
    '''
    # disregard any holes in this calculation.
    exterior_a = shapely.Polygon(poly_a.exterior)
    exterior_b = shapely.Polygon(poly_b.exterior)
    # Remove the part of the line inside of poly_b
    line_outside_b = shapely.difference(line, exterior_b)
    # Remove the part of the line outside of poly_a
    line_between_ab = shapely.intersection(line_outside_b, exterior_a)
    return shapely.length(line_between_ab)

# %% Margin related functions
def get_z_margins(structures, slice_table, precision=PRECISION):
    def centred_in(poly: StructureSlice, point: shapely.Point)->bool:
        return poly.contour.contains(point)

    def has_area(poly: StructureSlice, precision)->bool:
        area = poly.contour.area
        #area = round(poly.contour.area, precision)
        return area > 0

    def get_one_z_margin(slice_structures, structures, precision):
        roi_a, roi_b = structures
        # Select only the slices containing the second structure.
        contour_b_slices = slice_structures[roi_b].dropna()
        # Get the centre point of the end contour of the second structure.
        centre_point = contour_b_slices.iloc[0].contour.centroid
        # Identify the edge slice of the second structure.
        end_slice = contour_b_slices.index[0]
        # For the first structure replace the nan values with empty polygons.
        contour_a_slices = slice_structures[roi_a].fillna(StructureSlice([]))
        # Identify the slices beyond the edge of the second structure.
        ext = contour_a_slices.index <= end_slice
        # Identify the slices where the first structure contours contain the centre
        # point of the end contour of the second structure.
        aligned = contour_a_slices.apply(centred_in, point=centre_point)
        # Identify the slices where structure a is present
        has_a = contour_a_slices.apply(has_area, precision=PRECISION)
        # Identify the slices where there is a transition from aligned to not aligned.
        aligned_edge = aligned & ~aligned.shift(1, fill_value=False)
        struct_edge = has_a & ~has_a.shift(1, fill_value=False)
        # Select the transition point closest to the edge of the second structure.
        aligned_lim = aligned[ext][aligned_edge].index[-1]
        struct_lim = has_a[ext][struct_edge].index[-1]
        # The margin in the desired direction is the difference between the edge of
        # the second structure and the corresponding edge of the first structure.
        aligned_margin = end_slice - aligned_lim
        struct_margin = end_slice - struct_lim
        z_lim = {'aligned': round(aligned_margin, precision),
                 'struct': round(struct_margin, precision)}
        return z_lim

    roi_a, roi_b = structures
    slice_structures = slice_table.loc[:, [roi_a, roi_b]].copy()

    z_neg_margin = get_one_z_margin(slice_structures, structures, precision)

    slice_structures_rev = slice_structures.copy()
    slice_structures_rev.index = slice_structures_rev.index * -1
    slice_structures_rev.sort_index(inplace=True)
    z_pos_margin = get_one_z_margin(slice_structures_rev, structures, precision)
    margins = {'z_neg': z_neg_margin, 'z_pos': z_pos_margin}
    return pd.DataFrame(margins).T


def orthogonal_margins(poly_a: Contour, poly_b: Contour,
                       precision: int = PRECISION)->Dict[str, float]:
    '''Calculate the orthogonal margins between poly_a and poly_b.

    The orthogonal margins are the distances between the exterior of poly_b and
    the boundary of poly_a along lines that are parallel to the x and y axes and
    cross the centre point of poly_b.

    Args:
        poly_a (Contour): The outer polygon.
        poly_b (Contour): A polygon contained within poly_a
        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION constant.

    Returns:
        Dict[str, float]: A dictionary containing the orthogonal margins in
            each direction. The keys of the dictionary are:
                ['x_min', 'y_min', 'x_max', 'y_max']
    '''
    # The maximum extent of polygon a in orthogonal directions.
    a_limits = np.array(poly_a.bounds).reshape((2,-1))
    # Coordinates of the centre of polygon b.
    b_center = (shapely.centroid(poly_b))
    center_coords = shapely.get_coordinates(b_center)[0]
    # Points at the maximum extent of a in line with the centre of b.
    end_points = broadcast_coords(center_coords, a_limits)
    orthogonal_lengths = {}
    labels = ['x_neg', 'y_neg', 'x_pos', 'y_pos']
    for label, limit_point in zip(labels, end_points):
        # Make a line between the center of b and the limit of a.
        line = shapely.LineString([limit_point, center_coords])
        # Get the length of that line between the edges of b and a.
        length = length_between(line, poly_a, poly_b)
        orthogonal_lengths[label] = round(length, precision)
    return orthogonal_lengths


def min_margin(poly_a: Contour, poly_b: Contour,
               precision: int = PRECISION)->Dict[str, float]:
    boundary_a = poly_a.exterior
    boundary_b = poly_b.exterior
    distance = boundary_a.distance(boundary_b)
    rounded_distance = round(distance, precision)
    return rounded_distance


def max_margin(poly_a: Contour, poly_b: Contour,
               precision: int = PRECISION)->Dict[str, float]:
    boundary_a = poly_a.exterior
    boundary_b = poly_b.exterior
    distance = boundary_a.hausdorff_distance(boundary_b)
    rounded_distance = round(distance, precision)
    return rounded_distance


def agg_margins(margin_table: pd.DataFrame):
    if margin_table.empty:
        return pd.Series()
    margin_agg = margin_table.agg('min')
    margin_agg['max'] = margin_table['max'].max()
    return margin_agg


def margins(poly_a: StructureSlice, poly_b: StructureSlice,
            relation: RelationshipType,
            precision: int = PRECISION)->pd.Series:
    def calculate_margins(polygon_a: Contour, polygon_b: Contour,
                          precision: int = PRECISION)->Dict[str, float]:
        # Only calculate margins when the a polygon contains the b polygon.
        if polygon_a.contains(polygon_b):
            margin_dict = orthogonal_margins(polygon_a, polygon_b, precision)
            margin_dict['max'] = max_margin(polygon_a, polygon_b, precision)
            margin_dict['min'] = min_margin(polygon_a, polygon_b, precision)
            return margin_dict
        return {}

    margin_list = []
    # Compare all polygons on the same slice
    for polygon_a, polygon_b in product(poly_a.contour.geoms,
                                        poly_b.contour.geoms):
        if ((relation == RelationshipType.CONTAINS) |
            (relation == RelationshipType.PARTITION)):
            margin_dict = calculate_margins(polygon_a, polygon_b, precision)
        elif ((relation == RelationshipType.SURROUNDS) |
              (relation == RelationshipType.BORDERS_INTERIOR)):
            # Compare all holes in each a polygon with each b polygon.
            for hole_ring in polygon_a.interiors:
                hole = shapely.Polygon(hole_ring)
                margin_dict = calculate_margins(hole, polygon_b, precision)
                if margin_dict:
                    margin_list.append(margin_dict)
                margin_dict = {}  # Clear margin_dict so it is not added twice.
        elif relation == RelationshipType.SHELTERS:
            # The outer region to use for the margin is the "hole" formed by
            # closing the contour using the convex hull.  This can be obtained
            # by subtracting the contour polygon from its  convex hull polygon.
            hull = shapely.convex_hull(polygon_a)
            semi_hole = shapely.difference(hull, polygon_a)
            margin_dict = calculate_margins(semi_hole, polygon_b, precision)
        else:
            msg = ''.join([
                'Margins can only be calculated for the relations: ',
                '"Contains", "Partition", "Surrounds", "Borders_interior", ',
                'or "Shelters".\n',
                f'Supplied structures have relation of type: {str(relation)}.'
                ])
            raise InvalidContourRelation(msg)
        if margin_dict:
            margin_list.append(margin_dict)
    if margin_list:
        margin_table = pd.DataFrame(margin_list)
        return agg_margins(margin_table)
    return pd.Series()


def get_margins(structures, slice_table: pd.DataFrame,
                relation: RelationshipType,
                precision=PRECISION):
    def get_contour_margins(slice_structures: pd.Series,
                            relation: RelationshipType,
                            precision=PRECISION):
        margin_table = margins(slice_structures.iloc[0],
                               slice_structures.iloc[1],
                               relation, precision)
        return margin_table

    slice_structures = slice_table.loc[:, [structures[0], structures[1]]]
    # Remove Slices that have neither structure.
    slice_structures.dropna(how='all', inplace=True)
    # For slices that have only one of the two structures, replace the nan
    # values with empty polygons for duck typing.
    slice_structures.fillna(StructureSlice([]), inplace=True)
    # Get the relationships between the two structures for all slices.
    metric_seq = slice_structures.apply(get_contour_margins, axis='columns',
                                        result_type='expand', relation=relation,
                                        precision=precision)
    contour_margins = agg_margins(metric_seq)
    # Add the z margin components
    z_margins = get_z_margins(structures, slice_table, precision)
    max_magin = max([contour_margins['max'], max(z_margins['struct'])])
    contour_margins['max'] = max_magin
    min_magin = min([contour_margins['min'], min(z_margins['struct'])])
    contour_margins['min'] = min_magin
    contour_margins = pd.concat([contour_margins, z_margins['aligned']])
    return contour_margins


#%% Distance function
def distance(related_contours: pd.Series,
              precision: int = PRECISION)->float:
    '''Calculate the closest 3D distance between 2 contours.

    The shortest 2D distance between the exteriors of two polynomials is
    calculated and then converted into a 3D distance using the hight between
    the two contour slices. If the contours are on the same slice then the
    height is 0 and the 3D distance is the same as the 2D distance.

    This function is intended to be used with the pd.DataFrame.apply method.

    Args:
        related_contours (pd.Series): A series containing 3 items:
            ['a', 'b', 'height'], usually generated with the pandas apply
            method on a DataFrame with columns of those names.  The 'a' and 'b'
            columns have StructureSlice data type and the 'height' column
            contains float values of the distance in cm between the two slices.
        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION constant.

    Returns:
        float: The closest 3D distance between 2 contours.
    '''
    distance_list = []
    poly_a = related_contours['a']
    poly_b = related_contours['b']
    height = related_contours['height']
    # Compare all polygons on the same slice
    for polygon_a, polygon_b in product(poly_a.contour.geoms,
                                        poly_b.contour.geoms):
        boundary_a = polygon_a.exterior
        boundary_b = polygon_b.exterior
        distance_2d = boundary_a.distance(boundary_b)
        distance_list.append(distance_2d)
    if distance_list:
        distance_2d = min(distance_list)
        distance_3d = sqrt(height**2 + distance_2d**2)
    else:
        distance_3d = height
    rounded_distance = round(distance_3d, precision)
    return rounded_distance


# %% Relative volume related functions
def calculate_volume_ratio(related_contours: pd.Series,
                           relation: RelationshipType,
                           precision: int = PRECISION)->float:

    def get_area(contour: StructureSlice):
        if isinstance(contour, StructureSlice):
            return contour.area
        return np.nan

    def get_overlap_area(related_contours: pd.Series)->Dict:
        poly_a = related_contours['a']
        if not isinstance(poly_a, StructureSlice):
            return 0.0
        poly_b = related_contours['b']
        if not isinstance(poly_b, StructureSlice):
            return 0.0
        overlap_area_list = []
        for polygon_a, polygon_b in product(poly_a.contour.geoms,
                                            poly_b.contour.geoms):
            overlap_region = shapely.intersection(polygon_a, polygon_b)
            overlap_area_list.append(overlap_region.area)
        overlap_area = sum(overlap_area_list)
        return overlap_area

    areas = related_contours.map(get_area)
    overlap_area = related_contours.apply(get_overlap_area, axis='columns')
    overlap_area.name = 'Overlap'
    areas = pd.concat([areas, overlap_area], axis='columns')
    total_areas = areas.sum()
    if relation == RelationshipType.OVERLAPS:
        average_volume = (total_areas['a'] + total_areas['b']) / 2.0
        ratio = total_areas['Overlap'] / average_volume
    elif relation == RelationshipType.PARTITION:
        ratio = total_areas['Overlap'] / total_areas['a']
    else:
        ratio = np.nan
    rounded_ratio = round(ratio, precision)
    return rounded_ratio

# %% Relative surface area related functions
def related_lengths(poly_a: StructureSlice, poly_b: StructureSlice,
                    relation: RelationshipType)->List[shapely.LineString]:
    def get_perimeters(poly_a: StructureSlice, poly_b: StructureSlice):
        # FIXME I think the polygon `poly_a.exterior()` should not be used here.
        # Instead the line object `poly_a.boundary`

        overlap_region = shapely.shared_paths(boundary_a, boundary_b)
        perimeter_dict = {'overlapping_perimeter': overlap_region,
                          'perimeter_a': boundary_a,
                          'perimeter_b': boundary_b}
        return perimeter_dict

    perimeter_list = []
    # get relevant perimeters for all polygons on the same slice
    for polygon_a, polygon_b in product(poly_a.contour.geoms,
                                        poly_b.contour.geoms):
        if relation == RelationshipType.BORDERS:
            boundary_a = poly_a.contour.boundary.normalize()
            boundary_b = poly_b.contour.boundary.normalize()

            perimeter_dict = get_perimeters(polygon_a, polygon_b)
            perimeter_list.append(perimeter_dict)
        elif relation == RelationshipType.BORDERS_INTERIOR:
            # TODO Need to be able to identify which hole in a contains b even for
            # slices where b is not present. For now we use the perimeter of all
            # holes in a as a reasonable approximation.
            for hole_ring in polygon_a.interiors:
                hole = shapely.Polygon(hole_ring)
                perimeter_dict = get_perimeters(hole, polygon_b)
                perimeter_list.append(perimeter_dict)
    return perimeter_list


def length_ratio(poly_a: StructureSlice, poly_b: StructureSlice,
                 relation: RelationshipType,
                 precision: int = PRECISION)->pd.DataFrame:

    def get_length(perimeter: shapely.LineString)->float:
        return shapely.length(perimeter)

    perimeter_list = related_lengths(poly_a, poly_b, relation)
    if not perimeter_list:
        return np.nan
    perimeters = pd.DataFrame(perimeter_list)
    lengths = perimeters.apply(get_length)
    lengths_sum = lengths.apply(sum)
    if relation == RelationshipType.BORDERS:
        total_length = (lengths_sum.perimeter_a + lengths_sum.perimeter_b)
        reference_length = total_length / 2
    elif relation == RelationshipType.BORDERS_INTERIOR:
        reference_length = lengths_sum.perimeter_a
    ratio = lengths_sum.overlapping_perimeter / reference_length
    rounded_ratio = round(ratio, precision)
    return rounded_ratio

#        def get_contour_margins(slice_structures: pd.Series,
#                                relation: RelationshipType,
#                                precision=PRECISION):
#            margin_table = margins(slice_structures.iloc[0],
#                                slice_structures.iloc[1],
#                                relation, precision)
#            return margin_table
# %% Helper classes
class ValueFormat(defaultdict):
    '''String formatting templates for individual name, value pairs.
    The default value gives a string like:
        MetricName: 2.36%
    '''
    def __missing__(self, key: str) -> str:
        format_part = ''.join([key, ':\t{', key, ':2.0%}'])
        return format_part


class MetricType(Enum):
    MARGIN = auto()
    DISTANCE = auto()
    OVERLAP_VOLUME = auto()
    OVERLAP_SURFACE_AREA = auto()
    UNKNOWN = 999  # Used for initialization


# %% Metric Classes
class Metric(ABC):
    # Overwrite these class variables for all subclasses.
    metric_type: MetricType
    default_format_template: str

    def __init__(self, structures: StructurePair, **kwargs):
        self.structures = structures
        self.metric = {}
        self.calculate_metric(**kwargs)
        self.format_template = self.default_format_template

    def reset_formats(self):
        self.format_template = self.default_format_template

    def format_metric(self)-> str:
        # Overwritten in some subclasses
        # Returns str: Formatted metrics for report and display
        # Default is for single '%' formatted metric value:
        params = self.metric.copy()
        display_metric = self.format_template.format(**params)
        return display_metric

    @abstractmethod
    def calculate_metric(self, **kwargs):
        pass


class NoMetric(Metric):
    '''A relevant metric does not exist'''
    metric_type = MetricType.UNKNOWN
    default_format_template = 'No Metric'

    def calculate_metric(self, **kwargs):
        self.metric = {}


class DistanceMetric(Metric):
    r'''Distance between two structures.

    The shortest distance between any point in a and any point in b.

    **Used By:**
      - Disjoint
    '''
    metric_type = MetricType.DISTANCE
    default_format_template = 'Distance:\t{Distance:5.2f}'

    def calculate_metric(self, slice_table=pd.DataFrame(),
                         precision=PRECISION, **kwargs):
        roi_a, roi_b = self.structures
        slice_structures = slice_table.loc[:, [roi_a, roi_b]].copy()
        shift_up = structure_neighbours(slice_structures, shift_direction=-1)
        shift_down = structure_neighbours(slice_structures, shift_direction=1)
        no_shift = structure_neighbours(slice_structures, shift_direction=0)
        all_shifts = pd.concat([shift_up, shift_down, no_shift]).dropna()
        all_distances = all_shifts.apply(distance, axis='columns',
                                         precision=precision)
        min_distance = min(all_distances)
        self.metric = {'Distance': min_distance}


class OverlapVolumeMetric(Metric):
    '''The percentage of the volume overlapping for 2 structures.'''
    metric_type = MetricType.OVERLAP_VOLUME
    default_format_template = 'Percentage Overlap:\t{OverlapVolumeRatio:2.0%}'
    def calculate_metric(self, slice_table=pd.DataFrame(),
                         relation=RelationshipType.UNKNOWN,
                         precision=PRECISION, **kwargs):
        roi_a, roi_b = self.structures
        slice_structures = slice_table.loc[:, [roi_a, roi_b]].copy()
        slice_structures.columns = ['a', 'b']
        volume_ratio = calculate_volume_ratio(slice_structures,
                                              relation=relation,
                                              precision=precision)
        self.metric = {'OverlapVolumeRatio': volume_ratio}


class OverlapSurfaceMetric(Metric):
    '''OverlapSurface metric for testing.'''
    metric_type = MetricType.OVERLAP_SURFACE_AREA
    default_format_template = 'Percentage Overlap:\t{OverlapSurfaceRatio:2.0%}'

    def calculate_metric(self, **kwargs):
        # FIXME replace this stub with the OverlapSurface metric function.
        self.metric = {'OverlapSurfaceRatio': 0.23}


class MarginMetric(Metric):
    r'''Margins Between Structures.

        - $Margin_\perp = bounds(a) − bounds(b)$
        - $Margin_{min} = distance(a,b)$
        - $Margin_{max} = distance_{housdorff}(a,b)$

    **Z direction metrics**
    - Orthogonal:
      - The distance between the last slice containing $a$ and the last slice
            of $b$ where the last contour of $a$ overlaps with the contour of $b$
    - Max:
      - The larger of $\Delta Z$ or $d_{2D}$
    - Min:
      - The Smaller of $d_{min}^{2D}$ and $\Delta Z$,<br>

    **Used By:**
        - Contains
        - Surrounds
        - Shelters
    '''
    metric_type = MetricType.MARGIN
    orthogonal_format_template = '\n'.join([
        '        {ANT}   {SUP}  ',
        '        ANT  SUP       ',
        '         | /           ',
        '         |/            ',
        '{RT} RT--- ---LT {LT}  ',
        '        /|             ',
        '       / |             ',
        '   INF  POST           ',
        '  {INF}   {POST}       ',
        ])
    range_format_template = '{MIN}   {MAX}'
    default_format_template = '\n'.join([orthogonal_format_template,
                                         range_format_template])
    default_format_dict = {
        'SUP':  '{z_neg:3.1f}',
        'INF':  '{z_pos:3.1f}',
        'RT':   '{x_neg:3.1f}',
        'LT':   '{x_pos:3.1f}',
        'ANT':  '{y_pos:3.1f}',
        'POST': '{y_neg:3.1f}',
        'MIN':  'Min: {min:3.1f}',
        'MAX':  'Max: {max:3.1f}'
        }
    def __init__(self, structures: StructurePair, **kwargs):
        super().__init__(structures, **kwargs)
        self.format_dict = self.default_format_dict.copy()
        self.display_orthogonal_margins = True
        self.display_margin_range = True
        self.update_formats()

    def reset_formats(self):
        '''Return the metric display back to its default.
        Default formatting looks like this:

                1.2   0.8
                ANT  SUP
                 | /
                 |/
        1.1 RT--- ---LT 2.1
                /|
               / |
           INF  POST
          0.6   1.2

        MIN: 2.1   MAX: 1.2
        '''
        self.format_dict = self.default_format_dict.copy()
        self.display_orthogonal_margins = True
        self.display_margin_range = True

    def update_formats(self):
        '''Updates the a Formatted metrics string for display.

        This is called to update the complete display template when
        changes ar made to parts of the template.

        Individual margins can be removed by replacing the appropriate value in
        format_dict with an empty string.  Adding the margin back into the
        display is done by copying the appropriate value from
        default_format_dict into format_dict.

        The entire orthogonal display can be removed by setting
        display_orthogonal_margins to False.  Likewise, removing the entire
        margin range text and be done by setting display_margin_range to False.
        '''
        display_parts = []
        if self.display_orthogonal_margins:
            display_parts.append(self.orthogonal_format_template)
        if self.display_margin_range:
            display_parts.append(self.range_format_template)
        self.format_template = '\n\n'.join(display_parts)

    def format_metric(self)-> str:
        # Returns str: Formatted metrics for report and display
        format_params = {}
        for label, fmt_str in self.format_dict.items():
            format_params[label] = fmt_str.format(**self.metric)
        display_text = self.format_template.format(**format_params)
        return display_text

    def calculate_metric(self, slice_table=pd.DataFrame(),
                         relation=RelationshipType.UNKNOWN,
                         precision=PRECISION, **kwargs):
        def get_contour_margins(slice_structures: pd.Series,
                                relation: RelationshipType,
                                precision=PRECISION):
            margin_table = margins(slice_structures.iloc[0],
                                slice_structures.iloc[1],
                                relation, precision)
            return margin_table

        roi_a, roi_b = self.structures
        slice_structures = slice_table.loc[:, [roi_a, roi_b]].copy()
        # Remove Slices that have neither structure.
        slice_structures.dropna(how='all', inplace=True)
        # For slices that have only one of the two structures, replace the nan
        # values with empty polygons for duck typing.
        slice_structures.fillna(StructureSlice([]), inplace=True)
        # Get the relationships between the two structures for all slices.
        metric_seq = slice_structures.apply(get_contour_margins, axis='columns',
                                            result_type='expand',
                                            relation=relation,
                                            precision=precision)
        contour_margins = agg_margins(metric_seq)
        # Add the z margin components
        z_margins = get_z_margins(self.structures, slice_table, precision)
        max_margin = max([contour_margins['max'], max(z_margins['struct'])])
        contour_margins['max'] = max_margin
        min_margin = min([contour_margins['min'], min(z_margins['struct'])])
        contour_margins['min'] = min_margin
        contour_margins = pd.concat([contour_margins, z_margins['aligned']])
        self.metric = contour_margins.to_dict()


metric_match = {
    RelationshipType.DISJOINT: DistanceMetric,
    RelationshipType.BORDERS: OverlapSurfaceMetric,
    RelationshipType.BORDERS_INTERIOR: OverlapSurfaceMetric,
    RelationshipType.OVERLAPS: OverlapVolumeMetric,
    RelationshipType.PARTITION: OverlapVolumeMetric,
    RelationshipType.SHELTERS: MarginMetric,
    RelationshipType.SURROUNDS: MarginMetric,
    RelationshipType.CONTAINS: MarginMetric,
    RelationshipType.EQUALS: NoMetric,
    RelationshipType.UNKNOWN: NoMetric,
    }
