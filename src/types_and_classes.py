'''Structures from DICOM files

Types, Classes and utility function definitions.

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
import xlwings as xw
import pydicom
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line
import pygraphviz as pgv
import networkx as nx


# %%| Type definitions and Globals
ROI_Num = int  # Index to structures defined in Structure RT DICOM file
SliceIndex = float
Contour = shapely.Polygon
StructurePair =  Tuple[ROI_Num, ROI_Num]


# Global Settings
PRECISION = 2


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


def point_round(point: shapely.Point, precision: int = PRECISION)->List[float]:
    '''Round the coordinates of a polygon to the specified precision.

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
        # Apply requisite rounding to polygon
        contour_round = poly_round(contour, PRECISION)
        # Check for non-overlapping structures
        if self.contour.disjoint(contour_round):
            # Combine non-overlapping structures
            new_contours = self.contour.union(contour_round)
        # Check for hole contour
        elif self.contour.contains(contour_round):
            # Subtract hole contour
            new_contours = self.contour.difference(contour_round)
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

        Returns:
            shapely.MultiPolygon: The bounding contour for the entire contour
                MultiPolygon.
        '''
        hull = shapely.convex_hull(self.contour)
        return shapely.MultiPolygon([hull])


#%% Structure Class
class StructureCategory(Enum):
    '''Defines basic structure categories for sorting and display.
    Elements Are:
        TARGET
        ORGAN
        EXTERNAL
        PLANNING
        DOSE
        OTHER
        UNKNOWN (Used for initialization.)
    '''
    TARGET = auto()
    ORGAN = auto()
    EXTERNAL = auto()
    PLANNING = auto()
    DOSE = auto()
    OTHER = auto()
    UNKNOWN = 999  # Used for initialization

    def __str__(self)->str:
        name = self.name
        # Make into title case:
        title_str = name[0].upper() + name[1:].lower()
        return title_str


class ResolutionType(Enum):
    '''Defines Eclipse Structure Resolution categories.
    Elements Are:
        NORMAL
        HIGH
        UNKNOWN (Used for initialization.)
    '''
    NORMAL = 1
    HIGH = 2
    UNKNOWN = 999  # Used for initialization


@dataclass
class StructureInfo:
    '''Structure information obtained from the DICOM Structure file.
    Attributes Are:
        roi_number (ROI_Num): The unique indexing reference to the structure.
        structure_id (str): The unique label for the structure.
        structure_name (str, optional): A short description of the structure.
            Default is an empty string.
        structure_type (str, optional): DICOM Structure Type. Default is 'None'.
        structure_category (StructureCategory): Structure category determined
            from the name and DICOM Structure Type.
        color (Tuple[int,int,int], optional): The structure color. Default
            color is white (255,255,255).
        structure_code (str, optional): The Eclipse Structure Dictionary code
            assigned to the structure. Default is an empty string.
        structure_code_scheme (str, optional): The scheme references by the
            structure_code. Default is an empty string.
        structure_code_meaning (str, optional): The description of the
            Structure Dictionary item references by the structure_code. Default
            is an empty string.
        structure_density (float, optional): Assigned Hounsfield units for the
            structure. Default is np.nan
        structure_generation_algorithm (str, optional): The type of algorithm
            used to generate the structure.  Default is an empty string.
            Possible values are:
                'AUTOMATIC': Calculated ROI (auto contour)
                'SEMIAUTOMATIC': ROI calculated with user assistance. (e.g.
                    isodose line as structure).
                '': Algorithm is not known.
    '''
    roi_number: ROI_Num
    structure_id: str
    structure_name: str = ''
    structure_type: str = 'None'  # DICOM Structure Type
    color: Tuple[int,int,int] = (255,255,255)  # Default color is white
    structure_code: str = ''
    structure_code_scheme: str = ''
    structure_code_meaning: str = ''
    structure_generation_algorithm: str = ''
    structure_density: float = np.nan

    def get_roi_labels(self, roi_obs: pydicom.Dataset):
        '''Get label information for a structure from the RS DICOM file.

        Args:
            code_seq (pydicom.Dataset): An item from the RT ROI Observations Sequence
                in the dataset of an RS DICOM file.
        '''
        code_seq = roi_obs.get('RTROIIdentificationCodeSequence')
        if code_seq:
            roi_label = code_seq[0]
            self.roi_number = roi_obs.ReferencedROINumber
            self.structure_name = roi_obs.ROIObservationLabel
            self.structure_type = roi_obs.RTROIInterpretedType
            self.structure_code = roi_label.CodeValue
            self.structure_code_scheme = roi_label.CodingSchemeDesignator
            self.structure_code_meaning = roi_label.CodeMeaning


@dataclass
class StructureParameters:
    '''Structure parameters calculated from structure's contour points.
    Attributes Are:
        sup_slice (SliceIndex, optional): The SUP-most CT slice offset
            containing contour point for the structure. Default is np.nan:
        inf_slice (SliceIndex, optional): The INF-most CT slice offset
            containing contour point for the structure. Default is np.nan:
        length (float, optional): The distance between the SUP-most and INF-most
            CT slices.  It assumes a continuous structure. Default is np.nan:
        volume (float, optional): The volume enclosed by the structure's contour
            points. Default is np.nan:
        surface_area (float, optional): The surface area defined by the
            structure's contour points. Default is np.nan:
        thickness (float, optional): If the structure is hollow, the average
            thickness of the structure wall. Default is np.nan:
        sphericity (float, optional): A measure of the roundness of the
            structure relative to a sphere. The value range from 0 to 1, where
            a value of 1 indicates a perfect sphere.
        resolution (ResolutionType, optional): The Eclipse Structure Resolution
            estimated from the resolution value. Default is
            ResolutionType.NORMAL
        center_of_mass (Tuple[int,int,int], optional): The geometric centre of
            the structure (density is not taken into account. Default is tuple().
        structure_id (str): The unique label for the structure.
        structure_name (str, optional): A short description of the structure.
            Default is an empty string.
        structure_type (str, optional): DICOM Structure Type. Default is 'None'.
        structure_category (StructureCategory): Structure category determined
            from the name and DICOM Structure Type.
        color (Tuple[int,int,int], optional): The structure color. Default
            color is white (255,255,255).
        structure_code (str, optional): The Eclipse Structure Dictionary code
            assigned to the structure. Default is an empty string.
        structure_code_scheme (str, optional): The scheme references by the
            structure_code. Default is an empty string.
        structure_code_meaning (str, optional): The description of the
            Structure Dictionary item references by the structure_code. Default
            is an empty string.
        structure_density (float, optional): Assigned Hounsfield units for the
            structure. Default is np.nan
        structure_generation_algorithm (str, optional): The type of algorithm
            used to generate the structure.  Default is an empty string.
            Possible values are:
                'AUTOMATIC': Calculated ROI (auto contour)
                'SEMIAUTOMATIC': ROI calculated with user assistance. (e.g.
                    isodose line as structure).
                '': Algorithm is not known.
    '''
    sup_slice: SliceIndex = np.nan
    inf_slice: SliceIndex = np.nan
    length: float = np.nan
    volume: float = np.nan
    surface_area: float = np.nan
    sphericity: float = np.nan
    resolution: float = np.nan
    resolution_type: ResolutionType = ResolutionType.NORMAL
    thickness: float = np.nan
    center_of_mass: Tuple[float, float, float] = tuple()


class Structure():
    type_to_category = {
        'GTV': StructureCategory.TARGET,
        'CTV': StructureCategory.TARGET,
        'PTV': StructureCategory.TARGET,
        'EXTERNAL': StructureCategory.EXTERNAL,
        'ORGAN': StructureCategory.ORGAN,
        'NONE': StructureCategory.ORGAN,
        'AVOIDANCE': StructureCategory.PLANNING,
        'CONTROL': StructureCategory.PLANNING,
        'TREATED_VOLUME': StructureCategory.PLANNING,
        'IRRAD_VOLUME': StructureCategory.PLANNING,
        'DOSE_REGION': StructureCategory.DOSE,
        'CONTRAST_AGENT': StructureCategory.OTHER,
        'CAVITY': StructureCategory.OTHER,
        'SUPPORT': StructureCategory.EXTERNAL,
        'BOLUS': StructureCategory.EXTERNAL,
        'FIXATION': StructureCategory.EXTERNAL
        }
    str_template = '\n'.join([
        'ID: {structure_id}',
        'ROI: {roi_number}\n',
        'DICOM Type {structure_type}',
        'Code: {structure_code}',
        'Label: {structure_code_meaning}',
        'Scheme: {structure_code_scheme}',
        'Volume: {volume} cc',
        'Length: {length} cm',
        'Range: ({sup_slice}cm, {inf_slice}cm)'
        ])

    def __init__(self, roi: ROI_Num, struct_id: str, **kwargs) -> None:
        super().__setattr__('roi_num', roi)
        super().__setattr__('id', struct_id)
        super().__setattr__('show', True)
        super().__setattr__('info', StructureInfo(roi, structure_id=struct_id))
        # Set the Structure category, required for graphs.
        if 'structure_type' in kwargs:
            category = self.type_to_category.get(kwargs['structure_type'],
                                                 StructureCategory.UNKNOWN)
            super().__setattr__('structure_category', category)
        else:
            super().__setattr__('structure_category',
                                StructureCategory.UNKNOWN)
        super().__setattr__('parameters', StructureParameters())
        self.set(**kwargs)

    # Attribute Utility methods
    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __setattr__(self, attr: str, value: Any):
        if hasattr(self.info, attr):
            self.info.__setattr__(attr, value)
        elif hasattr(self.parameters, attr):
            self.parameters.__setattr__(attr, value)
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, atr_name:str):
        if hasattr(self.info, atr_name):
            return self.info.__getattribute__(atr_name)
        if hasattr(self.parameters, atr_name):
            return self.parameters.__getattribute__(atr_name)
        super().__getattr__(atr_name)   # pylint: disable=[no-member]

    def summary(self):
        data_dict = asdict(self.info)
        data_dict.update(asdict(self.parameters))
        return self.str_template.format(**data_dict)


# %% Metric Classes
class MetricType(Enum):
    MARGIN = auto()
    DISTANCE = auto()
    OVERLAP_AREA = auto()
    OVERLAP_SURFACE = auto()
    UNKNOWN = 999  # Used for initialization


class Metric(ABC):
    # Overwrite these class variables for all subclasses.
    metric_type: MetricType
    default_format_template: str

    def __init__(self, structures: StructurePair):
        self.structures = structures
        self.metric = {}
        self.calculate_metric()
        self.format_template = self.default_format_template

    def reset_formats(self):
        self.format_template = self.default_format_template

    def format_metric(self)-> str:
        # Overwritten in some subclasses
        # Returns str: Formatted metrics for report and display
        # Default is for single '%' formatted metric value:
        params = {value_name: value
                  for value_name, value in self.metric.items()}
        display_metric = self.format_template.format(**params)
        return display_metric

    @abstractmethod
    def calculate_metric(self)-> str:
        pass


class NoMetric(Metric):
    '''A relevant metric does not exist'''
    metric_type = MetricType.UNKNOWN
    default_format_template = 'No Metric'

    def calculate_metric(self)-> str:
        self.metric = {}


class DistanceMetric(Metric):
    '''Distance metric for testing.'''
    metric_type = MetricType.DISTANCE
    default_format_template = 'Distance:\t{Distance:5.2f}'

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the distance metric function.
        self.metric = {'Distance': 2.6}


class OverlapSurfaceMetric(Metric):
    '''OverlapSurface metric for testing.'''
    metric_type = MetricType.OVERLAP_SURFACE
    default_format_template = 'Percentage Overlap:\t{OverlapSurfaceRatio:2.0%}'

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the OverlapSurface metric function.
        self.metric = {'OverlapSurfaceRatio': 0.23}


class OverlapAreaMetric(Metric):
    '''OverlapArea metric for testing.'''
    metric_type = MetricType.OVERLAP_AREA
    default_format_template = 'Percentage Overlap:\t{OverlapAreaRatio:2.0%}'

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the OverlapArea metric function.
        self.metric = {'OverlapAreaRatio': 0.15}



class ValueFormat(defaultdict):
    '''String formatting templates for individual name, value pairs.
    The default value gives a string like:
        MetricName: 2.36%
    '''
    def __missing__(self, key: str) -> str:
        format_part = ''.join([key, ':\t{', key, ':2.0%}'])
        return format_part


class MarginMetric(Metric):
    '''Margin metric for testing.'''
    metric_type = MetricType.MARGIN
    default_format_template = ''
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
    default_format_dict = {
        'SUP':  '{sup_margin:3.1f}',
        'INF':  '{inf_margin:3.1f}',
        'RT':   '{rt_margin:3.1f}',
        'LT':   '{lt_margin:3.1f}',
        'ANT':  '{ant_margin:3.1f}',
        'POST': '{post_margin:3.1f}',
        'MIN':  'Min: {min_margin:3.1f}',
        'MAX':  'Max: {max_margin:3.1f}'
        }

    def __init__(self, structures: StructurePair):
        super().__init__(structures)
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

    def calculate_metric(self)-> str:
        # FIXME replace this stub with
        # the margin metric function.
        self.metric = {
            'sup_margin':  2.0,
            'inf_margin':  2.0,
            'rt_margin':   1.5,
            'lt_margin':   1.5,
            'ant_margin':  1.5,
            'post_margin': 1.0,
            'min_margin': 2.1,
            'max_margin': 0.9}


# %% Relationship class
def compare(mpoly1: shapely.MultiPolygon,
            mpoly2: shapely.MultiPolygon)->str:
    '''Get the DE-9IM relationship string for two contours

    The relationship string is converted to binary format, where 'F'
    is '0' and '1' or '2' is '1'.

    Args:
        mpoly1 (shapely.MultiPolygon): All contours for a structure on
            a single slice.
        mpoly2 (shapely.MultiPolygon): All contours for a second
            structure on the same slice.

    Returns:
        str: A length 9 string '1's and '0's reflecting the DE-9IM
            relationship between the supplied contours.
    '''
    relation_str = shapely.relate(mpoly1, mpoly2)
    # Convert relationship string in the form '212FF1FF2' into a
    # boolean string.
    relation_bool = relation_str.replace('F','0').replace('2','1')
    return relation_bool


def relate(contour1: StructureSlice, contour2: StructureSlice)->int:
    '''Get the 27 bit relationship integer for two polygons,

    When written in binary, the 27 bit relationship contains 3 9-bit
    parts corresponding to DE-9IM relationships. The left-most 9 bits
    are the relationship between the second structure's contour and the
    first structure's convex hull polygon. The middle 9 bits are the
    relationship between the second structure's contour and the first
    structure's exterior polygon (i.e. with any holes filled). The
    right-most 9 bits are the relationship between the second
    structure's contour and the first structure's contour.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

    Returns:
        int: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between the
            second contour and the struct1 convex hull, exterior and
            contour.
    '''
    primary_relation = compare(contour1.contour, contour2.contour)
    external_relation = compare(contour1.exterior, contour2.contour)
    convex_hull_relation = compare(contour1.hull, contour2.contour)
    full_relation = ''.join([convex_hull_relation,
                                external_relation,
                                primary_relation])
    binary_relation = int(full_relation, base=2)
    return binary_relation


def relate_structures(slice_structures: pd.DataFrame, structures: StructurePair)->int:
    '''Get the 27 bit relationship integer for two structures on a given slice.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

        structures (StructurePair): A tuple of ROI numbers which index
            columns in slice_structures.
    Returns:
        int: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between the
            second contour and the struct1 convex hull, exterior and
            contour.
    '''
    structure = slice_structures[structures[0]]
    other_contour = slice_structures[structures[1]]
    binary_relation = relate(structure, other_contour)
    return binary_relation


def relate_structs(slice_table: pd.DataFrame, structures: StructurePair) -> int:
    slice_structures = slice_table.loc[:, [structures[0],
                                           structures[1]]]
    # Remove Slices that have neither structure.
    slice_structures.dropna(how='all', inplace=True)
    # For slices that have only one of the two structures, replace the nan
    # values with empty polygons for duck typing.
    slice_structures.fillna(StructureSlice([]), inplace=True)
    # Get the relationships between the two structures for all slices.
    relation_seq = slice_structures.agg(relate_structures, structures=structures,
                                        axis='columns')
     # Get the overall relationship for the two structures by merging the
    # relationships for the individual slices.
    relation_binary = merge_rel(relation_seq)
    return relation_binary


class RelationshipType(Enum):
    '''The names for defines relationship types.'''
    DISJOINT = auto()
    SURROUNDS = auto()
    SHELTERS = auto()
    BORDERS = auto()
    BORDERS_INTERIOR = auto()
    OVERLAPS = auto()
    PARTITION = auto()
    CONTAINS = auto()
    EQUALS = auto()
    LOGICAL = auto()
    UNKNOWN = 999  # Used for initialization

    def __bool__(self):
        if self == self.UNKNOWN:
            return False
        return True

    def __str__(self):
        return f'Relationship: {self.name.capitalize()}'


@dataclass()
class RelationshipTest:
    '''The test binaries used to identify a relationship type.

    Each test definitions consists of 2 27-bit binaries, a mask and a value.
    Each of the 27-bit binaries contain 3 9-bit parts associated with DE-9IM
    relationships. The left-most 9 bits are associated with the relationship
    between one structure's convex hull and another structure's contour. The
    middle 9 bits are associated with the relationship between the first
    structure's exterior polygon (i.e. with any holes filled) and the second
    structure's contour. The right-most 9 bits are associated with the
    relationship between first structure's contour and the second structure's
    contour.

    Named relationships are identified by logical patterns such as: T*T*F*FF*
        The 'T' indicates the bit must be True.
        The 'F' indicates the bit must be False.
        The '*' indicates the bit can be either True or False.
    Ane example of a complete relationship logic is:
    Surrounds (One structure resides completely within a hole in another
               structure):
        Region Test =   FF*FF****  - The contours of the two structures have no
                                     regions in common.
        Exterior Test = T***F*F**  - With holes filled, one structure is within
                                     the other.
        Hull Test =     *********  - Together, the Region and Exterior Tests
                                     sufficiently identifies the relationship,
                                     so the Hull Test is not necessary.
    The mask binary is a sequence of 0s and 1s with every '*' as a '0' and every
    'T' or 'F' bit as a '1'.  The operation: relationship_integer & mask will
    set all of the bit that are allowed to be either True or False to 0.

    The value binary is a sequence of 0s and 1s with every 'T' as a '1' and
    every '*' or 'F' bit as a '0'. The relationship is identified when value
    binary is equal to the result of the `relationship_integer & mask`
    operation.
    '''
    relation_type: RelationshipType = RelationshipType.UNKNOWN
    mask: int = 0b000000000000000000000000000
    value: int = 0b000000000000000000000000000

    def __repr__(self) -> str:
        rep_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            ' ' * 4,
            f'mask =  0b{self.mask:0>27b}\n',
            ' ' * 4,
            f'value = 0b{self.value:0>27b}'
            ])
        return rep_str

    def test(self, relation: int)->RelationshipType:
        '''Apply the defined test to the supplied relation binary.

        Args:
            relation (int): The number corresponding to a 27-bit binary of
                relationship values.

        Returns:
            RelationshipType: The RelationshipType if the test passes,
                otherwise None.
        '''
        masked_relation = relation & self.mask
        if masked_relation == self.value:
            return self.relation_type
        return None


def identify_relation(relation_binary) -> RelationshipType:
    '''Applies a collection of definitions for named relationships to a supplied
    relationship binary.

    The defined relationships are:
        Relationship      Region Test   Exterior Test   Hull Test
        Disjoint          FF*FF****     FF*FF****       FF*FF****
        Shelters          FF*FF****     FF*FF****       T***F*F**
        Surrounds         FF*FF****     T***F*F**
        Borders_Interior  FF*FT****     T***T****
        Borders           FF*FT****     FF*FT****
        Contains	      T*T*F*FF*
        Incorporates	  T*T*T*FF*
        Equals	          T*F**FFF*
        Overlaps          TTTT*TTT*

    Args:
        relation_binary (int): An integer generated from the combined DE-9IM
            tests.

    Returns:
        RelationshipType: The identified RelationshipType if one of the tests
            passes, otherwise RelationshipType.UNKNOWN.
    '''
    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.SURROUNDS,        0b000000000100010110110110000, 0b000000000100000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,         0b111000100110110000110110000, 0b111000000000000000000000000),
        RelationshipTest(RelationshipType.DISJOINT,         0b110110000110110000110110000, 0b000000000000000000000000000),
        RelationshipTest(RelationshipType.BORDERS,          0b000000000001001110110110000, 0b000000000001001110000010000),
        RelationshipTest(RelationshipType.BORDERS_INTERIOR, 0b000000000101010110110110000, 0b000000000101000000000010000),
        RelationshipTest(RelationshipType.OVERLAPS,         0b000000000000000000101000100, 0b000000000000000000101000100),
        RelationshipTest(RelationshipType.PARTITION,        0b000000000000000000101010110, 0b000000000000000000101010000),
        RelationshipTest(RelationshipType.CONTAINS,         0b000000000000000000101010110, 0b000000000000000000101000000),
        RelationshipTest(RelationshipType.EQUALS,           0b000000000000000000101001110, 0b000000000000000000100000000)
        ]
    for rel_def in test_binaries:
        result = rel_def.test(relation_binary)
        if result:
            return result
    return RelationshipType.UNKNOWN


def merge_rel(relation_seq: pd.Series)->int:
    '''Aggregate all the relationship values from each slice to obtain
        one relationship value for the two structures.

    Args:
        relation_seq (pd.Series): The relationship values between the
        contours from each slice.

    Returns:
        int: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between struct2
            and the struct1 convex hulls, exteriors and contours.
    '''
    relation_seq.drop_duplicates(inplace=True)
    merged_rel = 0
    for rel in list(relation_seq):
        merged_rel = merged_rel | rel
    return merged_rel


class Relationship():
    symmetric_relations = [
        RelationshipType.DISJOINT,
        RelationshipType.OVERLAPS,
        RelationshipType.BORDERS,
        RelationshipType.EQUALS,
        RelationshipType.UNKNOWN  # If unknown structure order is irrelevant.
        ]
    transitive_relations = [
        RelationshipType.EQUALS,
        RelationshipType.SHELTERS,
        RelationshipType.SURROUNDS,
        RelationshipType.CONTAINS,
        ]
    metric_match = {
        RelationshipType.DISJOINT: DistanceMetric,
        RelationshipType.BORDERS: OverlapSurfaceMetric,
        RelationshipType.BORDERS_INTERIOR: OverlapSurfaceMetric,
        RelationshipType.OVERLAPS: OverlapAreaMetric,
        RelationshipType.PARTITION: OverlapAreaMetric,
        RelationshipType.SHELTERS: MarginMetric,
        RelationshipType.SURROUNDS: MarginMetric,
        RelationshipType.CONTAINS: MarginMetric,
        RelationshipType.EQUALS: NoMetric,
        RelationshipType.UNKNOWN: NoMetric,
        }

    def __init__(self, structures: StructurePair,
                 slice_table: pd.DataFrame = None, **kwargs) -> None:
        self.is_logical = False
        self.show = True
        self.metric = None
        if not slice_table:
            slice_table = pd.DataFrame()
        # Sets the is_logical and metric attributes, if supplied.  Ignores any
        # other items in kwargs.
        self.set(**kwargs)
        # Order the structures with the largest first.
        self.structures = None
        self.set_structures(structures)
        # Determine the relationship type.  Either set it from a kwargs value
        # or determine it by comparing the structures.
        self.relationship_type = RelationshipType.UNKNOWN
        if 'relationship' in kwargs:
            self.relationship_type = RelationshipType[kwargs['relationship']]
        else:
            self.identify_relationship(slice_table)
        self.get_metric()

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def get_metric(self):
        # Select the appropriate metric for the identified relationship.
        metric_class = self.metric_match[self.relationship_type]
        self.metric = metric_class(self.structures)

    @property
    def is_symmetric(self)-> bool:
        return self.relationship_type in self.symmetric_relations

    @property
    def is_transitive(self)-> bool:
        return self.relationship_type in self.transitive_relations

    def set_structures(self, structures: StructurePair) -> None:
        # FIXME Stub method to be replaced with set_structures function.
        # Order the structures with the larger one first
        self.structures = structures

    def identify_relationship(self, slice_table: pd.DataFrame) -> None:
        '''Get the 27 bit relationship integer for two structures,

            When written in binary, the 27 bit relationship contains 3 9 bit
            parts corresponding to DE-9IM relationships. The left-most 9 bits
            are the relationship between the second structure's contour and the
            first structure's convex hull.  The middle 9 bits are the
            relationship between the second structure's contour and the first
            structure's exterior. (The first structure's contour with any holes
            filled). The right-most 9 bits are the relationship between the
            second structure's contour and the first structure's actual contour.

            Note: The order of structures matters. For correct comparison, the
            first structure should always be the larger of the two structures.

            Args:
                slice_structures (pd.DataFrame): A table of structures, where the
                    values are the contours with type StructureSlice. The column
                    index contains the roi numbers for the structures.  The row
                    index contains the slice index distances.
        '''
        slice_structures = slice_table.loc[:, [self.structures[0],
                                               self.structures[1]]]
        # Remove Slices that have neither structure.
        slice_structures.dropna(how='all', inplace=True)
        # For slices that have only one of the two structures, replace the nan
        # values with empty polygons for duck typing.
        slice_structures.fillna(StructureSlice([]), inplace=True)
        # Get the relationships between the two structures for all slices.
        relation_seq = slice_structures.agg(relate_structures, structures=self.structures,
                                            axis='columns')
        # Get the overall relationship for the two structures by merging the
        # relationships for the individual slices.
        relation_binary = merge_rel(relation_seq)
        self.relationship_type = identify_relation(relation_binary)
        return relation_binary


# %% Structure Set class
@dataclass
class StructureSetInfo:
    structure_set_name: str = ''
    patient_id: str = ''
    patient_name: str = ''
    patient_lastname: str = ''
    study_id: str = ''
    study_description: str = ''
    series_number: int = None
    series_description: str = ''
    orientation: str = 'HFS'
    file: Path = None


class StructureSet():
    str_template = '\n'.join([
        'ID: {structure_set_name}',
        'Patient: {patient_name} ({patient_id}}\n',
        'Image Study: {study_id}\t{study_description}\n',
        'Image Series: {series_number}, {series_description}\n',
        'Image Orientation: {orientation}\n',
        'Data taken from: {file.name}'
        ])

    def __init__(self, **kwargs) -> None:
        super().__setattr__('info', StructureSetInfo())
        super().__setattr__('structure_data', pd.DataFrame())
        self.set(**kwargs)

    # Attribute Utility methods
    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __setattr__(self, attr: str, value: Any):
        if hasattr(self.info, attr):
            self.info.__setattr__(attr, value)
        elif hasattr(self.parameters, attr):
            self.parameters.__setattr__(attr, value)
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, atr_name:str):
        if hasattr(self.info, atr_name):
            return self.info.__getattribute__(atr_name)
        if hasattr(self.parameters, atr_name):
            return self.parameters.__getattribute__(atr_name)
        super().__getattr__(atr_name)   # pylint: disable=[no-member]

    def summary(self):
        data_dict = asdict(self.info)
        data_dict.update(asdict(self.parameters))
        return self.str_template.format(**data_dict)

    # DICOM read methods
    def read_structure_set_info(self, dataset: pydicom.Dataset) -> None:
        '''Get top-level info about a RS DICOM file.

        Extracts the following info from a structure set.
            structure_set_name
            patient_id
            patient_name
            study_id
            study_description
            series_number
            series_description

        Orientation is set to a default 'HFS', because this is not available
        in RS DICOM files.

        file is currently set to None.

        Args:
            struct_file (Path): The top level dataset from a DICOM RS file.
        '''
        self.info.structure_set_name = str(dataset.get('StructureSetLabel',''))
        self.info.patient_id = str(dataset.get('PatientID',''))
        self.info.patient_name = str(dataset.get('PatientName',''))
        self.info.patient_lastname = self.patient_name.split('^')[0]
        # FIXME This appears to be getting the study and series for the
        # structure set, not the image.
        # TODO Add method to get the orientation from the image or plan file.
        # TODO consider adding method to find the plan ID

        self.info.study_id = str(dataset.get('StudyID',''))
        self.info.study_description = str(dataset.get('StudyDescription',''))
        try:
            series_number = int(dataset.get('SeriesNumber'))
        except TypeError:
            series_number = None
        self.info.series_number = series_number
        self.info.series_description = str(dataset.get('SeriesDescription',''))


# %% StructureDiagram class
class StructureDiagram:
    graph_defaults = {
        'labelloc': 't',
        'clusterrank': 'none',
        'bgcolor': '#555555',
        'fontname': 'Helvetica,,Arial,sans-serif',
        'fontsize': 16,
        'fontcolor': 'white'
        }
    node_defaults = {
        'style': 'filled',
        'width': 1,
        'height': 0.6,
        'fixedsize': 'shape',
        'fontname': 'Helvetica-Bold',
        'fontsize': 12,
        'fontcolor': 'black',
        'labelloc': 'c',
        'nojustify': True,
        'penwidth': 3,
        }
    edge_defaults = {
        'style': 'solid',
        'penwidth': 3,
        'color': '#e27dd6ff',
        'arrowhead': 'none',
        'arrowtail': 'none',
        'labelfloat': False,
        'labelfontname': 'Cambria',
        'fontsize': '10',
        'fontcolor': '#55AAFF',
        }
    node_type_formatting = {
        'GTV': {'shape': 'pentagon', 'style': 'filled', 'penwidth': 3},
        'CTV': {'shape': 'hexagon', 'style': 'filled', 'penwidth': 3},
        'PTV': {'shape': 'octagon', 'style': 'filled', 'penwidth': 3},
        'EXTERNAL': {'shape': 'doublecircle', 'style': 'filled',
                     'fillcolor': 'white','penwidth': 2},
        'ORGAN': {'shape': 'rectangle', 'style': 'rounded, filled',
                  'penwidth': 3},
        'NONE': {'shape': 'trapezium', 'style': 'rounded, filled',
                 'penwidth': 3},
        'AVOIDANCE': {'shape': 'house', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CONTROL': {'shape': 'invhouse', 'style': 'rounded, filled',
                 'penwidth': 3},
        'TREATED_VOLUME': {'shape': 'parallelogram', 'style': 'rounded, filled',
                 'penwidth': 3},
        'IRRAD_VOLUME': {'shape': 'parallelogram', 'style': 'rounded, filled',
                 'penwidth': 3},
        'DOSE_REGION': {'shape': 'diamond', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CONTRAST_AGENT': {'shape': 'square', 'style': 'rounded, filled',
                 'penwidth': 3},
        'CAVITY': {'shape': 'square', 'style': 'rounded, filled',
                 'penwidth': 3},
        'SUPPORT': {'shape': 'triangle', 'style': 'rounded, bold',
                 'penwidth': 3},
        'BOLUS': {'shape': 'oval', 'style': 'bold', 'penwidth': 3},
        'FIXATION': {'shape': 'diamond', 'style': 'bold', 'penwidth': 3},
        }
    edge_type_formatting = {
        RelationshipType.DISJOINT: {'label': 'Disjoint', 'style': 'invis'},
        RelationshipType.SURROUNDS: {'label': 'Island', 'style': 'tapered',
                                     'dir': 'forward', 'penwidth': 3,
                                     'color': 'blue'},
        RelationshipType.SHELTERS: {'label': 'Shelter', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 3, 'color': 'blue'},
        RelationshipType.BORDERS: {'label': 'Borders', 'style': 'dashed',
                                   'dir': 'both', 'penwidth': 3,
                                   'color': 'green'},
        RelationshipType.BORDERS_INTERIOR: {'label': 'Cut-out', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 3,
                                    'color': 'magenta'},
        RelationshipType.OVERLAPS: {'label': 'Overlaps', 'style': 'tapered',
                                    'dir': 'both', 'penwidth': 6,
                                    'color': 'green'},
        RelationshipType.PARTITION: {'label': 'Group', 'style': 'tapered',
                                        'dir': 'forward', 'penwidth': 6,
                                        'color': 'white'},
        RelationshipType.CONTAINS: {'label': 'Contains', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 6,
                                    'color': 'cyan'},
        RelationshipType.EQUALS: {'label': 'Equals', 'style': 'both',
                                  'dir': 'both', 'penwidth': 5, 'color': 'red'},
        RelationshipType.LOGICAL: {'label': '', 'style': 'dotted',
                                   'penwidth': 0.5, 'color': 'yellow'},
        RelationshipType.UNKNOWN: {'label': '', 'style': 'invis'},
        }

    # The Formatting style for hidden structures and relationships
    hidden_node_format = {'shape': 'point', 'style': 'invis'}
    hidden_edge_format = {'style': 'invis'}
    logical_edge_format = {'style': 'dotted', 'penwidth': 0.5,
                           'color': 'yellow'}

    def __init__(self, name=r'Structure Relations') -> None:
        self.title = name
        self.display_graph = pgv.AGraph(label=name, **self.graph_defaults)
        self.display_graph.node_attr.update(self.node_defaults)
        self.display_graph.edge_attr.update(self.edge_defaults)

    @staticmethod
    def rgb_to_hex(rgb_tuple: Tuple[int,int,int]) -> str:
        '''Convert an RGB tuple to a hex string value.

        Args:
            rgb_tuple (Tuple[int,int,int]): A length-3 tuple of integers from 0 to
                255 corresponding to the RED, GREEN and BLUE color channels.

        Returns:
            str: The equivalent Hex color string.
        '''
        return '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)

    @staticmethod
    def text_color(color_rgb: Tuple[int])->Tuple[int]:
        '''Determine the appropriate text color for a given background color.

        Text color is either Black (0, 0, 0) or white (255, 255, 255)
        the cutoff between black and white is given by:
            brightness > 274.3 and green > 69
            (brightness is the length of the color vector $sqrt{R^2+G^2+B62}$

        Args:
            color_rgb (Tuple[int]): The 3-integer RGB tuple of the background color.

        Returns:
            Tuple[int]: The text color as an RGB tuple.
                One of (0, 0, 0) or (255, 255, 255).
        '''
        red, green, blue = color_rgb
        brightness = sqrt(red**2 + green**2 + blue**2)
        if brightness > 274.3:
            if green > 69:
                text_color = '#000000'
            else:
                text_color = '#FFFFFF'
        elif green > 181:
            text_color = '#000000'
        else:
            text_color = '#FFFFFF'
        return text_color

    def add_structure_nodes(self, structures: List[Structure]):
        structure_groups = defaultdict(list)
        for structure in structures:
            node_type = structure.structure_type
            node_id = structure.roi_num
            node_formatting = self.node_type_formatting[node_type].copy()
            node_text_color = self.text_color(structure.color)
            node_formatting['label'] = structure.info.structure_id
            node_formatting['color'] = self.rgb_to_hex(structure.color)
            node_formatting['fontcolor'] = node_text_color
            node_formatting['tooltip'] = structure.summary()
            if not structure.show:
                node_formatting.update(self.hidden_node_format)
            self.display_graph.add_node(node_id, **node_formatting)
            # Identify the subgroup that the node belongs to.
            group = structure.structure_category
            structure_groups[group].append(structure.roi_num)
        # Define the subgroups.
        for name, group_list in structure_groups.items():
            self.display_graph.add_subgraph(group_list, name=str(name),
                                            cluster=True)

    def add_structure_edges(self, relationships: List[Relationship]):
        for relationship in relationships:
            node1, node2 = relationship.structures
            edge_type = relationship.relationship_type
            edge_formatting = self.edge_type_formatting[edge_type].copy()
            edge_formatting['tooltip'] = relationship.metric.format_metric()
            # Override formatting
            hide_edge = any(self.is_hidden(node)
                            for node in relationship.structures)
            if (not relationship.show) | hide_edge:
                edge_formatting.update(self.hidden_edge_format)
            elif relationship.is_logical:
                edge_formatting.update(self.logical_edge_format)
            self.display_graph.add_edge(node1, node2, **edge_formatting)

    def node_attr(self, node_id: int)->Dict[str, any]:
        return self.display_graph.get_node(node_id).attr

    def is_hidden(self, node_id: int)->bool:
        node_attributes = self.node_attr(node_id)
        node_style = node_attributes.get('style', '')
        return 'invis' in node_style

# %% Debugging Display functions
# Eventually move these functions to their own module
