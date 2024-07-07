'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports

from typing import Any, Dict, List, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Standard Libraries
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from math import sqrt, pi
from statistics import mean
from itertools import zip_longest

# Shared Packages
import numpy as np
import pandas as pd
import xlwings as xw
import pydicom
import shapely
import pygraphviz as pgv
import networkx as nx


# %%| Type definitions and Globals
ROI_Num = int  # Index to structures defined in Structure RT DICOM file
SliceIndex = float
Contour = shapely.Polygon
StructureSlice = shapely.MultiPolygon
StructurePair =  Tuple[ROI_Num, ROI_Num]


# Global Settings
PRECISION = 3


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
class RelationshipType(Enum):
    DISJOINT = auto()
    SURROUNDS = auto()
    SHELTERS = auto()
    BORDERS = auto()
    CONFINES = auto()
    OVERLAPS = auto()
    INCORPORATES = auto()
    CONTAINS = auto()
    EQUALS = auto()
    LOGICAL = auto()
    UNKNOWN = 999  # Used for initialization


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
        RelationshipType.INCORPORATES,
        ]
    metric_match = {
        RelationshipType.DISJOINT: DistanceMetric,
        RelationshipType.BORDERS: OverlapSurfaceMetric,
        RelationshipType.CONFINES: OverlapSurfaceMetric,
        RelationshipType.OVERLAPS: OverlapAreaMetric,
        RelationshipType.INCORPORATES: OverlapAreaMetric,
        RelationshipType.SHELTERS: MarginMetric,
        RelationshipType.SURROUNDS: MarginMetric,
        RelationshipType.CONTAINS: MarginMetric,
        RelationshipType.EQUALS: NoMetric,
        RelationshipType.UNKNOWN: NoMetric,
        }

    def __init__(self, structures: StructurePair, **kwargs) -> None:
        self.is_logical = False
        self.show = True
        self.metric = None
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
            self.identify_relationship()

        self.get_metric()

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

    def identify_relationship(self) -> None:
        # FIXME Stub method to be replaced with identify_relationship function.
        # Re-order structures as necessary for for Surrounds and Shelters
        self.relationship_type = RelationshipType.UNKNOWN

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

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
    '''
    ContourData: Table
	Index: AutoInteger
	Columns: 
		ROI_Num, 
		SliceIndex,
		Area,
		Contour
Generated by: Read Contour Data

StructureData: Series:
	Index: ROI_Num, SliceIndex
	Values: StructureSlice
Generated by: Build StructureSet
    '''
    def __init__(self, **kwargs) -> None:
        self.info = StructureSetInfo()
        self.set(**kwargs)

    # Attribute Utility methods
    def set(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

    def __setattr__(self, attr: str, value: Any):
        if hasattr(self.info, attr):
            self.info.__setattr__(attr, value)
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, atr_name:str):
        if hasattr(self.info, atr_name):
            return self.info.__getattribute__(atr_name)
        getattr(self, atr_name)

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
        RelationshipType.CONFINES: {'label': 'Cut-out', 'style': 'tapered',
                                    'dir': 'forward', 'penwidth': 3,
                                    'color': 'magenta'},
        RelationshipType.OVERLAPS: {'label': 'Overlaps', 'style': 'tapered',
                                    'dir': 'both', 'penwidth': 6,
                                    'color': 'green'},
        RelationshipType.INCORPORATES: {'label': 'Group', 'style': 'tapered',
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

# %% Future functions
# Tuples to Strings
def colour_text(roi_colour):
    colour_fmt = ''.join([
        f'({roi_colour[0]:0d}, ',
        f'{roi_colour[1]:0d}, ',
        f'{roi_colour[2]:0d})'
        ])
    return colour_fmt


def com_text(com):
    com_fmt = ''.join([
        f'({com[0]:-5.2f}, ',
        f'{com[1]:-5.2f}, ',
        f'{com[2]:-5.2f})'
        ])
    return com_fmt
