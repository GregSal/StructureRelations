'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import Any, Dict, Tuple
from enum import Enum, auto
from dataclasses import dataclass

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
import PySimpleGUI as sg
import pydicom
import shapely


# %%| Type definitions and Globals
ROI_Num = int  # Index to structures defined in Structure RT DICOM file
SliceIndex = float
Contour = shapely.Polygon
StructureSlice = shapely.MultiPolygon
StructurePair =  Tuple[ROI_Num, ROI_Num]


# Global Settings
PRECISION = 3


# %% Enumeration Types
class ResolutionType(Enum):
    NORMAL = 1
    HIGH = 2
    UNKNOWN = 999  # Used for initialization


class StructureCategory(Enum):
    TARGET = auto()
    OAR = auto()
    EXTERNAL = auto()
    OTHER = auto()
    UNKNOWN = 999  # Used for initialization


class RelationshipType(Enum):
    DISJOINT = auto()
    OVERLAPS = auto()
    BORDERS = auto()
    EQUALS = auto()
    SHELTERS = auto()
    SURROUNDS = auto()
    CONTAINS = auto()
    INCORPORATES = auto()
    INTERIOR_BORDERS = auto()
    UNKNOWN = 999  # Used for initialization


class MetricType(Enum):
    MARGIN = auto()
    DISTANCE = auto()
    OVERLAP_AREA = auto()
    OVERLAP_SURFACE = auto()
    UNKNOWN = 999  # Used for initialization


# %% Dataclass Types
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


@dataclass
class StructureInfo:
    roi_number: ROI_Num
    structure_id: str
    structure_name: str = ''
    structure_category = StructureCategory.UNKNOWN
    structure_type: str = ''  # DICOM Structure Type
    color: Tuple[int,int,int] = (255,255,255)  # Default color is white
    structure_code: str = ''
    structure_code_scheme: str = ''
    structure_code_meaning: str = ''
    structure_generation_algorithm: str = ''
    structure_density: float = np.nan


@dataclass
class StructureParameters:
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


# %% class definitions
class ValueFormat(defaultdict):
    '''String formatting templates for individual name, value pairs.
    The default value gives a string like:
        MetricName: 2.36%
    '''
    def __missing__(self, key: str) -> str:
        format_part = ''.join([key, ':\t{', key, 'f2.0%}'])
        return format_part


class Metric(ABC):
    # Overwrite these class variables for all subclasses.
    metric_type: MetricType
    default_format = ValueFormat()
    default_format_template = ''

    def __init__(self, structures: StructurePair):
        self.structures = structures
        self.metric = {}
        self.calculate_metric()
        self.format_dict = self.default_format.copy()
        self.format_template = self.default_format_template
        self.set_formats()

    def set_formats(self):
        '''Set the default text formatting for displaying metrics.

        Generic self.default_format has a default value of:
            <MetricName>:\t{<MetricValue>f2.0%}])
        creates a string format with a line with the default_format for each
        metric in self.metric.

        Primarily modified by removing entries to exclude metric values from
        being displayed. To restore a metric the relevant display key value
        air is added from default_format.

        Overwrite for subclasses as necessary.
        '''
        for key in self.metric:
            self.format_dict[key] = self.default_format[key]
        if not self.format_template:
            self.format_template = '\n'.join(self.format_dict.values())

    def format_metric(self)-> str:
        # Overwritten in some subclasses
        # Returns str: Formatted metrics for report and display
        # Default is for single '%' formatted metric value:
        params = {value_name: value for value_name, value in self.metric}
        display_metric = self.format_template.format(**params)
        return display_metric

    @abstractmethod
    def calculate_metric(self)-> str:
        pass


class NoMetric(Metric):
    '''A relevant metric does not exist'''
    metric_type = MetricType.UNKNOWN
    default_format = ValueFormat()
    default_format_template = 'No Metric'

    def calculate_metric(self)-> str:
        self.metric = {}


class DistanceMetric(Metric):
    '''Distance metric for testing.'''
    metric_type = MetricType.DISTANCE
    default_format = ValueFormat()
    default_format_template = ''

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the distance metric function.
        self.metric = {'Distance': 1.0}


class OverlapSurfaceMetric(Metric):
    '''OverlapSurface metric for testing.'''
    metric_type = MetricType.OVERLAP_SURFACE
    default_format = ValueFormat()
    default_format_template = ''

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the OverlapSurface metric function.
        self.metric = {'OverlapSurfaceRatio': 1.0}


class OverlapAreaMetric(Metric):
    '''OverlapArea metric for testing.'''
    metric_type = MetricType.OVERLAP_AREA
    default_format = ValueFormat()
    default_format_template = ''

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the OverlapArea metric function.
        self.metric = {'OverlapAreaRatio': 1.0}


class MarginMetric(Metric):
    '''Margin metric for testing.'''
    metric_type = MetricType.MARGIN
    default_format = ValueFormat()
    default_format_template = ''

    def calculate_metric(self)-> str:
        # FIXME replace this stub with the OverlapArea metric function.
        self.metric = {'PerpendicularMargin': {'SUP':  1.0,
                                               'INF':  1.0,
                                               'RT':   1.0,
                                               'LT':   1.0,
                                               'ANT':  1.0,
                                               'POST': 1.0},
                       'MinimumDistance': 1.0,
                       'MaximumDistance': 1.0}


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
        RelationshipType.INTERIOR_BORDERS: OverlapSurfaceMetric,
        RelationshipType.OVERLAPS: OverlapAreaMetric,
        RelationshipType.INCORPORATES: OverlapAreaMetric,
        RelationshipType.SHELTERS: MarginMetric,
        RelationshipType.SURROUNDS: MarginMetric,
        RelationshipType.CONTAINS: MarginMetric,
        RelationshipType.EQUALS: NoMetric,
        RelationshipType.UNKNOWN: NoMetric,
        }


    def __init__(self, structures: StructurePair) -> None:
        self.relationship_type = RelationshipType.UNKNOWN
        self.structures = None
        # Order the structures with the largest first.
        self.set_structures(structures)
        self.identify_relationship()
        # Select the appropriate metric for the identified relationship.
        metric_class = self.metric_match[self.relationship_type]
        self.metric = metric_class(structures)

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


class Structure():
    def __init__(self, roi: ROI_Num, name: str, **kwargs) -> None:
        super().__setattr__('roi_num', roi)
        super().__setattr__('id', name)
        super().__setattr__('info', StructureInfo(roi, structure_id=name))
        super().__setattr__('parameters', StructureParameters())
        super().__setattr__('info_display_parameters', {})
        super().__setattr__('diagram_display_parameters', {})
        self.set(**kwargs)

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
        super().__getattr__(atr_name)


class StructureSet():
    def __init__(self, **kwargs) -> None:
        self.info = StructureSetInfo()
        self.set(**kwargs)

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

    def __read_structure_set_info__(self, dataset: pydicom.Dataset) -> None:
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


#%% Tuples to Strings
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
