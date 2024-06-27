'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Setup
# Imports
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


# Type definitions

ROI_Num = int  # Index to structures defined in Structure RT DICOM file
SliceIndex = float
Contour = shapely.Polygon
StructureSlice = shapely.MultiPolygon
StructurePair =  Tuple[ROI_Num, ROI_Num]


# Global Settings
PRECISION = 3


class ResolutionType(Enum):
    Normal = 1
    High = 2


class StructureCategory(Enum):
    Target = auto()
    OAR = auto()
    External = auto()
    Other = auto()


class MetricType(Enum):
    Margin = auto()
    Distance = auto()
    OverlapArea = auto()
    OverlapSurface = auto()


class ValueFormat(defaultdict):
    '''String formatting templates for individual name, value pairs.
    The default value gives a string like:
        MetricName: 2.36%
    '''
    def __missing__(self, key: str) -> str:
        format_part = ''.join([key, ':\t{', key, 'f2.0%}'])
        return format_part


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
        super().__init__(self)
        self.structure_set_name = str(dataset.get('StructureSetLabel',''))
        self.patient_id = str(dataset.get('PatientID',''))
        self.patient_name = str(dataset.get('PatientName',''))
        self.patient_lastname = self.patient_name.split('^')[0]
        self.study_id = str(dataset.get('StudyID',''))
        self.study_description = str(dataset.get('StudyDescription',''))
        try:
            series_number = int(dataset.get('SeriesNumber'))
        except TypeError:
            series_number = None
        self.series_number = series_number
        self.series_description = str(dataset.get('SeriesDescription',''))


@dataclass
class StructureInfo:
    roi_number: ROI_Num
    structure_id: str = ''
    structure_name: str = ''
    structure_category: StructureCategory = None
    structure_type: str = ''  # DICOM Structure Type
    color: Tuple[int,int,int] = (255,255,255)  # Default color is white
    structure_code: str = ''
    structure_code_scheme: str = ''
    structure_code_meaning: str = ''
    structure_generation_algorithm: str = ''
    structure_density: float = None


@dataclass
class StructureParameters:
    sup_slice: SliceIndex = None
    inf_slice: SliceIndex = None
    length: float = None
    volume: float = None
    surface_area: float = None
    sphericity: float = None
    resolution: float = None
    resolution_type: ResolutionType = ResolutionType.Normal
    thickness: float = None
    center_of_mass: Tuple[float, float, float] = None


class Structure:
    def __init__(self, roi: ROI_Num, name: str, **kwargs) -> None:
        self.roi_num = roi
        self.name = name
        self.info = StructureInfo(roi, structure_name=name)
        self.parameters = StructureParameters()
        self.info_display_parameters = {}
        self.diagram_display_parameters = {}


    def __setattribute__(self, **kwargs):
        for key, val in kwargs:
            if hasattr(self.info, key):
                self.info.__setattr__(key, val)
            if hasattr(self.parameters, key):
                self.parameters.__setattr__(key, val)

    def __getattribute__(self, atr_name:str):
        if hasattr(self.info, atr_name):
            return self.info.__getattribute__(atr_name)
        if hasattr(self.parameters, atr_name):
            return self.parameters.__getattribute__(atr_name)
        return None


class Metric(ABC):
    # Overwrite these class variables for all subclasses.
    metric_type: MetricType
    default_format = ValueFormat()
    default_format_template = ''

    def __init__(self, structures: StructurePair):
        self.structures = structures
        self.metric = {}
        self.calculate_metric()
        self.format_dict = {}
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
        for key in self.metric.keys():
            self.format_dict[key] = self.default_format[key]
        if not self.format_template:
            self.format_template = '\n'.join(self.format_dict.values())

    @abstractmethod
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
