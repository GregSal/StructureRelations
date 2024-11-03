'''Structure
'''
# %% Imports
# Type imports

from typing import List, Tuple, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Standard Libraries
from pathlib import Path
from itertools import chain
from math import ceil, sin, cos, radians, sqrt


# Shared Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
import pydicom
from shapely.plotting import plot_polygon, plot_line

from types_and_classes import ROI_Type, SliceIndexType, ContourType, StructurePairType, poly_round
from types_and_classes import InvalidContour

from types_and_classes import StructureSlice

# Global Default Settings
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
        roi_number (ROI_Type): The unique indexing reference to the structure.
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
    roi_number: ROI_Type
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
        sup_slice (SliceIndexType, optional): The SUP-most CT slice offset
            containing contour point for the structure. Default is np.nan:
        inf_slice (SliceIndexType, optional): The INF-most CT slice offset
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
    sup_slice: SliceIndexType = np.nan
    inf_slice: SliceIndexType = np.nan
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

    def __init__(self, roi: ROI_Type, struct_id: str, **kwargs) -> None:
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
