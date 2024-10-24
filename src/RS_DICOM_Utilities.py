'''Structures from DICOM files'''
# %% Setup
# Imports
# Type imports
from typing import Any, Dict, Tuple

# Standard Libraries
from pathlib import Path
from math import sqrt, pi
from statistics import mean
from itertools import zip_longest

# Shared Packages
import numpy as np
import pandas as pd
import xlwings as xw
import PySimpleGUI as sg
import pydicom
from shapely.geometry import Polygon


# Global Settings
PRECISION = 3

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


# %% File Selection Functions
def get_structure_file_info(struct_file: Path)->Dict[str, Any]:
    '''Get top-level info about a RS DICOM file.

    Extract sufficient info to identify a structure set.
    Returns a dictionary with the following items:
        - PatientName
        - PatientLastName
        - PatientID
        - StructureSet
        - StudyID
        - SeriesNumber
        - File

    Args:
        struct_file (Path): The full path to the DICOM RS file.

    Returns:
        Dict[str, Any]: a dictionary with sufficient info to identify a
            structure set, or an empty dictionary if the file is not a DICOM
            Structure file.
    '''
    try:
        dataset = pydicom.dcmread(struct_file)
    except pydicom.errors.InvalidDicomError:
        return {}
    if 'RTSTRUCT' in dataset.Modality:
        full_name = str(dataset.get('PatientName',''))
        last_name = full_name.split('^')[0]
        structure_set_info = {
            'PatientName': full_name,
            'PatientLastName': last_name,
            'PatientID': str(dataset.get('PatientID','')),
            'StructureSet': str(dataset.get('StructureSetLabel','')),
            'StudyID': str(dataset.get('StudyID','')),
            'SeriesNumber': str(dataset.get('SeriesNumber','')),
            'File': struct_file}
    else:
        structure_set_info = {}
    return structure_set_info


# %% DICOM Extraction Functions
def get_names_nums(struct_dataset: pydicom.Dataset)->pd.Series:
    '''Build lookup table of ROI number and ROI ID.

    Args:
        struct_dataset (pydicom.Dataset): The full dataset for an RS DICOM file.

    Returns:
        pd.Series: A lookup table of ROI number and ROI ID.
    '''
    roi_id = pd.Series({roi.ROINumber: roi.ROIName
                        for roi in struct_dataset.StructureSetROISequence})
    roi_id.name = 'StructureID'
    return roi_id


def get_gen_alg(struct_dataset: pydicom.Dataset)->pd.Series:
    '''Get the name of the generation algorithm for all structures.

    Args:
        struct_dataset (pydicom.Dataset): The full dataset for an RS DICOM file.

    Returns:
        pd.Series: A lookup table of ROI number and the corresponding
            Generation Algorithm.
    '''
    roi_gen = pd.Series({roi.ROINumber: roi.ROIGenerationAlgorithm
                        for roi in struct_dataset.StructureSetROISequence})
    roi_gen.name = 'GenerationAlgorithm'
    return roi_gen


def get_roi_labels(struct_dataset: pydicom.Dataset)->pd.DataFrame:
    '''Get label information about each structure in the RS DICOM file.

    The table index is the ROI Number.  The table columns are:
    - StructureID
    - StructureName
    - DICOM_Type
    - Code
    - CodeScheme
    - CodeMeaning
    - GenerationAlgorithm

    Args:
        struct_dataset (pydicom.Dataset): The full dataset for an RS DICOM file.

    Returns:
        pd.DataFrame: A table of Structure information with the corresponding
            ROI number.
    '''
    label_list = []
    roi_id = get_names_nums(struct_dataset)
    roi_gen = get_gen_alg(struct_dataset)
    obs_seq = struct_dataset.get('RTROIObservationsSequence')
    if obs_seq:
        for roi in struct_dataset.RTROIObservationsSequence:
            code_seq = roi.get('RTROIIdentificationCodeSequence')
            if code_seq:
                roi_label = code_seq[0]
                label_list.append({
                    'ROINumber': roi.ReferencedROINumber,
                    'StructureName': roi.ROIObservationLabel,
                    'DICOM_Type': roi.RTROIInterpretedType,
                    'Code': roi_label.CodeValue,
                    'CodeScheme': roi_label.CodingSchemeDesignator,
                    'CodeMeaning': roi_label.CodeMeaning
                    })
            roi_labels = pd.DataFrame(label_list)
            roi_labels.set_index('ROINumber', inplace=True)
            contour_labels = pd.concat([roi_id, roi_labels, roi_gen],
                                       axis='columns')
    else:
        contour_labels = pd.DataFrame()
    return contour_labels


# %% Contour Classes
class ContourSlice():
    '''Contours related to a specific structure at a specific slice.

    Class Attribute:
        default_thickness (float): The default slice thickness in cm assigned
            to any structure containing only one slice.

    Attributes:
        axial_position (float): The offset of the ContourSlice in the axial
            (Z) direction to the DICOM origin.
        thickness (float): The distance in cm to the next ContourSlice.
            to any structure containing only one slice. If the composite
            structure contains only one slice, default_thickness is used.
        contour (Polygon, MultiPolygon): The shapley polygon(s) defining the
            structure on the slice.
        exterior (Polygon): The solid shapley polygon covering the
            structure on the slice.  If the structure is solid, this will be
            identical to the contour.
        region_count (int): The number of distinct regions on the same slice.
        is_solid (bool): If False, one of the polygons completely surrounds a
            region which is not included as part of that polygon. i.g the
            polygon contains at least one hole.
    '''
    default_thickness = 0.2  # Default slice thickness in cm
    # This is assigned to ContourSlice rather than to ContourSet because the
    # slice thickness may vary between slices.
    precision = PRECISION  # The number of digits to use for measurement values.

    def __init__(self, contour_points: np.array):
        '''Create a new contour slice.

        A new contour starts as a single shapely polygon. By default it starts
        as a simple polygon.  The polygon is built from a nx3 numpy array.

        Args:
            contour_points (np.array): An nx3 numpy array that defines the
                points of a simple polygon.  The order of the points in the
                array matter: A set of lines connecting each point with the
                next must not contain any lines that cross.

        Raises:
            ValueError: When the points in the array do not form a simply
                polygon. i.e. A set of lines connecting each point with the
                next contains two lines that cross.
        '''
        shp = Polygon(contour_points)
        if not shp.is_valid:
            raise ValueError('Invalid Contour points')
        # Normalize orders the polygon coordinates in a standard way to simplify
        # comparisons.
        self.contour = shp.normalize()
        self.axial_position = round(contour_points[0,2], self.precision)
        self.thickness = self.default_thickness
        self.region_count = 1
        self.is_solid = True
        self.exterior = self.contour  # A single contour is always its exterior.

    @property
    def area(self)->float:
        '''The area of the composite polygon(s).

        Returns:
            float: The area of the composite polygon(s)
        '''
        return round(self.contour.area, self.precision)

    @property
    def areas(self) -> list:
        '''The areas of each polygon in the slice.

        Returns:
            List[float]: A list of the areas of each polygon in the slice.
        '''
        area_list = []
        if self.contour.geometryType() == 'Polygon':
            area = round(self.contour.area, self.precision)
            area_list.append(area)
        else:
            for contour in self.contour.geoms:
                area = round(contour.area, self.precision)
                area_list.append(area)
        return area_list

    @property
    def radius(self) -> float:
        '''The "circular" radius of the polygon

        The circular radius calculation is only correct if the ContourSlice
        contains a single polygon.  If not use radii instead to obtain a list
        of radii for each polygon in the slice.

        Returns:
            (float, None): The "circular" radius of the polygon if the
                ContourSlice contains only one contour (polygon), otherwise
                returns None.
        '''
        if self.contour.geometryType() == 'Polygon':
            return round(sqrt(self.contour.area / pi), self.precision)
        return None

    @property
    def radii(self):
        '''A list of the "circular" radii of each polygon in the ContourSlice.

        Returns:
            List[float]: A list of the "circular" radii of each polygon in the
                ContourSlice.
        '''
        radius_list = []
        if self.contour.geometryType() == 'Polygon':
            radius = round(sqrt(self.contour.area / pi), self.precision)
            radius_list.append(radius)
        else:
            for contour in self.contour.geoms:
                radius = round(sqrt(contour.area / pi), self.precision)
                radius_list.append(radius)
        return radius_list

    @property
    def centre_of_mass(self)->Tuple[float]:
        '''The geometric center of the composite polygon(s).

        Returns:
            Tuple[float]: The area of the composite polygon(s)
        '''
        com =  [round(num, self.precision)
                for num in list(self.contour.centroid.coords[0])]
        com.append(self.axial_position)
        return tuple(com)

    def centers(self):
        '''A list of the geometric centers of each polygon in the ContourSlice.

        Returns:
            List[Tuple[float]]: A list of the geometric centers of each polygon
            in the ContourSlice.
        '''
        centre_list = []
        if self.contour.geometryType() == 'Polygon':
            coord = list(self.contour.centroid.coords[0])
            coord.append(self.axial_position)
            centre_list.append(tuple(round(c, self.precision) for c in coord))
        else:
            for contour in self.contour.geoms:
                coord = list(contour.centroid.coords[0])
                coord.append(self.axial_position)
                centre_list.append(tuple(round(c, self.precision) for c in coord))
        return centre_list

    @property
    def perimeter(self)->float:
        '''The combined perimeters of the polygon(s) in the ContourSlice.

        Returns:
            float: The combined perimeters of the polygon(s) in the ContourSlice.
        '''
        return round(self.contour.length, self.precision)

    @property
    def perimeters(self) -> list:
        '''The perimeters of each polygon in the slice.

        Returns:
            List[float]: A list of the perimeters of each polygon in the slice.
        '''
        perimeter_list = []
        if self.contour.geometryType() == 'Polygon':
            perimeter = round(self.contour.length, self.precision)
            perimeter_list.append(perimeter)
        else:
            for contour in self.contour.geoms:
                perimeter = round(contour.length, self.precision)
                perimeter_list.append(perimeter)
        return perimeter_list

    @property
    def resolution(self)->float:
        '''Average number of points per contour length.'''
        res_list = []
        if self.contour.geom_type == 'MultiPolygon':
            for poly in self.contour.geoms:
                length = poly.exterior.length
                num_points = len(poly.exterior.coords) - 1
                res = num_points / length
                res_list.append(res)
                for contour in poly.interiors:
                    length = contour.length
                    num_points = len(contour.coords) - 1
                    res = num_points / length
                    res_list.append(res)
        else:
            length = self.contour.exterior.length
            num_points = len(self.contour.exterior.coords) - 1
            res = num_points / length
            res_list.append(res)
            for contour in self.contour.interiors:
                length = contour.length
                num_points = len(contour.coords) - 1
                res = num_points / length
                res_list.append(res)
        mean_res = mean(res_list)
        return round(mean_res, self.precision)

    def combine(self, other: "ContourSlice"):
        '''Add an additional contour for the same structure on the same slice.

        If on contour is inside the other, the interior contour is treated as a
        hole.  If the contours do not overlap at all they are treated as
        separate regions of the same structure.

        If the new contour surrounds the original contour(s), define the new
        contour as the exterior and the original contour(s) as holes.

        Args:
            other (ContourSlice): The additional contour.

        Raises:
            ValueError: If the additional contour is for a different slice.
                Or if the two contours overlap, but one is not contained within
                the other.
        '''
        if not other.axial_position == self.axial_position:
            raise ValueError("Can't combine contours from different slices")
        other_contour = other.contour
        # Check for non-overlapping structures
        if self.contour.disjoint(other_contour):
            # non-overlapping structures
            self.contour = self.contour.union(other_contour)
            self.region_count += 1  # Increment the number of separate regions.
        elif self.contour.contains(other_contour):
            # self contains other
            self.contour = self.contour.difference(other_contour)
            self.is_solid = False
        elif self.contour.within(other_contour):
            # other contains self
            self.exterior = other_contour # The exterior contour contains everything else
            self.contour = other_contour.difference(self.contour)
            self.is_solid = False
        else:
            raise ValueError('Cannot merge overlapping contours.')

    def __repr__(self) -> str:
        if self.is_solid:
            form = 'Solid'
        else:
            form = 'Hollow'
        if self.region_count > 1:
            quantity = 'Multi'
        else:
            quantity = 'Single'
        desc = ''.join([
            f'{quantity} {form} ContourSlice, at slice {self.axial_position}, ',
            f'containing {(self.contour.geom_type)}'
            ])
        return desc


class ContourSet():
    '''All contours for a given structure.
    '''
    def __init__(self, structure_id: str = None,
                 roi_num: int = None, end_effect='Interpolate') -> None:
        self.structure_id = structure_id
        self.roi_num = roi_num
        self.end_effect = end_effect
        self.contours: Dict[float, ContourSlice] = {}
        self.empty = True
        # Initialize summary parameters
        self.color = None
        self.volume: float = None
        self.spherical_radius: float = None
        self.center_of_mass: float = None
        self.resolution: float = None
        self.resolution_type = 'Normal'
        self.sup_slice: float = None
        self.inf_slice: float = None
        self.length: float = None

    @property
    def info(self):
        contour_info = {
            'ROI Num': self.roi_num,
            'StructureID': self.structure_id,
            'Sup Slice': self.sup_slice,
            'Inf Slice': self.inf_slice,
            'Length': self.length,
            'Thickness': round(self.length / len(self.contours), PRECISION),
            'Volume': self.volume,
            'Eq Sp Diam': self.spherical_radius * 2,
            'Center of Mass': self.center_of_mass,
            'Resolution': self.resolution,
            'Colour': self.color
            }
        return contour_info

    def add_contour(self, contour: ContourSlice):
        slice_position = contour.axial_position
        if slice_position in self.contours:
            new_contour = self.contours[slice_position]
            new_contour.combine(contour)
            self.contours[slice_position] = new_contour
        else:
            self.contours[slice_position] = contour
        self.empty = False

    def finalize(self):
        def sort_slices():
            slices = list(self.contours.keys())
            if len(slices) > 1:
                slices.sort()
                gaps = list(np.diff(slices))
                gaps.append(gaps[-1])
                new_dict = {}
                for gap, slc in zip(gaps, slices):
                    contour = self.contours[slc]
                    contour.thickness = round(gap, PRECISION)
                    new_dict[slc] = contour
                self.contours = new_dict
            else:
                self.contours[slices[0]].thickness = 0
            self.sup_slice = max(slices)
            self.inf_slice = min(slices)
            self.length = self.sup_slice - self.inf_slice

        def calculate_volume():
            slices = list(self.contours.keys())
            slices.sort()
            volume = 0
            area_sum = 0
            com_list = []
            for slc, next_slice in zip_longest(slices, slices[1:]):
                thickness = self.contours[slc].thickness
                area1 = self.contours[slc].area
                area_sum += area1
                com = np.array(self.contours[slc].centre_of_mass) * area1
                com_list.append(com)
                if next_slice:
                    area2 = self.contours[next_slice].area
                    slice_volume = (area1 + area2) / 2 * thickness
                else:
                    if self.end_effect == 'Extend':
                        # Structure as continues beyond end slice.
                        slice_volume = area1 * thickness
                    elif self.end_effect == 'Interpolate':
                        # Area goes to zero halfway past end slice.
                        slice_volume = area1 / 2 * thickness
                    else:  # 'Truncate'
                        # Volume ends at last slice
                        slice_volume = 0
                volume += slice_volume
            # Add required volume below lowest slice.
            slc = slices[0]
            thickness = self.contours[slc].thickness
            area1 = self.contours[slc].area
            if self.end_effect == 'Extend':
                # Treats structure as continues before starting slice.
                slice_volume = area1 * thickness
            elif self.end_effect == 'Interpolate':
                # Area starts from zero halfway to previous slice.
                slice_volume = area1 / 2 * thickness
            else:  # 'Truncate'
                # Volume starts at first slice
                slice_volume = 0
            volume += slice_volume
            self.volume = round(volume, PRECISION)
            s_radius = (3 * self.volume / (4 * pi)) ** (1 / 3)
            self.spherical_radius = round(s_radius, PRECISION)
            com_vector = np.array([0.0, 0.0, 0.0])
            for com in com_list:
                com_vector += com
            total_com = com_vector / area_sum
            self.center_of_mass = tuple(round(num, PRECISION)
                                        for num in list(total_com))

        def calculate_resolution():
            res_list = [slice.resolution for slice in self.contours.values()]
            self.resolution = round(mean(res_list), PRECISION)
            if self.resolution > 15:
                self.resolution_type = 'High'
            else:
                self.resolution_type = 'Normal'

        if not self.empty:
            sort_slices()
            calculate_volume()
            calculate_resolution()

    @property
    def neighbours(self):
        slices = pd.Series(self.contours.keys())
        inf1 = slices.shift(-1)
        inf2 = slices.shift(-2)
        sup1 = slices.shift(1)
        sup2 = slices.shift(2)
        neighbours = pd.concat([slices, inf1, inf2, sup1, sup2], axis='columns')
        neighbours.columns = ['slice', 'inf1', 'inf2', 'sup1', 'sup2']
        neighbours.set_index('slice', drop=False, inplace=True)
        return neighbours

    @property
    def gaps(self):
        slices = pd.Series(self.contours.keys())
        gap = slices.shift(-1) - slices
        gaps = pd.concat([slices, gap], axis='columns')
        gaps.columns = ['slice', 'gap']
        gaps.set_index('slice', inplace=True)
        return gaps


# %% read_contour data
def read_contours(struct_dataset: pydicom.Dataset)->Dict[int, ContourSet]:
    '''Load contours for each structure in the RS DICOM file.

    Extracts the structure data from an RS_DICOM dataset, building a dictionary
    of the structures for analysis.  They keys of the dictionary are the
    structure's ROI number.  The values are ContourSets

    Args:
        struct_dataset (pydicom.Dataset): The full dataset for an RS DICOM file.

    Returns:
        Dict[int, ContourSet]: A dictionary of structure data with the
            structure's ROI number as the key.
    '''
    roi_id = get_names_nums(struct_dataset)
    contour_sets = {}
    for roi in struct_dataset.ROIContourSequence:
        structure_num = roi.ReferencedROINumber
        if structure_num not in contour_sets:
            structure_id = roi_id.at[roi.ReferencedROINumber]
            contour_set = ContourSet(structure_id=structure_id,
                                     roi_num=structure_num)
        else:
            contour_set = contour_sets[structure_num]
        geo_type_list = []
        if hasattr(roi, 'ContourSequence'):
            for contour_points in roi.ContourSequence:
                geo_type_list.append(contour_points.ContourGeometricType)
                points = np.array(contour_points.ContourData).reshape(-1,3)
                points = points / 10  # Convert from mm to cm
                contour = ContourSlice(points)
                contour_set.add_contour(contour)
        contour_set.finalize()
        contour_set.color = tuple(roi.ROIDisplayColor)
        contour_sets[structure_num] = contour_set
    return contour_sets


# %% Make a table to CT slice indexes and the contours on that slice
# TODO Some of these index functions should be converted to ContourSet methods
def build_contour_index(contour_sets: Dict[int, ContourSet])->pd.DataFrame:
    '''Build an index of structures in a contour set.

    The table columns contain the structure names, the ROI number, and the
    slice positions where the contours for that structure are located.  There
    is one row for each slice and structure on that slice.  Multiple contours
    for a single structure on a given slice, have only one row in teh contour
    index

    Args:
        contour_sets (Dict[int, RS_DICOM_Utilities.ContourSet]): A dictionary
            of structure data.

    Returns:
        pd.DataFrame: An index of structures in a contour set indication which
            slices contains contours for each structure.
    '''
    slice_ref = {}
    name_ref = {}
    for structure in contour_sets.values():
        slice_ref[structure.roi_num] = list(structure.contours.keys())
        name_ref[structure.roi_num] = structure.structure_id
    slice_seq = pd.Series(slice_ref).explode()
    slice_seq.name = 'Slice'
    name_lookup = pd.Series(name_ref)
    name_lookup.name = 'StructureID'
    slice_lookup = pd.DataFrame(name_lookup).join(slice_seq, how='outer')
    return slice_lookup


# %% Find distance between slices with contours
def slice_spacing(contour):
    # Index is the slice position of all slices in the image set
    # Columns are structure IDs
    # Values are the distance (INF) to the next contour
    inf = contour.dropna().index.min()
    sup = contour.dropna().index.max()
    contour_range = (contour.index <= sup) & (contour.index >= inf)
    slices = contour.loc[contour_range].dropna().index.to_series()
    gaps = slices.shift(-1) - slices
    return gaps



def build_slice_table(contour_sets)->pd.DataFrame:
    def form_table(slice_index):
        slice_index.reset_index(inplace=True)
        slice_index.sort_values('Slice', inplace=True)
        slice_index.set_index(['Slice','StructureID'], inplace=True)
        slice_table = slice_index.unstack()
        slice_table.columns = slice_table.columns.droplevel()
        return slice_table

    slice_index = build_contour_index(contour_sets)
    slice_table = form_table(slice_index)
    contour_slices = slice_table.apply(slice_spacing)
    return contour_slices


# %% Identify contours with gaps
def has_gaps(structure_id, contour_slices) -> bool:
    inf = contour_slices[structure_id].dropna().index.min()
    sup = contour_slices[structure_id].dropna().index.max()

    contour_range = (contour_slices.index <= sup) & (contour_slices.index >= inf)
    missing_slices = contour_slices.loc[contour_range, structure_id].isna()
    return any(missing_slices)


# %% find contours surrounding missing conto0urs for interpolation
def neighbouring_slice(slice_index, missing_slices=None, shift_direction=1,
                       shift_start=0):
    shift_size = shift_start
    if missing_slices:
        ref = slice_index[missing_slices].copy()
        ref_missing = list(missing_slices)
        while ref_missing:
            shift_size += shift_direction
            shift_slice = slice_index.shift(shift_size)[missing_slices]
            ref_idx = ref.isin(ref_missing)
            ref[ref_idx] = shift_slice[ref_idx]
            ref_missing = list(set(ref) & set(missing_slices))
            ref_missing.sort()
    else:
        shift_size += shift_direction
        ref = slice_index.shift(shift_size).copy()
        ref.dropna(inplace=True)
    return ref


def next_slice(slice_index, missing_slices, previous,
               direction=1, shift_size=1):
    ref = neighbouring_slice(slice_index, missing_slices,
                             shift_direction=direction, shift_start=shift_size)
    used = (previous == ref)

    while any(used):
        shift_size += direction
        next_shift = neighbouring_slice(slice_index, missing_slices,
                                        shift_direction=direction,
                                        shift_start=shift_size)
        ref[used] = next_shift[used]
        used = (previous == ref)
    return ref


def find_neighbouring_slice(slice_index, missing_slices):
    inf1 = neighbouring_slice(slice_index, missing_slices, shift_direction=-1)
    inf2 = next_slice(slice_index, missing_slices, inf1, direction=-1,
                    shift_size=-1)
    sup1 = neighbouring_slice(slice_index, missing_slices, shift_direction=1)
    sup2 = next_slice(slice_index, missing_slices, sup1, direction=1,
                    shift_size=1)

    ref = pd.concat([inf1, inf2, sup1, sup2], axis='columns')
    ref.columns = ['inf1', 'inf2', 'sup1', 'sup2']
    return ref


# %% Old Functions
def build_contour_table(ds: pydicom.Dataset,
                        contour_sets: Dict[int, ContourSet])->pd.DataFrame:
    roi_gen = get_gen_alg(ds)
    roi_labels = get_roi_labels(ds)
    structure_info = {}
    for structure_num, contour_set in contour_sets.items():
        structure_id = contour_set.structure_id
        # Convert tuples to strings
        roi_colour = contour_set.color
        colour_text = ''.join([
            f'({roi_colour[0]:0d}, ',
            f'{roi_colour[1]:0d}, ',
            f'{roi_colour[2]:0d})'
            ])
        com = contour_set.center_of_mass
        if com:
            com_text = ''.join([
                f'({com[0]:-5.2f}, ',
                f'{com[1]:-5.2f}, ',
                f'{com[2]:-5.2f})'
                ])
        else:
            com_text = ''
        # Calculate spherical radius
        if contour_set.spherical_radius:
            diam = contour_set.spherical_radius * 2
        else:
            diam = None

        # Build table
        structure_dict = {
            'StructureID': structure_id,
            'Volume': contour_set.volume,
            'Eq Sp Diam': diam,
            'Thickness': None,
            'Length': contour_set.length,
            'Sup Slice': contour_set.sup_slice,
            'Inf Slice': contour_set.inf_slice,
            'Center of Mass': com_text,
            'Resolution': contour_set.resolution_type,
            'Colour': colour_text
            }
        structure_info[structure_num] = structure_dict
    contour_df = pd.DataFrame(structure_info).T
    contour_df.index.name = 'ROINumber'
    contour_data = pd.concat([roi_labels, contour_df, roi_gen],
                            axis='columns')
    contour_data.index.name = 'ROINumber'
    contour_data.reset_index(inplace=True)
    contour_data.set_index('StructureID', inplace=True)
    return contour_data


def drop_exclusions(structure_table):
    x_structs = structure_table.StructureID.str.lower().str.startswith('x')
    z_structs = structure_table.StructureID.str.lower().str.startswith('z')
    structure_ignore = x_structs | z_structs
    pruned_structure_table = structure_table.loc[~structure_ignore,:].copy()
    pruned_structure_table.dropna(subset=['Volume'], inplace=True)
    return pruned_structure_table
