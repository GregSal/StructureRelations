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
        thickness (float): The distance in cm to the next ContourSlice.
            to any structure containing only one slice. If the composite
            structure contains only one slice, default_thickness is used.
        axial_position (float): The offset of the ContourSlice in the axial
            (Z) direction to the DICOM origin.
        region_count (int): The number of distinct regions on the same slice.
        is_solid (bool): If False, one of the polygons completely surrounds a
            region which is not included as part of that polygon. i.g the
            polygon contains at least one hole.
    '''
    default_thickness = 0.2  # Default slice thickness in cm
    # This is assigned to ContourSlice rather than to ContourSet because the
    # slice thickness may vary between slices.

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
        self.axial_position = round(contour_points[0,2], PRECISION)
        self.thickness = self.default_thickness
        self.region_count = 1
        self.is_solid = True

    @property
    def area(self)->float:
        '''The area of the composite polygon(s).

        Returns:
            float: The area of the composite polygon(s)
        '''
        # TODO add a property that returns a list of areas if self.region_count > 1
        return round(self.contour.area, PRECISION)

    @property
    def radius(self) -> float:
        '''The "circular" radius of the polygon

        _extended_summary_

        Returns:
            float: _description_
        '''
        # FIXME The radius calculation is incorrect if the ContourSlice contains multiple polygons.
        # if self.region_count > 1, then iterate through the polygons, returning a tuple of radius.
        return round(sqrt(self.contour.area / pi), PRECISION)

    @property
    def perimeter(self):
        # FIXME The perimeter calculation is incorrect if the ContourSlice contains multiple polygons.
        # if self.region_count > 1, then iterate through the polygons, returning a tuple of perimeters.
        return round(self.contour.length, PRECISION)

    @property
    def centre_of_mass(self):
        # TODO add a property that returns a list of individual COMs for each polygon
        com =  [round(num, PRECISION)
                for num in list(self.contour.centroid.coords[0])]
        com.append(self.axial_position)
        return tuple(com)

    @property
    def resolution(self):
        '''Average number of points per contour length.'''
        res_list = []
        if self.contour.type == 'MultiPolygon':
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
        return round(mean_res, PRECISION)

    @property
    def center(self):
        centre_list = []
        if self.contour.geometryType() == 'Polygon':
            coord = list(self.contour.centroid.coords[0])
            coord.append(self.axial_position)
            centre_list.append(tuple(round(c, PRECISION) for c in coord))
        else:
            for contour in self.contour.geoms:
                coord = list(contour.centroid.coords[0])
                coord.append(self.axial_position)
                centre_list.append(tuple(round(c, PRECISION) for c in coord))
        return centre_list

    def combine(self, other: "ContourSlice"):
        if not other.axial_position == self.axial_position:
            raise ValueError("Can't combine contours from different slices")
        other_contour = other.contour
        # Check for non-overlapping structures
        if self.contour.relate_pattern(other_contour,'F*******2'):
            # non-overlapping structures
            self.contour = self.contour.union(other_contour)
            self.region_count += 1  # Increment the number of separate regions.
        elif self.contour.relate_pattern(other_contour,'212***FF2'):
            # self contains other
            self.contour = self.contour.difference(other_contour)
            self.is_solid = False
        elif self.contour.relate_pattern(other_contour,'2FF***212'):
            # other contains self
            self.contour = other_contour.difference(self.contour)
            if self.form == 'Simple':
                self.form = 'Rind'  # 'Multi' form beats 'Rind' form'
        else:
            raise ValueError('Cannot merge overlapping contours.')

    def __repr__(self) -> str:
        desc = ''.join([
            f'{self.form} ContourSlice, at slice {self.axial_position}, ',
            f'containing {(self.contour.type)}'
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

    def finalize(self):
        if not self.empty:
            self.sort_slices()
            self.calculate_volume()
            self.calculate_resolution()

    def add_contour(self, contour: ContourSlice):
        slice_position = contour.axial_position
        if slice_position in self.contours:
            new_contour = self.contours[slice_position]
            new_contour.combine(contour)
            self.contours[slice_position] = new_contour
        else:
            self.contours[slice_position] = contour
        self.empty = False

    def sort_slices(self):
        if not self.empty:
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

    def calculate_volume(self):
        if not self.empty:
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

    def calculate_resolution(self):
        if not self.empty:
            res_list = [slice.resolution for slice in self.contours.values()]
            self.resolution = round(mean(res_list), PRECISION)
            if self.resolution > 15:
                self.resolution_type = 'High'
            else:
                self.resolution = 'Normal'


# %% read_contour data
def read_contours(struct_dataset: pydicom.Dataset)->Dict[int, ContourSet]:
    '''Load contours for each structure in the RS DICOM file.

    _extended_summary_

    Args:
        struct_dataset (pydicom.Dataset): The full dataset for an RS DICOM file.

    Returns:
        Dict[int, ContourSet]: _description_
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
        # Calculate cylindrical radius
        appropriate_structure = structure_id in [
            'BODY',
            'BrainStem', 'PRV5 BrainStem',
            'SpinalCanal', 'PRV5 SpinalCanal', 'PRV8 SpinalCanal',
            'Esophagus', 'Larynx', 'opt Larynx', 'Trachea',
            'Aorta', 'Rectum', 'opt Rectum', 'PRV4 Rectum'
            ]
        has_volume = contour_set.volume is not None
        has_length = contour_set.length is not None
        if appropriate_structure & has_volume & has_length:
            cyl_rad = sqrt(contour_set.volume / (contour_set.length * pi))
        else:
            cyl_rad = None
        # Build table
        structure_dict = {
            'StructureID': structure_id,
            'Volume': contour_set.volume,
            'Eq Sp Diam': diam,
            'Radius': cyl_rad,
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


# %% Structure query functions
def get_structure_info(connection, selection, sql_path):
    query_file = sql_path / 'StructureLookup.sql'
    structure_lookup = vq.run_query(connection, query_file, selection)
    structure_lookup.set_index('StructureID', inplace=True)
    return structure_lookup


# %% Excel Table functions
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
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
    elif green > 181:
        color = (0, 0, 0)
    else:
        color = (255, 255, 255)
    return color


def color_format(sheet: xw.Sheet, color_info: pd.Series = None,
                 starting_cell='A1'):
    '''Apply background and text color for a given

    For each continuously filled cell in the column, starting with
    *starting_cell* A color and text color are obtained and applied.

    The color tuple is extracted from *color_info*, using the cell text as the
    row index.  If the search fails, the background color is
    white (255, 255, 255) and the text is black (0, 0, 0).

    If *color_info* is not supplied, then it tries to extract a color tuple
    from the cell text.  Again, if it fails, the background color is white
    (255, 255, 255) and the text is black (0, 0, 0).

    Args:
        sheet (xw.Sheet): The Excel worksheet to use
        structures_info (pd.Series, optional): A color lookup table.  The
            index must match the text in the Excel column.  If None, then the
            text in the Excel column is assumed to be RGB color tuples as a
            string. The format should be like: "(#, #, #)", where `#` is an
            integer between 0 and 255.   Defaults to None.
        starting_cell (str, optional): _description_. The top starting cell in
            the column to be colored. Defaults to 'A1'.
    '''
    def extract_color(rgb: str)->Tuple[int]:
        '''Convert RGB string to tuple.'''
        try:
            color_rgb = tuple(int(num) for num in rgb[1:-1].split(', '))
        except (ValueError, AttributeError, TypeError):
            color_rgb = (255,255,255)
        return color_rgb

    if color_info is None:
        text_as_color = True
    else:
        text_as_color = False

    start_range = sheet.range(starting_cell)
    end_range = start_range.end('down')
    num_rows = xw.Range(start_range, end_range).size
    for idx in range(num_rows):
        cell = start_range.offset(idx,0)
        color_index = cell.value.strip()
        if color_index:
            if text_as_color:
                color_rgb = extract_color(color_index)
            else:
                try:
                    rgb = color_info.at[color_index]
                except KeyError:
                    color_rgb = (255,255,255)
                else:
                    color_rgb = extract_color(rgb)
            text_rgb = text_color(color_rgb)
            cell.color = color_rgb
            cell.font.color = text_rgb


def add_fill(text: str, start='', end='', default='',
             drop_white_space=True)->str:
    '''Add spaces or other fillers around text.

    Add start to the beginning of the text if it is not already present.
    Add end to the ending of the text if it is not already present.
    If text is blank, return default.

    Args:
        text (str): The primary text.
        start (str, optional): If supplied, look for *start* at the beginning of
            *text*. If not present add *start* to the beginning of the text.
            Defaults to ''.
        end (str, optional): If supplied, look for *end* at the end of *text*.
            If not present add *end* after the text. Defaults to ''.
        default (str, optional): String to return if *text* is empty.
            Defaults to ''.
        drop_white_space (bool, optional): If True, strip whitespace from *text*
            before applying other tests. Defaults to True.

    Returns:
        str: _description_
    '''
    # Remove starting and trailing whitespace if requested.
    if drop_white_space:
        text = text.strip()
    # Use default if text is empty.  This comes after *drop_white_space* so that
    # text which is all whitespace will be treated as empty.
    if not text:
        return default
    # Set beginning if required.
    if start:
        if text.startswith(start):
            beginning = ''
        else:
            beginning = start
    else:
        beginning = ''
    # Set ending if required.
    if end:
        if text.endswith(end):
            ending = ''
        else:
            ending = end
    else:
        ending = ''
    # Combine text parts.
    full_text = ''.join([beginning, text, ending])
    return full_text


def set_file_name(selection: Dict[str, str] = None, extension='.pkl', prefix='',
                  folder: Path = None)->Path:
    '''Construct a custom file name.

    If supplied the file name will begin with prefix. Middle sections of the
    file name are obtained from the 'PatientLastName', 'CR_Number', 'Plan_Id',
    and 'Structure_Set' items in selection. If selection does not contain
    'CR_Number', add 'data' to the end of the file name in its place. If
    selection does not contain 'Plan_Id' check for 'Structure_Set'.
    Extension defaults to '.pkl'.
    # The final file name will have the form:
    '{prefix} {Plan_Id|Structure_Set} {PatientLastName} {CR_Number}.{extension}'.
    For missing name components, the surrounding spaces are also dropped.

    Args:
        selection (Dict[str, str], optional): Plan specific references.
            Defaults to None.  If supplied must contain the following items:
                'PatientLastName', 'CR_Number', 'Plan_Id'
        extension (str, optional): File extension to use (including '.').
            Defaults to '.pkl'.
        prefix (str, optional): Initial part of the file name. Defaults to ''.
        folder (Path, optional): Folder where the file is (to be) located.
            If None, Path.cwd() is used. Defaults to None.

    Returns:
        Path: Full path with the custom file name.
    '''
    # Get name parts from selection
    if selection:
        plan_id = selection.get('Plan_Id', '')
        if not plan_id:
            structure_set = selection.get('Structure_Set', '')
        else:
            structure_set = ''
        last_name = selection.get('PatientLastName', '')
        pt_id = selection.get('CR_Number', '')
    else:
        plan_id = ''
        last_name = ''
        pt_id = ''
    # Add required spaces etc. for name parts.
    name_parts = {
        'prefix': add_fill(prefix, end=' '),
        'extension': add_fill(extension, start='.', default='.pkl'),
        'plan_id': add_fill(plan_id, end=' '),
        'structure_set': add_fill(structure_set, end=' '),
        'last_name': add_fill(last_name, end=' '),
        'pt_id': add_fill(pt_id, default='data')
        }
    # Build the name
    template = '{prefix}{plan_id}{structure_set}{last_name}{pt_id}{extension}'
    data_file_name = template.format(**name_parts)
    # Build the full path
    if not folder:
        folder = Path.cwd()
    save_file =  folder / data_file_name
    return save_file


def save_data(contour_tables, contour_formatting, selection, data_path):
    workbook = xw.books.active
    struct_sheet = workbook.sheets['Contouring']
    dimensions_table = contour_tables['Dimensions']
    dim_range = struct_sheet.range('A3')
    dim_range.options(index=False, header=False).value = dimensions_table
    color_format(struct_sheet, contour_formatting, starting_cell='A3')

    settings_table = contour_tables['Settings']
    s_range = struct_sheet.range('K3')
    s_range.options(index=False, header=False).value = settings_table
    color_format(struct_sheet, contour_formatting, starting_cell='K3')

    info_sheet = workbook.sheets['Structure Ref']
    settings_table = contour_tables['Structure Ref']
    s_range = info_sheet.range('A1')
    s_range.options(index=False, header=True).value = settings_table
    color_format(info_sheet, contour_formatting, starting_cell='A2')

    save_file = set_file_name(selection, folder=data_path, extension='.xlsx',
                              prefix='Structures')
    new_book = xw.Book()
    for name, table in contour_tables.items():
        new_sheet = new_book.sheets.add(name)
        xw.view(table, sheet=new_sheet)
        color_format(new_sheet, contour_formatting, starting_cell='B2')
    new_book.save(save_file)
    update_pickle_data(contour_tables, data_path, selection,
                       prefix='plan_check_data')


def extract_structure_data(structure_set_info, sql_path, data_path):
    connection = vq.connect(DBSERVER)
    selection = {
        'CR_Number': structure_set_info['PatientID'],
        'StructureSet_ID': structure_set_info['StructureSet'],
        'PatientLastName': structure_set_info['PatientLastName'],
        }
    dataset = pydicom.dcmread(structure_set_info['File'])
    contour_sets = read_contours(dataset)
    contour_data = build_contour_table(dataset, contour_sets)

    structure_lookup = get_structure_info(connection, selection, sql_path)
    structure_table = contour_data.join(structure_lookup)

    structure_table.reset_index(inplace=True)
    structure_table = drop_exclusions(structure_table)


    struc_idx = structure_table.set_index('StructureID', drop=False)
    struc_idx = struc_idx.StructureID.copy()
    structure_parts = build_sort_index(struc_idx)
    structure_table = structure_table.join(structure_parts.SortOrder,
                                            on='StructureID')
    structure_table.sort_values('SortOrder', inplace=True)

    dimensions_columns = ['StructureID', 'Volume', 'Eq Sp Diam', 'Radius',
                            'Thickness', 'Length', 'Sup Slice', 'Inf Slice',
                            'Center of Mass']
    settings_columns = ['StructureID', 'ROINumber', 'StructureName',
                        'CodeMeaning', 'CodeScheme', 'VolumeType',
                        'DICOM_Type', 'Status', 'GenerationAlgorithm',
                        'Resolution', 'MaterialCTValue']
    ref_columns = ['StructureID', 'CodeMeaning', 'VolumeName', 'VolumeType',
                    'GenerationAlgorithm', 'Resolution', 'Colour',
                    'SortOrder', 'ROINumber', 'MaterialCTValue',
                    'Status', 'ColorName']
    dimensions_table = structure_table[dimensions_columns]
    settings_table = structure_table[settings_columns]
    structure_ref = structure_table[ref_columns]

    contour_formatting = structure_table.set_index('StructureID').Colour
    structure_parts.reset_index(inplace=True)
    contour_tables = {
        'Dimensions': dimensions_table,
        'Settings': settings_table,
        'Structures': structure_table,
        'Structure Parts': structure_parts,
        'Structure Ref': structure_ref
        }

    save_data(contour_tables, contour_formatting, selection, data_path)
def drop_exclusions(structure_table):
    x_structs = structure_table.StructureID.str.lower().str.startswith('x')
    z_structs = structure_table.StructureID.str.lower().str.startswith('z')
    structure_ignore = x_structs | z_structs
    pruned_structure_table = structure_table.loc[~structure_ignore,:].copy()
    pruned_structure_table.dropna(subset=['Volume'], inplace=True)
    return pruned_structure_table


# %% Main
def main():
    '''Main'''
    base_path = Path.cwd()
    base_path = base_path.resolve()
    data_path = Path(r'C:\Plan Checking Temp')
    dicom_folder = data_path / '_DICOM'
    sql_path = base_path / 'SQL'




    struct_files = list(dicom_folder.glob('**/RS*.dcm'))
    structure_set_info = select_structure_file(struct_files)

    #struct_file = dicom_folder / r'0338104\RS.0338104.BRAI.dcm'
    #structure_set_info = get_structure_file_info(struct_file)

    if structure_set_info:
        extract_structure_data(structure_set_info, sql_path, data_path)




if __name__ == '__main__':
    main()
