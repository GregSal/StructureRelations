'''DICOM Structure File handling module.
'''
# %% Imports
from typing import Any, Optional, List
from pathlib import Path
import logging
import json
import re

import numpy as np
import pandas as pd
import pydicom

from utilities import round_one_up
from contours import ContourPoints
from types_and_classes import ROI_Type, SliceIndexType

# Configure logging if not already configured
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DEFAULT_STRUCTURE_FILTERS_PATH = (
    Path(__file__).parent / 'webapp' / 'config' / 'structure_filter_rules.json'
)

STRUCTURE_FILTER_FIELDS = {
    'Structure Code',
    'Coding Scheme',
    'Code Meaning',
    'Structure Name',
    'DICOM Type',
    'Density',
    'ROI Physical Property',
    'Structure ID',
    'Generation Method',
    'Generation Description',
}

_UPLOAD_PREFIX_PATTERN = re.compile(
    r'^(?:'
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    r'|upload'
    r')_(.+)$'
)


def clean_uploaded_file_name(file_name: str | Path) -> str:
    '''Return a user-facing DICOM file name without temporary upload prefixes.'''
    cleaned_name = Path(str(file_name)).name
    while True:
        match = _UPLOAD_PREFIX_PATTERN.match(cleaned_name)
        if match is None:
            return cleaned_name
        cleaned_name = match.group(1)


def _stringify_filter_value(value: object) -> str:
    '''Normalize filter values to strings for matching.'''
    if value is None:
        return ''

    try:
        is_na_value = pd.isna(value)
        if isinstance(is_na_value, (bool, np.bool_)) and is_na_value:
            return ''
        if isinstance(is_na_value, (list, tuple, np.ndarray, pd.Series)):
            if bool(np.all(is_na_value)):
                return ''
    except TypeError:
        pass

    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=True)

    return str(value)


def _coerce_filter_bool(value: object, default: bool = False) -> bool:
    '''Normalize filter config values to booleans.'''
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'n', 'off'}:
            return False
    return bool(value)


class DicomStructureFile:
    '''Class for handling DICOM Structure files.

    This class provides a unified interface for loading and working with
    DICOM RT Structure files, handling both file path and dataset inputs.

    Attributes:
        top_dir (Path): The base directory for DICOM files.
        file_path (Path): The full path to the DICOM file.
        file_name (str): The name of the DICOM file.
        dataset (pydicom.Dataset): The loaded DICOM dataset.
        contour_points (List[ContourPoints]): List of extracted contour points.
        structure_names (pd.DataFrame): Table containing all roi number and
            structure names.
        resolution (Optional[float]): The calculated structure resolution in
            cm/pixel.
    '''

    def __init__(self, top_dir: Path,
                 file_path: Optional[Path] = None,
                 file_name: Optional[str] = None,
                 dataset: Optional[pydicom.Dataset] = None):
        '''Initialize the DicomStructureFile.

        Args:
            top_dir (Path): The folder to search for DICOM files.
            file_path (Path, optional): Full path to the DICOM file.
                If provided, file_name is extracted from this path.
            file_name (str, optional): Name of the DICOM file.
                If provided, path is created by combining with top_dir.
            dataset (pydicom.Dataset, optional): Pre-loaded DICOM dataset.
                If not provided, dataset will be loaded from the file.

        Raises:
            ValueError: If neither file_path nor file_name is provided.
            FileNotFoundError: If the specified file does not exist.
            pydicom.errors.InvalidDicomError: If the file is not a valid
                DICOM file.
        '''
        self.top_dir = Path(top_dir)

        # Handle file path/name logic
        if file_path is not None:
            self.file_path = Path(file_path)
            raw_file_name = self.file_path.name
            self.file_name = clean_uploaded_file_name(raw_file_name)
        elif file_name is not None:
            raw_file_name = file_name
            self.file_name = clean_uploaded_file_name(raw_file_name)
            self.file_path = self.top_dir / raw_file_name
        else:
            raise ValueError("Either file_path or file_name must be provided")

        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {self.file_path}")

        # Handle dataset
        if dataset is not None:
            self.dataset = dataset
            logger.debug('Using provided dataset for %s', self.file_name)
        else:
            logger.debug('Loading dataset from %s', self.file_path)
            self._load_dataset()

        # Initialize contour attributes
        self.structure_set_info = self.get_structure_set_info()
        self.contour_points = self.get_contour_points()
        self.structure_names = self.get_structure_names()

        # Calculate resolution (may use structure dimensions as fallback)
        self.resolution = self.calculate_structure_resolution()
        self.structure_filter_report = pd.DataFrame()
        self.structure_filter_config_path: Optional[Path] = None

        # Round contour points based on calculated resolution
        #if self.resolution is not None:
        #    self.round_contour_points()

    def _load_dataset(self) -> None:
        '''Load the DICOM dataset from file.

        Raises:
            pydicom.errors.InvalidDicomError: If the file is not a valid DICOM
                file.
        '''
        try:
            self.dataset = pydicom.dcmread(self.file_path)
            logger.info('Successfully loaded DICOM dataset from %s',
                        self.file_name)
        except pydicom.errors.InvalidDicomError as e:
            logger.error('Failed to load DICOM file %s: %s', self.file_path, e)
            raise

    def is_structure_file(self) -> bool:
        '''Check if the loaded dataset is a RT Structure file.

        Returns:
            bool: True if the dataset is a RT Structure file.
        '''
        if not hasattr(self.dataset, 'Modality'):
            return False
        return 'RTSTRUCT' in str(self.dataset.Modality)

    def get_structure_set_info(self) -> dict:
        '''Get basic structure file information.

        Returns:
            dict: Dictionary containing structure file information including:
                - PatientName
                - PatientID
                - StructureSet
                - StudyID
                - SeriesNumber
                - File path
        '''
        if not self.is_structure_file():
            logger.warning('File %s is not a RT Structure file', self.file_name)
            return {}

        full_name = str(self.dataset.get('PatientName', ''))
        last_name = full_name.split('^',1)[0] if '^' in full_name else full_name

        structure_info = {
            'PatientName': full_name,
            'PatientLastName': last_name,
            'PatientID': str(self.dataset.get('PatientID', '')),
            'StructureSet': str(self.dataset.get('StructureSetLabel', '')),
            'StudyID': str(self.dataset.get('StudyID', '')),
            'SeriesNumber': str(self.dataset.get('SeriesNumber', '')),
            'SeriesDescription': str(self.dataset.get('SeriesDescription', '')),
            'File': self.file_path
        }

        logger.debug('Retrieved structure info for %s',
                     structure_info['StructureSet'])
        self.structure_set_info = structure_info
        return structure_info

    def __repr__(self) -> str:
        '''String representation of the DicomStructureFile.'''
        text = (f"DicomStructureFile(file_name='{self.file_name}', "
                f"top_dir='{self.top_dir}')")
        return text

    def __str__(self) -> str:
        '''Human-readable string representation.'''
        info = self.get_structure_set_info()
        if info:
            return (f'DICOM RT Structure: {info.get("StructureSet", "Unknown")} '
                    f'for patient {info.get("PatientID", "Unknown")}')
        return f"DICOM file: {self.file_name}"

    def __enter__(self) -> 'DicomStructureFile':
        '''Support context manager usage for controlled dataset lifecycle.'''
        if not hasattr(self, 'dataset') or self.dataset is None:
            self._load_dataset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        '''Release dataset reference when leaving a context manager block.'''
        if hasattr(self, 'dataset'):
            del self.dataset
        return False

    def get_contour_points(self) -> List[ContourPoints]:
        '''Extract all contour points from the DICOM structure dataset.

        Reads all ROI contour sequences from the dataset and converts them to
        ContourPoints objects. The contour coordinates are converted from
        millimeters (DICOM standard) to centimeters.

        Returns:
            List[ContourPoints]: A list of ContourPoints objects containing
                all contours from all ROI structures in the dataset.

        Raises:
            ValueError: If the dataset is not a RT Structure file or does not
                contain ROIContourSequence.
        '''
        if not self.is_structure_file():
            raise ValueError(f"File {self.file_name} is not a RT Structure file")

        if not hasattr(self.dataset, 'ROIContourSequence'):
            raise ValueError("Dataset does not contain ROIContourSequence")

        contour_points_list = []

        for roi in self.dataset.ROIContourSequence:
            roi_number = ROI_Type(roi.ReferencedROINumber)

            # Check if this ROI has contour data
            if not hasattr(roi, 'ContourSequence'):
                logger.debug('ROI %s has no contour sequence', roi_number)
                continue

            for contour in roi.ContourSequence:
                # Get the geometric type (should be 'CLOSED_PLANAR' for most cases)
                geom_type = getattr(contour, 'ContourGeometricType', 'UNKNOWN')

                if geom_type not in ['CLOSED_PLANAR', 'POINT', 'OPEN_PLANAR']:
                    logger.warning('Unsupported geometric type %s in ROI %s',
                                   geom_type, roi_number)
                    continue
                if geom_type not in ['CLOSED_PLANAR']:
                    logger.warning('Skipping non-closed planar contour in ROI %s',
                                   roi_number)
                    continue

                # Extract contour data - it's a flat array of x,y,z coordinates
                if hasattr(contour, 'ContourData') and len(contour.ContourData) >= 9:
                    # Reshape the flat array into (n, 3) array of points
                    points_array = np.array(contour.ContourData).reshape(-1, 3)

                    # Convert from mm to cm (DICOM uses mm, our system uses cm)
                    points_array = points_array / 10.0

                    # Convert to list of tuples for ContourPoints constructor
                    #points_tuples = [tuple(point) for point in points_array]

                    # Extract slice index from the z-coordinate of the first point
                    slice_index = SliceIndexType(points_array[0][2])

                    # Create ContourPoints object with additional metadata
                    contour_points = ContourPoints(
                        points=points_array,
                        roi=ROI_Type(roi_number),
                        slice_index=slice_index,
                        #ContourGeometricType=geom_type
                        )

                    contour_points_list.append(contour_points)

                else:
                    logger.warning('Invalid or insufficient contour data in ROI %s',
                                   roi_number)

        logger.info('Extracted %d contours from %d ROIs',
                    len(contour_points_list),
                    len(self.dataset.ROIContourSequence))

        # Store contour points and convert to DataFrame
        return contour_points_list

    def get_structure_names(self) -> pd.DataFrame:
        '''Get ROI names and numbers from the structure set.

        Returns:
            pd.DataFrame: DataFrame with columns 'ROINumber' and 'StructureID'
                containing the mapping of ROI numbers to structure names.
        '''
        if not self.is_structure_file():
            logger.warning('File %s is not a RT Structure file', self.file_name)
            return pd.DataFrame()

        if not hasattr(self.dataset, 'StructureSetROISequence'):
            logger.warning("Dataset does not contain StructureSetROISequence")
            return pd.DataFrame()

        structure_data = {}
        for roi in self.dataset.StructureSetROISequence:
            structure_data[ROI_Type(roi.ROINumber)] = str(roi.ROIName)
        return structure_data

    def get_structure_filter_metadata(self) -> pd.DataFrame:
        '''Build a per-ROI metadata table for JSON-driven structure filters.'''
        if not self.is_structure_file():
            logger.warning('File %s is not a RT Structure file', self.file_name)
            return pd.DataFrame()

        contour_counts: dict[ROI_Type, int] = {}
        for contour_point in self.contour_points or []:
            roi_number = ROI_Type(contour_point['ROI'])
            contour_counts[roi_number] = contour_counts.get(roi_number, 0) + 1

        metadata_by_roi: dict[ROI_Type, dict[str, Any]] = {}

        for roi_number, structure_id in self.structure_names.items():
            metadata_by_roi[roi_number] = {
                'ROINumber': roi_number,
                'Structure ID': structure_id,
                'Structure Name': '',
                'DICOM Type': '',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'Density': '',
                'ROI Physical Property': '',
                'Generation Method': '',
                'Generation Description': '',
                'Contour Count': contour_counts.get(roi_number, 0),
                'Has Contours': contour_counts.get(roi_number, 0) > 0,
            }

        for roi in getattr(self.dataset, 'StructureSetROISequence', []):
            roi_number = ROI_Type(roi.ROINumber)
            roi_metadata = metadata_by_roi.setdefault(
                roi_number,
                {
                    'ROINumber': roi_number,
                    'Contour Count': contour_counts.get(roi_number, 0),
                    'Has Contours': contour_counts.get(roi_number, 0) > 0,
                },
            )
            roi_metadata['Structure ID'] = _stringify_filter_value(
                getattr(roi, 'ROIName', '')
            )
            roi_metadata['Generation Method'] = _stringify_filter_value(
                getattr(roi, 'ROIGenerationAlgorithm', '')
            )
            roi_metadata['Generation Description'] = _stringify_filter_value(
                getattr(roi, 'ROIGenerationDescription', '')
            )

        for roi in self.dataset.get('RTROIObservationsSequence', []):
            roi_number = ROI_Type(roi.ReferencedROINumber)
            roi_metadata = metadata_by_roi.setdefault(
                roi_number,
                {
                    'ROINumber': roi_number,
                    'Structure ID': self.structure_names.get(roi_number, ''),
                    'Contour Count': contour_counts.get(roi_number, 0),
                    'Has Contours': contour_counts.get(roi_number, 0) > 0,
                },
            )
            roi_metadata['Structure Name'] = _stringify_filter_value(
                getattr(roi, 'ROIObservationDescription', '')
            )
            roi_metadata['DICOM Type'] = _stringify_filter_value(
                getattr(roi, 'RTROIInterpretedType', '')
            )

            code_sequence = roi.get('RTROIIdentificationCodeSequence')
            if code_sequence:
                roi_code = code_sequence[0]
                roi_metadata['Structure Code'] = _stringify_filter_value(
                    getattr(roi_code, 'CodeValue', '')
                )
                roi_metadata['Coding Scheme'] = _stringify_filter_value(
                    getattr(roi_code, 'CodingSchemeDesignator', '')
                )
                roi_metadata['Code Meaning'] = _stringify_filter_value(
                    getattr(roi_code, 'CodeMeaning', '')
                )

            physical_properties = roi.get('ROIPhysicalPropertiesSequence')
            if physical_properties:
                physical_property = physical_properties[0]
                roi_metadata['ROI Physical Property'] = _stringify_filter_value(
                    getattr(physical_property, 'ROIPhysicalProperty', '')
                )
                roi_metadata['Density'] = _stringify_filter_value(
                    getattr(physical_property, 'ROIPhysicalPropertyValue', '')
                )

        if not metadata_by_roi:
            return pd.DataFrame()

        metadata = pd.DataFrame(metadata_by_roi.values())
        metadata.sort_values('ROINumber', inplace=True)
        metadata.set_index('ROINumber', inplace=True, drop=False)

        for field_name in STRUCTURE_FILTER_FIELDS:
            if field_name not in metadata.columns:
                metadata[field_name] = ''

        if 'Contour Count' not in metadata.columns:
            metadata['Contour Count'] = 0
        if 'Has Contours' not in metadata.columns:
            metadata['Has Contours'] = False

        return metadata

    def get_roi_labels(self):
        '''Get detailed ROI labels and codes from the structure set.

        Returns:
            pd.DataFrame: DataFrame with ROINumber as index and detailed ROI
                information as values including:
                    StructureName and DICOM_Type,
                If present, also includes:
                    Code, CodeScheme, and CodeMeaning.
        '''
        if not self.is_structure_file():
            logger.warning('File %s is not a RT Structure file', self.file_name)
            return pd.DataFrame()

        label_list = []
        obs_seq = self.dataset.get('RTROIObservationsSequence')
        if not obs_seq:
            logger.warning("Dataset does not contain RTROIObservationsSequence")
            return pd.DataFrame()
        for roi in self.dataset.RTROIObservationsSequence:
            roi_number = roi.ReferencedROINumber
            label_dict = {
                'ROINumber': roi_number,
                'StructureName': self.structure_names.get(roi_number, ''),
                'DICOM_Type': roi.RTROIInterpretedType
                }
            code_seq = roi.get('RTROIIdentificationCodeSequence')
            if code_seq:
                roi_label = code_seq[0]
                label_dict['Code'] = roi_label.CodeValue
                label_dict['CodeScheme'] = roi_label.CodingSchemeDesignator
                label_dict['CodeMeaning'] = roi_label.CodeMeaning
            label_list.append(label_dict)
        roi_labels = pd.DataFrame(label_list)
        roi_labels.set_index('ROINumber', inplace=True)
        return roi_labels

    def load_structure_filter_rules(
        self,
        filter_file: Optional[str | Path] = None,
    ) -> tuple[Optional[Path], dict]:
        '''Load structure filtering rules from a JSON file.'''
        config_path = Path(filter_file) if filter_file else DEFAULT_STRUCTURE_FILTERS_PATH

        if not config_path.exists():
            logger.info('Structure filter config not found at %s', config_path)
            return None, {'rules': []}

        try:
            with open(config_path, 'r', encoding='utf-8') as config_handle:
                config = json.load(config_handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning('Failed to load structure filter config %s: %s',
                           config_path, exc)
            return config_path, {'rules': []}

        if not isinstance(config, dict):
            logger.warning('Structure filter config %s must contain a JSON object',
                           config_path)
            return config_path, {'rules': []}

        rules = config.get('rules', [])
        if not isinstance(rules, list):
            logger.warning('Structure filter config %s has non-list rules',
                           config_path)
            config['rules'] = []

        return config_path, config

    def _filter_rule_matches_value(
        self,
        actual_value: object,
        rule: dict,
    ) -> bool:
        '''Return True when a single rule condition matches a value.'''
        field_value = _stringify_filter_value(actual_value)
        match_type = _stringify_filter_value(
            rule.get('match_type', rule.get('match'))
        ).lower()
        expected_value = _stringify_filter_value(
            rule.get('value', rule.get('pattern'))
        )
        case_sensitive = bool(rule.get('case_sensitive', False))

        if not match_type or not expected_value:
            return False

        if match_type == 'regex':
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                return re.search(expected_value, field_value, flags) is not None
            except re.error as exc:
                logger.warning('Invalid regex in structure filter rule %r: %s',
                               rule.get('id', '<unnamed>'), exc)
                return False

        if not case_sensitive:
            field_value = field_value.lower()
            expected_value = expected_value.lower()

        if match_type == 'exact':
            return field_value == expected_value
        if match_type == 'prefix':
            return field_value.startswith(expected_value)
        if match_type == 'suffix':
            return field_value.endswith(expected_value)
        if match_type == 'structure list':
            logger.warning(
                'structure list match type requires row-level handling in rule %r',
                rule.get('id', '<unnamed>'),
            )
            return False

        logger.warning('Unsupported structure filter match type %r in rule %r',
                       match_type, rule.get('id', '<unnamed>'))
        return False

    def _filter_rule_matches_structure_list(
        self,
        actual_value: object,
        rule: dict,
    ) -> bool:
        '''Return True when a structure ID matches a list-based rule.'''
        raw_value = rule.get('value', rule.get('pattern'))
        if not isinstance(raw_value, list):
            logger.warning('structure list match in rule %r must use a JSON list',
                           rule.get('id', '<unnamed>'))
            return False

        actual_text = _stringify_filter_value(actual_value)
        if not actual_text:
            return False

        expected_values = {
            _stringify_filter_value(item)
            for item in raw_value
            if _stringify_filter_value(item)
        }
        return actual_text in expected_values

    def _filter_rule_matches_row(
        self,
        row: pd.Series,
        rule: dict,
    ) -> bool:
        '''Return True when a structure row satisfies a filter rule.'''
        field_name = _stringify_filter_value(rule.get('field'))
        if field_name not in STRUCTURE_FILTER_FIELDS:
            logger.warning('Unsupported structure filter field %r in rule %r',
                           field_name, rule.get('id', '<unnamed>'))
            return False

        if not self._filter_rule_matches_value(row.get(field_name, ''), rule):
            return False

        companion_rule = rule.get('with')
        if companion_rule is None:
            return True

        if not isinstance(companion_rule, dict):
            logger.warning('Companion filter in rule %r must be a JSON object',
                           rule.get('id', '<unnamed>'))
            return False

        companion_field = _stringify_filter_value(companion_rule.get('field'))
        if companion_field not in STRUCTURE_FILTER_FIELDS:
            logger.warning('Unsupported companion filter field %r in rule %r',
                           companion_field, rule.get('id', '<unnamed>'))
            return False

        return self._filter_rule_matches_value(
            row.get(companion_field, ''),
            companion_rule,
        )

    def evaluate_structure_filters(
        self,
        filter_file: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        '''Evaluate JSON filter rules against structure metadata.'''
        metadata = self.get_structure_filter_metadata()
        if metadata.empty:
            self.structure_filter_report = pd.DataFrame()
            self.structure_filter_config_path = None
            return pd.DataFrame()

        config_path, filter_config = self.load_structure_filter_rules(filter_file)
        self.structure_filter_config_path = config_path

        _KNOWN_ACTIONS = {'exclude', 'hide', 'include', 'display'}
        include_by_default = _coerce_filter_bool(
            filter_config.get('include by default', filter_config.get('include_by_default')),
            default=True,
        )
        display_by_default = _coerce_filter_bool(
            filter_config.get('display by default', filter_config.get('display_by_default')),
            default=True,
        )
        rules = []
        for index, rule in enumerate(filter_config.get('rules', []), start=1):
            if not isinstance(rule, dict):
                logger.warning('Skipping non-object structure filter rule #%d',
                               index)
                continue
            if not _coerce_filter_bool(rule.get('enabled', True), default=True):
                continue
            action = _stringify_filter_value(
                rule.get('action', 'exclude')
            ).lower()
            if action not in _KNOWN_ACTIONS:
                logger.warning('Skipping unsupported structure filter action %r '
                               'in rule %r', rule.get('action'),
                               rule.get('id', f'rule_{index}'))
                continue
            rules.append((index, rule))

        report_rows = []
        for _, row in metadata.iterrows():
            selected_state = include_by_default
            display_state = display_by_default
            matched_rules = []
            final_match = None
            for index, rule in rules:
                field_name = _stringify_filter_value(rule.get('field'))
                match_type = _stringify_filter_value(
                    rule.get('match_type', rule.get('match'))
                ).lower()
                if match_type == 'structure list':
                    if field_name != 'Structure ID':
                        logger.warning(
                            'structure list match type is only supported for Structure ID in rule %r',
                            rule.get('id', '<unnamed>'),
                        )
                        continue
                    matches = self._filter_rule_matches_structure_list(
                        row.get(field_name, ''),
                        rule,
                    )
                else:
                    matches = self._filter_rule_matches_row(row, rule)

                if not matches:
                    continue

                rule_action = _stringify_filter_value(
                    rule.get('action', 'exclude')
                ).lower()
                rule_id = _stringify_filter_value(rule.get('id')) or f'rule_{index}'
                rule_value = _stringify_filter_value(
                    rule.get('value', rule.get('pattern'))
                )
                companion_rule = rule.get('with') if isinstance(
                    rule.get('with'), dict
                ) else None
                companion_text = ''
                if companion_rule is not None:
                    companion_text = (
                        f"; with {companion_rule.get('field')} "
                        f"{companion_rule.get('match_type', companion_rule.get('match'))} "
                        f"{companion_rule.get('value', companion_rule.get('pattern'))}"
                    )

                match_entry = {
                    'id': rule_id,
                    'action': rule_action,
                    'description': _stringify_filter_value(
                        rule.get('description')
                    ) or rule_id,
                    'field': field_name,
                    'match_type': match_type,
                    'value': rule_value,
                    'reason': (
                        f'{field_name} {match_type} {rule_value}{companion_text}'
                    ),
                }
                matched_rules.append(match_entry)
                final_match = match_entry

                if rule_action == 'exclude':
                    selected_state = False
                elif rule_action == 'include':
                    selected_state = True
                elif rule_action == 'hide':
                    display_state = False
                elif rule_action == 'display':
                    display_state = True

            filtered = not selected_state
            hidden = not display_state
            report_rows.append({
                **row.to_dict(),
                'IsFiltered': filtered,
                'IsHidden': hidden,
                'SelectedByDefault': selected_state,
                'DisplayedByDefault': display_state,
                'MatchedRules': matched_rules,
                'FinalMatch': final_match,
                'FilterReason': final_match['reason'] if final_match else '',
            })

        report = pd.DataFrame(report_rows)
        report.set_index('ROINumber', inplace=True, drop=False)
        self.structure_filter_report = report.copy()
        return report

    def filter_exclusions(self,
                         filter_file: Optional[str | Path] = None
                         ) -> List[ContourPoints]:
        '''Filter contour points using rules loaded from a JSON file.'''
        filter_report = self.evaluate_structure_filters(filter_file)
        if filter_report.empty:
            return self.contour_points or []

        excluded_rois = set(
            filter_report.loc[filter_report['IsFiltered'], 'ROINumber']
        )
        filtered_contour_points = [
            cp for cp in self.contour_points
            if cp['ROI'] not in excluded_rois
        ]
        self.structure_names = {
            roi: name for roi, name in self.structure_names.items()
            if roi not in excluded_rois
        }

        original_count = len(self.contour_points)
        filtered_count = len(filtered_contour_points)
        excluded_count = original_count - filtered_count

        logger.info('Filtered %d contours from %d excluded ROIs. '
                    'Remaining: %d contours from %d ROIs',
                   excluded_count, len(excluded_rois), filtered_count,
                   len(set(cp['ROI'] for cp in filtered_contour_points)))

        self.contour_points = filtered_contour_points
        return filtered_contour_points

    def get_excluded_structures(self,
                               filter_file: Optional[str | Path] = None
                               ) -> pd.DataFrame:
        '''Get the structures excluded by the configured JSON rules.'''
        filter_report = self.evaluate_structure_filters(filter_file)
        if filter_report.empty:
            return pd.DataFrame()

        excluded = filter_report[filter_report['IsFiltered']].copy()
        excluded.rename(columns={
            'Structure ID': 'StructureID',
            'Structure Name': 'StructureName',
        }, inplace=True)
        return excluded.reset_index(drop=True)

    def find_image_files(self,
                        modalities: Optional[List[str]] = None,
                        match_study: bool = True,
                        match_patient: bool = True) -> List[Path]:
        '''Find corresponding image files under top_dir that match this
        structure set.

        Searches for DICOM image files (CT, MR, etc.) that correspond to this
        RT Structure file. Prioritizes frame of reference UID matching for the
        most reliable spatial alignment, then falls back to study/patient
        matching.

        Args:
            modalities (List[str], optional): List of modalities to search for.
                Defaults to ['CT', 'MR', 'PT', 'RTIMAGE'] for common imaging
                modalities.
            match_study (bool, optional): Whether to match Study Instance UID.
                Defaults to True.
            match_patient (bool, optional): Whether to match Patient ID.
                Defaults to True.

        Returns:
            List[Path]: List of paths to matching DICOM image files, sorted by
                matching priority (frame of reference matches first).

        Example:
            >>> dicom_file = DicomStructureFile(top_dir=Path("data"),
                                                file_name="struct.dcm")
            >>> image_files = dicom_file.find_image_files()
            >>> print(f"Found {len(image_files)} matching image files")
        '''
        if modalities is None:
            modalities = ['CT', 'MR', 'PT', 'RTIMAGE']

        if not self.is_structure_file():
            logger.warning('File %s is not a RT Structure file', self.file_name)
            return []

        # Get reference information from structure file
        patient_id = str(self.dataset.get('PatientID', ''))
        study_instance_uid = str(self.dataset.get('StudyInstanceUID', ''))

        # Get frame of reference UID from structure set (prioritized matching
        # criterion)
        frame_of_ref_uid = None
        if hasattr(self.dataset, 'StructureSetROISequence'):
            for roi in self.dataset.StructureSetROISequence:
                if hasattr(roi, 'ReferencedFrameOfReferenceUID'):
                    frame_of_ref_uid = str(roi.ReferencedFrameOfReferenceUID)
                    break

        logger.debug('Searching for images matching Patient ID: %s', patient_id)
        if match_study:
            logger.debug('Study Instance UID: %s', study_instance_uid)
        if frame_of_ref_uid:
            logger.debug('Frame of Reference UID (PRIORITY): %s',
                         frame_of_ref_uid)

        frame_ref_matches = []  # Highest priority: frame of reference matches
        other_matches = []      # Lower priority: patient/study matches without frame ref

        # Search all DICOM files in top_dir and subdirectories
        for dcm_file in self.top_dir.rglob('*.dcm'):
            if dcm_file == self.file_path:
                continue  # Skip the structure file itself

            try:
                # Try to read the file efficiently (headers only)
                file_dataset = pydicom.dcmread(dcm_file, stop_before_pixels=True)

                # Check if it's an image file with the right modality
                file_modality = str(file_dataset.get('Modality', ''))
                if file_modality not in modalities:
                    continue

                # Check patient matching (required baseline)
                if match_patient:
                    file_patient_id = str(file_dataset.get('PatientID', ''))
                    if file_patient_id != patient_id:
                        continue

                # Check study matching if required
                study_matches = True
                if match_study:
                    file_study_uid = str(file_dataset.get('StudyInstanceUID', ''))
                    study_matches = file_study_uid == study_instance_uid
                    if not study_matches:
                        continue

                # PRIORITIZED: Check frame of reference matching (most reliable
                # for spatial alignment)
                if frame_of_ref_uid and hasattr(file_dataset,
                                                'FrameOfReferenceUID'):
                    file_frame_ref = str(file_dataset.get('FrameOfReferenceUID',
                                                          ''))
                    if file_frame_ref == frame_of_ref_uid:
                        frame_ref_matches.append(dcm_file)
                        logger.debug('Found PRIORITY match (Frame of Reference): %s',
                                     dcm_file.name)
                        continue

                # Secondary matches: patient/study criteria met but no frame
                # reference match
                if patient_id and (not match_study or study_matches):
                    other_matches.append(dcm_file)
                    logger.debug('Found secondary match (Patient/Study): %s',
                                  dcm_file.name)

            except (pydicom.errors.InvalidDicomError) as e:
                logger.debug('Skipping non-DICOM or invalid file %s: %s',
                              dcm_file, e)
                continue

        # Combine results with frame reference matches first (prioritized)
        all_matches = sorted(frame_ref_matches) + sorted(other_matches)

        logger.info('Found %d frame-of-reference matches and %d other matches '
                    'for structure set %s',
                    len(frame_ref_matches), len(other_matches), self.file_name)

        return all_matches

    def calculate_structure_resolution(self,
                                     modalities: Optional[List[str]] = None,
                                     use_priority_match: bool = True
                                     ) -> Optional[float]:
        '''Calculate the structure resolution based on corresponding image files.

        Finds matching image files and calculates resolution as:
        resolution = image_diameter / (2 * image_x_pixels)
        Where image_diameter is the field of view diameter and image_x_pixels
        is the number of pixels in the x-direction.
        This implies that the resolution is half of the distance between pixels.

        Args:
            modalities (List[str], optional): List of modalities to search for.
                Defaults to ['CT', 'MR', 'PT'] for common imaging modalities.
            use_priority_match (bool, optional): Whether to prefer
                frame-of-reference matches over other matches. Defaults to True.

        Returns:
            Optional[float]: The calculated resolution in cm/pixel, or None if no
                suitable image files are found or resolution cannot be calculated.

        Example:
            >>> dicom_file = DicomStructureFile(top_dir=Path("data"),
                                                file_name="struct.dcm")
            >>> resolution = dicom_file.calculate_structure_resolution()
            >>> if resolution:
            ...     print(f"Structure resolution: {resolution:.4f} cm/pixel")
        '''
        if modalities is None:
            modalities = ['CT', 'MR', 'PT']  # Exclude RTIMAGE for resolution calc

        # Find matching image files
        image_files = self.find_image_files(modalities=modalities)

        if not image_files:
            logger.warning('No matching image files found, attempting to use '
                           'structure dimensions')
            return self._calculate_resolution_from_structures()

        # Select the image file to use for resolution calculation
        selected_file = None

        if use_priority_match:
            # Try to find a frame-of-reference match first
            frame_of_ref_uid = None
            if hasattr(self.dataset, 'StructureSetROISequence'):
                for roi in self.dataset.StructureSetROISequence:
                    if hasattr(roi, 'ReferencedFrameOfReferenceUID'):
                        frame_of_ref_uid = str(roi.ReferencedFrameOfReferenceUID)
                        break

            # Check for frame-of-reference matches
            if frame_of_ref_uid:
                for img_file in image_files:
                    try:
                        img_dataset = pydicom.dcmread(img_file,
                                                      stop_before_pixels=True)
                        if hasattr(img_dataset, 'FrameOfReferenceUID'):
                            file_frame_ref = str(img_dataset.get(
                                'FrameOfReferenceUID', ''))
                            if file_frame_ref == frame_of_ref_uid:
                                selected_file = img_file
                                logger.debug('Using frame-of-reference match: %s',
                                             img_file.name)
                                break
                    except pydicom.errors.InvalidDicomError as e:
                        logger.debug('Error checking frame reference for %s: %s',
                                     img_file, e)
                        continue

        # If no priority match found, use the first available image file
        if selected_file is None:
            selected_file = image_files[0]
            logger.debug('Using first available image file: %s',
                         selected_file.name)

        try:
            # Load the selected image file
            img_dataset = pydicom.dcmread(selected_file,
                                          stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError as e:
            logger.error('Error calculating resolution from image file %s: %s',
                         selected_file, e)
            return None
        # Get image dimensions
        if not hasattr(img_dataset, 'Rows') or not hasattr(img_dataset,
                                                           'Columns'):
            logger.error('Image file %s missing dimension information',
                            selected_file.name)
            return None

        # Number of pixels in x-direction
        image_x_pixels = int(img_dataset.Columns)
        # Number of pixels in y-direction
        image_y_pixels = int(img_dataset.Rows)
        # Calculate image diameter (field of view)
        # Try to get pixel spacing first
        pixel_spacing = None
        if hasattr(img_dataset, 'PixelSpacing'):
            # PixelSpacing is [row_spacing, column_spacing] in mm
            pixel_spacing = img_dataset.PixelSpacing
            row_spacing_mm = float(pixel_spacing[0])
            col_spacing_mm = float(pixel_spacing[1])

            # Calculate field of view in mm, then convert to cm
            fov_x_mm = image_x_pixels * col_spacing_mm
            fov_y_mm = image_y_pixels * row_spacing_mm

            # Use the larger dimension as diameter and convert mm to cm
            image_diameter_cm = max(fov_x_mm, fov_y_mm) / 10.0

        elif hasattr(img_dataset, 'FOV') or hasattr(img_dataset,
                                                    'ReconstructionDiameter'):
            # Some scanners store FOV or reconstruction diameter directly
            if hasattr(img_dataset, 'ReconstructionDiameter'):
                # Convert mm to cm
                image_diameter_cm = float(img_dataset.ReconstructionDiameter) / 10.0
            elif hasattr(img_dataset, 'FOV'):
                # Convert mm to cm
                image_diameter_cm = float(img_dataset.FOV) / 10.0
            else:
                image_diameter_cm = None

        else:
            logger.warning('No pixel spacing or FOV information found in %s',
                            selected_file.name)
            # Use a reasonable default assumption for CT scans (50 cm diameter)
            image_diameter_cm = 50.0
            logger.warning('Using default image diameter of %.1f cm',
                            image_diameter_cm)
        if image_diameter_cm is None:
            logger.error('Could not determine image diameter for %s',
                            selected_file.name)
            return None

        # Calculate resolution: image_diameter / (2 * image_x_pixels)
        resolution = image_diameter_cm / (2.0 * image_x_pixels)

        # Round up to single decimal place
        resolution = round_one_up(resolution)

        logger.info('Calculated resolution: %.1f cm/pixel from image %s',
                    resolution, selected_file.name)
        logger.debug('Image dimensions: %d×%d pixels, diameter: %.2f cm',
                        image_x_pixels, image_y_pixels, image_diameter_cm)

        return resolution

    def _calculate_resolution_from_structures(self, default_pixels: int = 512
                                              ) -> Optional[float]:
        '''Calculate resolution using BODY or EXTERNAL structure dimensions.

        Using contour dimensions when image files are not available. Searches
        for BODY or EXTERNAL structures. If BODY or EXTERNAL structure is not
        found, falls back to using the largest structure.  The maximum x or y
        dimension of the structure is used to estimate the pixel resolution.
        maximum dimension / (2 *  # pixels) = resolution in cm/pixel.
        This implies that the resolution is half of the distance between pixels.

        Args:
            default_pixels (int, optional): Default number of pixels to assume.
                Defaults to 512.

        Returns:
            Optional[float]: The calculated resolution in cm/pixel, or None if no
                suitable structure is found.
        '''
        # Look for BODY or EXTERNAL structures in the structure names
        structure_names = self.structure_names
        if not structure_names:
            logger.warning('No structure names available for resolution '
                           'calculation')
            return None

        # Search for BODY or EXTERNAL structures (case-insensitive)
        target_structures = ['BODY', 'EXTERNAL', 'PATIENT', 'SKIN']
        body_roi = None
        body_name = None

        for roi_num, struct_name in structure_names.items():
            struct_name_upper = struct_name.upper()
            for target in target_structures:
                if target in struct_name_upper:
                    body_roi = roi_num
                    body_name = struct_name
                    logger.debug('Found structure for resolution calculation: '
                                 'ROI %s - %r',
                                 roi_num, struct_name)
                    break
            if body_roi is not None:
                break

        if body_roi is None:
            logger.warning('No BODY/EXTERNAL structure found among: %s',
                         list(structure_names.values()))
            # Fall back to using the largest structure
            body_roi = self._find_largest_structure()
            if body_roi is None:
                logger.error('No suitable structure found for resolution '
                             'calculation')
                return None
            logger.info('Using largest structure for resolution calculation: ROI %s',
                        body_roi)

        # Get contour points for the body structure
        if self.contour_points is None:
            self.get_contour_points()

        if not self.contour_points:
            logger.warning("No contour points available for resolution calculation")
            return None

        # Find contour points for the body ROI
        body_contours = [cp for cp in self.contour_points if cp['ROI'] == body_roi]

        if not body_contours:
            body_name = structure_names.get(body_roi, f"ROI_{body_roi}")
            logger.warning("No contour points found for structure ROI %s ('%s')",
                           body_roi, body_name)
            return None

        # Calculate the bounding box of all contour points for this structure
        all_x_coords = []
        all_y_coords = []

        for contour in body_contours:
            # contour['Points'] contains the coordinate points
            points = np.array(contour['Points'])
            x_coords = points[:, 0]  # x coordinates
            y_coords = points[:, 1]  # y coordinates

            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)

        if not all_x_coords or not all_y_coords:
            body_name = structure_names.get(body_roi, f"ROI_{body_roi}")
            logger.warning("No valid coordinates found for structure '%s'",
                           body_name)
            return None

        # Calculate the extent in x and y directions
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)

        x_extent = x_max - x_min  # in cm (contour coordinates are in cm)
        y_extent = y_max - y_min  # in cm

        # Calculate the maximum dimension
        max_dimension = max(x_extent, y_extent)

        logger.debug('Structure extents: x=%.2f cm, y=%.2f cm, '
                     'maximum dimension=%.2f cm, pixels=%d',
                     x_extent, y_extent, max_dimension, default_pixels)
        # Calculate resolution in cm/pixel: maximum dimension / ( 2 * # pixels)
        resolution = max_dimension / ( 2 * default_pixels)
        logger.debug('Un-Rounded Resolution calculated as: %.4f cm', resolution)
        # Round up to single decimal place
        resolution = round_one_up(resolution)

        body_name = structure_names.get(body_roi, f"ROI_{body_roi}")
        logger.info('Calculated resolution from structure %r: %.3f cm/pixel',
                    body_name, resolution)

        return resolution

    def round_contour_points(self) -> None:
        '''Round contour points based on the calculated resolution.

        Rounds the x and y coordinates of all contour points to the nearest
        resolution increment. Z coordinates are left unchanged as they typically
        represent slice positions that should remain precise.

        The rounding is applied in-place to the existing contour_points list.
        '''
        if self.resolution is None:
            logger.warning('No resolution available for rounding contour points')
            return

        if not self.contour_points:
            logger.debug('No contour points to round')
            return

        rounded_count = 0

        for contour in self.contour_points:
            # Round x and y coordinates to nearest resolution increment
            # Z coordinates (slice positions) are left unchanged
            points = np.array(contour['Points'])
            original_points = points.copy()

            # Round x and y coordinates
            points[:, 0] = np.round(points[:, 0] / self.resolution) * self.resolution
            points[:, 1] = np.round(points[:, 1] / self.resolution) * self.resolution
            # Z coordinates ([:, 2]) are left unchanged

            # Update the contour with rounded points
            contour['Points'] = points.tolist()

            # Count how many points were actually changed
            if not np.array_equal(original_points, points):
                rounded_count += 1

        logger.info('Rounded %d contours to resolution of %.1f cm/pixel',
                    rounded_count, self.resolution)
        logger.debug('Total contour points processed: %d',
                     len(self.contour_points))

    def _find_largest_structure(self) -> Optional[int]:
        '''Find the ROI of the largest structure by bounding box area.
        Returns:
            roi_number of the largest structure, or None if not found.
        '''
        if not self.contour_points:
            logger.warning('No contour points available to find largest structure')
            return None
        structure_names = self.structure_names
        if not structure_names:
            logger.warning("No structure names available")
            return None
        # Calculate bounding box area for each structure
        structure_areas = {}
        # Group contours by ROI
        roi_contours = {}
        for contour in self.contour_points:
            roi = contour['ROI']
            if roi not in roi_contours:
                roi_contours[roi] = []
            roi_contours[roi].append(contour)
        # Calculate bounding box area for each ROI
        for roi, contours in roi_contours.items():
            all_x_coords = []
            all_y_coords = []
            for contour in contours:
                points = np.array(contour['Points'])
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
            if all_x_coords and all_y_coords:
                x_extent = max(all_x_coords) - min(all_x_coords)
                y_extent = max(all_y_coords) - min(all_y_coords)
                area = x_extent * y_extent
                structure_areas[roi] = area
                struct_name = structure_names.get(roi, f'ROI_{roi}')
                logger.debug('Structure %s (ROI %s): '
                           'extent %.2fx%.2f cm, area %.2f cm²',
                           struct_name, roi, x_extent, y_extent, area)
        if not structure_areas:
            logger.warning("No structures with valid coordinates found")
            return None
        largest_roi = max(structure_areas, key=structure_areas.get)
        largest_area = structure_areas[largest_roi]
        largest_name = structure_names.get(largest_roi, f'ROI_{largest_roi}')
        logger.debug('Largest structure: %s (ROI %s) with area %.2f cm²',
                     largest_name, largest_roi, largest_area)
        return largest_roi
