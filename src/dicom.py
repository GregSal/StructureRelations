'''DICOM Structure File handling module.
'''
# %% Imports
from typing import Optional, Union, List
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import pydicom

from contours import ContourPoints
from types_and_classes import ROI_Type, SliceIndexType

# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        structure_names (pd.DataFrame): Table containing all roi number and structure names.
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
            pydicom.errors.InvalidDicomError: If the file is not a valid DICOM file.
        '''
        self.top_dir = Path(top_dir)
        
        # Handle file path/name logic
        if file_path is not None:
            self.file_path = Path(file_path)
            self.file_name = self.file_path.name
        elif file_name is not None:
            self.file_name = file_name
            self.file_path = self.top_dir / file_name
        else:
            raise ValueError("Either file_path or file_name must be provided")
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {self.file_path}")
        
        # Handle dataset
        if dataset is not None:
            self.dataset = dataset
            logger.debug(f"Using provided dataset for {self.file_name}")
        else:
            logger.debug(f"Loading dataset from {self.file_path}")
            self._load_dataset()
        
        # Initialize contour attributes
        self.structure_set_info = self.get_structure_set_info()
        self.contour_points = self.get_contour_points()
        self.structure_names = self.get_structure_names()

    def _load_dataset(self) -> None:
        '''Load the DICOM dataset from file.
        
        Raises:
            pydicom.errors.InvalidDicomError: If the file is not a valid DICOM file.
        '''
        try:
            self.dataset = pydicom.dcmread(self.file_path)
            logger.info(f"Successfully loaded DICOM dataset from {self.file_name}")
        except pydicom.errors.InvalidDicomError as e:
            logger.error(f"Failed to load DICOM file {self.file_path}: {e}")
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
            logger.warning(f"File {self.file_name} is not a RT Structure file")
            return {}
        
        full_name = str(self.dataset.get('PatientName', ''))
        last_name = full_name.split('^')[0] if '^' in full_name else full_name
        
        structure_info = {
            'PatientName': full_name,
            'PatientLastName': last_name,
            'PatientID': str(self.dataset.get('PatientID', '')),
            'StructureSet': str(self.dataset.get('StructureSetLabel', '')),
            'StudyID': str(self.dataset.get('StudyID', '')),
            'SeriesNumber': str(self.dataset.get('SeriesNumber', '')),
            'File': self.file_path
        }
        
        logger.debug(f"Retrieved structure info for {structure_info['StructureSet']}")
        self.structure_set_info = structure_info
        return structure_info
    
    def __repr__(self) -> str:
        '''String representation of the DicomStructureFile.'''
        return f"DicomStructureFile(file_name='{self.file_name}', top_dir='{self.top_dir}')"
    
    def __str__(self) -> str:
        '''Human-readable string representation.'''
        info = self.get_structure_set_info()
        if info:
            return f"DICOM RT Structure: {info.get('StructureSet', 'Unknown')} for patient {info.get('PatientID', 'Unknown')}"
        return f"DICOM file: {self.file_name}"
    
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
                logger.debug(f"ROI {roi_number} has no contour sequence")
                continue
            
            for contour in roi.ContourSequence:
                # Get the geometric type (should be 'CLOSED_PLANAR' for most cases)
                geom_type = getattr(contour, 'ContourGeometricType', 'UNKNOWN')
                
                if geom_type not in ['CLOSED_PLANAR', 'POINT', 'OPEN_PLANAR']:
                    logger.warning(f"Unsupported geometric type {geom_type} in ROI {roi_number}")
                    continue
                if geom_type not in ['CLOSED_PLANAR']:
                    logger.warning(f"Skipping non-closed planar contour in ROI {roi_number}")
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
                    logger.warning(f"Invalid or insufficient contour data in ROI {roi_number}")
        
        logger.info(f"Extracted {len(contour_points_list)} contours from {len(self.dataset.ROIContourSequence)} ROIs")
        
        # Store contour points and convert to DataFrame
        return contour_points_list
    
    def get_structure_names(self) -> pd.DataFrame:
        '''Get ROI names and numbers from the structure set.
        
        Returns:
            pd.DataFrame: DataFrame with columns 'ROINumber' and 'StructureID' 
                containing the mapping of ROI numbers to structure names.
        '''
        if not self.is_structure_file():
            logger.warning(f"File {self.file_name} is not a RT Structure file")
            return pd.DataFrame()
        
        if not hasattr(self.dataset, 'StructureSetROISequence'):
            logger.warning("Dataset does not contain StructureSetROISequence")
            return pd.DataFrame()
        
        structure_data = {}
        for roi in self.dataset.StructureSetROISequence:
            structure_data[ROI_Type(roi.ROINumber)] = str(roi.ROIName)        
        return structure_data
    
    def filter_exclusions(self, 
                         exclude_prefixes: List[str] = None,
                         case_sensitive: bool = False,
                         exclude_empty: bool = True) -> List[ContourPoints]:
        '''Filter out unwanted structures from the contour points list.
        
        This method is similar to drop_exclusions in RS_DICOM_Utilities.py.
        It removes structures based on naming patterns and optionally excludes 
        structures without contour data.
        
        Args:
            exclude_prefixes (List[str], optional): List of prefixes to exclude.
                Defaults to ['x', 'z'] which are commonly used for temporary
                or excluded structures.
            case_sensitive (bool, optional): Whether prefix matching should be 
                case sensitive. Defaults to False.
            exclude_empty (bool, optional): Whether to exclude ROIs that have
                no contour points. Defaults to True.
                
        Returns:
            List[ContourPoints]: Filtered list of contour points with excluded
                structures removed.
        '''
        if exclude_prefixes is None:
            exclude_prefixes = ['x', 'z']
        
        # Get structure names
        roi_name_lookup = self.structure_names
        if not roi_name_lookup:
            logger.warning("No structure names found, cannot filter exclusions")
            return self.contour_points or []
        
        # Get contour points if not already loaded
        if self.contour_points is None:
            self.get_contour_points()
        
        if not self.contour_points:
            logger.info("No contour points to filter")
            return []
                
        # Identify ROIs to exclude based on naming patterns
        excluded_rois = set()
        
        for roi_num, struct_name in roi_name_lookup.items():
            struct_name_check = struct_name if case_sensitive else struct_name.lower()
            
            for prefix in exclude_prefixes:
                prefix_check = prefix if case_sensitive else prefix.lower()
                
                if struct_name_check.startswith(prefix_check):
                    excluded_rois.add(roi_num)
                    logger.debug(f"Excluding ROI {roi_num} ('{struct_name}') - matches prefix '{prefix}'")
                    break
        
        # If excluding empty structures, identify ROIs with no contour data
        if exclude_empty:
            # Get ROIs that have contour points
            rois_with_contours = set(cp['ROI'] for cp in self.contour_points)
            
            # Find ROIs in structure set but not in contour points
            all_rois = set(self.structure_names.keys())
            empty_rois = all_rois - rois_with_contours
            
            for roi_num in empty_rois:
                excluded_rois.add(roi_num)
                struct_name = roi_name_lookup.get(roi_num, 'Unknown')
                logger.debug(f"Excluding ROI {roi_num} ('{struct_name}') - no contour data")
        
        # Filter contour points
        filtered_contour_points = [
            cp for cp in self.contour_points 
            if cp['ROI'] not in excluded_rois
        ]
        
        # Update stored contour points and DataFrame
        original_count = len(self.contour_points)
        filtered_count = len(filtered_contour_points)
        excluded_count = original_count - filtered_count
        
        logger.info(f"Filtered {excluded_count} contours from {len(excluded_rois)} excluded ROIs. "
                   f"Remaining: {filtered_count} contours from {len(set(cp['ROI'] for cp in filtered_contour_points))} ROIs")
        
        # Optionally update the stored data
        self.contour_points = filtered_contour_points        
        return filtered_contour_points
    
    def get_excluded_structures(self, 
                               exclude_prefixes: List[str] = None,
                               case_sensitive: bool = False) -> pd.DataFrame:
        '''Get information about structures that would be excluded by filtering.
        
        This method returns information about structures that would be excluded
        without actually filtering them, useful for inspection and validation.
        
        Args:
            exclude_prefixes (List[str], optional): List of prefixes to check for exclusion.
                Defaults to ['x', 'z'].
            case_sensitive (bool, optional): Whether prefix matching should be 
                case sensitive. Defaults to False.
                
        Returns:
            pd.DataFrame: DataFrame with information about excluded structures
                including ROINumber, StructureID, and exclusion reason.
        '''
        if exclude_prefixes is None:
            exclude_prefixes = ['x', 'z']
        
        structure_names = self.get_structure_names()
        if structure_names.empty:
            return pd.DataFrame()
        
        # Get contour points to check for empty structures
        if self.contour_points is None:
            self.get_contour_points()
        
        rois_with_contours = set(cp.roi for cp in self.contour_points) if self.contour_points else set()
        
        excluded_info = []
        
        for _, row in structure_names.iterrows():
            roi_num = row['ROINumber']
            struct_name = row['StructureID']
            exclusion_reasons = []
            
            # Check naming patterns
            struct_name_check = struct_name if case_sensitive else struct_name.lower()
            
            for prefix in exclude_prefixes:
                prefix_check = prefix if case_sensitive else prefix.lower()
                
                if struct_name_check.startswith(prefix_check):
                    exclusion_reasons.append(f"Name starts with '{prefix}'")
                    break
            
            # Check for empty contours
            if roi_num not in rois_with_contours:
                exclusion_reasons.append("No contour data")
            
            # Only include structures that would be excluded
            if exclusion_reasons:
                excluded_info.append({
                    'ROINumber': roi_num,
                    'StructureID': struct_name,
                    'ExclusionReason': '; '.join(exclusion_reasons)
                })
        
        return pd.DataFrame(excluded_info)