'''DICOM Structure File handling module.
'''
# %% Imports
from typing import Optional, Union, List
from pathlib import Path
import logging
import math

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
        resolution (Optional[float]): The calculated structure resolution in cm/pixel.
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
        
        # Calculate resolution (may use structure dimensions as fallback)
        self.resolution = self.calculate_structure_resolution()
        
        # Round contour points based on calculated resolution
        if self.resolution is not None:
            self.round_contour_points()

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
    
    def find_image_files(self, 
                        modalities: Optional[List[str]] = None,
                        match_study: bool = True,
                        match_patient: bool = True) -> List[Path]:
        '''Find corresponding image files under top_dir that match this structure set.
        
        Searches for DICOM image files (CT, MR, etc.) that correspond to this
        RT Structure file. Prioritizes frame of reference UID matching for the 
        most reliable spatial alignment, then falls back to study/patient matching.
        
        Args:
            modalities (List[str], optional): List of modalities to search for.
                Defaults to ['CT', 'MR', 'PT', 'RTIMAGE'] for common imaging modalities.
            match_study (bool, optional): Whether to match Study Instance UID.
                Defaults to True.
            match_patient (bool, optional): Whether to match Patient ID.
                Defaults to True.
                
        Returns:
            List[Path]: List of paths to matching DICOM image files, sorted by 
                matching priority (frame of reference matches first).
                
        Example:
            >>> dicom_file = DicomStructureFile(top_dir=Path("data"), file_name="struct.dcm")
            >>> image_files = dicom_file.find_image_files()
            >>> print(f"Found {len(image_files)} matching image files")
        '''
        if modalities is None:
            modalities = ['CT', 'MR', 'PT', 'RTIMAGE']
        
        if not self.is_structure_file():
            logger.warning(f"File {self.file_name} is not a RT Structure file")
            return []
        
        # Get reference information from structure file
        patient_id = str(self.dataset.get('PatientID', ''))
        study_instance_uid = str(self.dataset.get('StudyInstanceUID', ''))
        
        # Get frame of reference UID from structure set (prioritized matching criterion)
        frame_of_ref_uid = None
        if hasattr(self.dataset, 'StructureSetROISequence'):
            for roi in self.dataset.StructureSetROISequence:
                if hasattr(roi, 'ReferencedFrameOfReferenceUID'):
                    frame_of_ref_uid = str(roi.ReferencedFrameOfReferenceUID)
                    break
        
        logger.debug(f"Searching for images matching Patient ID: {patient_id}")
        if match_study:
            logger.debug(f"Study Instance UID: {study_instance_uid}")
        if frame_of_ref_uid:
            logger.debug(f"Frame of Reference UID (PRIORITY): {frame_of_ref_uid}")
        
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
                    study_matches = (file_study_uid == study_instance_uid)
                    if not study_matches:
                        continue
                
                # PRIORITIZED: Check frame of reference matching (most reliable for spatial alignment)
                if frame_of_ref_uid and hasattr(file_dataset, 'FrameOfReferenceUID'):
                    file_frame_ref = str(file_dataset.get('FrameOfReferenceUID', ''))
                    if file_frame_ref == frame_of_ref_uid:
                        frame_ref_matches.append(dcm_file)
                        logger.debug(f"Found PRIORITY match (Frame of Reference): {dcm_file.name}")
                        continue
                
                # Secondary matches: patient/study criteria met but no frame reference match
                if patient_id and (not match_study or study_matches):
                    other_matches.append(dcm_file)
                    logger.debug(f"Found secondary match (Patient/Study): {dcm_file.name}")
                    
            except (pydicom.errors.InvalidDicomError, Exception) as e:
                logger.debug(f"Skipping non-DICOM or invalid file {dcm_file}: {e}")
                continue
        
        # Combine results with frame reference matches first (prioritized)
        all_matches = sorted(frame_ref_matches) + sorted(other_matches)
        
        logger.info(f"Found {len(frame_ref_matches)} frame-of-reference matches and "
                   f"{len(other_matches)} other matches for structure set {self.file_name}")
        
        return all_matches
    
    def calculate_structure_resolution(self, 
                                     modalities: Optional[List[str]] = None,
                                     use_priority_match: bool = True) -> Optional[float]:
        '''Calculate the structure resolution based on corresponding image files.
        
        Finds matching image files and calculates resolution as:
        resolution = image_diameter / (2 * image_x_pixels)
        
        Where image_diameter is the field of view diameter and image_x_pixels 
        is the number of pixels in the x-direction.
        
        Args:
            modalities (List[str], optional): List of modalities to search for.
                Defaults to ['CT', 'MR', 'PT'] for common imaging modalities.
            use_priority_match (bool, optional): Whether to prefer frame-of-reference
                matches over other matches. Defaults to True.
                
        Returns:
            Optional[float]: The calculated resolution in cm/pixel, or None if no
                suitable image files are found or resolution cannot be calculated.
                
        Example:
            >>> dicom_file = DicomStructureFile(top_dir=Path("data"), file_name="struct.dcm")
            >>> resolution = dicom_file.calculate_structure_resolution()
            >>> if resolution:
            ...     print(f"Structure resolution: {resolution:.4f} cm/pixel")
        '''
        if modalities is None:
            modalities = ['CT', 'MR', 'PT']  # Exclude RTIMAGE for resolution calc
        
        # Find matching image files
        image_files = self.find_image_files(modalities=modalities)
        
        if not image_files:
            logger.warning(f"No matching image files found, attempting to use structure dimensions")
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
                        img_dataset = pydicom.dcmread(img_file, stop_before_pixels=True)
                        if hasattr(img_dataset, 'FrameOfReferenceUID'):
                            file_frame_ref = str(img_dataset.get('FrameOfReferenceUID', ''))
                            if file_frame_ref == frame_of_ref_uid:
                                selected_file = img_file
                                logger.debug(f"Using frame-of-reference match: {img_file.name}")
                                break
                    except Exception as e:
                        logger.debug(f"Error checking frame reference for {img_file}: {e}")
                        continue
        
        # If no priority match found, use the first available image file
        if selected_file is None:
            selected_file = image_files[0]
            logger.debug(f"Using first available image file: {selected_file.name}")
        
        try:
            # Load the selected image file
            img_dataset = pydicom.dcmread(selected_file, stop_before_pixels=True)
            
            # Get image dimensions
            if not hasattr(img_dataset, 'Rows') or not hasattr(img_dataset, 'Columns'):
                logger.error(f"Image file {selected_file.name} missing dimension information")
                return None
            
            image_x_pixels = int(img_dataset.Columns)  # Number of pixels in x-direction
            image_y_pixels = int(img_dataset.Rows)     # Number of pixels in y-direction
            
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
                
                # Use the larger dimension as diameter
                image_diameter_cm = max(fov_x_mm, fov_y_mm) / 10.0  # Convert mm to cm
                
            elif hasattr(img_dataset, 'FOV') or hasattr(img_dataset, 'ReconstructionDiameter'):
                # Some scanners store FOV or reconstruction diameter directly
                if hasattr(img_dataset, 'ReconstructionDiameter'):
                    image_diameter_cm = float(img_dataset.ReconstructionDiameter) / 10.0  # Convert mm to cm
                elif hasattr(img_dataset, 'FOV'):
                    image_diameter_cm = float(img_dataset.FOV) / 10.0  # Convert mm to cm
                else:
                    image_diameter_cm = None
                    
            else:
                logger.warning(f"No pixel spacing or FOV information found in {selected_file.name}")
                # Use a reasonable default assumption for CT scans (50 cm diameter)
                image_diameter_cm = 50.0
                logger.warning(f"Using default image diameter of {image_diameter_cm} cm")
            
            if image_diameter_cm is None:
                logger.error(f"Could not determine image diameter for {selected_file.name}")
                return None
            
            # Calculate resolution: image_diameter / (2 * image_x_pixels)
            resolution = image_diameter_cm / (2.0 * image_x_pixels)
            
            # Round up to single decimal place
            resolution = math.ceil(resolution * 10) / 10
            
            logger.info(f"Calculated resolution: {resolution:.1f} cm/pixel from image {selected_file.name}")
            logger.debug(f"Image dimensions: {image_x_pixels}×{image_y_pixels} pixels, "
                        f"diameter: {image_diameter_cm:.2f} cm")
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error calculating resolution from image file {selected_file}: {e}")
            return None
    
    def _calculate_resolution_from_structures(self, 
                                            default_pixels: int = 512) -> Optional[float]:
        '''Calculate resolution using BODY or EXTERNAL structure dimensions.
        
        Falls back to using structure contour dimensions when image files are not available.
        Searches for BODY or EXTERNAL structures and calculates diameter from their extent.
        
        Args:
            default_pixels (int, optional): Default number of pixels to assume.
                Defaults to 512.
                
        Returns:
            Optional[float]: The calculated resolution in cm/pixel, or None if no
                suitable structure is found.
        '''
        # Look for BODY or EXTERNAL structures in the structure names
        structure_names = self.get_structure_names()
        if not structure_names:
            logger.warning("No structure names available for resolution calculation")
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
                    logger.debug(f"Found structure for resolution calculation: ROI {roi_num} - '{struct_name}'")
                    break
            if body_roi is not None:
                break
        
        if body_roi is None:
            logger.warning(f"No BODY/EXTERNAL structure found among: {list(structure_names.values())}")
            # Fall back to using the largest structure
            body_roi, body_name = self._find_largest_structure()
            if body_roi is None:
                logger.error("No suitable structure found for resolution calculation")
                return None
            logger.info(f"Using largest structure for resolution calculation: ROI {body_roi} - '{body_name}'")
        
        # Get contour points for the body structure
        if self.contour_points is None:
            self.get_contour_points()
        
        if not self.contour_points:
            logger.warning("No contour points available for resolution calculation")
            return None
        
        # Find contour points for the body ROI
        body_contours = [cp for cp in self.contour_points if cp.roi == body_roi]
        
        if not body_contours:
            logger.warning(f"No contour points found for structure ROI {body_roi} ('{body_name}')")
            return None
        
        # Calculate the bounding box of all contour points for this structure
        all_x_coords = []
        all_y_coords = []
        
        for contour in body_contours:
            # contour.points is a numpy array of shape (n, 3) with x, y, z coordinates
            x_coords = contour.points[:, 0]  # x coordinates
            y_coords = contour.points[:, 1]  # y coordinates
            
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
        
        if not all_x_coords or not all_y_coords:
            logger.warning(f"No valid coordinates found for structure '{body_name}'")
            return None
        
        # Calculate the extent in x and y directions
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        
        x_extent = x_max - x_min  # in cm (contour coordinates are in cm)
        y_extent = y_max - y_min  # in cm
        
        # Calculate diameter using sqrt(x^2 + y^2)
        diameter_cm = np.sqrt(x_extent**2 + y_extent**2)
        
        # Calculate resolution: diameter / (2 * default_pixels)
        resolution = diameter_cm / (2.0 * default_pixels)
        
        # Round up to single decimal place
        resolution = math.ceil(resolution * 10) / 10
        
        logger.info(f"Calculated resolution from structure '{body_name}': {resolution:.1f} cm/pixel")
        logger.debug(f"Structure extents: x={x_extent:.2f} cm, y={y_extent:.2f} cm, "
                    f"diameter={diameter_cm:.2f} cm, pixels={default_pixels}")
        
        return resolution
    
    def round_contour_points(self) -> None:
        '''Round contour points based on the calculated resolution.
        
        Rounds the x and y coordinates of all contour points to the nearest
        resolution increment. Z coordinates are left unchanged as they typically
        represent slice positions that should remain precise.
        
        The rounding is applied in-place to the existing contour_points list.
        '''
        if self.resolution is None:
            logger.warning("No resolution available for rounding contour points")
            return
            
        if not self.contour_points:
            logger.debug("No contour points to round")
            return
        
        rounded_count = 0
        
        for contour in self.contour_points:
            # Round x and y coordinates to nearest resolution increment
            # Z coordinates (slice positions) are left unchanged
            original_points = contour.points.copy()
            
            # Round x and y coordinates
            contour.points[:, 0] = np.round(contour.points[:, 0] / self.resolution) * self.resolution
            contour.points[:, 1] = np.round(contour.points[:, 1] / self.resolution) * self.resolution
            # Z coordinates ([:, 2]) are left unchanged
            
            # Count how many points were actually changed
            if not np.array_equal(original_points, contour.points):
                rounded_count += 1
        
        logger.info(f"Rounded {rounded_count} contours to resolution of {self.resolution:.1f} cm/pixel")
        logger.debug(f"Total contour points processed: {len(self.contour_points)}")
    
    def _find_largest_structure(self) -> tuple[Optional[int], Optional[str]]:
        '''Find the largest structure by calculating bounding box area.
        
        Returns:
            tuple: (roi_number, structure_name) of the largest structure,
                or (None, None) if no suitable structure is found.
        '''
        if not self.contour_points:
            logger.warning("No contour points available to find largest structure")
            return None, None
        
        structure_names = self.get_structure_names()
        if not structure_names:
            logger.warning("No structure names available")
            return None, None
        
        # Calculate bounding box area for each structure
        structure_areas = {}
        
        # Group contours by ROI
        roi_contours = {}
        for contour in self.contour_points:
            roi = contour.roi
            if roi not in roi_contours:
                roi_contours[roi] = []
            roi_contours[roi].append(contour)
        
        # Calculate bounding box area for each ROI
        for roi, contours in roi_contours.items():
            all_x_coords = []
            all_y_coords = []
            
            for contour in contours:
                x_coords = contour.points[:, 0]  # x coordinates
                y_coords = contour.points[:, 1]  # y coordinates
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
            
            if all_x_coords and all_y_coords:
                x_extent = max(all_x_coords) - min(all_x_coords)
                y_extent = max(all_y_coords) - min(all_y_coords)
                area = x_extent * y_extent  # Bounding box area in cm²
                structure_areas[roi] = area
                
                struct_name = structure_names.get(roi, f"ROI_{roi}")
                logger.debug(f"Structure {struct_name} (ROI {roi}): "
                           f"extent {x_extent:.2f}×{y_extent:.2f} cm, area {area:.2f} cm²")
        
        if not structure_areas:
            logger.warning("No structures with valid coordinates found")
            return None, None
        
        # Find the ROI with the largest bounding box area
        largest_roi = max(structure_areas, key=structure_areas.get)
        largest_area = structure_areas[largest_roi]
        largest_name = structure_names.get(largest_roi, f"ROI_{largest_roi}")
        
        logger.debug(f"Largest structure: {largest_name} (ROI {largest_roi}) with area {largest_area:.2f} cm²")
        
        return largest_roi, largest_name