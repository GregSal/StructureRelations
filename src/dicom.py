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
        contour_dataframe (pd.DataFrame): DataFrame containing all contour point data.
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
        self.contour_points: Optional[List[ContourPoints]] = None
        self.contour_dataframe: Optional[pd.DataFrame] = None
    
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
    
    def get_structure_info(self) -> dict:
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
        return structure_info
    
    def __repr__(self) -> str:
        '''String representation of the DicomStructureFile.'''
        return f"DicomStructureFile(file_name='{self.file_name}', top_dir='{self.top_dir}')"
    
    def __str__(self) -> str:
        '''Human-readable string representation.'''
        info = self.get_structure_info()
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
            roi_number = int(roi.ReferencedROINumber)
            
            # Check if this ROI has contour data
            if not hasattr(roi, 'ContourSequence'):
                logger.debug(f"ROI {roi_number} has no contour sequence")
                continue
            
            for contour in roi.ContourSequence:
                # Get the geometric type (should be 'CLOSED_PLANAR' for most cases)
                geom_type = getattr(contour, 'ContourGeometricType', 'UNKNOWN')
                
                if geom_type not in ['CLOSED_PLANAR', 'POINT', 'OPEN_PLANAR']:
                    logger.warning(f"Unsupported geometric type {geom_type} in ROI {roi_number}")
                
                # Extract contour data - it's a flat array of x,y,z coordinates
                if hasattr(contour, 'ContourData') and len(contour.ContourData) >= 9:
                    # Reshape the flat array into (n, 3) array of points
                    points_array = np.array(contour.ContourData).reshape(-1, 3)
                    
                    # Convert from mm to cm (DICOM uses mm, our system uses cm)
                    points_array = points_array / 10.0
                    
                    # Convert to list of tuples for ContourPoints constructor
                    points_tuples = [tuple(point) for point in points_array]
                    
                    # Extract slice index from the z-coordinate of the first point
                    slice_index = SliceIndexType(points_tuples[0][2])
                    
                    # Create ContourPoints object with additional metadata
                    contour_points = ContourPoints(
                        points=points_tuples,
                        roi=ROI_Type(roi_number),
                        slice_index=slice_index,
                        ContourGeometricType=geom_type
                    )
                    
                    contour_points_list.append(contour_points)
                    
                else:
                    logger.warning(f"Invalid or insufficient contour data in ROI {roi_number}")
        
        logger.info(f"Extracted {len(contour_points_list)} contours from {len(self.dataset.ROIContourSequence)} ROIs")
        
        # Store contour points and convert to DataFrame
        self.contour_points = contour_points_list
        self.contour_dataframe = self._create_contour_dataframe(contour_points_list)
        
        return contour_points_list
    
    def _create_contour_dataframe(self, contour_points_list: List[ContourPoints]) -> pd.DataFrame:
        '''Convert list of ContourPoints to a pandas DataFrame.
        
        Args:
            contour_points_list (List[ContourPoints]): List of contour points to convert.
            
        Returns:
            pd.DataFrame: DataFrame with columns for ROI, slice_index, point_index, x, y, z, and geometry_type.
        '''
        if not contour_points_list:
            logger.warning("No contour points to convert to DataFrame")
            return pd.DataFrame()
        
        rows = []
        
        for contour in contour_points_list:
            roi = contour.roi
            slice_index = contour.slice_index
            geom_type = getattr(contour, 'ContourGeometricType', 'CLOSED_PLANAR')
            
            for point_idx, point in enumerate(contour.points):
                x, y, z = point[:3]  # Handle both 2D and 3D points
                if len(point) == 2:
                    # 2D point, use slice_index for z
                    z = float(slice_index)
                
                row = {
                    'roi': int(roi) if roi is not None else 0,
                    'slice_index': float(slice_index) if slice_index is not None else 0.0,
                    'point_index': point_idx,
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'geometry_type': geom_type
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Created DataFrame with {len(df)} points from {len(contour_points_list)} contours")
        return df
    
    def get_contour_dataframe(self) -> pd.DataFrame:
        '''Get the contour points as a pandas DataFrame.
        
        If contours haven't been loaded yet, this method will load them first.
        
        Returns:
            pd.DataFrame: DataFrame containing all contour points.
        '''
        if self.contour_dataframe is None:
            logger.info("Contour DataFrame not yet created, loading contours...")
            self.get_contour_points()
        
        return self.contour_dataframe if self.contour_dataframe is not None else pd.DataFrame()
    
    def get_roi_dataframe(self, roi_number: int) -> pd.DataFrame:
        '''Get contour points for a specific ROI as a DataFrame.
        
        Args:
            roi_number (int): The ROI number to filter by.
            
        Returns:
            pd.DataFrame: DataFrame containing points only for the specified ROI.
        '''
        df = self.get_contour_dataframe()
        if df.empty:
            return df
        
        roi_df = df[df['roi'] == roi_number].copy()
        logger.debug(f"Retrieved {len(roi_df)} points for ROI {roi_number}")
        return roi_df