'''Module for handling structures from DICOM files.

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports

from collections.abc import Hashable
from typing import Any, ClassVar, Dict, List, NewType, Union, Tuple
from dataclasses import dataclass

# Shared Packages
import shapely
import networkx as nx
import pandas as pd

# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Type = NewType('ROI_Type', int)  # pylint: disable=invalid-name

# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.        intp_node_label = (node_data['roi'], new_slice, intp_poly.wkt)
SliceIndexType = NewType('SliceIndexType', float)

# A list of contour points.
ContourPointsType = NewType('ContourPointsType', List[tuple[float]])

# The index for an individual contour.
# The index is a tuple of:
#   - The Region's ROI number,
#   - The Region's The slice index,
#   - An indexer value to force unique nodes.
ContourIndex = NewType('ContourIndex', Tuple[ROI_Type, SliceIndexType, int])

# The reference numbers for two different structures to be compared.
StructurePairType = NewType('StructurePairType', Tuple[ROI_Type, ROI_Type])

# Global Settings
PRECISION = 3

# Exception Types
class StructuresException(Exception):
    '''Base class for exceptions in this module.'''

class InvalidContour(ValueError, StructuresException):
    '''Exception raised for invalid contour data.'''

class InvalidContourRelation(ValueError, StructuresException):
    '''Exception raised for invalid contour relationship data.'''

class ContourPoints(dict):
    '''A dictionary of contour points.

    The dictionary has three required keys:
        'ROI': (ROI_Type) The value will contain the ROI number for the contour.
        'Slice': (SliceIndexType) The value will contain the slice index for
            the contour.
        'Points': A list of length 2 or three tuples for float containing the
            coordinates of the points that define the contour.

    The dictionary will accept additional keys related to the contour.

    Args:
        points (ContourPointsType: A list of tuples containing 2 or 3 float
            values representing 2D or 3D contour points.
        roi (ROI_Type): An integer referencing a structure.
        slice_index (SliceIndexType, optional): The slice index for the contour.
            If not provided, the slice index is extracted from the z coordinate
            of the first point or set to 0.0. Defaults to None.

    Raises:
        InvalidContour: Raised if the input data is invalid.
    '''

    def __init__(self, points: ContourPointsType, roi: ROI_Type,
                 slice_index: SliceIndexType = None, **dict_items):
        # Validate the ROI
        if not isinstance(roi, int):
            raise InvalidContour("ROI must be an integer.")
        slice_index = self.validate_slice_index(slice_index, points)
        points = self.validate_points(points, slice_index)
        dict_items['ROI'] = roi
        dict_items['Slice'] = slice_index
        dict_items['Points'] = points
        super().__init__(**dict_items)

    @staticmethod
    def validate_slice_index(slice_index: SliceIndexType,
                             points: ContourPointsType)-> SliceIndexType:
        '''Validate or set the slice_index for the contour.

    - slice_index if provided, must be float or integer.  If not provided and
        the points tuples are 3D, then slice_index is extracted from the z
        coordinate for the first point. Otherwise slice_index is set to 0.0.

    Args:
        slice_index (SliceIndexType): The slice index for the contour.  If None,
            the slice index is extracted from the z coordinate of the first
            point.
        points (ContourPointsType: A list of tuples containing 2 or 3 float
            values representing 2D or 3D contour points.

    Raises:
        InvalidContour: Raised if the input data is invalid.

    Returns:
        SliceIndexType: The validated or set slice_index value.
        '''
        if slice_index is None:
            if len(points[0]) == 3:
                slice_index = points[0][2]
            else:
                slice_index = 0.0
        else:
            # Verify that the slice is a float or integer
            try:
                slice_index = float(slice_index)
            except ValueError as err:
                raise InvalidContour("Slice must be an integer or float.") from err
        return slice_index

    @staticmethod
    def validate_points(points: ContourPointsType,
                        slice_index: SliceIndexType = None
                       )-> tuple[ContourPointsType, ROI_Type, SliceIndexType]:
        '''Validate the contour points and add a z coordinate if necessary.

        - points must be a list of 3 or more tuples containing 2 or 3 float
            values representing 2D or 3D contour points.
        - If the points are 2D, slice_index is added to the points as the
            z coordinate.

        Args:
            points (ContourPointsType: A list of tuples containing 2 or 3 float
                values representing 2D or 3D contour points.
            slice_index (SliceIndexType): The slice index for the contour.

        Raises:
            InvalidContour: Raised if the input data is invalid.

        Returns:
            ContourPointsType: The validated 3D points.
        '''
        # Validate the points
        if len(points) < 3:
            raise InvalidContour("Contour must have at least three points.")
        # Verify that the points are tuples of floats of length 2 or 3
        clean_points = []
        point_dim = None
        for point in points:
            # Verify that each point is a tuple of float.
            if not isinstance(point, tuple):
                raise InvalidContour("Points must be tuples of floats.")
            if not all(isinstance(coord, (float, int)) for coord in point):
                raise InvalidContour("Points must be tuples of floats.")
            # Verify that each point is a tuple of length 2 or 3.
            this_point_dim = len(point)
            if this_point_dim not in (2, 3):
                raise InvalidContour("Points must be tuples of length 2 or 3.")
            # Verify that all points have the same length.
            if not point_dim:
                point_dim = this_point_dim
            else:
                if point_dim != this_point_dim:
                    raise InvalidContour("All points must have the same length.")
            # If points are 2D, add slice as the z-coordinate.
            if this_point_dim == 2:
                point = point + (slice_index,)
            clean_points.append(point)
        return clean_points


@dataclass
class SliceNeighbours:
    this_slice: SliceIndexType
    previous_slice: SliceIndexType
    next_slice: SliceIndexType

    def __post_init__(self) -> None:
        self.force_types()

    def force_types(self):
        self.this_slice = SliceIndexType(float(self.this_slice))
        self.previous_slice = SliceIndexType(float(self.previous_slice))
        self.next_slice = SliceIndexType(float(self.next_slice))

    def is_neighbour(self, other_slice: SliceIndexType) -> bool:
        return self.previous_slice <= other_slice <= self.next_slice

    def gaps(self) -> Union[int, float]:
        return abs(self.next_slice - self.previous_slice) / 2


class SliceSequence:
    '''An ordered list of all slice indexes in use.

    Attributes:
        slices (pd.DataFrame): A DataFrame containing slice information.
    '''
    sequence: pd.DataFrame

    def __init__(self, slice_indices: Union[List[SliceIndexType], pd.Series]) -> None:
        '''Initialize the SliceSequence.

        Args:
            slice_indices (Union[List[SliceIndexType], pd.Series]): A list or Series of slice indices.
        '''
        # Convert to Series if not already
        slice_series = pd.Series(slice_indices).drop_duplicates().sort_values()

        # Create the DataFrame
        self.sequence = pd.DataFrame({
            'ThisSlice': slice_series,
            'NextSlice': slice_series.shift(-1),
            'PreviousSlice': slice_series.shift(1),
            'Original': True
        }).set_index('ThisSlice', drop=False)

    @property
    def slices(self) -> List[SliceIndexType]:
        '''Return a list of all slice indices in the sequence.'''
        return self.sequence['ThisSlice'].tolist()

    def add_slice(self, slice_index: SliceIndexType) -> None:
        '''Add a slice index to the sequence.'''
        if slice_index not in self.sequence.index:
            new_row = pd.DataFrame({
                'ThisSlice': [slice_index],
                'NextSlice': [None],
                'PreviousSlice': [None],
                'Original': [False]
            }).set_index('ThisSlice', drop=False)
            self.sequence = pd.concat([self.sequence, new_row]).sort_index()
            self.sequence['NextSlice'] = self.sequence['ThisSlice'].shift(-1)
            self.sequence['PreviousSlice'] = self.sequence['ThisSlice'].shift(1)

    def remove_slice(self, slice_index: SliceIndexType) -> None:
        '''Remove a slice index from the sequence.'''
        if slice_index in self.sequence.index:
            self.sequence = self.sequence.drop(slice_index)
            self.sequence['NextSlice'] = self.sequence['ThisSlice'].shift(-1)
            self.sequence['PreviousSlice'] = self.sequence['ThisSlice'].shift(1)

    def get_nearest_slice(self, value: float) -> SliceIndexType:
        '''Find the nearest slice index to a given value.'''
        return self.sequence.index[(self.sequence.index - value).abs().argmin()]

    def get_neighbors(self, slice_index: SliceIndexType) -> Union[SliceNeighbours, None]:
        '''Return a SliceNeighbours object for a given SliceIndex.

        Args:
            slice_index (SliceIndexType): The slice index to find neighbors for.

        Returns:
            SliceNeighbours: An object containing this slice, next slice, and previous slice.
            None: If the given SliceIndex is not in the sequence.
        '''
        if slice_index not in self.sequence.index:
            return None
        row = self.sequence.loc[slice_index]
        return SliceNeighbours(
            this_slice=slice_index,
            next_slice=row['NextSlice'],
            previous_slice=row['PreviousSlice']
        )

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index: int) -> SliceIndexType:
        return self.sequence.iloc[index]['ThisSlice']

    def __iter__(self):
        return iter(self.sequence['ThisSlice'])

    def __contains__(self, slice_index: SliceIndexType) -> bool:
        return slice_index in self.sequence.index


class Contour:
    '''Class representing a contour with associated metadata.

    Attributes:
       Index attributes:
        roi (ROI_Type): The ROI number of the contour.
        slice_index (SliceIndexType): The slice index of the contour.
        contour_index (int): Auto-incremented contour index.
        region_index (int): The region index of the contour.
            Defaults to None until regions are assigned.
        index (ContourIndex): A tuple of (ROI, SliceIndex, ContourIndex).
            This is a read-only property.
      shape:
        polygon (shapely.Polygon): The polygon generated from the contour.
        exterior (shapely.Polygon): The solid exterior polygon of the contour.
        hull (shapely.Polygon): The convex hull of the contour.
        thickness (float): The thickness of the contour.  Defaults to
            Contour.default_thickness (0.0).

      hole information:
        is_hole (bool): Whether the contour is a hole.
        hole_reference (ContourIndex): If is_hole is True, contains the index
            of the smallest non-hole contour that contains this hole.
            If is_hole is False, hole_reference is None.
        hole_type (str): If is_hole is True, the type of the hole. One of:
                Open,
                Closed, or
                Unknown (Default).
            If is_hole is False, hole_type is None.

      boundary information:
        is_boundary (bool): Whether the contour is a boundary.
            Defaults to False.
        is_interpolated (bool): Whether the contour is interpolated.
            Defaults to False.
    '''
    # Class Variables
    # Incremented counter for contour index
    counter = 0
    # Default thickness for contours
    default_thickness = 0.0

    def __init__(self, roi: ROI_Type, slice_index: SliceIndexType,
                 polygon: shapely.Polygon,
                 contours: List['Contour']) -> None:
        self.roi = roi
        self.slice_index = slice_index
        self.polygon = polygon
        self.thickness = self.default_thickness
        self.is_hole = False
        self.hole_reference = None
        self.hole_type = None
        self.is_boundary = False
        self.is_interpolated = False
        self.region_index = None
        self.contour_index = Contour.counter
        Contour.counter += 1
        self.validate_polygon()
        self.compare_with_existing_contours(contours)

    @property
    def index(self) -> ContourIndex:
        '''Return a tuple representing (ROI, SliceIndex, ContourIndex).'''
        return (self.roi, self.slice_index, self.contour_index)

    @property
    def exterior(self)-> shapely.Polygon:
        '''The solid exterior Polygon.

        Returns:
            shapely.Polygon: The contour Polygon with all holes
                filled in.
        '''
        ext_poly = shapely.Polygon(shapely.get_exterior_ring(self.polygon))
        return ext_poly

    @property
    def hull(self)-> shapely.Polygon:
        '''A bounding contour generated from the entire contour Polygon.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        Returns:
            shapely.Polygon: The bounding contour for the entire contour.
        '''
        hull = shapely.convex_hull(self.polygon)
        return hull

    def compare_with_existing_contours(self, contours: List['Contour']) -> None:
        '''Compare the polygon to each existing Contour in the list.

        If the polygon is within an existing contour, the new contour is a hole.
        If the polygon overlaps an existing contour, raise an error.

        Args:
            contours (List['Contour']): A list of existing contours.

        Raises:
            InvalidContour: Raised if the new contour overlaps an existing
                contour in the list.
        '''
        for existing_contour in reversed(contours):
            if existing_contour.is_hole:
                # If the existing contour is a hole, the new contour cannot be
                # its hole
                continue
            if self.polygon.within(existing_contour.polygon):
                # New contour is completely within the existing contour
                self.is_hole = True
                self.hole_reference = existing_contour.contour_index
                self.hole_type = "Unknown"
                break  # Stop checking once a containing contour is found
            if self.polygon.overlaps(existing_contour.polygon):
                # New contour overlaps an existing contour, raise an error
                raise InvalidContour("New contour overlaps an existing contour.")

    def validate_polygon(self) -> None:
        '''Validate the polygon to ensure it is valid.'''
        if not self.polygon.is_valid:
            raise InvalidContour("Invalid polygon provided for the contour.")

    def area(self) -> float:
        '''Calculate the area of the contour polygon.'''
        return self.polygon.area

    def centroid(self) -> Tuple[float, float]:
        '''Calculate the centroid of the contour polygon.'''
        return self.polygon.centroid.coords[0]


class ContourMatch:
    '''Class representing a match between two contours.

    Attributes:
        contour1 (Contour): The first contour.
        contour2 (Contour): The second contour.
        thickness (float): Half the difference between the two slice indices.
        combined_area (float): The sum of the areas of the two contours.
    '''

    def __init__(self, contour1: Contour, contour2: Contour) -> None:
        self.contour1 = contour1
        self.contour2 = contour2
        self.thickness = abs(contour1.slice_index - contour2.slice_index) / 2
        self.combined_area = contour1.area() + contour2.area()
