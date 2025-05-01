'''Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import List, NewType, Union, Tuple
from dataclasses import dataclass
import warnings

# Shared Packages
import numpy as np
import pandas as pd

# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Type = NewType('ROI_Type', int)  # pylint: disable=invalid-name

# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.        intp_node_label = (node_data['roi'], new_slice, intp_poly.wkt)
SliceIndexType = NewType('SliceIndexType', float)

# A sequence or individual slice index.
# This can be a list of slice indexes, a tuple of two slice indexes, or a
# single slice index.
SliceIndexSequenceType = Union[List[SliceIndexType],
                       Tuple[SliceIndexType, SliceIndexType],
                       SliceIndexType]

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

# TODO Everything after this should probably go into the Contours.py file
@dataclass
class SliceNeighbours:
    '''A class to represent the neighbours of a slice.

    Attributes:
        this_slice (SliceIndexType): The current slice index.
        previous_slice (SliceIndexType): The previous slice index.
        next_slice (SliceIndexType): The next slice index.

        force_types: Method to ensure the slice indices are of the correct type.
        is_neighbour: Method to check if another slice index is a neighbour.
        gaps: Method to calculate the gaps between slices.
    '''

    this_slice: SliceIndexType
    previous_slice: SliceIndexType
    next_slice: SliceIndexType

    def __post_init__(self) -> None:
        self.force_types()

    def force_types(self):
        '''Ensure the slice indices are of the correct type.'''
        # Convert to float and then to SliceIndexType
        # to ensure they are of the correct type.
        self.this_slice = SliceIndexType(float(self.this_slice))
        self.previous_slice = SliceIndexType(float(self.previous_slice))
        self.next_slice = SliceIndexType(float(self.next_slice))

    def gap(self, absolute=True) -> Union[int, float]:
        '''Calculate the gaps between slices.

        If one of the two neighbours is None, then calculate the gap based on
        the distance between the current slice and the other slice. if absolute
        is True, return the absolute value of the gap. If both neighbours are
        None, return NaN.

        Returns:
            Union[int, float]: The gap between slices.
        '''
        # If both neighbours are None, return NaN
        if pd.isna(self.previous_slice) and pd.isna(self.next_slice):
            return np.NaN
        # If one of the neighbours is None, calculate the gap based on the
        # distance between the current slice and the other slice.
        if pd.isna(self.previous_slice):
            gap = self.next_slice - self.this_slice
        elif pd.isna(self.next_slice):
            gap = self.this_slice - self.previous_slice
        else:
            # Calculate the gap between the previous and next slice
            # and divide by 2 to get the average gap.
            gap = (self.next_slice - self.previous_slice) / 2
        if absolute:
            return abs(gap)
        return gap

    def is_neighbour(self, other_slice: SliceIndexType) -> bool:
        '''Check if another slice index is a neighbour.'''
        return self.previous_slice <= other_slice <= self.next_slice

    def neighbour_list(self) -> List[SliceIndexType]:
        '''Return a list of neighbours, excluding the current slice.'''
        return [self.previous_slice, self.next_slice]


class SliceSequence:
    '''An ordered list of all slice indexes in use.

    Attributes:
        sequence (pd.DataFrame): A DataFrame containing slice information.
        slices (List[SliceIndexType]): A list of all slice indices in the
            sequence.
    Methods:
        add_slice: Method to add a slice index to the sequence.
        remove_slice: Method to remove a slice index from the sequence.
        get_nearest_slice: Method to find the nearest slice index to a given
            value.
        get_neighbors: Method to get the neighbors of a given slice index.
    '''
    sequence: pd.DataFrame

    def __init__(self, slice_indices: Union[List[SliceIndexType], pd.Series]) -> None:
        '''Initialize the SliceSequence.

        Args:
            slice_indices (Union[List[SliceIndexType], pd.Series]): A list or
                Series of slice indices.
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

    def add_slice(self, **slice_ref) -> None:
        '''Add a slice index to the sequence.

        Args:
            slice_ref (dict): A dictionary containing the slice reference
                information. Must contain the following keys:
                    - 'ThisSlice'
                    - 'PreviousSlice'
                    - 'NextSlice'.
                If 'Original' is not provided, it will be set to False.
        '''
        slice_index = slice_ref['ThisSlice']
        if 'Original' not in slice_ref:
            slice_ref['Original'] = False
        # Check if the slice index is already in the sequence
        if slice_index not in self.sequence.index:
            new_row = pd.DataFrame([{**slice_ref}])
            new_row.set_index('ThisSlice', drop=False, inplace=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.sequence = pd.concat([self.sequence, new_row]).sort_index()

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
