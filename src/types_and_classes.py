'''Module for handling structures from DICOM files.

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports

from typing import Any, Dict, NewType, Union, Tuple
from dataclasses import dataclass

# Shared Packages
import shapely
import networkx as nx

# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Type = NewType('ROI_Type', int)  # pylint: disable=invalid-name

# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.        intp_node_label = (node_data['roi'], new_slice, intp_poly.wkt)
SliceIndexType = NewType('SliceIndexType', float)

# The reference numbers for two different structures to be compared.
StructurePairType = NewType('StructurePairType', Tuple[ROI_Type, ROI_Type])

# A RegionIndexType is used as a locator for nodes in a region graph.
# The index is a tuple of:
#   - The Region's ROI number,
#   - The Region's The slice index,
#   - The Region's shape as a string in WKT format.
# a tuple of an ROI_Type and a string representing the
RegionIndexType = NewType('RegionIndexType', Tuple[ROI_Type, SliceIndexType, str])

# RegionType is a node in a region graph.
# The node will have the following attributes:
#  - polygon: shapely.Polygon,0
#  - roi: ROI_Type,
#  - slice_index: SliceIndexType,
#  - is_hole: bool,
#  - is_boundary: bool,
#  - is_interpolated: bool,
#  - is_empty: bool,
#  - slice_neighbours: SliceNeighbours
RegionType = NewType('RegionType', Dict[str, Any])

# RegionGraph is a nx.Graph object with RegionType nodes.
# The edges indicate matching regions on neighbouring slices.
RegionGraph = NewType('RegionGraph', nx.Graph)


# A length 9 string of '1's and '0's representing a DE-9IM relationship.
DE9IM_Type = NewType('DE9IM_Type', str)  # pylint: disable=invalid-name

# A 27 bit binary value composed of three DE9IM_Values concatenated.
# The right-most 9 binary digits represent the DE-9IM relationship between
#    polygon b and polygon a.
# The middle 9 binary digits represent the DE-9IM relationship between
#    polygon b and the *exterior* of polygon a.
# The left 9 binary digits represent the DE-9IM relationship between
#    polygon b and the *convex hull* of polygon a.
DE27IM_Type = NewType('DE27IM_Type', int)  # pylint: disable=invalid-name


# Global Settings
PRECISION = 3

# Exception Types
class StructuresException(Exception):
    '''Base class for exceptions in this module.'''

class InvalidContour(ValueError, StructuresException):
    '''Exception raised for invalid contour data.'''

class InvalidContourRelation(ValueError, StructuresException):
    '''Exception raised for invalid contour relationship data.'''

@dataclass
class SliceNeighbours:
    this_slice: SliceIndexType
    previous_slice: SliceIndexType
    next_slice: SliceIndexType

    def is_neighbour(self, other_slice: SliceIndexType) -> bool:
        return self.previous_slice <= other_slice <= self.next_slice

    def gaps(self) -> Union[int, float]:
        return abs(self.next_slice - self.previous_slice) / 2
