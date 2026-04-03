'''Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import List, NewType, Union, Tuple
from enum import Enum
from dataclasses import dataclass

# Shared Packages
import networkx as nx
import shapely

# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Type = NewType('ROI_Type', int)  # pylint: disable=invalid-name

# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.
SliceIndexType = NewType('SliceIndexType', float)

# A sequence or individual slice index.
# This can be a list of slice indexes, a tuple of two slice indexes, or a
# single slice index.
SliceIndexSequenceType = Union[List[SliceIndexType],
                               Tuple[SliceIndexType, SliceIndexType],
                               SliceIndexType]

# A list of contour points.
Coordinate = Union[Tuple[float, float], Tuple[float, float, float]]
ContourPointsType = NewType('ContourPointsType', List[Coordinate])

# The index for an individual contour.
# The index contains:
#   - The Region's ROI number,
#   - The Region's slice index,
#   - An indexer value to force unique nodes.
@dataclass(frozen=True, order=True)
class ContourIndex:
    '''Stable contour identifier used as graph node labels and dict keys.'''

    roi: ROI_Type
    slice_index: SliceIndexType
    uniqueness_int: int

    def __iter__(self):
        yield self.roi
        yield self.slice_index
        yield self.uniqueness_int

    def __getitem__(self, idx: int):
        return (self.roi, self.slice_index, self.uniqueness_int)[idx]

    def __len__(self) -> int:
        return 3

# A link between two contours.
# The link is a tuple of two ContourIndexes.
ContourLink = NewType('ContourLink', Tuple[ContourIndex, ContourIndex])

# The index of a a unique contiguous 3D region.
RegionIndex = NewType('RegionIndex', str)

# A Networkx DiGraph object containing contour information.
# Each node in the graph represents a contour and has a 'contour' attribute
# that is an instance of the contours.Contour class.  The node labels are
# ContourIndexes. The graph edges are directed from lower to higher slice
# indices and indicate contours that are on neighbouring slices (based on the
# slice sequence), have the same hole type, and have intersecting convex hulls.
# The edges have a 'match' attribute that is a contours.ContourMatch object.
ContourGraph = NewType('ContourGraph', nx.DiGraph)

# The reference numbers for two different structures to be compared.
StructurePairType = NewType('StructurePairType', Tuple[ROI_Type, ROI_Type])

# PolygonType is a type alias for shapely.Polygon or shapely.MultiPolygon.
PolygonType = Union[shapely.Polygon, shapely.MultiPolygon]

# Global Settings
DEFAULT_TRANSVERSE_TOLERANCE = 0.01  # The default resolution in the transverse plane in cm
SLICE_INDEX_PRECISION = 0.01  # The default precision for slice indexes in cm
PRECISION = 3


class HoleType(str, Enum):
    '''Canonical hole/boundary labels used across contour processing.'''

    NONE = 'None'
    OPEN = 'Open'
    CLOSED = 'Closed'
    BOUNDARY = 'Boundary'

# Exception Types
class StructuresException(Exception):
    '''Base class for exceptions in this module.'''

class InvalidSlice(ValueError, StructuresException):
    '''Exception raised for an invalid slice index.'''

class InvalidContour(ValueError, StructuresException):
    '''Exception raised for invalid contour data.'''

class InvalidContourRelation(ValueError, StructuresException):
    '''Exception raised for invalid contour relationship data.'''
