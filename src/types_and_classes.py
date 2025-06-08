'''Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import Any, Dict, List, NewType, Union, Tuple
from dataclasses import dataclass
import warnings

# Shared Packages
import numpy as np
import pandas as pd
import networkx as nx

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

# The index of a a unique contiguous 3D region.
RegionIndex = NewType('RegionIndex', str)

# A Networkx Graph object containing contour information.
# Each node in the graph represents a contour and has a 'contour' attribute
# that is an instance of the contours.Contour class.  The node labels are
# ContourIndexes. The graph edges indicate contours that are on neighbouring
# slices (based on the slice sequence), have the same hole type, and have
# intersecting convex hulls.  The edges have and has a 'match' attribute that
# is a contours.ContourMatch object.
ContourGraph = NewType('ContourGraph', nx.Graph)

# The reference numbers for two different structures to be compared.
StructurePairType = NewType('StructurePairType', Tuple[ROI_Type, ROI_Type])

# Global Settings
PRECISION = 3

# Exception Types
class StructuresException(Exception):
    '''Base class for exceptions in this module.'''

class InvalidSlice(ValueError, StructuresException):
    '''Exception raised for an invalid slice index.'''

class InvalidContour(ValueError, StructuresException):
    '''Exception raised for invalid contour data.'''

class InvalidContourRelation(ValueError, StructuresException):
    '''Exception raised for invalid contour relationship data.'''
