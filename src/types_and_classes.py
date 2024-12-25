'''Module for handling structures from DICOM files.

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports

from typing import NewType, Tuple

# Shared Packages
import shapely


# %% Type definitions and Globals
# Index to structures defined in Structure RT DICOM file
ROI_Type = NewType('ROI_Type', int)  # pylint: disable=invalid-name

RegionIndexType = NewType('RegionIndexType', Tuple[int, str])  # pylint: disable=invalid-name
# The offset in cm between a given image slice and the DICOm origin in the
# `Z` direction.
SliceIndexType = NewType('SliceIndexType', float)

# An enclosed region representing either a structure area, or a hole within .
# that structure.
ContourType = NewType('ContourType', shapely.Polygon)

# The reference numbers for two different structures to be compared.
StructurePairType = NewType('StructurePairType', Tuple[ROI_Type, ROI_Type])

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
