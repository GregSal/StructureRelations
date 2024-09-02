'''Relationships between structures.
'''
# %% Imports
# Type imports

from typing import Any, Dict, List, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Standard Libraries
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from math import sqrt, pi, sin, cos, tan, radians
from statistics import mean
from itertools import zip_longest
from itertools import product

# Shared Packages
import numpy as np
import pandas as pd
import xlwings as xw
import pydicom
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line
import pygraphviz as pgv
import networkx as nx

from types_and_classes import ROI_Num, SliceIndex, Contour, StructurePair
from types_and_classes import DE9IM_Value, DE27IM_Value
from types_and_classes import poly_round
from types_and_classes import InvalidContour
from types_and_classes import StructureSlice, RelationshipType
from metrics import MarginMetric, DistanceMetric, NoMetric
from metrics import OverlapAreaMetric, OverlapSurfaceMetric
from utilities import find_boundary_slices


# Global Settings
PRECISION = 3

def compare(mpoly1: shapely.MultiPolygon,
            mpoly2: shapely.MultiPolygon)->DE9IM_Value:
    '''Get the DE-9IM relationship string for two contours

    The relationship string is converted to binary format, where 'F'
    is '0' and '1' or '2' is '1'.

    Args:
        mpoly1 (shapely.MultiPolygon): All contours for a structure on
            a single slice.
        mpoly2 (shapely.MultiPolygon): All contours for a second
            structure on the same slice.

    Returns:
        DE9IM_Value: A length 9 string of '1's and '0's reflecting the DE-9IM
            relationship between the supplied contours.
    '''
    relation_str = shapely.relate(mpoly1, mpoly2)
    # Convert relationship string in the form '212FF1FF2' into a
    # boolean string.
    relation_bool = relation_str.replace('F','0').replace('2','1')
    return relation_bool


def relate(contour1: StructureSlice, contour2: StructureSlice)->DE27IM_Value:
    '''Get the 27 bit relationship integer for two polygons,

    When written in binary, the 27 bit relationship contains 3 9-bit
    parts corresponding to DE-9IM relationships. The left-most 9 bits
    are the relationship between the second structure's contour and the
    first structure's convex hull polygon. The middle 9 bits are the
    relationship between the second structure's contour and the first
    structure's exterior polygon (i.e. with any holes filled). The
    right-most 9 bits are the relationship between the second
    structure's contour and the first structure's contour.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

    Returns:
        DE9IM_Value: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between contour2 and
            contour1's convex hull, exterior and polygon.
    '''
    primary_relation = compare(contour1.contour, contour2.contour)
    external_relation = compare(contour1.exterior, contour2.contour)
    convex_hull_relation = compare(contour1.hull, contour2.contour)
    full_relation = ''.join([convex_hull_relation,
                                external_relation,
                                primary_relation])
    binary_relation = int(full_relation, base=2)
    return binary_relation


def relate_structures(slice_structures: pd.DataFrame,
                      structures: StructurePair)->DE9IM_Value:
    '''Get the 27 bit relationship integer for two structures on a given slice.

    Args:
        slice_structures (pd.DataFrame): A table of structures, where
            the values are the contours with type StructureSlice. The
            column index contains the roi numbers for the structures.
            The row index contains the slice index distances.

        structures (StructurePair): A tuple of ROI numbers which index
            columns in slice_structures.
    Returns:
        DE9IM_Value: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between the
            second contour and the first contour convex hull, exterior and
            contour.
    '''
    structure = slice_structures[structures[0]]
    other_contour = slice_structures[structures[1]]
    binary_relation = relate(structure, other_contour)
    return binary_relation


def adjust_slice_boundary_relations(relation_seq: pd.Series,
                                    structures: StructurePair,
                                    boundary_slices: pd.DataFrame)->pd.Series:
    '''Adjust the DE-9IM relationship metric for the boundary slices of both
    structures.

    For the beginning and ending slices of a structure the entire contour must
    be treated as a boundary.  The structure does not have an interior on these
    slices. In this case the “Interior” relations become “Boundary” relations.
    For the `b` structure the, first three values of the DE-9IM relationship
    metric are shifted to become the second three.  For the `a` structure,
    every third value of the DE-9IM relationship metric is shifted by 1.

    Args:
        relation_seq (pd.Series): A series with SliceIndex as the index and
            DE-9IM relationship metrics as the values.
        structures (StructurePair): A tuple of ROI numbers which index
            columns in boundary_slices.
        boundary_slices (pd.DataFrame): _description_

    Returns:
        pd.Series: The supplied relation_seq with adjusted DE-9IM relationship
        metrics for the boundary slices of both structures.
    '''
    def shift_values(value: DE9IM_Value, mask: DE9IM_Value,
                     shift: int)->DE9IM_Value:
        # Select the DE-9IM relationship values related to the interior.
        interior_relations = value & mask
        # Select the DE-9IM relationship values related to the exterior.
        # (Not interior and not Boundary.)
        value_mask = (mask >> shift) + mask
        other_relations = value & ~value_mask
        # Convert the interior relations into corresponding boundary relations.
        boundary_relations = interior_relations >> shift
        # Combine the interior (zeros), boundary and exterior values to form
        # the adjusted DE-9IM relationship metric.
        relations_bin = boundary_relations + other_relations
        return relations_bin

    roi_a, roi_b = structures
    # Boundaries of b
    # Select the first three values of the DE-9IM relationship metric.
    b_mask = 0b111000000111000000111000000
    for boundary_slice in list(boundary_slices[roi_b]):
        value = relation_seq[boundary_slice]
        relations_bin = shift_values(value, b_mask, 3)
        relation_seq[boundary_slice] = relations_bin
    # Boundaries of a
    a_mask = 0b100100100100100100100100100
    for boundary_slice in list(boundary_slices[roi_a]):
        value = relation_seq[boundary_slice]
        relations_bin = shift_values(value, a_mask, 1)
        relation_seq[boundary_slice] = relations_bin
    return relation_seq


def relate_structs(slice_table: pd.DataFrame,
                   structures: StructurePair)->DE9IM_Value:
    '''Calculate the overall 27 bit relationship integer for two structures.

    When written in binary, the 27 bit relationship contains 3 9-bit
    parts corresponding to DE-9IM relationships. The left-most 9 bits
    are the relationship between the second structure's contour and the
    first structure's convex hull polygon. The middle 9 bits are the
    relationship between the second structure's contour and the first
    structure's exterior polygon (i.e. with any holes filled). The
    right-most 9 bits are the relationship between the second
    structure's contour and the first structure's contour.

    Args:
        slice_table (pd.DataFrame): A table of with StructureSlice or na as the
            values, SliceIndex as the index and ROI_Num as the Columns.

        structures (StructurePair): A tuple of ROI numbers which index
            columns in slice_table.

    Returns:
        DE9IM_Value: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between the
            second contour and the first contour convex hull, exterior and
            contour.
    '''
    slice_structures = slice_table.loc[:, [structures[0],
                                           structures[1]]]
    # Remove Slices that have neither structure.
    slice_structures.dropna(how='all', inplace=True)
    boundary_slices = slice_table.apply(find_boundary_slices)

    # For slices that have only one of the two structures, replace the nan
    # values with empty polygons for duck typing.
    slice_structures.fillna(StructureSlice([]), inplace=True)
    # Get the relationships between the two structures for all slices.
    relation_seq = slice_structures.agg(relate_structures,
                                        structures=structures,
                                        axis='columns')
    # Adjust the relationship metrics for the boundary slices of both structures.
    relation_seq = adjust_slice_boundary_relations(relation_seq, structures,
                                                   boundary_slices)
     # Get the overall relationship for the two structures by merging the
    # relationships for the individual slices.
    relation_binary = merge_rel(relation_seq)
    return relation_binary


@dataclass()
class RelationshipTest:
    '''The test binaries used to identify a relationship type.

    Each test definitions consists of 2 27-bit binaries, a mask and a value.
    Each of the 27-bit binaries contain 3 9-bit parts associated with DE-9IM
    relationships. The left-most 9 bits are associated with the relationship
    between one structure's convex hull and another structure's contour. The
    middle 9 bits are associated with the relationship between the first
    structure's exterior polygon (i.e. with any holes filled) and the second
    structure's contour. The right-most 9 bits are associated with the
    relationship between first structure's contour and the second structure's
    contour.

    Named relationships are identified by logical patterns such as: T*T*F*FF*
        The 'T' indicates the bit must be True.
        The 'F' indicates the bit must be False.
        The '*' indicates the bit can be either True or False.
    Ane example of a complete relationship logic is:
    Surrounds (One structure resides completely within a hole in another
               structure):
        Region Test =   FF*FF****  - The contours of the two structures have no
                                     regions in common.
        Exterior Test = T***F*F**  - With holes filled, one structure is within
                                     the other.
        Hull Test =     *********  - Together, the Region and Exterior Tests
                                     sufficiently identifies the relationship,
                                     so the Hull Test is not necessary.
    The mask binary is a sequence of 0s and 1s with every '*' as a '0' and every
    'T' or 'F' bit as a '1'.  The operation: relationship_integer & mask will
    set all of the bit that are allowed to be either True or False to 0.

    The value binary is a sequence of 0s and 1s with every 'T' as a '1' and
    every '*' or 'F' bit as a '0'. The relationship is identified when value
    binary is equal to the result of the `relationship_integer & mask`
    operation.
    '''
    relation_type: RelationshipType = RelationshipType.UNKNOWN
    mask: int = 0b000000000000000000000000000
    value: int = 0b000000000000000000000000000

    def __repr__(self) -> str:
        rep_str = ''.join([
            f'RelationshipTest({self.relation_type}\n',
            ' ' * 4,
            f'mask =  0b{self.mask:0>27b}\n',
            ' ' * 4,
            f'value = 0b{self.value:0>27b}'
            ])
        return rep_str

    def test(self, relation: int)->RelationshipType:
        '''Apply the defined test to the supplied relation binary.

        Args:
            relation (int): The number corresponding to a 27-bit binary of
                relationship values.

        Returns:
            RelationshipType: The RelationshipType if the test passes,
                otherwise None.
        '''
        masked_relation = relation & self.mask
        if masked_relation == self.value:
            return self.relation_type
        return None


def identify_relation(relation_binary) -> RelationshipType:
    '''Applies a collection of definitions for named relationships to a supplied
    relationship binary.

    The defined relationships are:
        Relationship      Region Test   Exterior Test   Hull Test
        Disjoint          FF*FF****     FF*FF****       FF*FF****
        Shelters          FF*FF****     FF*FF****       T***F*F**
        Surrounds         FF*FF****     T***F*F**
        Borders_Interior  FF*FT****     T***T****
        Borders           FF*FT****     FF*FT****
        Contains	      T*T*F*FF*
        Incorporates	  T*T*T*FF*
        Equals	          T*F**FFF*
        Overlaps          TTTT*TTT*

    Args:
        relation_binary (int): An integer generated from the combined DE-9IM
            tests.

    Returns:
        RelationshipType: The identified RelationshipType if one of the tests
            passes, otherwise RelationshipType.UNKNOWN.
    '''
    # Relationship Test Definitions
    test_binaries = [
        RelationshipTest(RelationshipType.SURROUNDS,        0b000000000100010110110110000, 0b000000000100000000000000000),
        RelationshipTest(RelationshipType.SHELTERS,         0b111000100110110000110110000, 0b111000000000000000000000000),
        RelationshipTest(RelationshipType.DISJOINT,         0b110110000110110000110110000, 0b000000000000000000000000000),
        RelationshipTest(RelationshipType.BORDERS,          0b000000000001001110110110000, 0b000000000001001110000010000),
        RelationshipTest(RelationshipType.BORDERS_INTERIOR, 0b000000000101010110110110000, 0b000000000101000000000010000),
        RelationshipTest(RelationshipType.OVERLAPS,         0b000000000000000000101000100, 0b000000000000000000101000100),
        RelationshipTest(RelationshipType.PARTITION,        0b000000000000000000101010110, 0b000000000000000000101010000),
        RelationshipTest(RelationshipType.CONTAINS,         0b000000000000000000101010110, 0b000000000000000000101000000),
        RelationshipTest(RelationshipType.EQUALS,           0b000000000000000000101001110, 0b000000000000000000100000000)
        ]
    for rel_def in test_binaries:
        result = rel_def.test(relation_binary)
        if result:
            return result
    return RelationshipType.UNKNOWN


def merge_rel(relation_seq: pd.Series)->int:
    '''Aggregate all the relationship values from each slice to obtain
        one relationship value for the two structures.

    Args:
        relation_seq (pd.Series): The relationship values between the
        contours from each slice.

    Returns:
        int: An integer corresponding to a 27 bit binary value
            reflecting the combined DE-9IM relationship between struct2
            and the struct1 convex hulls, exteriors and contours.
    '''
    relation_seq.drop_duplicates(inplace=True)
    merged_rel = 0
    for rel in list(relation_seq):
        merged_rel = merged_rel | rel
    return merged_rel




# %% Relationship class
class Relationship():
    symmetric_relations = [
        RelationshipType.DISJOINT,
        RelationshipType.OVERLAPS,
        RelationshipType.BORDERS,
        RelationshipType.EQUALS,
        RelationshipType.UNKNOWN  # If unknown structure order is irrelevant.
        ]
    transitive_relations = [
        RelationshipType.EQUALS,
        RelationshipType.SHELTERS,
        RelationshipType.SURROUNDS,
        RelationshipType.CONTAINS,
        ]
    metric_match = {
        RelationshipType.DISJOINT: DistanceMetric,
        RelationshipType.BORDERS: OverlapSurfaceMetric,
        RelationshipType.BORDERS_INTERIOR: OverlapSurfaceMetric,
        RelationshipType.OVERLAPS: OverlapAreaMetric,
        RelationshipType.PARTITION: OverlapAreaMetric,
        RelationshipType.SHELTERS: MarginMetric,
        RelationshipType.SURROUNDS: MarginMetric,
        RelationshipType.CONTAINS: MarginMetric,
        RelationshipType.EQUALS: NoMetric,
        RelationshipType.UNKNOWN: NoMetric,
        }

    def __init__(self, structures: StructurePair,
                 slice_table: pd.DataFrame = pd.DataFrame(), **kwargs) -> None:
        self.is_logical = False
        self.show = True
        self.metric = None
        # Sets the is_logical and metric attributes, if supplied.  Ignores any
        # other items in kwargs.
        self.set(**kwargs)
        # Order the structures with the largest first.
        self.structures = None
        self.set_structures(structures)
        # Determine the relationship type.  Either set it from a kwargs value
        # or determine it by comparing the structures.
        self.relationship_type = RelationshipType.UNKNOWN
        if 'relationship' in kwargs:
            self.relationship_type = RelationshipType[kwargs['relationship']]
        else:
            self.identify_relationship(slice_table)
        self.get_metric()

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def get_metric(self):
        # Select the appropriate metric for the identified relationship.
        metric_class = self.metric_match[self.relationship_type]
        self.metric = metric_class(self.structures)

    @property
    def is_symmetric(self)-> bool:
        return self.relationship_type in self.symmetric_relations

    @property
    def is_transitive(self)-> bool:
        return self.relationship_type in self.transitive_relations

    def set_structures(self, structures: StructurePair,
                       slice_table: pd.DataFrame = pd.DataFrame()) -> None:
        # FIXME Stub method to be replaced with set_structures function.
        # Order the structures with the larger one first
        if slice_table.empty:
            self.structures = structures

    def identify_relationship(self, slice_table: pd.DataFrame) -> None:
        '''Get the 27 bit relationship integer for two structures,

            When written in binary, the 27 bit relationship contains 3 9 bit
            parts corresponding to DE-9IM relationships. The left-most 9 bits
            are the relationship between the second structure's contour and the
            first structure's convex hull.  The middle 9 bits are the
            relationship between the second structure's contour and the first
            structure's exterior. (The first structure's contour with any holes
            filled). The right-most 9 bits are the relationship between the
            second structure's contour and the first structure's actual contour.

            Note: The order of structures matters. For correct comparison, the
            first structure should always be the larger of the two structures.

            Args:
                slice_structures (pd.DataFrame): A table of structures, where the
                    values are the contours with type StructureSlice. The column
                    index contains the roi numbers for the structures.  The row
                    index contains the slice index distances.
        '''
        slice_structures = slice_table.loc[:, [self.structures[0],
                                               self.structures[1]]]
        # Remove Slices that have neither structure.
        slice_structures.dropna(how='all', inplace=True)
        boundary_slices = slice_table.apply(find_boundary_slices)

        # For slices that have only one of the two structures, replace the nan
        # values with empty polygons for duck typing.
        slice_structures.fillna(StructureSlice([]), inplace=True)
        # Get the relationships between the two structures for all slices.
        relation_seq = slice_structures.agg(relate_structures, structures=self.structures,
                                            axis='columns')
        # Adjust the relationship metrics for the boundary slices of both structures.
        relation_seq = adjust_slice_boundary_relations(relation_seq,
                                                       self.structures,
                                                       boundary_slices)

        # Get the overall relationship for the two structures by merging the
        # relationships for the individual slices.
        relation_binary = merge_rel(relation_seq)
        self.relationship_type = identify_relation(relation_binary)
        return relation_binary
