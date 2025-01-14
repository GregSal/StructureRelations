import pytest
import shapely

from structure_slice import StructureSlice
from types_and_classes import RegionNode, SliceNeighbours

from relations import DE27IM, RelationshipType
from debug_tools import box_points, circle_points

def test_de27im_initialization_with_contours():
    poly_a = shapely.Polygon(circle_points(2))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation is not None
    assert de27im.int is not None

def test_de27im_padding_with_contours():
    poly_a = shapely.Polygon(circle_points(2))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation[9:]  == '0' * 18
    assert de27im.relation[:9]  == '100010001'

def test_de27im_initialization_with_regions():
    roi = 1
    slice_index = 0.5
    slice_neighbours = SliceNeighbours(slice_index, 0, 1)
    poly_a = RegionNode(roi, slice_index, slice_neighbours,
                        shapely.Polygon(circle_points(2)))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation is not None
    assert de27im.int is not None

def test_de27im_initialization_with_relation_str():
    relation_str = '111000000111000000111000000'
    de27im = DE27IM(relation_str=relation_str)
    assert de27im.relation == relation_str
    assert de27im.int == int(relation_str, 2)

def test_de27im_initialization_with_relation_int():
    relation_str = '111000000111000000111000000'
    relation_int = int(relation_str, 2)
    de27im = DE27IM(relation_int=relation_int)
    assert de27im.relation == relation_str
    assert de27im.int == relation_int

def test_de27im_to_str():
    relation_str = '111000000111000000111000000'
    relation_int = int(relation_str, 2)
    relation_str = DE27IM.to_str(relation_int)
    assert relation_str == relation_str

def test_de27im_to_int():
    relation_str = '111000000111000000111000000'
    relation_int = DE27IM.to_int(relation_str)
    assert relation_int == int(relation_str, 2)

def test_de27im_identify_relation():
    relation_int = DE27IM.test_binaries[7].value
    de27im = DE27IM(relation_int=relation_int)
    assert de27im.identify_relation() == DE27IM.test_binaries[7].relation_type

def test_de27im_merge():
    relation_str1 = '111000000111000000111000000'
    relation_str2 = '000111000000111000000111000'
    de27im1 = DE27IM(relation_str=relation_str1)
    de27im2 = DE27IM(relation_str=relation_str2)
    merged_de27im = de27im1.merge(de27im2)
    expected_relation = '111111000111111000111111000'
    assert merged_de27im.relation == expected_relation
    assert merged_de27im.int == int(expected_relation, 2)


# TODO test DE9IM.apply_adjustments
