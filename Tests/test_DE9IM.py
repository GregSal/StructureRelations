import pytest
import shapely

from region_slice import RegionSlice

from relations import DE9IM
from debug_tools import box_points, circle_points

def test_box_6_in_box4():
    box6 = shapely.MultiPolygon([shapely.Polygon(box_points(6))])
    box4 = shapely.MultiPolygon([shapely.Polygon(box_points(4))])
    relation = DE9IM(box6, box4)
    relation_string = '212FF1FF2'
    matrix = '\n'.join([
        '|111|',
        '|001|',
        '|001|'
        ])
    assert str(relation) == matrix
    assert relation.relation_str == relation_string

def test_box_6_in_box4_boundary_a():
    box6 = shapely.MultiPolygon([shapely.Polygon(box_points(6))])
    box4 = shapely.MultiPolygon([shapely.Polygon(box_points(4))])
    relation = DE9IM(box6, box4)
    boundary_relation = relation.boundary_adjustment('a')
    relation_string = 'FFF212FF2'
    matrix = '\n'.join([
        '|000|',
        '|111|',
        '|001|'
        ])
    assert str(boundary_relation) == matrix
    assert boundary_relation.relation_str == relation_string

def test_box_6_in_box4_transpose():
    box6 = shapely.MultiPolygon([shapely.Polygon(box_points(6))])
    box4 = shapely.MultiPolygon([shapely.Polygon(box_points(4))])
    relation = DE9IM(box6, box4)
    transpose_relation = relation.transpose()
    relation_string = '2FF1FF212'
    matrix = '\n'.join([
        '|100|',
        '|100|',
        '|111|'
        ])
    assert str(transpose_relation) == matrix
    assert transpose_relation.relation_str == relation_string

def test_box_4_contains_box6():
    box6 = shapely.MultiPolygon([shapely.Polygon(box_points(6))])
    box4 = shapely.MultiPolygon([shapely.Polygon(box_points(4))])
    relation = DE9IM(box4, box6)
    relation_string = '2FF1FF212'
    matrix = '\n'.join([
        '|100|',
        '|100|',
        '|111|'
        ])
    assert str(relation) == matrix
    assert relation.relation_str == relation_string

def test_box_4_contains_box6_boundary_b():
    box6 = shapely.MultiPolygon([shapely.Polygon(box_points(6))])
    box4 = shapely.MultiPolygon([shapely.Polygon(box_points(4))])
    relation = DE9IM(box4, box6)
    boundary_relation = relation.boundary_adjustment('b')
    relation_string = 'F2FF1FF22'
    matrix = '\n'.join([
        '|010|',
        '|010|',
        '|011|'
        ])
    assert str(boundary_relation) == matrix
    assert boundary_relation.relation_str == relation_string

def test_island():
    circle6 = shapely.Polygon(circle_points(3))
    circle4 = shapely.Polygon(circle_points(2))
    circle2 = shapely.Polygon(circle_points(1))
    circle3 = shapely.Polygon(circle_points(1.5))
    a = RegionSlice([circle6, circle4, circle2])
    b = RegionSlice([circle3])
    relation = DE9IM(b.contour, a.contour)
    relation_string = '212FF1212'
    matrix = '\n'.join([
        '|111|',
        '|001|',
        '|111|'
        ])
    assert str(relation) == matrix
    assert relation.relation_str == relation_string

def test_transpose_island():
    circle6 = shapely.Polygon(circle_points(3))
    circle4 = shapely.Polygon(circle_points(2))
    circle2 = shapely.Polygon(circle_points(1))
    circle3 = shapely.Polygon(circle_points(1.5))
    a = RegionSlice([circle6, circle4, circle2])
    b = RegionSlice([circle3])
    relation = DE9IM(b.contour, a.contour)
    transpose_relation = relation.transpose()
    relation_string = '2F21F1212'
    matrix = '\n'.join([
        '|101|',
        '|101|',
        '|111|'
        ])
    assert str(transpose_relation) == matrix
    assert transpose_relation.relation_str == relation_string
