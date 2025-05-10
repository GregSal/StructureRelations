import pytest
import shapely
import numpy as np
import pandas as pd
import networkx as nx
from types_and_classes import SliceIndexType
from region_slice import RegionSlice, empty_structure
from utilities import make_multi

class DummyContour:
    def __init__(self, area=1.0, is_empty=False):
        self.area = area
        self.is_empty = is_empty
        self.geoms = [shapely.Polygon([(0,0),(1,0),(1,1),(0,1)])] if not is_empty else []
        self.is_empty = is_empty

    def convex_hull(self):
        return shapely.Polygon([(0,0),(1,0),(1,1),(0,1)])

    @property
    def bounds(self):
        return (0,0,1,1)

    def __bool__(self):
        return not self.is_empty


def make_minimal_region_slice(empty=False):
    # Create a minimal RegionSlice with or without regions
    g = nx.Graph()
    slice_index = SliceIndexType(0.0)
    rs = RegionSlice.__new__(RegionSlice)
    if empty:
        rs.regions = []
        rs.boundaries = []
        rs.open_holes = []
        rs.contour_indexes = []
        rs.region_indexes = []
        rs.is_interpolated = False
    else:
        poly = shapely.Polygon([(0,0),(1,0),(1,1),(0,1)])
        rs.regions = [make_multi(poly)]
        rs.boundaries = [make_multi(shapely.Polygon())]
        rs.open_holes = [make_multi(shapely.Polygon())]
        rs.contour_indexes = [[0]]
        rs.region_indexes = [[0]]
        rs.is_interpolated = False
    return rs


def test_region_slice_is_empty_true():
    rs = make_minimal_region_slice(empty=True)
    assert rs.is_empty is True
    assert not rs

def test_region_slice_is_empty_false():
    rs = make_minimal_region_slice(empty=False)
    assert rs.is_empty is False
    assert rs

def test_region_slice_exterior_and_hull():
    rs = make_minimal_region_slice(empty=False)
    exteriors = rs.exterior
    hulls = rs.hull
    assert isinstance(exteriors, list)
    assert isinstance(hulls, list)
    assert all(hasattr(e, 'is_empty') for e in exteriors)
    assert all(hasattr(h, 'is_empty') for h in hulls)

def test_empty_structure_none():
    assert empty_structure(None) is True

def test_empty_structure_nan():
    assert empty_structure(float('nan')) is True
    assert empty_structure(np.nan) is True

def test_empty_structure_region_slice():
    rs_empty = make_minimal_region_slice(empty=True)
    rs_full = make_minimal_region_slice(empty=False)
    assert empty_structure(rs_empty) is True
    assert empty_structure(rs_full) is False

def test_empty_structure_invert():
    rs_empty = make_minimal_region_slice(empty=True)
    assert empty_structure(rs_empty, invert=True) is False
    rs_full = make_minimal_region_slice(empty=False)
    assert empty_structure(rs_full, invert=True) is True

def test_empty_structure_dict_key():
    d = {'is_empty': True}
    assert empty_structure(d) is True
    d = {'is_empty': False}
    assert empty_structure(d) is False

def test_empty_structure_area():
    class Dummy:
        area = 0
    assert empty_structure(Dummy()) is True
    Dummy.area = 1
    assert empty_structure(Dummy()) is False

def test_empty_structure_unknown():
    class Dummy: pass
    assert empty_structure(Dummy()) is True
