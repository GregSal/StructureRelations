import pytest
import shapely
import numpy as np
import networkx as nx
from types_and_classes import RegionIndex, SliceIndexType
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
    region_index = RegionIndex('A')
    rs = RegionSlice.__new__(RegionSlice)
    rs.slice_index = slice_index
    if empty:
        rs.regions = {region_index: shapely.MultiPolygon()}
        rs.boundaries = {region_index: shapely.MultiPolygon()}
        rs.open_holes = {region_index: shapely.MultiPolygon()}
        rs.contour_indexes = {region_index: []}
        rs.region_holes = {region_index: []}
        rs.embedded_regions = {region_index: []}
        rs.is_interpolated = False
    else:
        poly = shapely.Polygon([(0,0),(1,0),(1,1),(0,1)])
        rs.regions = {region_index: make_multi(poly)}
        rs.boundaries = {region_index: shapely.MultiPolygon()}
        rs.open_holes = {region_index: shapely.MultiPolygon()}
        rs.contour_indexes = {region_index: [(0,0,0)]}
        rs.region_holes = {region_index: [DummyContour()]}
        rs.embedded_regions = {region_index: [DummyContour()]}
        rs.is_interpolated = False
    return rs

class TestEmptyRegionSlice():
    '''Test cases for empty RegionSlice and empty_structure function'''
    def test_region_slice_is_empty_true(self):
        '''Test empty RegionSlice'''
        rs = make_minimal_region_slice(empty=True)
        assert rs.is_empty is True
        assert not rs

    def test_region_slice_is_empty_false(self):
        '''Test non-empty RegionSlice'''
        rs = make_minimal_region_slice(empty=False)
        assert rs.is_empty is False
        assert rs

    def test_empty_structure_none(self):
        '''Test empty_structure with None'''
        assert empty_structure(None) is True

    def test_empty_structure_nan(self):
        '''Test empty_structure with NaN values'''
        assert empty_structure(float('nan')) is True
        assert empty_structure(np.nan) is True

    def test_empty_structure_region_slice(self):
        '''Test empty_structure with RegionSlice'''
        rs_empty = make_minimal_region_slice(empty=True)
        rs_full = make_minimal_region_slice(empty=False)
        assert empty_structure(rs_empty) is True
        assert empty_structure(rs_full) is False

    def test_empty_structure_invert(self):
        '''Test empty_structure with invert=True'''
        rs_empty = make_minimal_region_slice(empty=True)
        assert empty_structure(rs_empty, invert=True) is False
        rs_full = make_minimal_region_slice(empty=False)
        assert empty_structure(rs_full, invert=True) is True

    def test_empty_structure_dict_key(self):
        '''Test empty_structure with dict key'''
        d = {'is_empty': True}
        assert empty_structure(d) is True
        d = {'is_empty': False}
        assert empty_structure(d) is False

    def test_empty_structure_area(self):
        '''Test empty_structure with area attribute'''
        class Dummy:
            area = 0
        assert empty_structure(Dummy()) is True
        Dummy.area = 1
        assert empty_structure(Dummy()) is False

    def test_empty_structure_unknown(self):
        '''Test empty_structure with no area attribute'''
        class Dummy: pass
        assert empty_structure(Dummy()) is True
