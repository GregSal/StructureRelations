import pytest
import shapely
from structure_slice import StructureSlice
from types_and_classes import ROI_Type, SliceIndexType, SliceNeighbours

def test_structure_slice_initialization():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    slice_position = SliceIndexType(1.0)
    roi = ROI_Type(1)
    structure_slice = StructureSlice(contours, slice_position=slice_position, roi=roi)

    assert structure_slice.roi == roi
    assert structure_slice.slice_position == slice_position
    assert structure_slice.contour.is_valsid

def test_add_contour():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    new_contour = shapely.geometry.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    structure_slice = StructureSlice(contours)
    structure_slice.add_contour(new_contour)

    assert len(structure_slice.contour.geoms) == 2

def test_exterior_property():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    structure_slice = StructureSlice(contours)

    assert structure_slice.exterior.is_valid

def test_hull_property():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    structure_slice = StructureSlice(contours)

    assert structure_slice.hull.is_valid

def test_interiors_property():
    contours = [shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)], holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]])]
    structure_slice = StructureSlice(contours)

    assert len(structure_slice.interiors) == 1

def test_area_property():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    structure_slice = StructureSlice(contours)

    assert structure_slice.area == 1.0

def test_is_empty_property():
    contours = [shapely.geometry.Polygon()]
    structure_slice = StructureSlice(contours)

    assert structure_slice.is_empty

def test_set_slice_neighbours():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    structure_slice = StructureSlice(contours)
    previous_slice = SliceIndexType(0.5)
    next_slice = SliceIndexType(1.5)
    structure_slice.set_slice_neighbours(previous_slice, next_slice)

    assert structure_slice.slice_neighbours.previous_slice == previous_slice
    assert structure_slice.slice_neighbours.next_slice == next_slice
