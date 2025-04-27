import pytest
import shapely

from types_and_classes import InvalidContourRelation, ROI_Type, SliceIndexType, SliceNeighbours
from region_slice import RegionSlice

def test_structure_slice_initialization_with_position():
    '''Test the initialization of a StructureSlice object with a specific
    slice_position.
    '''
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    slice_position = SliceIndexType(1.0)
    roi = ROI_Type(1)
    structure_slice = RegionSlice(contours, slice_position=slice_position, roi=roi)
    # Check that the slice_position and roi are set correctly
    assert structure_slice.roi == roi
    assert structure_slice.slice_position == slice_position
    assert structure_slice.contour.is_valid

def test_structure_slice_initialization_with_z():
    '''Test the initialization of a StructureSlice object with a provided z
    coordinate.
    '''
    contours = [shapely.geometry.Polygon([(0, 0, 1), (1, 0, 1),
                                          (1, 1, 1), (0, 1, 1)])]
    roi = ROI_Type(1)
    structure_slice = RegionSlice(contours, roi=roi)
    assert structure_slice.roi == roi
    assert structure_slice.slice_position == 1.0
    assert structure_slice.contour.is_valid

def test_is_empty_property():
    '''Test the is_empty property of a StructureSlice object.
    '''
    contours = [shapely.geometry.Polygon()]
    structure_slice = RegionSlice(contours)
    assert structure_slice.is_empty

def test_add_hole():
    '''Test that adding a polygon that is contained within the contour of a
    StructureSlice object results in the polygon being added as a hole in the
    contour.
    '''
    contours = [shapely.geometry.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
    new_contour = shapely.geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    structure_slice = RegionSlice(contours)
    structure_slice.add_contour(new_contour)
    assert len(structure_slice.contour.geoms) == 1
    assert len(structure_slice.contour.geoms[0].interiors) == 1

def test_add_2nd_region():
    '''Test that adding a polygon that is not contained within the contour of a
    StructureSlice object results in the polygon being added as a separate
    contour.
    '''
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    new_contour = shapely.geometry.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    structure_slice = RegionSlice(contours)
    structure_slice.add_contour(new_contour)
    assert len(structure_slice.contour.geoms) == 2

def test_add_overlapping_contour_raises_error():
    '''Test that adding an overlapping polygon raises an InvalidContourRelation
    error.
    '''
    contours = [shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    overlapping_contour = shapely.geometry.Polygon(
        [(1, 1), (3, 1), (3, 3), (1, 3)])
    structure_slice = RegionSlice(contours)
    with pytest.raises(InvalidContourRelation):
        structure_slice.add_contour(overlapping_contour)

def test_inverted_order_raises_error():
    '''Test that adding an overlapping polygon raises an InvalidContourRelation
    error.
    '''
    contours = [shapely.geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])]
    new_contour = shapely.geometry.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    structure_slice = RegionSlice(contours)
    with pytest.raises(InvalidContourRelation):
        structure_slice.add_contour(new_contour)

def test_area_property():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    structure_slice = RegionSlice(contours)
    assert structure_slice.area == 1.0

def test_area_with_hole():
    contours = [shapely.geometry.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
    new_contour = shapely.geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    structure_slice = RegionSlice(contours)
    structure_slice.add_contour(new_contour)
    area = 3*3-1*1
    assert structure_slice.area == area

def test_area_with_two_regions():
    contours = [shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                shapely.geometry.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
    structure_slice = RegionSlice(contours)
    area = 1*1+1*1
    assert structure_slice.area == area

def test_hull_property():
    exterior_contour = shapely.geometry.Polygon(
        [(0, 0), (3, 0), (3, 3), (0, 3)])
    cut_region = shapely.geometry.Polygon(
        [(3, 1), (2, 1), (2, 2), (3, 2)])
    contours = [exterior_contour - cut_region]
    structure_slice = RegionSlice(contours)
    assert shapely.equals(structure_slice.hull, exterior_contour)

def test_exterior_property():
    contours = [shapely.geometry.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
    new_contour = shapely.geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    structure_slice = RegionSlice(contours)
    structure_slice.add_contour(new_contour)
    assert shapely.equals(structure_slice.exterior, contours[0])
