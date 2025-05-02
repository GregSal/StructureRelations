'''Test for the contours.py module.'''
import math

import numpy as np
import pytest
from pytest import approx
import shapely
from shapely.geometry import Polygon

from debug_tools import box_points
from types_and_classes import InvalidContour
from contours import SliceNeighbours, SliceSequence
from contours import points_to_polygon, calculate_new_slice_index
from contours import interpolate_polygon, ContourPoints, build_contour_table
from contours import Contour


class TestSliceNeighbours():
    '''Test the SliceNeighbours class.'''
    def test_initialization_and_types(self):
        '''Test that the SliceNeighbours class initializes correctly and
        that the slice indices are of the correct type.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=4.0, next_slice=6.0)
        assert isinstance(sn.this_slice, float)
        assert isinstance(sn.previous_slice, float)
        assert isinstance(sn.next_slice, float)

    def test_gap_two_neighbours(self):
        '''Test that the gap is calculated correctly when both neighbours are
        present.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=4.0, next_slice=6.0)
        assert sn.gap() == 1.0

    def test_gap_one_neighbour(self):
        '''Test that the gap is calculated correctly when only one neighbour is
        present.'''
        sn2 = SliceNeighbours(this_slice=5.0, previous_slice=None,
                              next_slice=7.0)
        assert sn2.gap() == 2.0
        sn3 = SliceNeighbours(this_slice=5.0, previous_slice=3.0,
                              next_slice=None)
        assert sn3.gap() == 2.0

    def test_gap_no_neighbour(self):
        '''Test that the gap is NaN when no neighbours are present.'''
        sn4 = SliceNeighbours(this_slice=5.0, previous_slice=None,
                              next_slice=None)
        assert math.isnan(sn4.gap())

    def test_gap_negative(self):
        '''Test that the gap is negative if previous_slice > next_slice.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=6.0, next_slice=4.0)
        assert sn.gap(absolute=False) == -1.0

    def test_gap_abs(self):
        '''Test that the gap is forced to be positive if absolute=True.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=6.0, next_slice=4.0)
        assert sn.gap(absolute=True) == 1.0


    def test_is_neighbour(self):
        '''Test that the is_neighbour method works correctly.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=4.0, next_slice=6.0)
        assert sn.is_neighbour(4.0)
        assert sn.is_neighbour(6.0)
        assert not sn.is_neighbour(7.0)

    def test_neighbour_list(self):
        '''Test that the neighbour_list method returns the correct list of
        neighbours.'''
        sn = SliceNeighbours(this_slice=5.0, previous_slice=4.0, next_slice=6.0)
        assert sn.neighbour_list() == [4.0, 6.0]


class TestSliceSequence():
    '''Test the SliceSequence class.'''
    def test_initialization_and_slices(self):
        '''Test that the SliceSequence class initializes correctly and that
        the slices are of the correct type.'''
        ss = SliceSequence([1.0, 2.0, 3.0])
        assert ss.slices == [1.0, 2.0, 3.0]
        assert len(ss) == 3
        assert 2.0 in ss
        assert 4.0 not in ss

    def test_drop_duplicate_slices(self):
        '''Test that the SliceSequence class drops duplicates.'''
        ss = SliceSequence([1.0, 2.0, 2.0, 3.0])
        assert ss.slices == [1.0, 2.0, 3.0]
        assert len(ss) == 3

    def test_neighbour_slices(self):
        '''Test that the SliceSequence class contains the correct neighbour
        slices.'''
        ss = SliceSequence([1.0, 2.0, 3.0, 4.0])
        # Test the neighbours of the first slice
        # Drop the nan values from the NextSlice and PreviousSlice lists
        # because np.nan is not equal to any number.
        assert list(ss.sequence.NextSlice.dropna()) == [2.0, 3.0, 4.0]
        assert list(ss.sequence.PreviousSlice.dropna()) == [1.0, 2.0, 3.0]

    def test_add_slice_from_dict(self):
        '''Test that the add_slice method works correctly when adding a slice
        from a dictionary.'''
        ss = SliceSequence([1.0, 2.0])
        slice_ref = {'ThisSlice': 3.0,
                     'NextSlice': np.nan,
                     'PreviousSlice': 2.0,
                     'Original': True}
        ss.add_slice(**slice_ref)
        assert 3.0 in ss
        assert ss[3.0]['Original'] is True
        assert np.isnan(ss[3.0]['NextSlice'])
        assert ss[3.0]['PreviousSlice'] == 2.0
        assert ss[3.0]['ThisSlice'] == 3.0

    def test_add_slice_defaults(self):
        '''Test that the add_slice method works correctly when adding a slice
        with default values.'''
        ss = SliceSequence([1.0, 2.0])
        ss.add_slice(3.0)
        assert 3.0 in ss
        assert np.isnan(ss[3.0]['NextSlice'])
        assert np.isnan(ss[3.0]['PreviousSlice'])
        assert ss[3.0]['Original'] is False
        assert ss[3.0]['ThisSlice'] == 3.0

    def test_get_nearest_slice(self):
        '''Test that the get_nearest_slice method works correctly.'''
        # get_nearest_slice is currently not used
        ss = SliceSequence([1.0, 2.0, 3.0])
        assert ss.get_nearest_slice(2.1) == 2.0
        assert ss.get_nearest_slice(2.9) == 3.0

    def test_get_neighbors(self):
        '''Test that the get_neighbors method works correctly.'''
        ss = SliceSequence([1.0, 2.0, 3.0])
        sn = ss.get_neighbors(2.0)
        assert sn.this_slice == 2.0
        assert sn.previous_slice == 1.0
        assert sn.next_slice == 3.0


class TestPointsToPolygon():
    '''Test the points_to_polygon function.'''
    def test_valid_polygon(self):
        '''Test that a valid set of points creates a valid polygon.'''
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = points_to_polygon(points)
        assert polygon.is_valid

    def test_invalid_polygon(self):
        '''Test that an invalid set of points raises an InvalidContour error.'''
        points = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 0)]  # Self-intersecting
        with pytest.raises(InvalidContour):
            points_to_polygon(points)


class TestCalculateNewSliceIndex():
    '''Test the calculate_new_slice_index function.'''
    def test_single_slice(self):
        '''Test that a single slice index is returned correctly.'''
        assert calculate_new_slice_index(5.0) ==5.0

    def test_multiple_slices(self):
        '''Test that two slice indices return the correct average.'''
        assert calculate_new_slice_index([1.0, 2.0]) == 1.5

    def test_precision_parameter(self):
        '''Test that the precision parameter rounds the slice index correctly.
        '''
        slices = [1.12345, 1.12355]
        result = calculate_new_slice_index(slices, precision=4)
        assert result == 1.1235

    def test_precision_parameter_with_excessive_rounding(self):
        '''Test that the precision parameter raises an error when the
        calculated slice is out of bounds.
        '''
        slices = [1.12345, 1.12355]
        with pytest.raises(ValueError):
            calculate_new_slice_index(slices, precision=0)


class TestInterpolatePolygon():
    '''Test the interpolate_polygon function.'''
    def test_interpolation_offset(self):
        '''Test that interpolation between two identical sized, but shifted
        polygons returns a correct interpolation.

        First polygon is 2x2 centred on 0,0 and the second is 2x2 centred on 1,1.
        The expected result is a 1x1 polygon centred on 0.5,0.5.
        '''
        p1 = Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        p2 = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        expected_polygon = Polygon([(-0.5, -0.5, 0.5), (1.5, -0.5, 0.5),
                                    (1.5, 1.5, 0.5), (-0.5, 1.5, 0.5),
                                    (-0.5, -0.5, 0.5)])
        interpolated = interpolate_polygon([0, 1], p1, p2)
        assert interpolated.is_valid
        assert interpolated.equals(expected_polygon)

    def test_interpolation_scale(self):
        '''Test that interpolation between two polygons returns a valid
        polygon.

        First polygon is 1x1 centred on 0,0 and the second is 3x3 centred on 0,0.
        The expected result is a 2x2 polygon centred on 0,0.
        '''
        p1 = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
        p2 = Polygon([(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)])
        expected_polygon = Polygon([(-1.0, -1.0, 0.5), (1.0, -1.0, 0.5),
                                    (1.0, 1.0, 0.5), (-1.0, 1.0, 0.5),
                                    (-1.0, -1.0, 0.5)])
        interpolated = interpolate_polygon([0, 1], p1, p2)
        assert interpolated.is_valid
        assert interpolated.equals(expected_polygon)

    def test_identical_polygons(self):
        '''Test that interpolation of two identical polygons returns the same
        polygon.
        '''
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        interpolated = interpolate_polygon([0, 1], p1, p2)
        assert interpolated.equals(p1)
        assert interpolated.equals(p2)

    def test_single_polygon_half_size(self):
        '''Test that interpolation with a single polygon returns a polygon with
        the same shape, but half the size.
        '''
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        interpolated = interpolate_polygon([0], p1)
        shape_difference = shapely.affinity.scale(p1, 0.5, 0.5)  - interpolated
        assert interpolated.area == approx(p1.area / 4, rel=1e-9)
        assert shape_difference.area == pytest.approx(0.0, rel=1e-9)

    def test_polygon_with_hole(self):
        '''Test that interpolation works for a polygon with a hole.

        Test interpolation of a constant outer polygon with a hole in the
        first polygon.
        '''
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
        p1 = Polygon(shell=exterior, holes=[hole])
        p2 = Polygon(shell=exterior)
        interpolated = interpolate_polygon([0, 1], p1, p2)
        poly_hole = Polygon(hole)
        interp_hole = Polygon(interpolated.interiors[0])
        shape_difference = shapely.affinity.scale(interp_hole, 0.5, 0.5)  - interp_hole
        assert shape_difference.area == approx(0.0, rel=1e-9)
        assert interp_hole.area == approx(poly_hole.area / 4, rel=1e-9)


class TestContourPoints():
    '''Test the ContourPoints class.'''
    def test_initialization(self):
        '''Test that the ContourPoints class initializes correctly.'''
        points = [(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5)]
        contour_points = ContourPoints(points, roi=1)
        assert contour_points['ROI'] == 1
        assert contour_points['Slice'] == 0.5
        assert len(contour_points['Points']) == 3

    def test_error_when_no_slice_index_and_invalid_points(self):
        '''Test that ContourPoints raises an error when points are 2D and
        slice_index is not provided.
        '''
        points = [(0, 0), (1, 0), (1, 1)]  # 2D points
        with pytest.raises(InvalidContour):
            ContourPoints(points, roi=1)

    def test_error_when_z_coordinate_not_constant(self):
        '''Test that ContourPoints raises an error when the z-coordinate
        is not constant.
        '''
        points = [(0, 0, 0), (1, 0, 1), (1, 1, 0)]  # Non-constant z-coordinates
        with pytest.raises(InvalidContour):
            ContourPoints(points, roi=1)


class TestBuildContourTable():
    '''Test the build_contour_table function.'''
    def test_table_creation(self):
        '''Test that build_contour_table creates a table with the correct
        columns and values.'''
        box1 = box_points(width=1)
        slice_data = [ContourPoints(box1, roi=1, slice_index= 0.0)]
        table, _ = build_contour_table(slice_data)
        assert len(table) == 1
        assert table['ROI'][0] == 1
        assert table['Slice'][0] == 0.0
        assert table['Area'][0] == 1.0  # Area of the box is 1.0

    def test_sorting_by_roi_slice_area(self):
        '''Test that build_contour_table sorts by ROI, Slice, and
        descending Area .'''
        box1 = box_points(width=1)
        box2 = box_points(width=2)
        box3 = box_points(width=3)
        slice_data = [
            ContourPoints(box1, roi=2, slice_index=0.0),  # ROI 2, Area 4
            ContourPoints(box1, roi=2, slice_index=1.0),  # ROI 2, Area 1
            ContourPoints(box2, roi=2, slice_index=1.0),  # ROI 2, Area 4
            ContourPoints(box3, roi=1, slice_index=1.0),  # ROI 1, Area 9
            ContourPoints(box3, roi=1, slice_index=2.0),  # ROI 1, Area 9
            ]
        table, sequence = build_contour_table(slice_data)
        # Check sorting by ROI, then Slice, then Area descending
        rois = list(table['ROI'])
        slices = list(table['Slice'])
        areas = list(table['Area'])
        slice_sequence = sequence.slices
        # Should be sorted:
        #   - ROI 1, Slice 1.0, Area 9;
        #   - ROI 1, Slice 2.0, Area 9;
        #   - ROI 2, Slice 0.0, Area 4;
        #   - ROI 2, Slice 1.0, Area 4;
        #   - ROI 2, Slice 1.0, Area 1;
        assert rois == [1, 1, 2, 2, 2]
        assert slices == [1.0, 2.0, 0.0, 1.0, 1.0]
        assert slice_sequence == [0.0, 1.0, 2.0]
        assert areas[3] > areas[4]  # Area 4 > Area 1 for ROI 1, Slice 1.0


class TestContour():
    '''Test the Contour class.'''
    def test_initialization(self):
        '''Test that the Contour class initializes correctly.'''
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        contour = Contour(roi=1, slice_index=0.0, polygon=polygon, contours=[])
        assert contour.roi == 1
        assert contour.slice_index == 0.0

    def test_sequential_contour_index(self):
        '''Test that contour_index is assigned sequentially for each new
        Contour instance.'''
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        contour1 = Contour(roi=1, slice_index=0.0, polygon=polygon1, contours=[])
        contour2 = Contour(roi=1, slice_index=1.0, polygon=polygon1, contours=[])
        assert contour1.contour_index + 1 == contour2.contour_index

    def test_identify_holes(self):
        '''Test that the Contour class correctly identifies holes.
        '''
        outer_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
        hole_polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

        # Create the outer contour first
        outer_contour = Contour(roi=1, slice_index=0.0, polygon=outer_polygon,
                                contours=[])
        # Create the hole contour and pass the outer contour in the contours list
        hole_contour = Contour(roi=1, slice_index=0.0, polygon=hole_polygon,
                               contours=[outer_contour])
        # Check that the hole contour is identified correctly
        assert hole_contour.is_hole
        # Check that related_contours are set correctly for both contours
        assert hole_contour.related_contours == [outer_contour.contour_index]
        assert outer_contour.related_contours == [hole_contour.contour_index]

    def test_identify_islands(self):
        '''Test that the Contour class correctly identifies islands.
        '''
        outer_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
        hole_polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

        # Create the outer contour first
        outer_contour = Contour(roi=1, slice_index=0.0, polygon=outer_polygon,
                                contours=[])
        # Create the hole contour and pass the outer contour in the contours list
        hole_contour = Contour(roi=1, slice_index=0.0, polygon=hole_polygon,
                               contours=[outer_contour])
        # Check that the hole contour is identified correctly
        assert hole_contour.is_hole
        # Check that related_contours are set correctly for both contours
        assert hole_contour.related_contours == [outer_contour.contour_index]
        assert outer_contour.related_contours == [hole_contour.contour_index]

    def test_error_on_overlapping_contours(self):
        '''Test that an error is raised if a contour overlaps with an existing contour.'''
        polygon1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        polygon2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # Overlaps with polygon1
        contour1 = Contour(roi=1, slice_index=0.0, polygon=polygon1, contours=[])
        with pytest.raises(InvalidContour):
            Contour(roi=1, slice_index=0.0, polygon=polygon2, contours=[contour1])
