import pytest
from pytest import approx
import shapely
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon

from debug_tools import box_points
from types_and_classes import InvalidContour
from contours import *


class TestPointsToPolygon():
    '''Test the points_to_polygon function.'''
    def test_valid_polygon(self):
        '''Test that a valid set of points creates a valid polygon.'''
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = points_to_polygon(points)
        assert polygon.is_valid

    def test_invalid_polygon(self):
        '''Test that an invalid set of points raises an InvalidContour error.'''
        points = [(0, 0), (1, 0), (0, 1), (0, 0)]  # Self-intersecting
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
        assert table['Points'][0] == box1
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
        slice_sequence = list(sequence['ThisSlice'])
        # Should be sorted:
        #   - ROI 1, Slice 1.0, Area 9;
        #   - ROI 1, Slice 2.0, Area 9;
        #   - ROI 2, Slice 0.0, Area 4;
        #   - ROI 2, Slice 1.0, Area 4;
        #   - ROI 2, Slice 1.0, Area 1;
        assert rois == [1, 1, 2, 2, 2]
        assert slices == [1.0, 2.0, 0.0, 1.0, 1.0]
        assert slice_sequence == [0.0, 1.0, 2, 0]
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


class TestBuildContours():
    def test_contour_building(self):
        contour_table = pd.DataFrame({
            'ROI': [1, 1],
            'Slice': [0.0, 1.0],
            'Points': [[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                       [(0, 0, 1), (1, 0, 1), (1, 1, 1)]]
        })
        contour_table['Polygon'] = contour_table['Points'].apply(points_to_polygon)
        contours = build_contours(contour_table, roi=1)
        assert 0.0 in contours
        assert 1.0 in contours


class TestContourMatch():
    '''Test the ContourMatch class.'''
    def test_initialization_and_thickness(self):
        '''Test ContourMatch initialization and thickness calculation.'''
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        contour1 = Contour(roi=1, slice_index=0.0, polygon=polygon1, contours=[])
        contour2 = Contour(roi=1, slice_index=2.0, polygon=polygon2, contours=[])
        match = ContourMatch(contour1, contour2)
        assert match.gap == 2.0

    def test_direction(self):
        '''Test the direction method of ContourMatch.'''
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        contour1 = Contour(roi=1, slice_index=0.0, polygon=polygon1, contours=[])
        contour2 = Contour(roi=1, slice_index=2.0, polygon=polygon2, contours=[])
        match = ContourMatch(contour1, contour2)
        assert match.direction(contour1) == 1
        assert match.direction(contour2) == -1

    def test_direction_invalid_node(self):
        '''Test that direction raises ValueError for a node not in the match.'''
        polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygon3 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        contour1 = Contour(roi=1, slice_index=0.0, polygon=polygon1, contours=[])
        contour2 = Contour(roi=1, slice_index=2.0, polygon=polygon2, contours=[])
        contour3 = Contour(roi=1, slice_index=4.0, polygon=polygon3, contours=[])
        match = ContourMatch(contour1, contour2)
        with pytest.raises(ValueError):
            match.direction(contour3)


class TestBuildContourGraph():
    def test_graph_building(self):
        contour_table = pd.DataFrame({
            'ROI': [1, 1],
            'Slice': [0.0, 1.0],
            'Points': [[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                       [(0, 0, 1), (1, 0, 1), (1, 1, 1)]]
        })
        contour_table['Polygon'] = contour_table['Points'].apply(points_to_polygon)
        slice_sequence = pd.Series([0.0, 1.0])
        graph, _ = build_contour_graph(contour_table, slice_sequence, roi=1)
        assert isinstance(graph, nx.Graph)
