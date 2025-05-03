import networkx as nx
import pandas as pd
import pytest

from shapely.geometry import Polygon

from contours import Contour, ContourMatch, points_to_polygon
from contours import ContourPoints, build_contour_table
from debug_tools import box_points

from contour_graph import *


class TestBuildContours():
    '''Test the build_contours function.'''
    def make_test_contour_table(self):
        '''Create a test contour table.

        Create a contour table for testing build_contours function.
        The contour table contains the following columns:
            ROI, Slice, Points, Polygon, Area.
        The table is sorted by ROI, Slice and by descending area.

        The test table contains the following data:
            ROI 1:
                slices, 1.0, 2.0,
                1 contour per slice with area 9.0.
            ROI 2:
                slices, 0.0, 1.0,
                  2 contours on slice 0.0 with areas 1.0, 4.0
                  4 contours on slice 1.0,
                    two with area 1.0, and
                    two with area 4.0.
        '''
        box1_left = box_points(width=1, offset_x=1.5)
        box1_right = box_points(width=1, offset_x=-1.5)
        box2_left = box_points(width=2, offset_x=1.5)
        box2_right = box_points(width=2, offset_x=-1.5)
        box3_right = box_points(width=3, offset_x=-1.5)
        slice_data = [
            ContourPoints(box1_left, roi=2, slice_index=0.0),  # ROI 2, Area 1
            ContourPoints(box2_left, roi=2, slice_index=0.0),  # ROI 2, Area 4
            ContourPoints(box1_left, roi=2, slice_index=1.0),  # ROI 2, Area 1
            ContourPoints(box1_right, roi=2, slice_index=1.0),  # ROI 2, Area 1
            ContourPoints(box2_left, roi=2, slice_index=1.0),  # ROI 2, Area 4
            ContourPoints(box2_right, roi=2, slice_index=1.0),  # ROI 2, Area 4
            ContourPoints(box3_right, roi=1, slice_index=1.0),  # ROI 1, Area 9
            ContourPoints(box3_right, roi=1, slice_index=2.0),  # ROI 1, Area 9
            ]
        contour_table, _ = build_contour_table(slice_data)
        return contour_table

    def test_contour_building(self):
        '''Test the build_contours function with a simple example.

        '''
        contour_table = self.make_test_contour_table()
        contours = build_contours(contour_table, roi=1)
        assert list(contours.keys()) == [1.0, 2.0]

    def test_contour_sorting(self):
        '''Verify that the build_contours function sorts the contours by
        decreasing area.
        '''
        contour_table = self.make_test_contour_table()
        contours = build_contours(contour_table, roi=2)
        assert list(contours.keys()) == [0.0, 1.0]
        first_slice = contours[0.0]
        second_slice = contours[1.0]
        assert len(first_slice) == 2
        assert len(second_slice) == 4
        assert first_slice[0].area() == 4.0
        assert first_slice[1].area() == 1.0
        assert second_slice[0].area() == 4.0
        assert second_slice[1].area() == 4.0
        assert second_slice[2].area() == 1.0
        assert second_slice[3].area() == 1.0




class TestBuildContourGraph():
    def make_test_contour_table(self):
        '''Create a test contour table.

        Create a contour table for testing build_contours function.
        The contour table contains the following columns:
            ROI, Slice, Points, Polygon, Area.
        The table is sorted by ROI, Slice and by descending area.

        The test table contains the following data:
            ROI 1:
                slices, 1.0, 2.0,
                1 contour per slice with area 9.0.
            ROI 2:
                slices, 0.0, 1.0,
                  2 contours on slice 0.0 with areas 1.0, 4.0
                  4 contours on slice 1.0,
                    two with area 1.0, and
                    two with area 4.0.
        '''
        box1_left = box_points(width=1, offset_x=1.5)
        box1_right = box_points(width=1, offset_x=-1.5)
        box2_left = box_points(width=2, offset_x=1.5)
        box2_right = box_points(width=2, offset_x=-1.5)
        box3_right = box_points(width=3, offset_x=-1.5)
        slice_data = [
            ContourPoints(box1_left, roi=2, slice_index=0.0),   # ROI 2, Area 1
            ContourPoints(box2_left, roi=2, slice_index=0.0),   # ROI 2, Area 4
            ContourPoints(box1_left, roi=2, slice_index=1.0),   # ROI 2, Area 1
            ContourPoints(box1_right, roi=2, slice_index=1.0),  # ROI 2, Area 1
            ContourPoints(box2_left, roi=2, slice_index=1.0),   # ROI 2, Area 4
            ContourPoints(box2_right, roi=2, slice_index=1.0),  # ROI 2, Area 4
            ContourPoints(box3_right, roi=1, slice_index=1.0),  # ROI 1, Area 9
            ContourPoints(box3_right, roi=1, slice_index=2.0),  # ROI 1, Area 9
            ]
        contour_table, slice_sequence = build_contour_table(slice_data)
        return contour_table, slice_sequence

    def test_graph_building(self):
        contour_table, slice_sequence = self.make_test_contour_table()
        contours = build_contours(contour_table, roi=1)
        contour_indices = [cn.index for cl in contours.values() for cn in cl]
        graph, slice_sequence = build_contour_graph(contour_table,
                                                    slice_sequence, roi=1)
        assert contour_indices == list(graph.nodes())
