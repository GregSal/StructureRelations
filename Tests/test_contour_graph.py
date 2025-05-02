import networkx as nx
import pandas as pd
import pytest

from shapely.geometry import Polygon

from contours import Contour, ContourMatch, points_to_polygon
from contour_graph import *


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
