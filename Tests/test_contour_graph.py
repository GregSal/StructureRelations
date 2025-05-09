from random import uniform
import networkx as nx
import pandas as pd
import pytest

from shapely.geometry import Polygon

from contours import Contour, ContourMatch, SliceSequence, points_to_polygon
from contours import ContourPoints, build_contour_table
from debug_tools import box_points

from contour_graph import *


def basic_contour_table():
    '''Create a contour table for testing the build_contours and add_graph_edges
    functions.

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

def boundary_test_contour_table():
    '''Create a test contour table.

    Create a contour table for testing add_boundary_contours function.
    The contour table contains the following columns:
        ROI, Slice, Points, Polygon, Area.
    The table is sorted by ROI, Slice and by descending area.

    The test table contains the following data:
        ROI 0:
            slices, 0.0, 1.0, 2.0, 3.0
            1 contour per slice with area 1.0.
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
    box1= box_points(width=1)
    box3_right = box_points(width=3, offset_x=-1.5)
    slice_data = [
        ContourPoints(box1, roi=0, slice_index=0.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=1.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=2.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=3.0),        # ROI 0, Area 1
        ContourPoints(box3_right, roi=1, slice_index=1.0),  # ROI 1, Area 9
        ContourPoints(box3_right, roi=1, slice_index=2.0),  # ROI 1, Area 9
        ]
    contour_table, slice_sequence = build_contour_table(slice_data)
    return contour_table, slice_sequence

def region_test_contour_table():
    '''Create a contour table for testing set_enclosed_regions function.

    The test table contains the following data:
        ROI 0: The background region that defines the set of all slice indexes.
            slices, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        ROI 1:  Simple region with 2 slices and 1 contour per slice.
        ROI 2: 2 contours and one hole on slices 1.0 to 4.0,
            results in 3 regions
        ROI 3: 2 contours on slices 1.0 to 3.0,
               1 large contour on slice 4.0 that matches with both contours
               on slice 3.0,
               results in a single region.
    '''
    box1= box_points(width=1)
    box1_right = box_points(width=1, offset_x=-1.5)
    box2_left = box_points(width=2, offset_x=1.5)
    box2_right = box_points(width=2, offset_x=-1.5)
    box3_right = box_points(width=3, offset_x=-1.5)
    box4= box_points(width=4)
    slice_data = [
        ContourPoints(box1, roi=0, slice_index=0.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=1.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=2.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=3.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=4.0),        # ROI 0, Area 1
        ContourPoints(box1, roi=0, slice_index=5.0),        # ROI 0, Area 1

        ContourPoints(box3_right, roi=1, slice_index=1.0),  # ROI 1, Area 1
        ContourPoints(box3_right, roi=1, slice_index=2.0),  # ROI 1, Area 1

        ContourPoints(box2_right, roi=2, slice_index=1.0),  # ROI 2, Area 4
        ContourPoints(box1_right, roi=2, slice_index=1.0),  # ROI 2, Hole
        ContourPoints(box2_left,  roi=2, slice_index=1.0),  # ROI 2, Area 4
        ContourPoints(box2_right, roi=2, slice_index=2.0),  # ROI 2, Area 4
        ContourPoints(box1_right, roi=2, slice_index=2.0),  # ROI 2, Hole
        ContourPoints(box2_left,  roi=2, slice_index=2.0),  # ROI 2, Area 4
        ContourPoints(box2_right, roi=2, slice_index=3.0),  # ROI 2, Area 4
        ContourPoints(box1_right, roi=2, slice_index=3.0),  # ROI 2, Hole
        ContourPoints(box2_left,  roi=2, slice_index=3.0),  # ROI 2, Area 4
        ContourPoints(box2_right, roi=2, slice_index=4.0),  # ROI 2, Area 4
        ContourPoints(box1_right, roi=2, slice_index=4.0),  # ROI 2, Hole
        ContourPoints(box2_left,  roi=2, slice_index=4.0),  # ROI 2, Area 4

        ContourPoints(box2_right, roi=3, slice_index=1.0),  # ROI 4, Area 4
        ContourPoints(box2_left,  roi=3, slice_index=1.0),  # ROI 4, Area 4
        ContourPoints(box2_right, roi=3, slice_index=2.0),  # ROI 4, Area 4
        ContourPoints(box2_left,  roi=3, slice_index=2.0),  # ROI 4, Area 4
        ContourPoints(box2_right, roi=3, slice_index=3.0),  # ROI 4, Area 4
        ContourPoints(box2_left,  roi=3, slice_index=3.0),  # ROI 4, Area 4
        ContourPoints(box4,       roi=3, slice_index=4.0),  # ROI 4, Area 16
        ]
    contour_table, slice_sequence = build_contour_table(slice_data)
    return contour_table, slice_sequence


def hole_test_contour_table():
    '''Create a contour table for testing set_enclosed_regions function.

    The test table contains the following data:
        ROI 0: The background region that defines the set of all slice indexes.
            slices, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        ROI 1: 1 contour with an embedded hole on slices 1.0 to 4.0,
            results in open hole
        ROI 2: 1 contour on slices 1.0 to 4.0,
               an embedded hole on slices 2.0 and 3.0,
               results in a closed hole.
        ROI 3: 1 contour on slices 1.0 to 4.0,
               an embedded hole on slices 1.0 and 3.0,
               results in a open hole.
    '''
    box1= box_points(width=1)
    box3_right = box_points(width=3, offset_x=-1.5)
    box1_right = box_points(width=1, offset_x=-1.5)
    box3_left = box_points(width=3, offset_x=1.5)
    box1_left = box_points(width=1, offset_x=1.5)
    box3_ant = box_points(width=3, offset_y=1.5)
    box1_ant = box_points(width=1, offset_y=1.5)
    slice_data = [
       ContourPoints(box1, roi=0, slice_index=0.0),        # ROI 0, Area 1
       ContourPoints(box1, roi=0, slice_index=1.0),        # ROI 0, Area 1
       ContourPoints(box1, roi=0, slice_index=2.0),        # ROI 0, Area 1
       ContourPoints(box1, roi=0, slice_index=3.0),        # ROI 0, Area 1
       ContourPoints(box1, roi=0, slice_index=4.0),        # ROI 0, Area 1
       ContourPoints(box1, roi=0, slice_index=5.0),        # ROI 0, Area 1

       ContourPoints(box3_right, roi=1, slice_index=1.0),  # ROI 1, Area 4
       ContourPoints(box1_right, roi=1, slice_index=1.0),  # ROI 1, Hole
       ContourPoints(box3_right, roi=1, slice_index=2.0),  # ROI 1, Area 4
       ContourPoints(box1_right, roi=1, slice_index=2.0),  # ROI 1, Hole
       ContourPoints(box3_right, roi=1, slice_index=3.0),  # ROI 1, Area 4
       ContourPoints(box1_right, roi=1, slice_index=3.0),  # ROI 1, Hole
       ContourPoints(box3_right, roi=1, slice_index=4.0),  # ROI 1, Area 4
       ContourPoints(box1_right, roi=1, slice_index=4.0),  # ROI 1, Hole

       ContourPoints(box3_left, roi=2, slice_index=1.0),  # ROI 2, Area 4
       ContourPoints(box3_left, roi=2, slice_index=2.0),  # ROI 2, Area 4
       ContourPoints(box1_left, roi=2, slice_index=2.0),  # ROI 2, Hole
       ContourPoints(box3_left, roi=2, slice_index=3.0),  # ROI 2, Area 4
       ContourPoints(box1_left, roi=2, slice_index=3.0),  # ROI 2, Hole
       ContourPoints(box3_left, roi=2, slice_index=4.0),  # ROI 2, Area 4

       ContourPoints(box3_ant, roi=3, slice_index=1.0),  # ROI 3, Area 4
       ContourPoints(box1_ant, roi=3, slice_index=1.0),  # ROI 3, Hole
       ContourPoints(box3_ant, roi=3, slice_index=2.0),  # ROI 3, Area 4
       ContourPoints(box1_ant, roi=3, slice_index=2.0),  # ROI 3, Hole
       ContourPoints(box3_ant, roi=3, slice_index=3.0),  # ROI 3, Area 4
       ContourPoints(box1_ant, roi=3, slice_index=3.0),  # ROI 3, Hole
       ContourPoints(box3_ant, roi=3, slice_index=4.0),  # ROI 3, Area 4
    ]
    contour_table, slice_sequence = build_contour_table(slice_data)
    return contour_table, slice_sequence


def contour_graph_for_testing(contour_table: pd.DataFrame,
                                         roi: ROI_Type) -> ContourGraph:
    contour_by_slice = build_contours(contour_table, roi)
    # Create an empty graph
    contour_graph = nx.Graph()
    # Add nodes to the graph
    for contour_data in contour_by_slice.values():
        for contour in contour_data:
            contour_label = contour.index
            contour_graph.add_node(contour_label, contour=contour)
    return contour_graph


class TestBuildContours():
    '''Test the build_contours function.'''
    def test_contour_building(self):
        '''Test the build_contours function with a simple example.

        '''
        box3_right = box_points(width=3, offset_x=-1.5)
        slice_data = [ContourPoints(box3_right, roi=1, slice_index=1.0),
                      ContourPoints(box3_right, roi=1, slice_index=2.0)]
        contour_table, _ = build_contour_table(slice_data)
        contours = build_contours(contour_table, roi=1)
        assert list(contours.keys()) == [1.0, 2.0]

    def test_contour_sorting(self):
        '''Verify that the build_contours function sorts the contours by
        decreasing area.
        '''
        contour_table, _ = basic_contour_table()
        contours = build_contours(contour_table, roi=2)
        assert list(contours.keys()) == [0.0, 1.0]
        first_slice = contours[0.0]
        second_slice = contours[1.0]
        assert len(first_slice) == 2
        assert len(second_slice) == 4
        assert first_slice[0].area == 4.0
        assert first_slice[1].area == 1.0
        assert second_slice[0].area == 4.0
        assert second_slice[1].area == 4.0
        assert second_slice[2].area == 1.0
        assert second_slice[3].area == 1.0


class TestAddGraphEdges():
    '''Test the add_graph_edges function.
    '''
    def test_add_graph_edges(self):
        '''Test the add_graph_edges function generates the correct number of
        edges.'''
        contour_table, slice_sequence = basic_contour_table()
        contour_graph = contour_graph_for_testing(contour_table, roi=1)
        contour_graph = add_graph_edges(contour_graph, slice_sequence)
        # Check that the graph for ROI 1 has a single edge between the two
        # slices.
        assert contour_graph.number_of_edges() == 1
        contour_graph = contour_graph_for_testing(contour_table, roi=2)
        contour_graph = add_graph_edges(contour_graph, slice_sequence)
        # Check that the graph for ROI 2 has two edges between the two slices.
        assert contour_graph.number_of_edges() == 2

    def test_edge_match(self):
        '''Test the match data is correct for an edge.'''
        contour_table, slice_sequence = basic_contour_table()
        contour_graph = contour_graph_for_testing(contour_table, roi=1)
        contour_graph = add_graph_edges(contour_graph, slice_sequence)
        # Check that the polygons in the match are correct.
        contour1_selection = ((contour_table['ROI'] == 1) &
                              (contour_table['Slice'] == 1.0))
        contour2_selection = ((contour_table['ROI'] == 1) &
                              (contour_table['Slice'] == 1.0))
        polygon1 = contour_table[contour1_selection].Polygon.tolist()[0]
        polygon2 = contour_table[contour2_selection].Polygon.tolist()[0]
        contour_match = list(contour_graph.edges.data('match'))[0][2]
        assert (polygon1 - contour_match.contour1.polygon).is_empty
        assert (polygon2 - contour_match.contour2.polygon).is_empty
        assert contour_match.gap == 1.0

class TestBoundaryContourGeneration():
    '''Test the add_boundary_contours function.
        Test the boundary contour:
            - The slice index of the boundary contour is half way between the
                boundary contour slice and the neighbour slice that is not
                linked with an edge.
            - IsInterpolated is True
            - IsBoundary is True
            - IsHole matches the IsHole value of the non-interpolated Contour
            - The area of the interpolated contour is less than the area of the
                non-interpolated Contour
        Test the edge between the original contour and the interpolated contour:
            - The ContourMatch contains references to the original contour
                and the interpolated contour.
            - The gap is the distance between the original contour and the
                interpolated contour.
        Test the changes to the SliceSequence:
            - The SliceSequence contains the interpolated contour slice index.
            - The interpolated contour slice index has 'Original' = False.
            - The 'PreviousSlice' and 'NextSlice' are set appropriately (one to
                None, the other to the SliceIndex of the original end contour).
        '''
    # pylint: disable=attribute-defined-outside-init
    def setup_method(self):
        contour_table, slice_sequence = boundary_test_contour_table()
        contour_graph = contour_graph_for_testing(contour_table, roi=1)
        contour_graph = add_graph_edges(contour_graph, slice_sequence)
        contour_graph, slice_sequence = add_boundary_contours(contour_graph,
                                                              slice_sequence)
        self.contour_graph = contour_graph
        self.slice_sequence = slice_sequence

    def test_interpolated_boundary_contour_properties(self):
        # Find two interpolated boundary contours
        # (IsInterpolated and IsBoundary True)
        interpolated = [node for node, data in self.contour_graph.nodes.data('contour')
                        if data.is_interpolated and data.is_boundary]
        assert len(interpolated) == 2
        intp_contour1 = self.contour_graph.nodes[interpolated[0]]['contour']
        intp_contour2 = self.contour_graph.nodes[interpolated[1]]['contour']
        # Check flags
        assert intp_contour1.is_interpolated is True
        assert intp_contour1.is_boundary is True
        assert intp_contour2.is_interpolated is True
        assert intp_contour2.is_boundary is True
        # Find original contours (should be two for ROI 1)
        originals = [node for node, data in self.contour_graph.nodes.data('contour')
                        if not data.is_interpolated and not data.is_boundary]
        assert len(originals) == 2
        orig_contour_1 = self.contour_graph.nodes[originals[0]]['contour']
        orig_contour_2 = self.contour_graph.nodes[originals[1]]['contour']
        # Check slice index is halfway between original and neighbor
        slice_indices = [orig_contour_1.slice_index, orig_contour_2.slice_index]
        expected_slices = [slice_indices[0] - 0.5, slice_indices[1] + 0.5]
        actual_slices = [intp_contour1.slice_index, intp_contour2.slice_index]
        assert actual_slices == expected_slices
        # verify that is_hole matches original
        assert intp_contour1.is_hole == orig_contour_1.is_hole
        assert intp_contour2.is_hole == orig_contour_2.is_hole
        # verify that region_index matches original
        assert intp_contour1.region_index == orig_contour_1.region_index
        assert intp_contour2.region_index == orig_contour_2.region_index
        # Area is less than original
        assert intp_contour1.area < orig_contour_1.area
        assert intp_contour2.area < orig_contour_2.area

    def test_edge_between_original_and_interpolated(self):
        '''Test the edge between the original and interpolated contours.
        verify that:
            - The ContourMatch contains references to the original contour
                and the interpolated contour.
            - The gap is the distance between the original contour and the
                interpolated contour.
        '''
        # Find interpolated boundary contour
        interpolated = [node for node, data in self.contour_graph.nodes.data('contour')
                        if data.is_interpolated and data.is_boundary]
        intp_contour1 = self.contour_graph.nodes[interpolated[0]]['contour']
        intp_contour2 = self.contour_graph.nodes[interpolated[1]]['contour']
        # Find original contours (should be two for ROI 1)
        originals = [node for node, data in self.contour_graph.nodes.data('contour')
                        if not data.is_interpolated and not data.is_boundary]
        orig_contour_1 = self.contour_graph.nodes[originals[0]]['contour']
        orig_contour_2 = self.contour_graph.nodes[originals[1]]['contour']
        # Find edge between original and interpolated
        edges1 = [edge[2] for edge in self.contour_graph.edges.data('match')
                 if interpolated[0] in edge[:2]]
        match1 = edges1[0]
        edges2 = [edge[2] for edge in self.contour_graph.edges.data('match')
                 if interpolated[1] in edge[:2]]
        match2 = edges2[0]
        # Check that the original boundary contour is in the match.
        assert ((match1.contour1 == orig_contour_1) or
                (match1.contour2 == orig_contour_1))
        assert ((match2.contour1 == orig_contour_2) or
                (match2.contour2 == orig_contour_2))
        # Check that the interpolated boundary contour is in the match.
        assert ((match1.contour1 == intp_contour1) or
                (match1.contour2 == intp_contour1))
        assert ((match2.contour1 == intp_contour2) or
                (match2.contour2 == intp_contour2))
        # Verify that the match gap is the distance between the two slices.
        expected_gap = abs(orig_contour_1.slice_index -
                           intp_contour1.slice_index)
        assert abs(match1.gap - expected_gap) < 1e-6
        assert abs(match2.gap - expected_gap) < 1e-6

    def test_slice_sequence_for_interpolated(self):
        '''Test the changes to the SliceSequence:
            - The SliceSequence contains the interpolated contour slice index.
            - The interpolated contour slice index has 'Original' = False.
            - One of 'PreviousSlice' and 'NextSlice' are set to the
                SliceIndex of the original end contour.
            - 'PreviousSlice' and 'NextSlice' of the original end contour do
                not contain the interpolated slice index.
        '''
        # Find interpolated boundary contour
        interpolated = [node for node, data in self.contour_graph.nodes.data('contour')
                        if data.is_interpolated and data.is_boundary]
        intp_contour1 = self.contour_graph.nodes[interpolated[0]]['contour']
        intp_contour2 = self.contour_graph.nodes[interpolated[1]]['contour']
        # Find original contours (should be two for ROI 1)
        originals = [node for node, data in self.contour_graph.nodes.data('contour')
                        if not data.is_interpolated and not data.is_boundary]
        orig_contour_1 = self.contour_graph.nodes[originals[0]]['contour']
        orig_contour_2 = self.contour_graph.nodes[originals[1]]['contour']
        # Check SliceSequence contains interpolated slice
        slices = self.slice_sequence.slices
        assert intp_contour1.slice_index in slices
        assert intp_contour2.slice_index in slices
        # Check that interpolated contour slice indexes have 'Original' = False.
        original_flag = self.slice_sequence.sequence.loc[[
            intp_contour1.slice_index,
            intp_contour2.slice_index],
            'Original']
        assert all(original_flag == False)
        # Check that 'PreviousSlice' and 'NextSlice' of the original end
        # contour do not contain the interpolated slice index.
        linked_slices1 = list(self.slice_sequence.sequence.loc[
            orig_contour_1.slice_index,
            ['PreviousSlice', 'NextSlice']
            ])
        linked_slices2 = list(self.slice_sequence.sequence.loc[
            orig_contour_2.slice_index,
            ['PreviousSlice', 'NextSlice']
            ])
        assert intp_contour1.slice_index not in linked_slices1
        assert intp_contour2.slice_index not in linked_slices2
        # Check that one of 'PreviousSlice' and 'NextSlice' are set to the
        # SliceIndex of the original end contour.
        linked_slices1 = list(self.slice_sequence.sequence.loc[
            intp_contour1.slice_index,
            ['PreviousSlice', 'NextSlice']
            ])
        linked_slices2 = list(self.slice_sequence.sequence.loc[
            intp_contour2.slice_index,
            ['PreviousSlice', 'NextSlice']
            ])
        assert orig_contour_1.slice_index in linked_slices1
        assert orig_contour_2.slice_index in linked_slices2

class TestBuildContourGraph():
    '''Test the build_contour_graph function.'''
    @pytest.mark.xfail
    def test_add_graph_nodes(self):
        contour_table, slice_sequence = basic_contour_table()
        contours = build_contours(contour_table, roi=1)
        contour_indices = [cn.index for cl in contours.values() for cn in cl]
        graph, slice_sequence = build_contour_graph(contour_table,
                                                    slice_sequence, roi=1)
        assert contour_indices == list(graph.nodes())


class TestRegionIdentification():
    '''Test the region identification function.

    - Verify that all nodes in ContourGraph that have a path between them have
        the same RegionIndex.
    - Verify that all nodes in ContourGraph that are not connected by a path
        have different RegionIndexes.
    - Verify that changes to the Contour nodes are reflected in the nodes of
        the ContourGraph.



    '''
    # pylint: disable=attribute-defined-outside-init
    def region_graph_prep(self, roi):
        contour_table, slice_sequence = region_test_contour_table()
        contour_graph = contour_graph_for_testing(contour_table, roi=roi)
        contour_graph = add_graph_edges(contour_graph, slice_sequence)
        contour_graph = set_enclosed_regions(contour_graph)
        self.slice_sequence = slice_sequence
        return contour_graph


    def test_uniform_region_identification(self):
        '''Test that all nodes in ContourGraph that have a path between them
        have the same RegionIndex.
        '''
        # ROI 1:  Single region with 2 slices and 1 contour per slice.
        contour_graph = self.region_graph_prep(roi=1)
        contours_data = dict(contour_graph.nodes.data('contour'))
        region_indexes = {contour.region_index
                          for contour in contours_data.values()}
        assert len(region_indexes) == 1

        # ROI 2 has 2 contours and one hole on slices 1.0 to 4.0, which results
        # in 3 regions.
        contour_graph = self.region_graph_prep(roi=2)
        contours_data = dict(contour_graph.nodes.data('contour'))
        region_indexes = {contour.region_index
                          for contour in contours_data.values()}
        assert len(region_indexes) == 3

        # ROI 3 has 2 contours on slices 1.0 to 3.0, and 1 large contour on
        # slice 4.0 that matches with both contours on slice 3.0, which results
        # in a single region.
        contour_graph = self.region_graph_prep(roi=3)
        contours_data = dict(contour_graph.nodes.data('contour'))
        region_indexes = {contour.region_index
                          for contour in contours_data.values()}
        assert len(region_indexes) == 1

class TestHoleType():
    '''Test the hole type identification.

    Test that 'Open' and 'Closed' regions that are holes are identified
    correctly.

    Define two structures. One that has an 'Open' hole and the other that
    has a 'Closed' hole.
    '''
    # pylint: disable=attribute-defined-outside-init
    def setup_method(self):
        self.contour_table, self.slice_sequence = hole_test_contour_table()


    def test_hole_type(self):
        '''Test that the hole type of contours is set correctly.'''
        contour_graph = contour_graph_for_testing(self.contour_table, roi=2)
        contour_graph = add_graph_edges(contour_graph, self.slice_sequence)
        contour_graph = set_enclosed_regions(contour_graph)
        # Find contours with holes
        hole_contours = [node for node, data in contour_graph.nodes.data('contour')
                         if data.is_hole]
        assert len(hole_contours) == 4
        # Check that the hole type is set correctly
        for node in hole_contours:
            contour = contour_graph.nodes[node]['contour']
            assert contour.is_hole is True
