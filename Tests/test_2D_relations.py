'''This file contains unit tests for the DE27IM relation identification
algorithm, specifically for 2D spatial relations using Shapely geometries.

'''
import shapely
import pytest

from relations import DE27IM, RelationshipType
from debug_tools import circle_points, box_points
from utilities import poly_round

class TestContains:
    '''Tests for the "contains" relationship between geometries.'''
    def test_contains_centered(self):
        '''Test the "contains" relationship for centered circles.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        relation_type = DE27IM(circle6, circle4).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_offset_x(self):
        '''Test the "contains" relationship with an offset circle.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle3_offset_x = shapely.Polygon(circle_points(1.5, offset_x=1.2))
        relation_type = DE27IM(circle6, circle3_offset_x).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_island(self):
        '''Test the "contains" relationship with a circle inside of an island.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle4).union(circle3)
        relation_type = DE27IM(a, circle2).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_embedded_ring(self):
        '''Test the "contains" relationship with a ring embedded in another ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle2)
        b = (circle5 - circle3)
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_offset_xy(self):
        '''Test the "contains" relationship with an offset circle in both x and y.'''
        circle6_offset = shapely.Polygon(circle_points(3, offset_y=-1))
        circle3_offset = shapely.Polygon(circle_points(1.5, offset_x=0.5,
                                                       offset_y=-2))
        relation_type = DE27IM(circle6_offset, circle3_offset).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_multi_region(self):
        '''Test the "contains" relationship with multiple regions.'''
        circle4_left = shapely.Polygon(circle_points(2, offset_x=-3))
        circle4_right = shapely.Polygon(circle_points(2, offset_x=3))
        circle3_left = shapely.Polygon(circle_points(1.5, offset_x=-3))
        circle3_right = shapely.Polygon(circle_points(1.5, offset_x=3))
        circle5_up = shapely.Polygon(circle_points(2.5, offset_y=4))
        circle2_up = shapely.Polygon(circle_points(1, offset_y=3))
        circle3_down = shapely.Polygon(circle_points(1.5, offset_y=-2.5))
        circle1_down = shapely.Polygon(circle_points(0.5, offset_y=-2))
        a = shapely.union_all([circle4_left, circle4_right,
                               circle5_up, circle3_down])
        b = shapely.union_all([circle3_left, circle3_right,
                               circle2_up, circle1_down])
        relation_type = DE27IM(a, b).identify_relation()
        print(relation_type)
        assert relation_type == RelationshipType.CONTAINS

class TestSurrounds:
    '''Tests for the "surrounds" relationship between geometries.'''
    def test_simple_surrounds(self):
        '''Test the "surrounds" relationship with a circle inside of a ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = circle6 - circle4
        relation_type = DE27IM(a, circle2).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def test_surrounds_middle_ring(self):
        '''Test the "surrounds" relationship with a ring in between an island
        and another ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle5).union(circle2)
        b = circle4 - circle3
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def surrounds_two_holes_example(self):
        '''Test the "surrounds" relationship with a complex shape containing
        two holes.'''
        box10x5 = shapely.Polygon(box_points(10,5))
        circle4_left = shapely.Polygon(circle_points(2, offset_x=-3))
        circle3_right = shapely.Polygon(circle_points(1.5, offset_x=3))
        circle2_left = shapely.Polygon(circle_points(1, offset_x=-3,
                                                     offset_y=0.5))
        circle2_right = shapely.Polygon(circle_points(1, offset_x=3))
        a = ((box10x5 - circle4_left) - circle3_right).union(circle2_right)
        relation_type = DE27IM(a, circle2_left).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

class TestShelters:
    '''Tests for the "shelters" relationship between geometries.'''
    def test_shelters_big_hole(self):
        '''Test the "shelters" relationship with a crescent shape sheltering a
        circle.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle4_offset = shapely.Polygon(circle_points(2, offset_x=3.5))
        shell = shapely.difference(circle6, circle5)
        cove = shapely.difference(shell, circle4_offset)
        circle2 = shapely.Polygon(circle_points(1, offset_x=1))
        relation_type = DE27IM(cove, circle2).identify_relation()
        assert relation_type == RelationshipType.SHELTERS

    def test_shelters_circle(self):
        '''Test the "shelters" relationship with a minimal crescent shape
        sheltering a smaller circle.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle3 = shapely.Polygon(circle_points(1.5, offset_x=1.6))
        crescent = shapely.difference(circle6, circle3)
        circle2 = shapely.Polygon(circle_points(1, offset_x=1.5))
        relation_type = DE27IM(crescent, circle2).identify_relation()
        assert relation_type == RelationshipType.SHELTERS

class TestDisjoint:
    '''Tests for the "disjoint" relationship between geometries.'''
    def test_disjoint(self):
        '''Test the "disjoint" relationship with two separate boxes.'''
        circle4_left = shapely.Polygon(circle_points(4, offset_x=-4.5))
        circle4_right = shapely.Polygon(circle_points(4, offset_x=4.5))
        relation_type = DE27IM(circle4_left, circle4_right).identify_relation()
        assert relation_type == RelationshipType.DISJOINT

class TestBorders:
    '''Tests for the "borders" relationship between geometries.'''
    def test_borders_simple(self):
        '''Test the "borders" relationship with two boxes that touch at one
        edge.'''
        box4_left = shapely.Polygon(box_points(4, offset_x=-2))
        box4_right = shapely.Polygon(box_points(4, offset_x=2))
        relation_type = DE27IM(box4_left, box4_right).identify_relation()
        assert relation_type == RelationshipType.BORDERS

    def test_borders_insert(self):
        '''Test the "borders" relationship with a box inserted into another box.'''
        box6 = shapely.Polygon(box_points(6))
        box5_up = shapely.Polygon(box_points(5, offset_y=3))
        box6_cropped = shapely.difference(box6, box5_up)
        relation_type = DE27IM(box6_cropped, box5_up).identify_relation()
        assert relation_type == RelationshipType.BORDERS

class TestConfines:
    '''Tests for the "confines" relationship between geometries.'''
    def test_confines_inner_circle(self):
        '''Test the "confines" relationship with a partial circle inside of a ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box4_offset = shapely.Polygon(box_points(4, offset_x=2))
        ring = shapely.difference(circle6, circle4)
        cropped_circle = shapely.difference(circle4, box4_offset)
        relation_type = DE27IM(ring, cropped_circle).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_ring(self):
        '''Test the "confines" relationship with a ring containing an island and
        another ring filling the gap between the outer ring and the island.

        The inner ring has internal borders with the ring portion of the outer
        ring, but has an external border with the island part of the first
        shape. The internal borders relation wins.
        '''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle4).union(circle2)
        b = circle4 - circle2
        # b has internal borders with the ring portion of a, but has an external
        # border with the island part of a. The internal borders relation wins.
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_embedded_box(self):
        '''Test the "confines" relationship with a box embedded in another box.'''
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        box_with_hole = box6.difference(box4)
        relation_type = DE27IM(box_with_hole, box4).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_corner_box(self):
        '''Test the "confines" relationship with a box that has an offset corner.'''
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        box2_offset = shapely.Polygon(box_points(2, offset_x=-1, offset_y=-1))
        box_with_hole = box6.difference(box4)
        relation_type = DE27IM(box_with_hole, box2_offset).identify_relation()
        assert relation_type == RelationshipType.CONFINES

class TestPartition:
    '''Tests for the "partition" relationship between geometries.'''
    def test_partition_simple(self):
        '''Test the "partition" relationship with a box that is partitioned by
        another box.'''
        box4 = shapely.Polygon(box_points(4))
        box4_cropped = shapely.Polygon(box_points(2, 4, offset_x=-1))
        relation_type = DE27IM(box4, box4_cropped).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_side_box(self):
        '''Test the "partition" relationship with a box that is partitioned by
        another box on one side.'''
        box6 = poly_round(shapely.Polygon(box_points(6)))
        box4_offset = shapely.Polygon(box_points(4, offset_x=-1))
        relation_type = DE27IM(box6, box4_offset).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_island(self):
        '''Test the "partition" relationship with a circle that is partitioned
        by another circle, creating an island.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle4).union(circle2)
        relation_type = DE27IM(a, circle2).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_partial_ring(self):
        '''Test the "partition" relationship with a ring that is partitioned
        by another shape.

        Note: Rounding required here because of floating point inaccuracies.
        '''
        # Rounding required because of floating point inaccuracies.
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box6_offset = shapely.Polygon(box_points(6, offset_x=2))
        ring = shapely.difference(circle6, circle4)
        cropped_ring = poly_round(shapely.difference(ring, box6_offset))
        relation_type = DE27IM(ring, cropped_ring).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    @pytest.mark.xfail
    def test_partition_embedded_circle(self):
        '''Test the "partition" relationship with a circle that is partitioned
        by another circle, creating an island.

        Note: This test exposes a known bug that is likely related to rounding
        errors. The expected relation type is "PARTITION", but due to the bug
        it may not return the correct result.'''
        # This test exposes a known bug that is likely related to rounding errors.
        # Rounding required because of floating point inaccuracies.
        circle6 = poly_round(shapely.Polygon(circle_points(3)))
        circle4_offset = shapely.Polygon(circle_points(2, offset_x=2))
        cropped_circle = poly_round(shapely.intersection(circle6,
                                                         circle4_offset))
        relation_type = DE27IM(circle6, cropped_circle).identify_relation()
        assert relation_type == RelationshipType.PARTITION

class TestOverlaps:
    '''Tests for the "overlaps" relationship between geometries.'''
    def test_overlaps_box(self):
        '''Test the "overlaps" relationship with two boxes that overlap.'''
        box4 = shapely.Polygon(box_points(4))
        box4_offset = shapely.Polygon(box_points(4, offset_x=2))
        relation_type = DE27IM(box4, box4_offset).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_box_circle(self):
        '''Test the "overlaps" relationship with a box and a circle that
        overlap.'''
        circle6 = shapely.Polygon(circle_points(3))
        box6_offset = shapely.Polygon(box_points(6, offset_x=3))
        relation_type = DE27IM(circle6, box6_offset).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_circles(self):
        '''Test the "overlaps" relationship with two circles that overlap.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle6_offset = shapely.Polygon(circle_points(3, offset_x=2))
        relation_type = DE27IM(circle6, circle6_offset).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_box(self):
        '''Test the "overlaps" relationship with a ring and a box that
        overlap.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box6_offset = shapely.Polygon(box_points(6, offset_x=3))
        ring = shapely.difference(circle6, circle4)
        relation_type = DE27IM(ring, box6_offset).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_circle(self):
        '''Test the "overlaps" relationship with a ring and a circle that
        overlap.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle6_offset = shapely.Polygon(circle_points(3, offset_x=2.5))
        ring = shapely.difference(circle6, circle4)
        relation_type = DE27IM(ring, circle6_offset).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_surrounded(self):
        '''Test the "overlaps" relationship with a ring that is surrounded by
        another ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1.5, offset_x=1))
        ring = shapely.difference(circle6, circle4)
        relation_type = DE27IM(ring, circle2).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_surrounded_box(self):
        '''Test the "overlaps" relationship with a box that is surrounded by
        another box.'''
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        box3 = shapely.Polygon(box_points(3, offset_x=1))
        box_with_hole = box6.difference(box4)
        relation_type = DE27IM(box_with_hole, box3).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_surrounded(self):
        '''Test the "overlaps" relationship with a ring that is surrounded by
        another ring.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle4 = shapely.Polygon(circle_points(2))
        ring = shapely.difference(circle6, circle3)
        relation_type = DE27IM(ring, circle4).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_circle_island(self):
        '''Test the "overlaps" relationship with a circle that has an island
        inside it.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle4).union(circle2)
        relation_type = DE27IM(a, circle4).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_concentric_rings(self):
        '''Test the "overlaps" relationship with concentric rings that overlap.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = (circle6 - circle3).union(circle2)
        relation_type = DE27IM(a, circle4).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS


class TestEquals:
    '''Tests for the "equals" relationship between geometries.'''
    def test_equals_box(self):
        '''Test the "equals" relationship with two boxes that are equal.'''
        box6 = shapely.Polygon(box_points(6))
        relation_type = DE27IM(box6, box6).identify_relation()
        assert relation_type == RelationshipType.EQUALS

    def test_equals_circle(self):
        '''Test the "equals" relationship with two circles that are equal.'''
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        cropped_circle = shapely.intersection(circle6, circle5)
        relation_type = DE27IM(circle5, cropped_circle).identify_relation()
        assert relation_type == RelationshipType.EQUALS
