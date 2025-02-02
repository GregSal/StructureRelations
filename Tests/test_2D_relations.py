import shapely
import pytest

from structure_slice import StructureSlice
from relations import DE27IM, RelationshipType
from debug_tools import circle_points, box_points
from utilities import poly_round

class TestContains:
    def test_contains_centered(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        a = StructureSlice([circle6])
        b = StructureSlice([circle4])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_offset_x(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle3_offset_x = shapely.Polygon(circle_points(1.5, offset_x=1.2))
        a = StructureSlice([circle6])
        b = StructureSlice([circle3_offset_x])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_island(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle4, circle3])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_embedded_ring(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle2])
        b = StructureSlice([circle5, circle3])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_offset_xy(self):
        circle6_offset = shapely.Polygon(circle_points(3, offset_y=-1))
        circle3_offset = shapely.Polygon(circle_points(1.5, offset_x=0.5,
                                                       offset_y=-2))
        a = StructureSlice([circle6_offset])
        b = StructureSlice([circle3_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_multi_region(self):
        circle4_left = shapely.Polygon(circle_points(2, offset_x=-3))
        circle4_right = shapely.Polygon(circle_points(2, offset_x=3))
        circle3_left = shapely.Polygon(circle_points(1.5, offset_x=-3))
        circle3_right = shapely.Polygon(circle_points(1.5, offset_x=3))
        circle5_up = shapely.Polygon(circle_points(2.5, offset_y=4))
        circle2_up = shapely.Polygon(circle_points(1, offset_y=3))
        circle3_down = shapely.Polygon(circle_points(1.5, offset_y=-2.5))
        circle1_down = shapely.Polygon(circle_points(0.5, offset_y=-2))
        a = StructureSlice([circle4_left, circle4_right, circle5_up, circle3_down])
        b = StructureSlice([circle3_left, circle3_right, circle2_up, circle1_down])
        relation_type = DE27IM(a, b).identify_relation()
        print(relation_type)
        assert relation_type == RelationshipType.CONTAINS

class TestSurrounds:
    def test_simple_surrounds(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def test_surrounds_middle_ring(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle5, circle2])
        b = StructureSlice([circle4, circle3])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def surrounds_two_holes_example(self):
        box10x5 = shapely.Polygon(box_points(10,5))
        circle4_left = shapely.Polygon(circle_points(2, offset_x=-3))
        circle3_right = shapely.Polygon(circle_points(1.5, offset_x=3))
        circle2_left = shapely.Polygon(circle_points(1, offset_x=-3,
                                                     offset_y=0.5))
        circle2_right = shapely.Polygon(circle_points(1, offset_x=3))
        a = StructureSlice([box10x5, circle4_left, circle3_right,
                            circle2_right])
        b = StructureSlice([circle2_left])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

class TestShelters:
    def test_shelters_big_hole(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        circle4_offset = shapely.Polygon(circle_points(2, offset_x=3.5))
        shell = shapely.difference(circle6, circle5)
        cove = shapely.difference(shell, circle4_offset)
        circle2 = shapely.Polygon(circle_points(1, offset_x=1))
        a = StructureSlice([cove])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SHELTERS

    def test_shelters_circle(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle3 = shapely.Polygon(circle_points(1.5, offset_x=1.6))
        crescent = shapely.difference(circle6, circle3)
        circle2 = shapely.Polygon(circle_points(1, offset_x=1.5))
        a = StructureSlice([crescent])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.SHELTERS

class TestDisjoint:
    def test_disjoint(self):
        circle4_left = shapely.Polygon(circle_points(4, offset_x=-4.5))
        circle4_right = shapely.Polygon(circle_points(4, offset_x=4.5))
        a = StructureSlice([circle4_left])
        b = StructureSlice([circle4_right])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.DISJOINT

class TestBorders:
    def test_borders_simple(self):
        box4_left = shapely.Polygon(box_points(4, offset_x=-2))
        box4_right = shapely.Polygon(box_points(4, offset_x=2))
        a = StructureSlice([box4_left])
        b = StructureSlice([box4_right])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.BORDERS

    def test_borders_insert(self):
        box6 = shapely.Polygon(box_points(6))
        box5_up = shapely.Polygon(box_points(5, offset_y=3))
        box6_cropped = shapely.difference(box6, box5_up)
        a = StructureSlice([box6_cropped])
        b = StructureSlice([box5_up])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.BORDERS

class TestConfines:
    def test_confines_inner_circle(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box4_offset = shapely.Polygon(box_points(4, offset_x=2))
        cropped_circle = shapely.difference(circle4, box4_offset)
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([cropped_circle])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_ring(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        # b has internal borders with the ring portion of a, but has an external
        # border with the island part of a. The internal borders relation wins.
        a = StructureSlice([circle6, circle4, circle2])
        b = StructureSlice([circle4, circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_embedded_box(self):
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        a = StructureSlice([box6, box4])
        b = StructureSlice([box4])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONFINES

    def test_confines_corner_box(self):
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        box2_offset = shapely.Polygon(box_points(2, offset_x=-1, offset_y=-1))
        a = StructureSlice([box6, box4])
        b = StructureSlice([box2_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.CONFINES

class TestPartition:
    def test_partition_simple(self):
        box4 = shapely.Polygon(box_points(4))
        box4_cropped = shapely.Polygon(box_points(2, 4, offset_x=-1))
        a = StructureSlice([box4])
        b = StructureSlice([box4_cropped])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_side_box(self):
        box6 = poly_round(shapely.Polygon(box_points(6)))
        box4_offset = shapely.Polygon(box_points(4, offset_x=-1))
        a = StructureSlice([box6])
        b = StructureSlice([box4_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_island(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle4, circle2])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    def test_partition_partial_ring(self):
        # Rounding required because of floating point inaccuracies.
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box6_offset = shapely.Polygon(box_points(6, offset_x=2))
        ring = shapely.difference(circle6, circle4)
        cropped_ring = poly_round(shapely.difference(ring, box6_offset))
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([cropped_ring])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.PARTITION

    @pytest.mark.xfail
    def test_partition_embedded_circle(self):
        # This test exposes a known bug that is likely related to rounding errors.
        # Rounding required because of floating point inaccuracies.
        circle6 = poly_round(shapely.Polygon(circle_points(3)))
        circle4_offset = shapely.Polygon(circle_points(2, offset_x=2))
        cropped_circle = poly_round(shapely.intersection(circle6,
                                                         circle4_offset))
        a = StructureSlice([circle6])
        b = StructureSlice([cropped_circle])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.PARTITION

class TestOverlaps:
    def test_overlaps_box(self):
        box4 = shapely.Polygon(box_points(4))
        box4_offset = shapely.Polygon(box_points(4, offset_x=2))
        a = StructureSlice([box4])
        b = StructureSlice([box4_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_box_circle(self):
        circle6 = shapely.Polygon(circle_points(3))
        box6_offset = shapely.Polygon(box_points(6, offset_x=3))
        a = StructureSlice([circle6])
        b = StructureSlice([box6_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_circles(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle6_offset = shapely.Polygon(circle_points(3, offset_x=2))
        a = StructureSlice([circle6])
        b = StructureSlice([circle6_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_box(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        box6_offset = shapely.Polygon(box_points(6, offset_x=3))
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([box6_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_circle(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle6_offset = shapely.Polygon(circle_points(3, offset_x=2.5))
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([circle6_offset])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_surrounded(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1.5, offset_x=1))
        a = StructureSlice([circle6, circle4])
        b = StructureSlice([circle2])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_surrounded_box(self):
        box6 = shapely.Polygon(box_points(6))
        box4 = shapely.Polygon(box_points(4))
        box3 = shapely.Polygon(box_points(3, offset_x=1))
        a = StructureSlice([box6, box4])
        b = StructureSlice([box3])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_ring_surrounded(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle4 = shapely.Polygon(circle_points(2))
        a = StructureSlice([circle6, circle3])
        b = StructureSlice([circle4])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlaps_circle_island(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle4, circle2])
        b = StructureSlice([circle4])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

    def overlaps_concentric_rings(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle4 = shapely.Polygon(circle_points(2))
        circle3 = shapely.Polygon(circle_points(1.5))
        circle2 = shapely.Polygon(circle_points(1))
        a = StructureSlice([circle6, circle3, circle2])
        b = StructureSlice([circle4])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.OVERLAPS

class TestEquals:
    def test_equals_box(self):
        box6 = shapely.Polygon(box_points(6))
        a = StructureSlice([box6])
        b = StructureSlice([box6])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.EQUALS

    def test_equals_circle(self):
        circle6 = shapely.Polygon(circle_points(3))
        circle5 = shapely.Polygon(circle_points(2.5))
        cropped_circle = shapely.intersection(circle6, circle5)
        a = StructureSlice([circle5])
        b = StructureSlice([cropped_circle])
        relation_type = DE27IM(a, b).identify_relation()
        assert relation_type == RelationshipType.EQUALS
