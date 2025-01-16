import shapely

from structure_slice import StructureSlice
from relations import DE27IM, RelationshipType
from debug_tools import circle_points

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
