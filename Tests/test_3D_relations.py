import pandas as pd
import pytest

from structure_set import generate_region_graph, make_slice_table
from relations import RelationshipType, find_relations
from debug_tools import make_vertical_cylinder, make_horizontal_cylinder
from debug_tools import make_sphere, make_box


class TestContains:
    def test_contains_embedded_spheres(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30, offset_z=-15,
                                    spacing=slice_spacing)
        # embedded boxes
        sphere6 = make_sphere(roi_num=1, radius=3, spacing=slice_spacing)
        sphere3 = make_sphere(roi_num=2, radius=1.5, spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([sphere6, sphere3, body])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_contains_simple_cylinders(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=1.0,
                                    offset_z=0,
                                    spacing=slice_spacing)
        # Centred cylinder
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=5, length=0.8,
                                                offset_z=0,
                                                spacing=slice_spacing)
        # cylinder within primary
        contained_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=0.6,
                                                    offset_x=0, offset_z=0,
                                                    spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, primary_cylinder, contained_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_embedded_boxes(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        cube6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        cube3 = make_box(roi_num=2, width=3, offset_z=0, spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([cube6, cube3, body])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_parallel_cylinders_example(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=10,
                                      offset_z=0,
                                      spacing=slice_spacing)
        left_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                               offset_x=-2.5, offset_z=0,
                                               spacing=slice_spacing)
        right_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                                offset_x=2.5, offset_z=0,
                                                spacing=slice_spacing)
        right_middle_cylinder = make_vertical_cylinder(roi_num=2, radius=1,
                                                       length=6,
                                                       offset_x=2.5, offset_z=0,
                                                       spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, left_cylinder, right_cylinder,
                                right_middle_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.CONTAINS

    def test_nested_spheres(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=16, length=20, offset_z=0,
                                    spacing=slice_spacing)
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        hole10 = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=1, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)

        # combine the contours
        slice_data = pd.concat([body, sphere12, hole10, sphere8, sphere6])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.CONTAINS

class TestSurrounds:
    def test_surrounded_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(roi_num=1, radius=5, length=8,
                                            spacing=slice_spacing)
        surrounded_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=6,
                                                    spacing=slice_spacing)

        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                surrounded_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def test_surrounded_horizontal_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_horizontal_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_horizontal_cylinder(roi_num=1, radius=5, length=8,
                                            spacing=slice_spacing)
        surrounded_cylinder = make_horizontal_cylinder(roi_num=2, radius=3, length=6,
                                                    spacing=slice_spacing)

        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                surrounded_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def test_sphere_in_shell(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=16, length=20, offset_z=0,
                                    spacing=slice_spacing)
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        hole10 = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)

        # combine the contours
        slice_data = pd.concat([body, sphere12, hole10, sphere6])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

    def test_sphere_in_cylinders_in_box(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_box(roi_num=0, width=20, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        cube6 = make_box(roi_num=1, width=10, length=10, height=10, spacing=slice_spacing)
        left_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                            offset_x=-2.5, offset_z=0,
                                            spacing=slice_spacing)
        right_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                                offset_x=2.5, offset_z=0,
                                                spacing=slice_spacing)
        right_sphere = make_sphere(roi_num=2, radius=1,
                                offset_x=2.5, offset_z=0,
                                spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, cube6, left_cylinder, right_cylinder,
                                right_sphere])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SURROUNDS

@pytest.mark.xfail
class TestShelters:
    def test_shelters_horizontal_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_box(roi_num=0, width=20, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        cube6 = make_box(roi_num=1, width=10, length=10, height=10, spacing=slice_spacing)
        left_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                            offset_x=-2.5, offset_z=0,
                                            spacing=slice_spacing)
        right_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                                offset_x=2.5, offset_z=0,
                                                spacing=slice_spacing)
        right_sphere = make_sphere(roi_num=2, radius=1,
                                offset_x=2.5, offset_z=0,
                                spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, cube6, left_cylinder, right_cylinder,
                                right_sphere])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SHELTERS

    def test_shelters_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(roi_num=1, radius=5, length=10,
                                            spacing=slice_spacing)
        surrounded_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=6,
                                                    spacing=slice_spacing)

        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                surrounded_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SHELTERS

    def test_sphere_in_cylinders_in_box(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_box(roi_num=0, width=12, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        cube6 = make_box(roi_num=1, width=10, length=10, height=8, spacing=slice_spacing)
        left_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                            offset_x=-2.5, offset_z=0,
                                            spacing=slice_spacing)
        right_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=8,
                                                offset_x=2.5, offset_z=0,
                                                spacing=slice_spacing)
        right_sphere = make_sphere(roi_num=2, radius=1,
                                offset_x=2.5, offset_z=0,
                                spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, cube6, left_cylinder, right_cylinder,
                                right_sphere])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.SHELTERS

class TestDisjoint:
    def test_disjoint_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=20, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        left_cube = make_box(roi_num=1, width=2, offset_x=-3,
                            spacing=slice_spacing)
        right_cube = make_box(roi_num=2, width=2, offset_x=3,
                            spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([left_cube, right_cube, body])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.DISJOINT

    @pytest.mark.xfail
    def test_extended_inner_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(roi_num=1, radius=5, length=10,
                                            spacing=slice_spacing)
        inner_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=12,
                                                    spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                inner_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.DISJOINT

    @pytest.mark.xfail
    def test_parallel_disjoint_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=25, length=16,
                                      offset_z=0, spacing=slice_spacing)
        outer_cylinder = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(roi_num=1, radius=5, length=8,
                                            spacing=slice_spacing)
        inner_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=6,
                                                    spacing=slice_spacing)
        disjoint_cylinder = make_vertical_cylinder(roi_num=2, radius=3,
                                                   length=6, offset_x=10,
                                                   spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                inner_cylinder, disjoint_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.DISJOINT

    @pytest.mark.xfail
    def test_axial_disjoint_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=30,
                                      offset_z=0, spacing=slice_spacing)
        outer_cylinder = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(roi_num=1, radius=5, length=8,
                                               spacing=slice_spacing)
        inner_cylinder = make_vertical_cylinder(roi_num=2, radius=3, length=6,
                                                spacing=slice_spacing)
        disjoint_cylinder = make_vertical_cylinder(roi_num=2, radius=3,
                                                   length=6, offset_z=12,
                                                   spacing=slice_spacing)
        # combine the contours
        slice_data = pd.concat([body, outer_cylinder, cylinder_hole,
                                inner_cylinder, disjoint_cylinder])
        # convert contour slice data into a table of slices and structures
        slice_table = make_slice_table(slice_data, ignore_errors=True)
        regions = generate_region_graph(slice_table)
        selected_roi = [1, 2]
        relation = find_relations(slice_table, regions, selected_roi)
        relation_type = relation.identify_relation()
        assert relation_type == RelationshipType.DISJOINT
