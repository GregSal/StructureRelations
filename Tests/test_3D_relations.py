#%%
import pandas as pd
import pytest

import shapely

from structure_set import StructureSet
from relations import DE27IM, RelationshipType
from debug_tools import make_vertical_cylinder, make_horizontal_cylinder
from debug_tools import make_sphere, make_box, box_points
from utilities import poly_round

# %% Utility functions
def get_relation_type(slice_data, roi1=1, roi2=2)->DE27IM:
    # build the structure set
    structures = StructureSet(slice_data)
    structure_a = structures.structures[roi1]
    structure_b = structures.structures[roi2]
    relation = structure_a.relate(structure_b)
    relation_type = relation.identify_relation()
    return relation_type

# %% Tests
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
        slice_data = sphere6 + sphere3 + body
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + primary_cylinder + contained_cylinder
        relation_type = get_relation_type(slice_data)
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
        slice_data = cube6 + cube3 + body
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + left_cylinder + right_cylinder + right_middle_cylinder
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + sphere12 + hole10 + sphere8 + sphere6
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + outer_cylinder + cylinder_hole + surrounded_cylinder
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + outer_cylinder + cylinder_hole + surrounded_cylinder
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + sphere12 + hole10 + sphere6
        relation_type = get_relation_type(slice_data)
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
        slice_data = body + cube6 + left_cylinder + right_cylinder + right_sphere
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.SURROUNDS

class TestShelters:
    def test_shelters_horizontal_cylinder_single_side(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_horizontal_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_horizontal_cylinder(roi_num=1, radius=4, length=8,
                                                offset_x=1, offset_z=0,
                                                spacing=slice_spacing)
        surrounded_cylinder = make_horizontal_cylinder(roi_num=2, radius=3,
                                                    length=6, offset_x=1,
                                                    spacing=slice_spacing)

        # combine the contours
        slice_data = body + outer_cylinder + cylinder_hole + surrounded_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.SHELTERS

    #@pytest.mark.xfail
    def test_shelters_sphere_in_cylinders_in_box(self):
        slice_spacing = 0.5
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
        slice_data = body + cube6 + left_cylinder + right_cylinder + right_sphere
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.SHELTERS

    #@pytest.mark.xfail
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
        slice_data = body + outer_cylinder + cylinder_hole + surrounded_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.SHELTERS

    #@pytest.mark.xfail
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
        slice_data = body + cube6 + left_cylinder + right_cylinder + right_sphere
        relation_type = get_relation_type(slice_data)
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
        slice_data = left_cube + right_cube + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

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
        slice_data = body + outer_cylinder + cylinder_hole + inner_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

    def test_disjoint_horizontal_cylinder(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=16, offset_z=0,
                                    spacing=slice_spacing)
        outer_cylinder = make_horizontal_cylinder(roi_num=1, radius=6, length=10,
                                                spacing=slice_spacing)
        cylinder_hole = make_horizontal_cylinder(roi_num=1, radius=5, length=10,
                                                spacing=slice_spacing)
        surrounded_cylinder = make_horizontal_cylinder(roi_num=2, radius=3,
                                                    length=12,
                                                    spacing=slice_spacing)
        # combine the contours
        slice_data = outer_cylinder + cylinder_hole + surrounded_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

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
        slice_data = body + outer_cylinder + cylinder_hole + inner_cylinder + disjoint_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

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
        slice_data = body + outer_cylinder + cylinder_hole + inner_cylinder + disjoint_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

    def test_disjoint_concentric_cylinders(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=10,
                                    spacing=slice_spacing)
        # Centred cylinder
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=3, length=8,
                                                offset_z=0,
                                                spacing=slice_spacing)
        # cylinder 2 slices above primary cylinder
        upper_cylinder1 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=7,
                                                spacing=slice_spacing)
        # cylinder 2 slices below primary cylinder
        lower_cylinder2 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=-7,
                                                spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + upper_cylinder1 + lower_cylinder2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.DISJOINT

class TestBorders:
    def test_bordering_concentric_cylinders(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=10,
                                    spacing=slice_spacing)
        # Centred cylinder with two embedded cylinders
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=3, length=8,
                                                offset_z=0,
                                                spacing=slice_spacing)
        # cylinder bordering primary cylinder
        bordering_cylinder1 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=6,
                                                spacing=slice_spacing)
        # cylinder bordering primary cylinder
        bordering_cylinder2 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=-6,
                                                spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + bordering_cylinder1 + bordering_cylinder2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.BORDERS

    def test_lateral_borders_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=20, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        left_cube = make_box(roi_num=1, width=2, offset_x=-1,
                            spacing=slice_spacing)
        right_cube = make_box(roi_num=2, width=2, offset_x=1,
                            spacing=slice_spacing)
        # combine the contours
        slice_data = left_cube + right_cube + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.BORDERS

    def test_concentric_cylinders_sup_offset(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                      offset_z=-0.5, spacing=slice_spacing)
        # Two concentric cylinders different z offsets
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=0.2,
                                                  length=0.4,  offset_z=-0.5,
                                                  spacing=slice_spacing)
        sup_cylinder = make_vertical_cylinder(roi_num=2, radius=0.2, length=0.4,
                                              offset_z=0, spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + sup_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.BORDERS

    def test_lateral_borders_two_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=20, offset_z=0,
                                    spacing=slice_spacing)
        # embedded boxes
        left_cube = make_box(roi_num=1, width=2, offset_x=-1,
                            spacing=slice_spacing)
        right_cube = make_box(roi_num=2, width=2, offset_x=1,
                            spacing=slice_spacing)
        disjoint_cube = make_box(roi_num=2, width=2, offset_x=-2,
                            spacing=slice_spacing)
        # combine the contours
        slice_data = left_cube + right_cube + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.BORDERS

    def test_inserted_cylinder(self):
        '''Test cylinder inserted in open hole of first cylinder.'''
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1, offset_z=-0.6,
                                    spacing=slice_spacing)
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=4, length=0.8,
                                                offset_z=-0.3,
                                                spacing=slice_spacing)
        center_hole = make_vertical_cylinder(roi_num=1, radius=2, length=0.6,
                                            offset_z=-0.2, spacing=slice_spacing)
        # Two concentric cylinders different z offsets
        middle_cylinder = make_vertical_cylinder(roi_num=2, radius=1, length=0.6,
                                                offset_z=-0.2,
                                                spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + center_hole + middle_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.BORDERS

class TestConfines:
    def test_confined_bordering_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                      spacing=slice_spacing)
        # embedded boxes
        box6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        hole4 = make_box(roi_num=1, width=4,  spacing=slice_spacing)
        Box4 = make_box(roi_num=2, width=4,  spacing=slice_spacing)

        # combine the contours
        slice_data = body + box6 + hole4 + Box4
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES

    def test_confines_dual_cylinders(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=1.2,
                                      spacing=slice_spacing)
        # Centred cylinder with two cylindrical holes
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=5,
                                                  length=0.8,
                                                  spacing=slice_spacing)
        left_hole = make_vertical_cylinder(roi_num=1, radius=2, length=0.6,
                                           offset_x=-2.5, spacing=slice_spacing)
        right_hole = make_vertical_cylinder(roi_num=1, radius=2, length=0.6,
                                            offset_x=2.5, spacing=slice_spacing)
        # cylinder with interior borders
        confines_cylinder = make_vertical_cylinder(roi_num=2, radius=1,
                                                   length=0.6, offset_x=2.5,
                                                   spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + left_hole + right_hole + confines_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES

    def test_embedded_cylinder(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=4,
                                                  length=0.8,
                                                  spacing=slice_spacing)
        center_hole = make_vertical_cylinder(roi_num=1, radius=2, length=0.6,
                                             spacing=slice_spacing)
        # Two concentric cylinders different z offsets
        middle_cylinder = make_vertical_cylinder(roi_num=2, radius=1,
                                                 length=0.6,
                                                 spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + center_hole + middle_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES

    def test_embedded_spheres(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1, offset_z=-0.6,
                                    spacing=slice_spacing)

        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        hole10 = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=1, radius=4, spacing=slice_spacing)

        sphere10 = make_sphere(roi_num=2, radius=5, spacing=slice_spacing)
        hole8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)

        # combine the contours
        slice_data = body + sphere12 + hole10 + sphere8 + sphere10 + hole8
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES

    def test_confined_box_z_border(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1, offset_z=-0.2,
                                    spacing=slice_spacing)
        # embedded boxes
        box6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        hole4 = make_box(roi_num=1, width=4,  spacing=slice_spacing)
        Box2 = make_box(roi_num=2, width=2, offset_z=1,  spacing=slice_spacing)

        # combine the contours
        slice_data = body + box6 + hole4 + Box2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES

    def test_confined_box_y_border(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1, offset_z=-0.2,
                                    spacing=slice_spacing)
        # embedded boxes
        box6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        hole4 = make_box(roi_num=1, width=4,  spacing=slice_spacing)
        Box2 = make_box(roi_num=2, width=2, offset_y=1,  spacing=slice_spacing)

        # combine the contours
        slice_data = body + box6 + hole4 + Box2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.CONFINES


class TestPartition:
    def test_partition_embedded_box_on_y_surface(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30, spacing=slice_spacing)
        # embedded boxes    # 6 cm x 6 cm box
        box6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        box6_3 = make_box(roi_num=2, width=6, length=3, height=6, offset_y=1.5,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = box6 + box6_3 + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_partition_embedded_box_on_z_surface(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30,
                                    spacing=slice_spacing)
        # embedded boxes    # 6 cm x 6 cm box
        box6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        box6_3 = make_box(roi_num=2, width=6, length=6, height=3, offset_z=1.5,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = box6 + box6_3 + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION


    def test_horizontal_cylinders(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_box(roi_num=0, width=6, length=6, height=8, offset_z=-4,
                        spacing=slice_spacing)
        cylinder2h = make_horizontal_cylinder(radius=2, length=5, roi_num=1,
                                            spacing=slice_spacing)
        cylinder1h = make_horizontal_cylinder(radius=1, length=5, roi_num=2,
                                            spacing=slice_spacing)
        # combine the contours
        slice_data = body + cylinder1h + cylinder2h
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_vertical_concentric_cylinders(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30, offset_z=-15,
                                    spacing=slice_spacing)
        cylinder6 = make_vertical_cylinder(roi_num=1, radius=6, length=10,
                                        spacing=slice_spacing)
        cylinder4 = make_vertical_cylinder(roi_num=2, radius=4, length=10,
                                        spacing=slice_spacing)
        # combine the contours
        slice_data = body + cylinder6 + cylinder4
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_concentric_cylinders_same_start(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # Concentric cylinders starting on the same slice
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=7,
                                                offset_z=-3.5,
                                                spacing=slice_spacing)
        sup_partition = make_vertical_cylinder(roi_num=2, radius=1, length=3.0,
                                            offset_z=-1.5,
                                            spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + sup_partition
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_concentric_cylinders_same_end(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=10,
                                    spacing=slice_spacing)
        # Concentric cylinders ending on the same slice
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=7.0,
                                                offset_z=3.5,
                                                spacing=slice_spacing)
        inf_partition = make_vertical_cylinder(roi_num=2, radius=1, length=4,
                                            offset_z=2, spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + inf_partition
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_concentric_cylinders_same_start_end(self):
        slice_spacing = 0.05
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # Concentric cylinders starting and ending on the same slice
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=2, length=0.7,
                                                offset_z=0.0,
                                                spacing=slice_spacing)
        mid_partition = make_vertical_cylinder(roi_num=2, radius=1, length=0.7,
                                            offset_z=-0.0, spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + mid_partition
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION

    def test_partition_sphere_island(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1, offset_z=-0.6,
                                    spacing=slice_spacing)

        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        hole8 = make_sphere(roi_num=1, radius=4, spacing=slice_spacing)
        sphere4 = make_sphere(roi_num=1, radius=2, spacing=slice_spacing)

        sphere4_2 = make_sphere(roi_num=2, radius=2, spacing=slice_spacing)

        # combine the contours
        slice_data = body + sphere12 + hole8 + sphere4 + sphere4_2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.PARTITION


class TestOverlaps:
    def test_overlapping_spheres_example(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                    spacing=slice_spacing)

        right_sphere6 = make_sphere(roi_num=1, radius=6, offset_x=-2,
                                    spacing=slice_spacing)
        left_sphere6 = make_sphere(roi_num=2, radius=6, offset_x=2,
                                    spacing=slice_spacing)

        # combine the contours
        slice_data = body + right_sphere6 + left_sphere6
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlapping_boxes_y(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # overlapping boxes    # 6 cm x 6 cm box
        box6 = make_box(roi_num=1, width=0.6, spacing=slice_spacing)
        box6_y = make_box(roi_num=2, width=0.6, offset_y=0.2,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = box6 + box6_y + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlapping_boxes_z(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # overlapping boxes    # 6 cm x 6 cm box
        box6 = make_box(roi_num=1, width=0.6, spacing=slice_spacing)
        box6_y = make_box(roi_num=2, width=0.6, offset_z=0.3,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = box6 + box6_y + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_stacked_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # overlapping boxes    # 6 cm x 6 cm box
        box6 = make_box(roi_num=1, width=0.6, spacing=slice_spacing)
        box6_y = make_box(roi_num=2, width=0.6, offset_z=0.6,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = box6 + box6_y + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlapping_concentric_cylinders_example(self):
        slice_spacing = 1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=10,
                                    spacing=slice_spacing)
        # Centred cylinder with two embedded cylinders
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=3, length=8,
                                                offset_z=0,
                                                spacing=slice_spacing)
        # cylinder overlapping primary cylinder
        overlapping_cylinder1 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=5,
                                                spacing=slice_spacing)
        # cylinder overlapping primary cylinder
        overlapping_cylinder2 = make_vertical_cylinder(roi_num=2, radius=1, length=2,
                                                offset_z=-5,
                                                spacing=slice_spacing)
        # combine the contours
        slice_data = body + primary_cylinder + overlapping_cylinder1 + overlapping_cylinder2
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlapping_cubes_inf_rt(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=30, offset_z=-15,
                                    spacing=slice_spacing)
        #
        cube6 = make_box(roi_num=1, width=6, spacing=slice_spacing)
        cube6_inf_rt = make_box(roi_num=2, width=6, offset_z=3, offset_x=3,
                                offset_y=3, spacing=slice_spacing)
        # combine the contours
        slice_data = body + cube6 + cube6_inf_rt
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS

    def test_overlapping_extended_cylinder(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=12, length=1.1,
                                      spacing=slice_spacing)
        # Centred cylinder with two embedded cylinders
        primary_cylinder = make_vertical_cylinder(roi_num=1, radius=5,
                                                  length=0.7,
                                                  spacing=slice_spacing)
        # cylinder with interior borders
        overlapping_cylinder = make_vertical_cylinder(roi_num=2, radius=3,
                                                      length=0.9,
                                                      spacing=slice_spacing)

        # combine the contours
        slice_data = body + primary_cylinder + overlapping_cylinder
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.OVERLAPS


class TestEquals:
    def test_equal_spheres(self):
        slice_spacing = 0.5
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                    spacing=slice_spacing)

        a_sphere6 = make_sphere(roi_num=1, radius=6,
                                    spacing=slice_spacing)
        b_sphere6 = make_sphere(roi_num=2, radius=6,
                                    spacing=slice_spacing)

        # combine the contours
        slice_data = body + a_sphere6 + b_sphere6
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.EQUALS

    def test_equal_boxes(self):
        slice_spacing = 0.1
        # Body structure defines slices in use
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        # overlapping boxes    # 6 cm x 6 cm box
        a_box6 = make_box(roi_num=1, width=0.6, spacing=slice_spacing)
        b_box6 = make_box(roi_num=2, width=0.6, spacing=slice_spacing)
        # combine the contours
        slice_data = a_box6 + b_box6 + body
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.EQUALS

    def test_equal_boxes_by_crop(self):
        def apply_crop(p):
            # polygon made from offset boxed resulting in a 4x4 square hole in the
            # middle.
            left_xy_points = box_points(width=0.8, offset_x=0.6, offset_y=0)
            left_crop = shapely.Polygon(left_xy_points)
            right_xy_points = box_points(width=0.8, offset_x=-0.6, offset_y=0)
            right_crop = shapely.Polygon(right_xy_points)
            up_xy_points = box_points(width=0.8, offset_x=0, offset_y=0.6)
            up_crop = shapely.Polygon(up_xy_points)
            down_xy_points = box_points(width=0.8, offset_x=0, offset_y=-0.6)
            down_crop = shapely.Polygon(down_xy_points)
            crop_poly = shapely.union_all([left_crop, right_crop,
                                        up_crop, down_crop])
            cropped = p - crop_poly
            return poly_round(cropped)

        def get_cropped_box(box8):# -> List[Dict[str, Any]]:
            cropped_box = []
            for contour_slice in box8:
                contour = contour_slice['Points']
                poly = shapely.Polygon(contour)
                cropped_poly = apply_crop(poly)
                cropped_coords = list(cropped_poly.boundary.coords)
                cropped_contour = {
                    'ROI': contour_slice['ROI'],
                    'Slice': contour_slice['Slice'],
                    'Points': cropped_coords,
                }
                cropped_box.append(cropped_contour)
            return cropped_box
        slice_spacing = 0.1
        body = make_vertical_cylinder(roi_num=0, radius=10, length=1,
                                    spacing=slice_spacing)
        box8 = make_box(roi_num=1, width=0.8, spacing=slice_spacing)
        box4 = make_box(roi_num=2, width=0.4, spacing=slice_spacing)
        # apply the crop to box8 to create a copy of box4
        cropped_box = get_cropped_box(box8)
        slice_data = body + box4 + cropped_box
        relation_type = get_relation_type(slice_data)
        assert relation_type == RelationshipType.EQUALS
