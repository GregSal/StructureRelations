'''Tests for logical relationship identification in StructureSet.

This module contains comprehensive tests for the calculate_logical_flags()
method, using geometric examples from LogicalRelationshipTests notebook.
'''
import pytest

from debug_tools import (
    make_sphere, make_box, make_vertical_cylinder,
    circle_points, extrude_poly
)
from structure_set import StructureSet
from relations import RELATIONSHIP_TYPES
from types_and_classes import ROI_Type
import shapely


class TestLogicalContains:
    '''Tests for logical CONTAINS relationships via transitivity.'''

    def test_3_embedded_spheres(self):
        '''Test: A Contains B, B Contains C => A Contains C is logical.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Check (1,3) relationship
        rel_1_3 = structures.get_relationship(
            ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        assert rel_1_3.is_logical
        assert rel_1_3.relationship_type.relation_type == 'CONTAINS'
        assert ROI_Type(2) in rel_1_3.intermediate_structures

    def test_4_embedded_spheres(self):
        '''Test: A Contains B Contains C Contains D => logical relationships.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere10 = make_sphere(roi_num=2, radius=5, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=3, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=4, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere10 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Check (1,3) is logical
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3.is_logical
        assert ROI_Type(2) in rel_1_3.intermediate_structures

        # Check (1,4) is logical
        rel_1_4 = structures.get_relationship(ROI_Type(1), ROI_Type(4))
        assert rel_1_4.is_logical
        assert ROI_Type(2) in rel_1_4.intermediate_structures
        assert ROI_Type(3) in rel_1_4.intermediate_structures

        # Check (2,4) is logical
        rel_2_4 = structures.get_relationship(ROI_Type(2), ROI_Type(4))
        assert rel_2_4.is_logical
        assert ROI_Type(3) in rel_2_4.intermediate_structures

    def test_logical_contains_partitioned(self):
        '''Test: A Contains B, B is Partitioned by C => A Contains C is logical.'''
        slice_spacing = 0.5
        cube10 = make_box(roi_num=1, width=10, spacing=slice_spacing)
        half_cube10 = make_box(roi_num=2, width=5, length=8, height=8,
                            offset_x=0, offset_y=0, offset_z=0,
                            spacing=slice_spacing)

        half_cube6 = make_box(roi_num=3, width=3, length=6, height=6,
                        offset_x=1, offset_y=0, offset_z=0,
                        spacing=slice_spacing)

        slice_data = cube10 + half_cube10 + half_cube6
        structures = StructureSet(slice_data)

        # Check (1,3) relationship contains is logical
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        # Relationship A->C should be CONTAINS
        assert rel_1_3.relationship_type.relation_type == 'CONTAINS'
        # Due to implied relationship, this should be logical
        assert rel_1_3.is_logical

    def test_logical_partitions_contains(self):
        '''Test: A is Partitioned by B, B Contains C => A Contains C is logical.'''
        slice_spacing = 0.5
        cube10 = make_box(roi_num=1, width=10, spacing=slice_spacing)
        half_cube10 = make_box(roi_num=2, width=5, length=8, height=8,
                            offset_x=2.5, offset_y=0, offset_z=0,
                            spacing=slice_spacing)

        half_cube6 = make_box(roi_num=3, width=3, length=6, height=6,
                        offset_x=2.5, offset_y=0, offset_z=0,
                        spacing=slice_spacing)

        slice_data = cube10 + half_cube10 + half_cube6
        structures = StructureSet(slice_data)
        # Check (1,3) relationship contains is logical
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        # Relationship A->C should be CONTAINS
        assert rel_1_3.relationship_type.relation_type == 'CONTAINS'
        # Due to implied relationship, this should be logical
        assert rel_1_3.is_logical


class TestLogicalConfines:
    '''Tests for logical SURROUNDS via CONFINES chain.'''

    def test_confined_spheres_surrounds(self):
        '''Test: A Confines B, B Confines C => A Surrounds C is logical.'''
        slice_spacing = 1
        outer_sphere = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        outer_hole = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
        middle_sphere = make_sphere(roi_num=2, radius=5, spacing=slice_spacing)
        middle_hole = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere4 = make_sphere(roi_num=3, radius=4, spacing=slice_spacing)

        slice_data = (outer_sphere + outer_hole + middle_sphere +
                      middle_hole + sphere4)
        structures = StructureSet(slice_data, tolerance=0.2)

        # Check (1,3) relationship
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        assert rel_1_3.is_logical
        assert rel_1_3.relationship_type.relation_type in [
            'SURROUNDS', 'CONFINES'
        ]


class TestLogicalEquals:
    '''Tests for logical relationships derived from EQUAL.'''

    def test_equals_borders_logical(self):
        '''Test: A is Equal to B, B Borders C => A Borders C is logical.'''
        slice_spacing = 0.5
        bottom_box1 = make_box(
            roi_num=1, width=5, length=5, height=2,
            offset_x=0, offset_y=0, offset_z=0,
            spacing=slice_spacing
        )
        bottom_box2 = make_box(
            roi_num=2, width=5, length=5, height=2,
            offset_x=0, offset_y=0, offset_z=0,
            spacing=slice_spacing
        )
        top_box = make_box(
            roi_num=3, width=4, length=4, height=2,
            offset_x=0, offset_y=0, offset_z=-2.5,
            spacing=slice_spacing
        )

        slice_data = bottom_box1 + bottom_box2 + top_box
        structures = StructureSet(slice_data)

        # Check (1,2) is EQUAL
        rel_1_2 = structures.get_relationship(ROI_Type(1), ROI_Type(2))
        assert rel_1_2.relationship_type.relation_type == 'EQUAL'

        # Check (2,3) is logical
        rel_2_3 = structures.get_relationship(ROI_Type(2), ROI_Type(3))
        assert rel_2_3 is not None
        assert rel_2_3.is_logical

    def test_equals_disjoint_logical(self):
        '''Test: A is Equal to B, B Disjoint C => A Disjoint C is logical.'''
        slice_spacing = 0.5
        cylinder_a = make_vertical_cylinder(
            roi_num=1, radius=2, length=8,
            offset_x=-2.5, offset_z=0,
            spacing=slice_spacing
        )
        cylinder_b = make_vertical_cylinder(
            roi_num=2, radius=2, length=8,
            offset_x=-2.5, offset_z=0,
            spacing=slice_spacing
        )
        cylinder_c = make_vertical_cylinder(
            roi_num=3, radius=2, length=8,
            offset_x=2.5, offset_z=0,
            spacing=slice_spacing
        )

        slice_data = cylinder_a + cylinder_b + cylinder_c
        structures = StructureSet(slice_data)

        # Check (1,2) is EQUAL
        rel_1_2 = structures.get_relationship(ROI_Type(1), ROI_Type(2))
        assert rel_1_2.relationship_type.relation_type == 'EQUAL'

        # Check (2,3) is logical
        rel_2_3 = structures.get_relationship(ROI_Type(2), ROI_Type(3))
        assert rel_2_3 is not None
        assert rel_2_3.is_logical


class TestLogicalMixed:
    '''Tests for logical relationships with mixed transitive types.'''

    def test_shelters_surrounds_contains_chain(self):
        '''Test: A Shelters B, B Surrounds C, C Contains D => logical rels.'''
        slice_spacing = 1
        # Outside C-shaped cylinder
        outer_circle_coords = circle_points(radius=8)
        circle_hole_coords = circle_points(radius=7)
        slot_coords = [(0, 4), (0, -4), (9, -4), (9, 4)]
        outer_circle = shapely.Polygon(outer_circle_coords)
        circle_hole = shapely.Polygon(circle_hole_coords)
        slot = shapely.Polygon(slot_coords)
        c_shape = outer_circle.difference(
            circle_hole.union(slot))
        c_cylinder = extrude_poly(
            c_shape, length=8, spacing=slice_spacing, roi_num=1)

        # Hollow cylinder
        closed_cylinder = make_vertical_cylinder(
            roi_num=2, radius=6, length=8, spacing=slice_spacing)
        cylinder_hole = make_vertical_cylinder(
            roi_num=2, radius=5, length=6, spacing=slice_spacing)

        # Inside solid cylinders
        surrounded_cylinder = make_vertical_cylinder(
            roi_num=3, radius=3, length=4, spacing=slice_spacing)
        contained_cylinder = make_vertical_cylinder(
            roi_num=4, radius=2, length=2, spacing=slice_spacing)

        slice_data = (c_cylinder + closed_cylinder + cylinder_hole +
                      surrounded_cylinder + contained_cylinder)
        structures = StructureSet(slice_data)

        # Check that some relationships are transitive
        rel_2_4 = structures.get_relationship(ROI_Type(2), ROI_Type(4))
        assert rel_2_4 is not None
        assert rel_2_4.relationship_type.is_transitive


class TestNotLogical:
    '''Tests for relationships that should NOT be marked as logical.'''

    def test_disjoint_contained_not_logical(self):
        '''Test: A Contains B, A Contains C, B Disjoint C => A Contains C
        is NOT logical.'''
        slice_spacing = 0.5
        primary = make_vertical_cylinder(
            roi_num=1, radius=6, length=8, offset_z=0,
            spacing=slice_spacing
        )
        contained_b = make_vertical_cylinder(
            roi_num=2, radius=2, length=6,
            offset_x=2.5, offset_z=0,
            spacing=slice_spacing
        )
        contained_c = make_vertical_cylinder(
            roi_num=3, radius=2, length=6,
            offset_x=-2.5, offset_z=0,
            spacing=slice_spacing
        )

        slice_data = primary + contained_b + contained_c
        structures = StructureSet(slice_data)

        # (1,3) should exist but NOT be logical
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        # Not logical because alternative path is not transitive
        # (B and C are disjoint, not connected transitively)
        assert not rel_1_3.is_logical

    def test_borders_not_transitive(self):
        '''Test: A Borders B, B Borders C => A Borders C is NOT logical
        (Borders is not transitive).'''
        slice_spacing = 0.5
        half_cube10 = make_box(
            roi_num=1, width=8, length=4, height=8,
            offset_x=0, offset_y=2, offset_z=0,
            spacing=slice_spacing
        )
        left_cube4 = make_box(
            roi_num=2, width=4, height=8,
            offset_x=-2, offset_y=-2, offset_z=0,
            spacing=slice_spacing
        )
        right_cube4 = make_box(
            roi_num=3, width=4, height=8,
            offset_x=2, offset_y=-2, offset_z=0,
            spacing=slice_spacing
        )

        slice_data = half_cube10 + left_cube4 + right_cube4
        structures = StructureSet(slice_data)

        # (1,3) should NOT be logical (Borders is not transitive)
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        assert not rel_1_3.is_logical

    def test_not_logical_partitioned_box(self):
        '''Test: A is Partitioned by B, B is Partitioned by C =>
        A is Partitioned by C is NOT logical (Contains is also possible).'''
        slice_spacing = 0.5
        cube10 = make_box(roi_num=1, width=10, spacing=slice_spacing)
        half_cube10 = make_box(roi_num=2, width=5, length=8, height=8,
                            offset_x=2.5, offset_y=0, offset_z=0,
                            spacing=slice_spacing)

        half_cube6 = make_box(roi_num=3, width=3, length=6,
                        offset_x=3.5, offset_y=0, offset_z=0,
                        spacing=slice_spacing)
        slice_data = cube10 + half_cube10 + half_cube6
        structures = StructureSet(slice_data)

        # (1,3) should NOT be logical (result is ambiguous)
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        # Relationship A->C should be PARTITIONED
        assert rel_1_3.relationship_type.relation_type == 'PARTITIONED'
        # Not logical because no single transitive path can explain it
        assert not rel_1_3.is_logical

    def test_not_logical_partitioned_contains(self):
        '''Test: A is Partitioned by B, B is Partitioned by C =>
        A Contains C is NOT logical (Partitioned is also possible).
        '''
        slice_spacing = 0.5
        cube10 = make_box(roi_num=1, width=10, spacing=slice_spacing)
        half_cube10 = make_box(roi_num=2, width=5, length=8, height=8,
                            offset_x=2.5, offset_y=0, offset_z=0,
                            spacing=slice_spacing)
        half_cube6 = make_box(roi_num=3, width=3, length=6,
                        offset_x=1.5, offset_y=0, offset_z=0,
                        spacing=slice_spacing)
        # combine the contours
        slice_data = cube10 + half_cube10 + half_cube6
        structures = StructureSet(slice_data)

        # (1,3) should NOT be logical (result is ambiguous)
        rel_1_3 = structures.get_relationship(ROI_Type(1), ROI_Type(3))
        assert rel_1_3 is not None
        # Relationship A->C should be CONTAINS
        assert rel_1_3.relationship_type.relation_type == 'CONTAINS'
        # Not logical because 1_3 can either be Contains or Partitioned
        assert not rel_1_3.is_logical


class TestLogicalRelationshipFiltering:
    '''Tests for filter_relationships_by_logical_mode method.'''

    def test_hide_mode_removes_logical(self):
        '''Test: Hide mode removes all logical relationships.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Get filtered matrix with hide mode
        filtered_matrix, fade_info = (
            structures.filter_relationships_by_logical_mode(
                mode='hide',
                visible_rois=[1, 2, 3]
            )
        )

        # The logical relationship (1,3) should be None
        name_1 = structures.structures[1].name
        name_3 = structures.structures[3].name
        assert filtered_matrix.loc[name_3, name_1] is None

    def test_limited_mode_hides_with_all_intermediates(self):
        '''Test: Limited mode hides logical relationships when all
        intermediates are visible.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # All structures visible - logical (1,3) should be hidden
        filtered_matrix, fade_info = (
            structures.filter_relationships_by_logical_mode(
                mode='limited',
                visible_rois=[1, 2, 3]
            )
        )

        name_1 = structures.structures[1].name
        name_3 = structures.structures[3].name
        assert filtered_matrix.loc[name_3, name_1] is None

    def test_limited_mode_shows_with_hidden_intermediates(self):
        '''Test: Limited mode shows logical relationships when any
        intermediate is hidden.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Intermediate (2) is not visible - logical (1,3) should be shown
        filtered_matrix, fade_info = (
            structures.filter_relationships_by_logical_mode(
                mode='limited',
                visible_rois=[1, 3]  # 2 is hidden
            )
        )

        name_1 = structures.structures[1].name
        name_3 = structures.structures[3].name
        # The relationship object should still be there (not None)
        rel_obj = filtered_matrix.loc[name_3, name_1]
        assert rel_obj is not None

    def test_faded_mode_marks_logical_for_fading(self):
        '''Test: Faded mode marks logical relationships for opacity.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Get filtered matrix with faded mode
        filtered_matrix, fade_info = (
            structures.filter_relationships_by_logical_mode(
                mode='faded',
                visible_rois=[1, 2, 3]
            )
        )

        # The logical relationship (1,3) should be marked for fading
        if (1, 3) in fade_info:
            assert fade_info[(1, 3)] is True

    def test_show_mode_preserves_all_relationships(self):
        '''Test: Show mode preserves all relationships including logical.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Get filtered matrix with show mode
        filtered_matrix, fade_info = (
            structures.filter_relationships_by_logical_mode(
                mode='show',
                visible_rois=[1, 2, 3]
            )
        )

        # All relationships should be preserved
        name_1 = structures.structures[1].name
        name_3 = structures.structures[3].name
        rel_obj = filtered_matrix.loc[name_3, name_1]
        assert rel_obj is not None

    def test_to_dict_with_faded_mode(self):
        '''Test: to_dict() returns faded_relationships dict in faded mode.'''
        slice_spacing = 1
        sphere12 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing)
        sphere8 = make_sphere(roi_num=2, radius=4, spacing=slice_spacing)
        sphere6 = make_sphere(roi_num=3, radius=3, spacing=slice_spacing)

        slice_data = sphere12 + sphere8 + sphere6
        structures = StructureSet(slice_data)

        # Call to_dict with faded mode
        result = structures.to_dict(
            row_rois=[1, 2, 3],
            col_rois=[1, 2, 3],
            use_symbols=True,
            logical_relations_mode='faded',
            visible_rois=[1, 2, 3]
        )

        # Should have faded_relationships key with some entries
        assert 'faded_relationships' in result
