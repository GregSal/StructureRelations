'''Tests for the composite_structure module.'''
from pytest import approx

from contours import ContourPoints
from debug_tools import box_points
from structure_set import StructureSet
from composite_structure import CompositeStructure, structure_boolean


def _composite_area(composite: CompositeStructure, slice_index: float) -> float:
    '''Sum exterior contour areas minus hole areas for a composite on one slice.'''
    lookup = composite.contour_lookup
    on_slice = ((lookup['SliceIndex'] == slice_index)
                & ~lookup['Boundary'] & ~lookup['Interpolated'])
    rows = lookup.loc[on_slice]
    ext_area = sum(
        composite.contour_graph.nodes[label]['contour'].polygon.area
        for label in rows.loc[rows['HoleType'] == 'None', 'Label']
    )
    hole_area = sum(
        composite.contour_graph.nodes[label]['contour'].polygon.area
        for label in rows.loc[rows['HoleType'] != 'None', 'Label']
    )
    return ext_area - hole_area


def _set_with_two_disjoint_boxes() -> StructureSet:
    '''Build a StructureSet with two disjoint 1x1 boxes on one slice.'''
    box_a = box_points(width=1, offset_x=0)
    box_b = box_points(width=1, offset_x=10)
    slice_data = [
        ContourPoints(box_a, roi=1, slice_index=0.0),
        ContourPoints(box_b, roi=2, slice_index=0.0),
    ]
    return StructureSet(slice_data=slice_data, auto_calculate_relationships=False,
                        auto_calculate_logical_flags=False)


def _set_with_two_overlapping_boxes() -> StructureSet:
    '''Build a StructureSet with two overlapping 2x2 boxes on one slice.'''
    box_a = box_points(width=2, offset_x=0)
    box_b = box_points(width=2, offset_x=1)
    slice_data = [
        ContourPoints(box_a, roi=1, slice_index=0.0),
        ContourPoints(box_b, roi=2, slice_index=0.0),
    ]
    return StructureSet(slice_data=slice_data, auto_calculate_relationships=False,
                        auto_calculate_logical_flags=False)


class TestStructureBoolean():
    '''Test the structure_boolean function.'''

    def test_union_of_disjoint_boxes(self):
        '''UNION area should equal the sum of the two disjoint box areas.'''
        structure_set = _set_with_two_disjoint_boxes()
        composite = structure_boolean(structure_set, '1 UNION 2')
        assert isinstance(composite, CompositeStructure)
        assert _composite_area(composite, 0.0) == approx(2.0)

    def test_intersection_of_overlapping_boxes(self):
        '''INTERSECTION area should equal the overlap of the two boxes.'''
        structure_set = _set_with_two_overlapping_boxes()
        composite = structure_boolean(structure_set, '1 INTERSECTION 2')
        # Box A: x in [-1, 1]; Box B: x in [0, 2]; overlap x in [0, 1], y in [-1, 1]
        assert _composite_area(composite, 0.0) == approx(2.0)

    def test_difference_of_overlapping_boxes(self):
        '''DIFFERENCE area should equal box A minus the overlapping region.'''
        structure_set = _set_with_two_overlapping_boxes()
        composite = structure_boolean(structure_set, '1 DIFFERENCE 2')
        # Box A area 4.0, minus overlap area 2.0
        assert _composite_area(composite, 0.0) == approx(2.0)

    def test_chained_expression(self):
        '''A three-operand expression evaluates left-to-right.'''
        box_a = box_points(width=2, offset_x=0)
        box_b = box_points(width=2, offset_x=1)
        box_c = box_points(width=1, offset_x=0)
        slice_data = [
            ContourPoints(box_a, roi=1, slice_index=0.0),
            ContourPoints(box_b, roi=2, slice_index=0.0),
            ContourPoints(box_c, roi=3, slice_index=0.0),
        ]
        structure_set = StructureSet(slice_data=slice_data,
                                    auto_calculate_relationships=False,
                                    auto_calculate_logical_flags=False)
        composite = structure_boolean(structure_set, '1 UNION 2 DIFFERENCE 3')
        assert isinstance(composite, CompositeStructure)

    def test_unique_negative_roi(self):
        '''Each new CompositeStructure gets a unique negative ROI.'''
        structure_set = _set_with_two_disjoint_boxes()
        composite1 = structure_boolean(structure_set, '1 UNION 2')
        composite2 = structure_boolean(structure_set, '1 UNION 2')
        assert composite1.roi < 0
        assert composite2.roi < 0
        assert composite1.roi != composite2.roi

    def test_name_substitution_for_base_structures(self):
        '''Base structures are shown as [ROI] in the composite name.'''
        structure_set = _set_with_two_disjoint_boxes()
        composite = structure_boolean(structure_set, '1 UNION 2')
        assert composite.name == '[1] UNION [2]'
        assert composite.expression == composite.name

    def test_name_substitution_for_nested_composite(self):
        '''A nested CompositeStructure contributes its own name.'''
        structure_set = _set_with_two_disjoint_boxes()
        composite1 = structure_boolean(structure_set, '1 UNION 2')
        composite1_roi = composite1.roi
        composite2 = structure_boolean(
            structure_set, f'{composite1_roi} DIFFERENCE 1'
        )
        assert composite1.name in composite2.name

    def test_registered_in_structure_set(self):
        '''The resulting CompositeStructure is added to the StructureSet.'''
        structure_set = _set_with_two_disjoint_boxes()
        composite = structure_boolean(structure_set, '1 UNION 2')
        assert structure_set.structures[composite.roi] is composite

    def test_hole_is_subtracted_before_union(self):
        '''A hole in one structure is preserved through a UNION.'''
        outer = box_points(width=4)
        inner = box_points(width=2)
        box_far = box_points(width=1, offset_x=20)
        slice_data = [
            ContourPoints(outer, roi=1, slice_index=0.0),
            ContourPoints(inner, roi=1, slice_index=0.0),
            ContourPoints(box_far, roi=2, slice_index=0.0),
        ]
        structure_set = StructureSet(slice_data=slice_data,
                                    auto_calculate_relationships=False,
                                    auto_calculate_logical_flags=False)
        composite = structure_boolean(structure_set, '1 UNION 2')
        # Donut area (16 - 4) plus disjoint box area (1)
        assert _composite_area(composite, 0.0) == approx(13.0)
