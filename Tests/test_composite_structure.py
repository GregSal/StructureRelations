'''Tests for the composite_structure module.'''
from pytest import approx

from contours import ContourPoints
from debug_tools import box_points, make_box, make_sphere
from debug_tools import make_vertical_cylinder
from structure_set import StructureSet
from composite_structure import CompositeStructure, structure_boolean

# Allowance for boundary-slice effects in volume calculations.  Boundary
# contours (half-size polygons half a slice spacing beyond each end of a
# structure) add volume not present in the ideal geometry, and composites
# with holes accumulate additional boundary contributions.
VOLUME_TOLERANCE = 0.05


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


def _embedded_boxes_example() -> list:
    '''Cube-in-cube (CONTAINS) test geometry.'''
    slice_spacing = 0.1
    # Body structure defines slices in use.  This is required to get the
    # correct boundary slices for the outer cube.
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    inner_cube = make_box(roi_num=2, width=2, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    return outer_cube + inner_cube + body


def _embedded_spheres_example() -> list:
    '''Embedded spheres (CONTAINS) test geometry.'''
    slice_spacing = 0.1
    sphere6 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing,
                          num_points=100)
    sphere3 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing,
                          num_points=100)
    return sphere6 + sphere3


def _overlapping_boxes_example() -> list:
    '''Overlapping cubes (OVERLAPS) test geometry.'''
    slice_spacing = 0.1
    # Body structure defines slices in use.  This is required to get the
    # correct boundary slices for the cubes.
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    left_cube = make_box(roi_num=1, width=4, offset_x=-1, offset_z=0,
                         spacing=slice_spacing)
    right_cube = make_box(roi_num=2, width=4, offset_x=1, offset_z=0,
                          spacing=slice_spacing)
    return left_cube + right_cube + body


def _partitioning_boxes_example() -> list:
    '''Cube and half-height cube (PARTITIONED) test geometry.'''
    slice_spacing = 0.1
    # Body structure defines slices in use.  This is required to get the
    # correct boundary slices for the outer cube.
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    inner_cube = make_box(roi_num=2, width=4, length=4, height=2, offset_x=0,
                          offset_z=1, spacing=slice_spacing)
    return outer_cube + inner_cube + body


def _disjoint_boxes_example() -> list:
    '''Disjoint cubes (DISJOINT) test geometry.'''
    slice_spacing = 0.1
    # Body structure defines slices in use.  This is required to get the
    # correct boundary slices for the cubes.
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    left_cube = make_box(roi_num=1, width=2, offset_x=-2, offset_z=0,
                         spacing=slice_spacing)
    right_cube = make_box(roi_num=2, width=2, offset_x=2, offset_z=0,
                          spacing=slice_spacing)
    return left_cube + right_cube + body


class TestCompositeVolumeCalculations():
    '''Volume checks for composite structures.

    Expected values are derived from the operand structure volumes, and a
    relative tolerance (VOLUME_TOLERANCE) allows for the volume contributed
    by boundary contours at the ends of each structure.
    '''

    @staticmethod
    def _composite_volumes(structure_set: StructureSet) -> dict:
        '''Return the physical volumes of the UNION, INTERSECTION and
        DIFFERENCE composites of structures 1 and 2.'''
        union = structure_boolean(structure_set, expression='1 UNION 2')
        intersection = structure_boolean(structure_set,
                                         expression='1 INTERSECTION 2')
        difference = structure_boolean(structure_set,
                                       expression='1 DIFFERENCE 2')
        return {
            'union': union.structure_volumes.physical,
            'intersection': intersection.structure_volumes.physical,
            'difference': difference.structure_volumes.physical,
        }

    def test_contains_box_volumes(self):
        '''CONTAINS: Union = V_A, Intersection = V_B, Difference = V_A - V_B.'''
        slice_data = _embedded_boxes_example()
        structures = StructureSet(slice_data=slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        volumes = self._composite_volumes(structures)
        assert volumes['union'] == approx(volume_a, rel=VOLUME_TOLERANCE)
        assert volumes['intersection'] == approx(volume_b, rel=VOLUME_TOLERANCE)
        assert volumes['difference'] == approx(volume_a - volume_b,
                                               rel=VOLUME_TOLERANCE)

    def test_contains_sphere_volumes(self):
        '''CONTAINS (spheres): Union = V_A, Intersection = V_B,
        Difference = V_A - V_B.'''
        slice_data = _embedded_spheres_example()
        structures = StructureSet(slice_data=slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        volumes = self._composite_volumes(structures)
        assert volumes['union'] == approx(volume_a, rel=VOLUME_TOLERANCE)
        assert volumes['intersection'] == approx(volume_b, rel=VOLUME_TOLERANCE)
        assert volumes['difference'] == approx(volume_a - volume_b,
                                               rel=VOLUME_TOLERANCE)

    def test_overlaps_volumes(self):
        '''OVERLAPS: Union = V_A + V_B - overlap, Intersection = overlap,
        Difference = V_A - overlap.'''
        slice_data = _overlapping_boxes_example()
        structures = StructureSet(slice_data=slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        overlap = volume_a / 2
        volumes = self._composite_volumes(structures)
        assert volumes['union'] == approx(volume_a + volume_b - overlap,
                                          rel=VOLUME_TOLERANCE)
        assert volumes['intersection'] == approx(overlap, rel=VOLUME_TOLERANCE)
        assert volumes['difference'] == approx(volume_a - overlap,
                                               rel=VOLUME_TOLERANCE)

    def test_partitioned_volumes(self):
        '''PARTITIONED: Union = V_A, Intersection = V_B,
        Difference = V_A - V_B.'''
        slice_data = _partitioning_boxes_example()
        structures = StructureSet(slice_data=slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        volumes = self._composite_volumes(structures)
        assert volumes['union'] == approx(volume_a, rel=VOLUME_TOLERANCE)
        assert volumes['intersection'] == approx(volume_b, rel=VOLUME_TOLERANCE)
        assert volumes['difference'] == approx(volume_a - volume_b,
                                               rel=VOLUME_TOLERANCE)

    def test_disjoint_volumes(self):
        '''DISJOINT: Union = V_A + V_B, Intersection = 0, Difference = V_A.'''
        slice_data = _disjoint_boxes_example()
        structures = StructureSet(slice_data=slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        volumes = self._composite_volumes(structures)
        assert volumes['union'] == approx(volume_a + volume_b,
                                          rel=VOLUME_TOLERANCE)
        assert volumes['intersection'] == approx(0.0, abs=1e-6)
        assert volumes['difference'] == approx(volume_a, rel=VOLUME_TOLERANCE)
