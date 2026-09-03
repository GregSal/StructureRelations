"""Tests for margin metrics (orthogonal and minimum margins).

Converted from the examples in
src/notebooks/metrics/StructureMarginMetricTests.ipynb, which defines the
margin metric algorithms.

Margins measure clearance distances inside containing structures.  Applicable
to CONTAINS, PARTITIONED, SURROUNDS, SHELTERS, CONFINES and EQUAL
relationships; NaN for OVERLAPS, BORDERS and DISJOINT.

Note on two examples where the notebook's commented-out expected values did
not match its own example code (which is why its asserts were commented out):
- Box in cylinder: the notebook's formulas (1.46 orthogonal, 1.17 minimum)
  assume a square 4x4 cross-section, but the example code used length=2.
  The test uses length=4 to match the documented intent.
- Vertical sheltered cylinder: the notebook expects the inner half-cylinder
  inside the C-shaped shell's cavity (x_neg 1.0, y 1.0), which requires the
  LEFT half-disc; the example code produced the right half-disc.  The test
  uses the left half-disc to match the documented intent.
- Two boxes in sphere: the notebook's expected y margins (0.24) reused the X
  margin formula; the geometrically correct vertex-based Y margin for the
  right box is ~0.66.  The test uses the geometric values.
"""

import math
from math import nan, isnan

import pytest
from shapely.geometry import Polygon, box as shapely_box

from structure_set import StructureSet
from relations import DE9IM
from metrics.margins import Z_MARGIN_CHANGE_TESTS
from debug_tools import (
    make_sphere, make_box, make_vertical_cylinder,
    make_horizontal_cylinder, make_hourglass_polygon,
    circle_points, box_points, extrude_poly,
)


def get_relation_and_margins(slice_data, roi_a=1, roi_b=2, tolerance=0.0):
    """Build the structure set, identify the relationship, and get margins.

    Args:
        slice_data: Combined contour data for all structures.
        roi_a: ROI number of the outer structure.
        roi_b: ROI number of the inner structure.
        tolerance: Structure set tolerance used for the margin calculation.

    Returns:
        Tuple of (structure_set, relation_type, MarginMetrics).
    """
    structures = StructureSet(slice_data)
    if tolerance:
        structures.tolerance = tolerance
    structure_a = structures.structures[roi_a]
    structure_b = structures.structures[roi_b]
    relation = structure_a.relate(structure_b)
    relation_type = relation.identify_relation()
    margin_result = structures.calculate_metric(roi_a, roi_b,
                                                'minimum_margins')
    return structures, relation_type, margin_result


def assert_margins_close(margins, expected, tolerance):
    """Assert that each margin value is within tolerance of expected.

    NaN expectations are checked with isnan.

    Args:
        margins: Dict of direction -> margin value.
        expected: Dict of direction -> expected margin value (or NaN).
        tolerance: Maximum absolute difference allowed.
    """
    for direction, value in expected.items():
        assert direction in margins, (
            f'{direction} was not found in margins: {margins}.')
        actual = margins[direction]
        if isinstance(value, float) and isnan(value):
            assert isnan(actual), (
                f'Expected NaN in {direction} direction, got {actual}.')
        else:
            assert not isnan(actual), (
                f'Expected {value} in {direction} direction, got NaN.')
            # The epsilon guards against float representation noise when the
            # difference lands exactly on the tolerance (slice spacing).
            assert abs(actual - value) <= tolerance + 1e-9, (
                f'Expected {value} in {direction} direction, got {actual}.')


# %% Basic margin metric functionality
class TestEmbeddedBoxes:
    """Cube in Cube (Embedded Boxes): all margins are the same."""

    def test_embedded_boxes(self):
        def embedded_boxes_example():
            slice_spacing = 0.2
            # Body structure defines slices in use
            # This is required to get the correct boundary slices for the
            # outer cube.
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                                  spacing=slice_spacing)
            inner_cube = make_box(roi_num=2, width=2, offset_x=0, offset_z=0,
                                  spacing=slice_spacing)
            return outer_cube + inner_cube + body

        structures, relation_type, margin_result = get_relation_and_margins(
            embedded_boxes_example())

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.0, 'x_pos': 1.0,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': 1.0, 'z_pos': 1.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, 0.001)
        assert margin_result.minimum_margin == 1.0


class TestEmbeddedSpheres:
    """Simple embedded spheres: all margins are the same."""

    def test_embedded_spheres(self):
        def embedded_sphere_example():
            slice_spacing = 0.1
            sphere6 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing,
                                  num_points=100)
            sphere3 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing,
                                  num_points=100)
            return sphere6 + sphere3

        tolerance = 0.05
        structures, relation_type, margin_result = get_relation_and_margins(
            embedded_sphere_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 3.0, 'x_pos': 3.0,
                            'y_neg': 3.0, 'y_pos': 3.0,
                            'z_neg': 3.0, 'z_pos': 3.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        # Pseudo-boundary slices affect only the Z orthogonal margins, so the
        # minimum margin is calculated from the original contours.
        assert abs(margin_result.minimum_margin - 3.0) <= tolerance
    """Equal structures: all margins are 0, minimum distance is NaN."""

    def test_equal_boxes(self):
        slice_spacing = 0.1
        cube_1 = make_box(roi_num=1, width=4, offset_x=0, offset_y=0,
                          offset_z=0, spacing=slice_spacing)
        cube_2 = make_box(roi_num=2, width=4, offset_x=0, offset_y=0,
                          offset_z=0, spacing=slice_spacing)
        slice_data = cube_1 + cube_2

        structures, relation_type, margin_result = get_relation_and_margins(
            slice_data)

        assert relation_type.relation_type == 'EQUAL'
        expected_margins = {'x_neg': 0.0, 'x_pos': 0.0,
                            'y_neg': 0.0, 'y_pos': 0.0,
                            'z_neg': 0.0, 'z_pos': 0.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, 0.001)
        assert margin_result.minimum_margin == 0.0

        # Minimum distance is NaN for EQUAL structures.
        distance_result = structures.calculate_metric(1, 2,
                                                      'minimum_distance')
        assert isnan(distance_result.minimum_distance)


class TestOffsetBoxes:
    """Planar margins correctly handle different positive/negative margins."""

    def test_offset_boxes(self):
        def offset_boxes_example():
            slice_spacing = 0.2
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            outer_cube = make_box(roi_num=1, width=4, length=4,
                                  offset_x=0, offset_y=-0.5, offset_z=0,
                                  spacing=slice_spacing)
            inner_cube = make_box(roi_num=2, width=2, length=2,
                                  offset_x=0.3, offset_y=0, offset_z=0,
                                  spacing=slice_spacing)
            return outer_cube + inner_cube + body

        structures, relation_type, margin_result = get_relation_and_margins(
            offset_boxes_example())

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.3, 'x_pos': 0.7,
                            'y_neg': 1.5, 'y_pos': 0.5,
                            'z_neg': 1.0, 'z_pos': 1.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, 0.001)
        assert margin_result.minimum_margin == 0.5


class TestZOffsetBoxes:
    """Z margins correctly handle different positive and negative margins."""

    def test_z_offset_boxes(self):
        def z_offset_boxes_example():
            slice_spacing = 0.2
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            outer_cube = make_box(roi_num=1, width=4, length=4, height=4,
                                  offset_x=0, offset_y=0, offset_z=0,
                                  spacing=slice_spacing)
            inner_cube = make_box(roi_num=2, width=2, length=2, height=1,
                                  offset_x=0, offset_y=0, offset_z=-0.5,
                                  spacing=slice_spacing)
            return outer_cube + inner_cube + body

        structures, relation_type, margin_result = get_relation_and_margins(
            z_offset_boxes_example())

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.0, 'x_pos': 1.0,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': 1.0, 'z_pos': 2.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, 0.001)
        assert margin_result.minimum_margin == 1.0


class TestBoxInCylinder:
    """Minimum margin is not an orthogonal margin (box in cylinder).

    For cylinder radius 4 and a box with a 4x4 cross-section, the minimum
    margin is in a diagonal direction: 4 - sqrt(2*(4/2)**2) = 1.17, while the
    orthogonal X and Y margins are sqrt(4**2 - (4/2)**2) - 4/2 = 1.46.
    """

    def test_box_in_cylinder(self):
        def box_in_cylinder_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            outer_cylinder = make_vertical_cylinder(roi_num=1, radius=4,
                                                    length=8, offset_z=0,
                                                    num_points=360,
                                                    spacing=slice_spacing)
            inner_box = make_box(roi_num=2, width=4, length=4, height=6,
                                 offset_x=0, offset_y=0, offset_z=0,
                                 spacing=slice_spacing)
            return outer_cylinder + inner_box + body

        tolerance = 0.01
        _, relation_type, margin_result = get_relation_and_margins(
            box_in_cylinder_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.46, 'x_pos': 1.46,
                            'y_neg': 1.46, 'y_pos': 1.46,
                            'z_neg': 1.0, 'z_pos': 1.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 1.17) <= tolerance


class TestCubeInSphere:
    """Minimum margin is off-plane (cube in sphere).

    For sphere radius 4 and a cube of width 4, the minimum margin is in a 3D
    diagonal direction: 4 - sqrt(3*(4/2)**2) = 0.54.  The orthogonal X and Y
    margins are sqrt(4**2 - 2**2 - 2**2) - 2 = 0.83.  The ideal Z margin
    (0.83) is rounded up to the next slice: 0.9 with a slice spacing of 0.1.
    """

    def test_cube_in_sphere(self):
        def cube_in_sphere_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            outer_sphere = make_sphere(roi_num=1, radius=4,
                                       spacing=slice_spacing, num_points=300)
            inner_cube = make_box(roi_num=2, width=4, offset_x=0, offset_y=0,
                                  offset_z=0, spacing=slice_spacing)
            return outer_sphere + inner_cube + body

        tolerance = 0.1
        _, relation_type, margin_result = get_relation_and_margins(
            cube_in_sphere_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 0.83, 'x_pos': 0.83,
                            'y_neg': 0.83, 'y_pos': 0.83,
                            'z_neg': 0.90, 'z_pos': 0.90}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 0.54) <= tolerance


# %% Multiple regions
class TestTwoBoxesInSphere:
    """Structure A has 1 region, structure B has 2 regions.

    Sphere radius 4 containing a 1x4x4 box offset left by 1.5 and a 2x2x2
    box offset right by 2.5.
    """

    def test_two_boxes_in_sphere(self):
        def two_boxes_in_sphere_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=10, length=20,
                                          offset_z=0, spacing=slice_spacing)
            outer_sphere = make_sphere(roi_num=1, radius=4,
                                       spacing=slice_spacing, offset_x=0,
                                       offset_y=0, offset_z=0,
                                       num_points=360)
            left_inner_box = make_box(roi_num=2, width=1, length=4, height=4,
                                      spacing=slice_spacing, offset_x=-1.5,
                                      offset_y=0, offset_z=0)
            right_inner_box = make_box(roi_num=2, width=2, length=2, height=2,
                                       spacing=slice_spacing, offset_x=2.5,
                                       offset_y=0, offset_z=0)
            return outer_sphere + left_inner_box + right_inner_box + body

        tolerance = 0.1
        _, relation_type, margin_result = get_relation_and_margins(
            two_boxes_in_sphere_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        # Final values are the minimum across the two region pairs.
        expected_margins = {'x_neg': 0.8, 'x_pos': 0.2,
                            'y_neg': 0.7, 'y_pos': 0.7,
                            'z_neg': 0.7, 'z_pos': 0.7}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 0.2) <= tolerance

        # Per-region-pair checks.
        per_region = margin_result.per_region_orthogonal_margins
        per_region_min = margin_result.per_region_minimum_margin
        assert len(per_region) == 2
        # The right box (2x2x2 at x=2.5) has the small x_pos margin.
        right_pair = min(per_region, key=lambda p: per_region[p]['x_pos'])
        left_pair = max(per_region, key=lambda p: per_region[p]['x_pos'])

        assert_margins_close(per_region[left_pair],
                             {'x_neg': 0.83, 'x_pos': 3.83,
                              'y_neg': 0.83, 'y_pos': 0.83,
                              'z_neg': 0.9, 'z_pos': 0.9}, tolerance)
        assert abs(per_region_min[left_pair] - 0.54) <= tolerance

        assert_margins_close(per_region[right_pair],
                             {'x_neg': 5.24, 'x_pos': 0.24,
                              'y_neg': 0.66, 'y_pos': 0.66,
                              'z_neg': 0.7, 'z_pos': 0.7}, tolerance)
        assert abs(per_region_min[right_pair] - 0.23) <= tolerance


class TestCubesInSpheres:
    """Structure A has 2 regions, structure B has 2 regions.

    2 spheres offset horizontally from each other, each with a box embedded
    and centered in its sphere (4-wide box on the left, 2-wide on the right).
    """

    def test_cubes_in_spheres(self):
        def cubes_in_spheres_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            left_outer_sphere = make_sphere(roi_num=1, radius=4,
                                            spacing=slice_spacing,
                                            offset_x=-5, num_points=300)
            left_inner_cube = make_box(roi_num=2, width=4,
                                       spacing=slice_spacing, offset_x=-5)
            right_outer_sphere = make_sphere(roi_num=1, radius=4,
                                             spacing=slice_spacing,
                                             offset_x=5, num_points=300)
            right_inner_cube = make_box(roi_num=2, width=2,
                                        spacing=slice_spacing, offset_x=5)
            return (left_outer_sphere + left_inner_cube +
                    right_outer_sphere + right_inner_cube + body)

        tolerance = 0.1
        structures, relation_type, margin_result = get_relation_and_margins(
            cubes_in_spheres_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        # Final values are the minimum across the valid region pairs.
        expected_margins = {'x_neg': 0.83, 'x_pos': 0.83,
                            'y_neg': 0.83, 'y_pos': 0.83,
                            'z_neg': 0.9, 'z_pos': 0.9}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 0.54) <= tolerance

        # Per-region-pair checks (disjoint region pairs are excluded).
        per_region = margin_result.per_region_orthogonal_margins
        per_region_min = margin_result.per_region_minimum_margin
        assert len(per_region) == 2
        left_pair = min(per_region_min, key=per_region_min.get)
        right_pair = max(per_region_min, key=per_region_min.get)

        assert_margins_close(per_region[left_pair],
                             {'x_neg': 0.83, 'x_pos': 0.83,
                              'y_neg': 0.83, 'y_pos': 0.83,
                              'z_neg': 0.9, 'z_pos': 0.9}, tolerance)
        assert abs(per_region_min[left_pair] - 0.54) <= tolerance

        assert_margins_close(per_region[right_pair],
                             {'x_neg': 2.74, 'x_pos': 2.74,
                              'y_neg': 2.74, 'y_pos': 2.74,
                              'z_neg': 2.8, 'z_pos': 2.8}, tolerance)
        assert abs(per_region_min[right_pair] - 2.27) <= tolerance


# %% Non-applicable relationships
class TestOverlappingBoxes:
    """Overlapping boxes: all margins are NaN (OVERLAPS is not applicable)."""

    def test_overlapping_boxes(self):
        def overlapping_boxes_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=20,
                                          offset_z=0, spacing=slice_spacing)
            left_cube = make_box(roi_num=1, width=2, offset_x=0,
                                 offset_z=-0.5, spacing=slice_spacing)
            right_cube = make_box(roi_num=2, width=2, offset_x=0,
                                  offset_z=0.5, spacing=slice_spacing)
            return left_cube + right_cube + body

        structures, relation_type, margin_result = get_relation_and_margins(
            overlapping_boxes_example())

        assert relation_type.relation_type == 'OVERLAPS'
        for value in margin_result.orthogonal_margins.values():
            assert isnan(value)
        assert isnan(margin_result.minimum_margin)

        # Minimum distance is NaN for OVERLAPS structures as well.
        distance_result = structures.calculate_metric(1, 2,
                                                      'minimum_distance')
        assert isnan(distance_result.minimum_distance)


class TestDisjointBoxes:
    """Disjoint boxes: all margins are NaN (DISJOINT is not applicable)."""

    def test_disjoint_boxes(self):
        slice_spacing = 0.1
        body = make_vertical_cylinder(roi_num=0, radius=20, length=20,
                                      offset_z=0, spacing=slice_spacing)
        left_cube = make_box(roi_num=1, width=2, offset_x=-5,
                             spacing=slice_spacing)
        right_cube = make_box(roi_num=2, width=2, offset_x=5,
                              spacing=slice_spacing)
        slice_data = left_cube + right_cube + body

        structures, relation_type, margin_result = get_relation_and_margins(
            slice_data)

        assert relation_type.relation_type == 'DISJOINT'
        for value in margin_result.orthogonal_margins.values():
            assert isnan(value)
        assert isnan(margin_result.minimum_margin)

        # Minimum distance IS applicable for DISJOINT structures.
        distance_result = structures.calculate_metric(1, 2,
                                                      'minimum_distance')
        assert not isnan(distance_result.minimum_distance)
        assert distance_result.minimum_distance > 0


# %% Orthogonal lines with multiple intersections with the outer structure
class TestSphereInSphereInShell:
    """Sphere inside a solid sphere inside a shell (multiple intersections).

    Structure A is a shell (radius 6..8) plus a solid inner sphere (radius 4);
    structure B is a sphere of radius 3 inside the inner sphere.  Orthogonal
    lines from B's contour intersect A's boundary multiple times; the margin
    is the closest intersection.
    """

    def test_sphere_in_sphere_in_shell(self):
        def sphere_in_sphere_in_shell_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            # outer shell
            shell8 = make_sphere(roi_num=1, radius=8, spacing=slice_spacing,
                                 num_points=100)
            # hole converts sphere into a shell
            hole6 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing,
                                num_points=100)
            # inner sphere
            sphere4 = make_sphere(roi_num=1, radius=4, spacing=slice_spacing,
                                  num_points=100)
            # innermost sphere (separate structure)
            sphere3 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing,
                                  num_points=100)
            return shell8 + hole6 + sphere4 + sphere3 + body

        tolerance = 0.05
        structures, relation_type, margin_result = get_relation_and_margins(
            sphere_in_sphere_in_shell_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.0, 'x_pos': 1.0,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': 1.0, 'z_pos': 1.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 1.0) <= tolerance
class TestSphereInShell:
    """Sphere surrounded by a hollow sphere: margins are the same."""

    def test_sphere_in_shell(self):
        def sphere_in_shell_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            # outer shell
            shell8 = make_sphere(roi_num=1, radius=8, spacing=slice_spacing,
                                 num_points=100)
            # hole converts sphere into a shell
            hole6 = make_sphere(roi_num=1, radius=6, spacing=slice_spacing,
                                num_points=100)
            # inner sphere
            sphere3 = make_sphere(roi_num=2, radius=3, spacing=slice_spacing,
                                  num_points=100)
            return shell8 + hole6 + sphere3 + body

        tolerance = 0.05
        structures, relation_type, margin_result = get_relation_and_margins(
            sphere_in_shell_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'SURROUNDS'

        expected_margins = {'x_neg': 3.0, 'x_pos': 3.0,
                            'y_neg': 3.0, 'y_pos': 3.0,
                            'z_neg': 3.0, 'z_pos': 3.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 3.0) <= tolerance


# %% Shelters
class TestVerticalShelteredCylinder:
    """Shelters with opening on the plane of the slices.

    Half-cylinder sheltered by a C-shaped shell: the X_pos margin and the Z
    margins are NaN (the cavity is open in those directions).
    """

    def test_vertical_sheltered_cylinder(self):
        def vertical_sheltered_cylinder_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)

            outer_circle = Polygon(circle_points(radius=4.0, num_points=256))
            outer_hole = Polygon(circle_points(3.0, num_points=256))
            # A large rectangle keeps slightly more than the left half of the
            # C polygon.  Note: the notebook example used the default height
            # (4), which only removes the middle band |y| <= 2 and leaves the
            # annulus intact for |y| > 2, so the cavity is not actually open
            # in the +X direction.  A taller rectangle is used here to match
            # the documented intent (x_pos margin is NaN).
            left_half_plane = Polygon(box_points(width=4.0, height=12.0,
                                                 offset_x=2.5,
                                                 offset_y=0))
            outer_c_polygon = (outer_circle.difference(outer_hole)
                               - left_half_plane)
            outer_c = extrude_poly(outer_c_polygon, length=4.0,
                                   spacing=slice_spacing, roi_num=1)

            inner_circle = Polygon(circle_points(radius=2.0, num_points=256,
                                                 offset_x=0, offset_y=0))
            # Note: the notebook example code intersected with the rectangle
            # x in [0, 4], producing the RIGHT half-disc, but its expected
            # margins (x_neg 1.0, y 1.0) require the LEFT half-disc sitting
            # inside the C-shaped cavity, which is used here.
            left_inner_half_plane = Polygon(box_points(width=4.0,
                                                       offset_x=-2.0,
                                                       offset_y=0))
            inner_semicircle_polygon = inner_circle.intersection(
                left_inner_half_plane)
            inner_semicircle = extrude_poly(inner_semicircle_polygon,
                                            length=4.0,
                                            spacing=slice_spacing, roi_num=2)

            return outer_c + inner_semicircle + body

        tolerance = 0.05
        structures, relation_type, margin_result = get_relation_and_margins(
            vertical_sheltered_cylinder_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'SHELTERS'

        expected_margins = {'x_neg': 1.0, 'x_pos': nan,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': nan, 'z_pos': nan}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 1.0) <= tolerance


class TestHorizontalShelteredCylinder:
    """Shelters with opening in the Z direction.

    Half-cylinder sheltered by a hollow half-cylinder: the Z_pos margin is
    NaN (the cavity is open in the positive Z direction).
    """

    def test_horizontal_sheltered_cylinder(self):
        def horizontal_sheltered_cylinder_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)

            outer_cylinder = make_horizontal_cylinder(
                roi_num=1, radius=4.0, length=10.0, offset_x=0, offset_y=0,
                offset_z=0, spacing=slice_spacing)
            # Exclude slices above the Z=0 plane to cut the cylinder in half.
            outer_cylinder = [contour for contour in outer_cylinder
                              if contour['Slice'] <= 0.0]

            cylinder_hole = make_horizontal_cylinder(
                roi_num=1, radius=3.0, length=10.0, offset_x=0, offset_y=0,
                offset_z=0, spacing=slice_spacing)
            # Exclude slices above the Z=0 plane to match the cylinder.
            cylinder_hole = [contour for contour in cylinder_hole
                             if contour['Slice'] <= 0.0]

            surrounded_cylinder = make_horizontal_cylinder(
                roi_num=2, radius=2.0, length=6.0, spacing=slice_spacing)
            # Exclude slices above the Z=-0.5 plane to shelter the
            # half-cylinder.
            surrounded_cylinder = [contour for contour in surrounded_cylinder
                                   if contour['Slice'] <= -0.5]
            return outer_cylinder + cylinder_hole + surrounded_cylinder + body

        tolerance = 0.1
        structures, relation_type, margin_result = get_relation_and_margins(
            horizontal_sheltered_cylinder_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'SHELTERS'

        expected_margins = {'x_neg': nan, 'x_pos': nan,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': 1.0, 'z_pos': nan}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 1.0) <= tolerance


# %% X margin is in a central location (hourglass shape)
class TestHourglass:
    """Hourglass containing a cylinder shifted towards the neck.

    Margins are approximate because the inner cylinder is shifted towards the
    middle of the hourglass.
    """

    def test_hourglass(self):
        def hourglass_example():
            slice_spacing = 0.1
            body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                          offset_z=0, spacing=slice_spacing)
            # Hourglass polygon
            outer_poly, geometry_report = make_hourglass_polygon(
                circle_radius=4.0,
                neck_thickness=1.0,
                transition_steepness=1.5,
                neck_half_width=1.0,
                offset_x=0.0,
                offset_y=0.0,
                report_points=False,
            )
            hour_glass_stacked = extrude_poly(roi_num=1, polygon=outer_poly,
                                              length=6.0, offset_z=0,
                                              spacing=slice_spacing)
            # Inner cylinder (shifted towards the centre of the hourglass)
            left_offset = geometry_report['left_circle_center'][0] + 0.5
            left_circle = make_vertical_cylinder(roi_num=2, radius=3.0,
                                                 length=4.0,
                                                 offset_x=left_offset,
                                                 offset_y=0, offset_z=0,
                                                 num_points=256)
            return hour_glass_stacked + left_circle + body

        tolerance = 0.1
        structures, relation_type, margin_result = get_relation_and_margins(
            hourglass_example(), tolerance=tolerance)

        assert relation_type.relation_type == 'CONTAINS'

        expected_margins = {'x_neg': 1.5, 'x_pos': 0.5,
                            'y_neg': 1.0, 'y_pos': 1.0,
                            'z_neg': 1.0, 'z_pos': 1.0}
        assert_margins_close(margin_result.orthogonal_margins,
                             expected_margins, tolerance)
        assert abs(margin_result.minimum_margin - 0.5) <= tolerance


# %% Z-margin change test masks
class TestZMarginTriggerTables:
    """Unit tests pinning the Z-margin change test mask semantics.

    The change tests are applied to the cross-slice 2D DE-9IM between the
    outer region on the test slice and the inner region on the reference
    slice.  A Z margin triggers when any masked cell is True.
    """

    @staticmethod
    def _change_detected(poly_a, poly_b, reference_class):
        mask = Z_MARGIN_CHANGE_TESTS[reference_class]
        return bool(DE9IM(poly_a, poly_b).int & mask)

    def test_contains_column(self):
        # Outer square [0, 10] x [0, 10].
        poly_a = shapely_box(0, 0, 10, 10)
        # Fully contained: no change.
        assert not self._change_detected(poly_a, shapely_box(2, 2, 4, 4),
                                         'CONTAINS')
        # Crossing the boundary: change (B-B and E-I both True).
        assert self._change_detected(poly_a, shapely_box(8, 2, 12, 4),
                                     'CONTAINS')
        # Touching the boundary from inside: change (B-B True).
        assert self._change_detected(poly_a, shapely_box(8, 2, 10, 4),
                                     'CONTAINS')
        # Fully outside: change (E-I True).
        assert self._change_detected(poly_a, shapely_box(20, 20, 22, 22),
                                     'CONTAINS')

    def test_partitioned_column(self):
        poly_a = shapely_box(0, 0, 10, 10)
        # Fully contained: no change.
        assert not self._change_detected(poly_a, shapely_box(2, 2, 4, 4),
                                         'PARTITIONED')
        # Poking out: change (E-I True).
        assert self._change_detected(poly_a, shapely_box(8, 2, 12, 4),
                                     'PARTITIONED')
        # Touching the boundary from inside without poking out is not a
        # change for a partitioned reference (E-I False).
        assert not self._change_detected(poly_a, shapely_box(8, 2, 10, 4),
                                         'PARTITIONED')

    def test_surrounds_shelters_column(self):
        # Outer square with a square hole [4, 6] x [4, 6].
        poly_a = Polygon(shell=shapely_box(0, 0, 10, 10).exterior.coords,
                         holes=[shapely_box(4, 4, 6, 6).exterior.coords])
        # Inside the hole: no change.
        assert not self._change_detected(poly_a,
                                         shapely_box(4.25, 4.25, 5.75, 5.75),
                                         'SURROUNDS')
        # Touching the hole wall: change (B-B True).
        assert self._change_detected(poly_a,
                                     shapely_box(4.0, 4.25, 4.5, 5.75),
                                     'SURROUNDS')
        # Cavity closed (solid outer): change (I-I True).
        solid_a = shapely_box(0, 0, 10, 10)
        assert self._change_detected(solid_a, shapely_box(4.5, 4.5, 5.5, 5.5),
                                     'SURROUNDS')
        # Same tests apply to the SHELTERS reference class.
        assert not self._change_detected(poly_a,
                                         shapely_box(4.25, 4.25, 5.75, 5.75),
                                         'SHELTERS')
        assert self._change_detected(solid_a, shapely_box(4.5, 4.5, 5.5, 5.5),
                                     'SHELTERS')

    def test_confines_column(self):
        # Outer square with a square hole [4, 6] x [4, 6].
        poly_a = Polygon(shell=shapely_box(0, 0, 10, 10).exterior.coords,
                         holes=[shapely_box(4, 4, 6, 6).exterior.coords])
        # Inside the hole: no change.
        assert not self._change_detected(poly_a,
                                         shapely_box(4.25, 4.25, 5.75, 5.75),
                                         'CONFINES')
        # Cavity closed (solid outer): change (I-I True).
        solid_a = shapely_box(0, 0, 10, 10)
        assert self._change_detected(solid_a, shapely_box(4.5, 4.5, 5.5, 5.5),
                                     'CONFINES')
        # Touching the hole wall without entering the material is not a
        # change for a confined reference (I-I False).
        assert not self._change_detected(poly_a,
                                         shapely_box(4.0, 4.25, 4.5, 5.75),
                                         'CONFINES')


# %% Metric storage structure
class TestMetricStorageShape:
    """Check the structure of the margin metric result tables."""

    def test_storage_shape(self):
        slice_spacing = 0.2
        body = make_vertical_cylinder(roi_num=0, radius=20, length=10,
                                      offset_z=0, spacing=slice_spacing)
        outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                              spacing=slice_spacing)
        inner_cube = make_box(roi_num=2, width=2, offset_x=0, offset_z=0,
                              spacing=slice_spacing)
        slice_data = outer_cube + inner_cube + body

        structures = StructureSet(slice_data)
        margin_result = structures.calculate_metric(1, 2, 'minimum_margins')

        # Per-region-pair tables: one pair for single-region structures.
        assert len(margin_result.per_region_orthogonal_margins) == 1
        pair = next(iter(margin_result.per_region_orthogonal_margins))
        assert len(pair) == 2

        # Detailed orthogonal table: (reference_slice, test_slice) keys;
        # planar (X/Y) entries have reference_slice == test_slice.
        ortho_records = margin_result.slice_orthogonal_margins[pair]
        assert ortho_records
        for (ref, test), margins in ortho_records.items():
            for direction in margins:
                if direction in ('x_neg', 'x_pos', 'y_neg', 'y_pos'):
                    assert ref == test

        # Detailed minimum margin table: (reference_slice, test_slice,
        # direction) keys with direction 'planar', 'neg', or 'pos'.
        min_records = margin_result.slice_minimum_margins[pair]
        assert min_records
        for ref, test, tag in min_records:
            assert tag in ('planar', 'neg', 'pos')

        # Per-reference-slice summaries are populated.
        assert margin_result.per_slice_orthogonal_margins
        assert margin_result.per_slice_minimum_margin

        # The result is stored in the relationship graph.
        relationship = structures.get_relationship(1, 2)
        assert relationship.metrics.margin is margin_result

        # JSON serialization keys for the webapp are unchanged.
        metrics_dict = relationship.metrics.to_dict()
        assert 'orthogonal_margins' in metrics_dict['margin']
        assert 'minimum_margin' in metrics_dict['margin']
