"""Tests for volume ratio metrics (overlapping and non-overlapping ratios).

Converted from the examples in
src/notebooks/metrics/StructureVolumeMetricTests.ipynb, which defines the
volume ratio algorithms and the composite structures used to compute them.

Volume ratios are only meaningful for "Shared" relationships (EQUAL,
CONTAINS, PARTITIONED, OVERLAPS).  Both ratios are stored in a single
shared VolumeRatioMetrics instance at relationship.metrics.volume_ratio;
each calculator fills only its own fields and reuses CompositeStructures
already registered in the StructureSet.
"""

import logging
from math import isnan

import pytest

from structure_set import StructureSet
from metrics.data_structures import VolumeRatioMetrics
from debug_tools import make_sphere, make_box, make_vertical_cylinder


def embedded_boxes_example():
    """4 cm cube containing a 2 cm cube (CONTAINS)."""
    slice_spacing = 0.1
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    inner_cube = make_box(roi_num=2, width=2, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    return outer_cube + inner_cube + body


def overlapping_boxes_example():
    """Two 4 cm cubes shifted +/-1 cm in x (OVERLAPS, intersection = V/2)."""
    slice_spacing = 0.1
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    left_cube = make_box(roi_num=1, width=4, offset_x=-1, offset_z=0,
                         spacing=slice_spacing)
    right_cube = make_box(roi_num=2, width=4, offset_x=1, offset_z=0,
                          spacing=slice_spacing)
    return left_cube + right_cube + body


def partitioning_boxes_example():
    """4 cm cube partitioned by a 4x4x2 cm box (PARTITIONED)."""
    slice_spacing = 0.1
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    outer_cube = make_box(roi_num=1, width=4, offset_x=0, offset_z=0,
                          spacing=slice_spacing)
    inner_cube = make_box(roi_num=2, width=4, length=4, height=2, offset_x=0,
                          offset_z=1, spacing=slice_spacing)
    return outer_cube + inner_cube + body


def disjoint_boxes_example():
    """Two 2 cm cubes at x = -2 and x = +2 (DISJOINT)."""
    slice_spacing = 0.1
    body = make_vertical_cylinder(roi_num=0, radius=20, length=10, offset_z=0,
                                  spacing=slice_spacing)
    left_cube = make_box(roi_num=1, width=2, offset_x=-2, offset_z=0,
                         spacing=slice_spacing)
    right_cube = make_box(roi_num=2, width=2, offset_x=2, offset_z=0,
                          spacing=slice_spacing)
    return left_cube + right_cube + body


def equal_spheres_example():
    """Two identical spheres with different ROIs (EQUAL)."""
    slice_spacing = 0.1
    sphere_a = make_sphere(roi_num=1, radius=6, spacing=slice_spacing,
                           num_points=100)
    sphere_b = make_sphere(roi_num=2, radius=6, spacing=slice_spacing,
                           num_points=100)
    return sphere_a + sphere_b


class TestContains:
    """Volume ratios for a CONTAINS relationship (cube in cube)."""

    def test_overlapping_ratio(self):
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        relationship = structures.get_relationship(1, 2)
        assert relationship.relationship_type.relation_type == 'CONTAINS'
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert isinstance(result, VolumeRatioMetrics)
        # CONTAINS: Intersection = V_B, Union = V_A
        assert result.overlapping_ratio == pytest.approx(
            volume_b / volume_a, abs=0.02)
        assert result.non_overlapping_ratio is None

    def test_non_overlapping_ratio(self):
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        result = structures.calculate_metric(1, 2,
                                             'non_overlapping_volume_ratio')
        # CONTAINS: Difference = V_A - V_B, Union = V_A.  The difference
        # composite is built from non-boundary contours only (per the
        # CompositeStructure specification), so its volume slightly
        # exceeds V_A - V_B.
        assert result.non_overlapping_ratio == pytest.approx(
            (volume_a - volume_b) / volume_a, abs=0.03)
        # Internal consistency: ratio matches the recorded composite volumes.
        assert result.non_overlapping_ratio == pytest.approx(
            result.difference_volume / result.union_volume, abs=1e-3)
        assert result.overlapping_ratio is None

    def test_ratios_sum_to_one(self):
        # CONTAINS: Difference / Union = 1 - Intersection / Union.  The
        # difference composite is built from non-boundary contours only,
        # so the sum deviates slightly from exactly 1.
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        structures.calculate_metric(1, 2, 'non_overlapping_volume_ratio')
        result = structures.get_relationship(1, 2).metrics.volume_ratio
        total = result.overlapping_ratio + result.non_overlapping_ratio
        assert total == pytest.approx(1.0, abs=0.03)


class TestOverlaps:
    """Volume ratios for an OVERLAPS relationship (shifted cubes)."""

    def test_overlapping_ratio(self):
        slice_data = overlapping_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        relationship = structures.get_relationship(1, 2)
        assert relationship.relationship_type.relation_type == 'OVERLAPS'
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        # Intersection = V_A / 2; Union = V_A + V_B - V_A / 2
        expected = (volume_a / 2) / (volume_a + volume_b - volume_a / 2)
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert result.overlapping_ratio == pytest.approx(expected, abs=0.02)

    def test_non_overlapping_ratio(self):
        slice_data = overlapping_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        # Difference = V_A / 2; Union = V_A + V_B - V_A / 2
        expected = (volume_a / 2) / (volume_a + volume_b - volume_a / 2)
        result = structures.calculate_metric(1, 2,
                                             'non_overlapping_volume_ratio')
        assert result.non_overlapping_ratio == pytest.approx(expected,
                                                           abs=0.02)


class TestPartitioned:
    """Volume ratios for a PARTITIONED relationship (cube and half cube)."""

    def test_overlapping_ratio(self):
        slice_data = partitioning_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        relationship = structures.get_relationship(1, 2)
        assert relationship.relationship_type.relation_type == 'PARTITIONED'
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        # PARTITIONED: Intersection = V_B, Union = V_A
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert result.overlapping_ratio == pytest.approx(
            volume_b / volume_a, abs=0.02)

    def test_non_overlapping_ratio(self):
        slice_data = partitioning_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        volume_a = structures.structures[1].structure_volumes.physical
        volume_b = structures.structures[2].structure_volumes.physical
        # PARTITIONED: Difference = V_A - V_B, Union = V_A
        result = structures.calculate_metric(1, 2,
                                             'non_overlapping_volume_ratio')
        assert result.non_overlapping_ratio == pytest.approx(
            (volume_a - volume_b) / volume_a, abs=0.02)


class TestEqual:
    """Volume ratios for an EQUAL relationship (identical spheres)."""

    def test_equal_ratios_without_composites(self):
        slice_data = equal_spheres_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        relationship = structures.get_relationship(1, 2)
        assert relationship.relationship_type.relation_type == 'EQUAL'
        structure_count = len(structures.structures)
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert result.overlapping_ratio == 1.0
        assert result.non_overlapping_ratio is None
        # EQUAL relationships do not need composite structures.
        assert len(structures.structures) == structure_count
        result = structures.calculate_metric(1, 2,
                                             'non_overlapping_volume_ratio')
        assert result.non_overlapping_ratio == 0.0
        assert len(structures.structures) == structure_count

    def test_equal_per_region_values(self):
        slice_data = equal_spheres_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert result.per_region_overlapping_ratio
        for ratio in result.per_region_overlapping_ratio.values():
            assert ratio == 1.0
        result = structures.calculate_metric(1, 2,
                                             'non_overlapping_volume_ratio')
        assert result.per_region_non_overlapping_ratio
        for ratio in result.per_region_non_overlapping_ratio.values():
            assert ratio == 0.0


class TestDisjoint:
    """Volume ratios for a DISJOINT relationship are not applicable."""

    def test_not_applicable(self, caplog):
        slice_data = disjoint_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        relationship = structures.get_relationship(1, 2)
        assert relationship.relationship_type.relation_type == 'DISJOINT'
        structure_count = len(structures.structures)
        with caplog.at_level(logging.WARNING):
            result = structures.calculate_metric(
                1, 2, 'overlapping_volume_ratio')
        assert isnan(result.overlapping_ratio)
        assert result.non_overlapping_ratio is None
        assert 'not applicable' in caplog.text
        # Non-applicable relationships do not create composite structures.
        assert len(structures.structures) == structure_count


class TestSharedStorageAndReuse:
    """Both ratios share one VolumeRatioMetrics and reuse composites."""

    def test_shared_metrics_instance(self):
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        overlapping = structures.calculate_metric(
            1, 2, 'overlapping_volume_ratio')
        non_overlapping = structures.calculate_metric(
            1, 2, 'non_overlapping_volume_ratio')
        assert non_overlapping is overlapping
        stored = structures.get_relationship(1, 2).metrics.volume_ratio
        assert stored is overlapping
        assert stored.overlapping_ratio is not None
        assert stored.non_overlapping_ratio is not None

    def test_composites_reused_between_calculators(self):
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        structure_count = len(structures.structures)
        structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        # Intersection + Union composites added.
        assert len(structures.structures) == structure_count + 2
        structures.calculate_metric(1, 2, 'non_overlapping_volume_ratio')
        # Only the Difference composite is added; the Union is reused.
        assert len(structures.structures) == structure_count + 3

    def test_per_region_values_reference_composites(self):
        slice_data = embedded_boxes_example()
        structures = StructureSet(slice_data, logging_enabled=False)
        result = structures.calculate_metric(1, 2, 'overlapping_volume_ratio')
        assert result.per_region_overlapping_ratio
        assert result.per_region_intersection_volumes
        assert result.per_region_union_volumes
        # Single-region structures: the only region-pair ratio matches the
        # overall ratio.
        for ratio in result.per_region_overlapping_ratio.values():
            assert ratio == pytest.approx(result.overlapping_ratio,
                                          abs=0.02)
