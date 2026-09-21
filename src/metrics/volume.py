"""Volume ratio metric calculators for structures with shared volume.

Two calculators are provided:

- OverlappingVolumeRatioCalculator ('overlapping_volume_ratio'):
    Intersection volume / Union volume.
- NonOverlappingVolumeRatioCalculator ('non_overlapping_volume_ratio'):
    Difference (A - B) volume / Union volume.

Both are applicable only to "Shared" relationships (EQUAL, CONTAINS, WITHIN,
PARTITIONED, PARTITIONS, OVERLAPS).  For EQUAL relationships no
CompositeStructure is required: the overlapping ratio is 1.0 and the
non-overlapping ratio is 0.0.  For the remaining shared relationships the
required composites (Intersection/Union or Difference/Union) are obtained
from the StructureSet, reusing composites already registered there (for
example by a prior call to the companion calculator) and creating them with
structure_boolean when absent.

Both calculators share a single VolumeRatioMetrics instance stored at
relationship.metrics.volume_ratio.  Each calculator fills only its own
ratio fields; fields belonging to the other calculator remain None
("not evaluated") until that calculator is run.
"""

import logging
import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from structures import StructureShape
from relationships import StructureRelationship
from composite_structure import CompositeStructure, structure_boolean
from types_and_classes import ROI_Type
from metrics.base import MetricCalculator, register_calculator
from metrics.data_structures import VolumeRatioMetrics

if TYPE_CHECKING:
    from structure_set import StructureSet

logger = logging.getLogger(__name__)

# Volume ratios are only meaningful for "Shared" relationships.
SHARED_RELATIONSHIP_TYPES = (
    'EQUAL', 'CONTAINS', 'WITHIN', 'PARTITIONED', 'PARTITIONS', 'OVERLAPS'
)


def _get_or_create_composite(
    structure_set: 'StructureSet',
    operator: str,
    roi_a: ROI_Type,
    roi_b: ROI_Type,
) -> CompositeStructure:
    """Find or create the composite for `roi_a operator roi_b`.

    Scans the structure set for a CompositeStructure with matching operator
    and operands before creating a new one via structure_boolean.  This lets
    the overlapping and non-overlapping calculators share composites (e.g.
    the Union) when both are called for the same structure pair.

    Args:
        structure_set: The StructureSet containing the operand structures.
        operator: 'UNION', 'INTERSECTION', or 'DIFFERENCE'.
        roi_a: ROI of the first operand.
        roi_b: ROI of the second operand.

    Returns:
        The existing or newly created CompositeStructure.
    """
    for structure in structure_set.structures.values():
        if not isinstance(structure, CompositeStructure):
            continue
        if (structure.operators == (operator,)
                and structure.operand_rois == (roi_a, roi_b)):
            return structure
    return structure_boolean(structure_set, f'{roi_a} {operator} {roi_b}')


class _VolumeRatioCalculator(MetricCalculator):
    """Shared calculation flow for the volume ratio calculators."""

    # Subclass configuration
    ratio_field = ''  # VolumeRatioMetrics field for the overall ratio
    per_region_ratio_field = ''  # Field for the per-region-pair ratios
    numerator_operator = ''  # 'INTERSECTION' or 'DIFFERENCE'
    numerator_volume_field = ''  # 'intersection_volume'/'difference_volume'
    numerator_regions_field = ''  # Per-region volumes field for the numerator
    equal_ratio = math.nan  # Ratio value for EQUAL relationships

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Volume ratios apply only to "Shared" relationships.

        Args:
            relationship: The spatial relationship to check.

        Returns:
            True for EQUAL, CONTAINS, WITHIN, PARTITIONED, PARTITIONS and
            OVERLAPS; False otherwise.
        """
        rel_type = relationship.relationship_type.relation_type
        return rel_type in SHARED_RELATIONSHIP_TYPES

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship,
        tolerance: Optional[float] = None,
        structure_set: Optional['StructureSet'] = None
    ) -> VolumeRatioMetrics:
        """Calculate the volume ratio for the given structure pair.

        Args:
            structure_a: First structure.
            structure_b: Second structure.
            relationship: StructureRelationship with type information.
            tolerance: Optional structure set tolerance, passed to the
                per-region volume calculation.
            structure_set: StructureSet containing the structures.  Required
                for non-EQUAL shared relationships (used to build the
                composite structures).

        Returns:
            The shared VolumeRatioMetrics instance for the relationship
            with this calculator's ratio fields filled in.

        Raises:
            ValueError: If structure_set is required but not supplied.
        """
        metrics = self._get_or_create_metrics(relationship)
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            setattr(metrics, self.ratio_field,
                    self.get_non_applicable_value())
            return metrics
        if relationship.relationship_type.relation_type == 'EQUAL':
            return self._apply_equal(structure_a, metrics)
        if structure_set is None:
            raise ValueError(
                f'{self.get_name()} requires the structure_set argument '
                'to build composite structures.'
            )
        return self._calculate_from_composites(structure_a, structure_b,
                                               metrics, structure_set,
                                               tolerance)

    @staticmethod
    def _get_or_create_metrics(
        relationship: StructureRelationship
    ) -> VolumeRatioMetrics:
        """Reuse the shared VolumeRatioMetrics on the relationship if present.

        Args:
            relationship: The relationship whose metrics container may hold
                an existing VolumeRatioMetrics.

        Returns:
            The existing shared instance, or a new empty one.
        """
        if relationship.metrics is not None:
            existing = relationship.metrics.volume_ratio
            if existing is not None:
                return existing
        return VolumeRatioMetrics()

    def _apply_equal(
        self,
        structure_a: StructureShape,
        metrics: VolumeRatioMetrics
    ) -> VolumeRatioMetrics:
        """Fill ratio values for EQUAL relationships without composites.

        Union = Intersection = V(A) = V(B); Difference = 0.  Per-region
        values reference the regions of structure A (which are identical to
        those of structure B).

        Args:
            structure_a: First structure (identical to the second).
            metrics: The shared VolumeRatioMetrics to fill.

        Returns:
            The filled VolumeRatioMetrics.
        """
        volume = structure_a.structure_volumes.physical
        metrics.intersection_volume = volume
        metrics.union_volume = volume
        metrics.difference_volume = 0.0
        setattr(metrics, self.ratio_field, self.equal_ratio)
        region_volumes = {
            str(region_index): region_volume
            for region_index, region_volume in
            structure_a.calculate_region_volumes('Physical').items()
        }
        metrics.per_region_intersection_volumes = region_volumes
        metrics.per_region_union_volumes = dict(region_volumes)
        metrics.per_region_difference_volumes = {
            region_index: 0.0 for region_index in region_volumes
        }
        setattr(metrics, self.per_region_ratio_field, {
            (region_index, region_index): self.equal_ratio
            for region_index in region_volumes
        })
        return metrics

    def _calculate_from_composites(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        metrics: VolumeRatioMetrics,
        structure_set: 'StructureSet',
        tolerance: Optional[float]
    ) -> VolumeRatioMetrics:
        """Calculate the ratio from composite structure volumes.

        The overall ratio uses the total physical volumes of the composite
        structures (not region aggregation, to avoid rounding and boundary
        issues).  Per-region ratios pair each region of the numerator
        composite with the union region of maximum spatial overlap.

        Args:
            structure_a: First structure.
            structure_b: Second structure.
            metrics: The shared VolumeRatioMetrics to fill.
            structure_set: StructureSet containing the structures.
            tolerance: Optional tolerance for per-region volume rounding.

        Returns:
            The filled VolumeRatioMetrics.
        """
        tolerance = tolerance if tolerance else 0.0
        union = _get_or_create_composite(structure_set, 'UNION',
                                         structure_a.roi, structure_b.roi)
        numerator = _get_or_create_composite(
            structure_set, self.numerator_operator,
            structure_a.roi, structure_b.roi)
        union_volume = union.structure_volumes.physical
        numerator_volume = numerator.structure_volumes.physical
        metrics.union_volume = self._round_volume(union_volume)
        setattr(metrics, self.numerator_volume_field,
                self._round_volume(numerator_volume))
        if union_volume > 0:
            ratio = numerator_volume / union_volume
        else:
            ratio = self.get_non_applicable_value()
        setattr(metrics, self.ratio_field, self._round_ratio(ratio))
        # Per-region values reference composite regions.
        union_regions = union.calculate_region_volumes('Physical', tolerance)
        numerator_regions = numerator.calculate_region_volumes('Physical',
                                                               tolerance)
        metrics.per_region_union_volumes = union_regions
        setattr(metrics, self.numerator_regions_field, numerator_regions)
        setattr(metrics, self.per_region_ratio_field,
                self._pair_region_ratios(numerator, numerator_regions,
                                         union, union_regions))
        return metrics

    def _pair_region_ratios(
        self,
        numerator: CompositeStructure,
        numerator_regions: Dict[str, float],
        denominator: CompositeStructure,
        denominator_regions: Dict[str, float]
    ) -> Dict[Tuple[str, str], float]:
        """Pair numerator regions to denominator regions by spatial overlap.

        For each region of the numerator composite, find the region of the
        denominator composite with the largest summed per-slice overlap area
        and record the ratio of the two per-region volumes for that pair.

        Args:
            numerator: The numerator composite (Intersection or Difference).
            numerator_regions: Per-region physical volumes of the numerator.
            denominator: The denominator composite (Union).
            denominator_regions: Per-region physical volumes of the
                denominator.

        Returns:
            Dict mapping (numerator_region, denominator_region) pairs to
            the region-pair ratio.
        """
        pairs: Dict[Tuple[str, str], float] = {}
        if not numerator_regions or not denominator_regions:
            return pairs
        overlap_areas = self._region_overlap_areas(numerator, denominator)
        for num_idx, num_volume in numerator_regions.items():
            denom_idx = self._best_overlap_match(num_idx, overlap_areas)
            if denom_idx is None:
                continue
            denom_volume = denominator_regions.get(denom_idx, 0.0)
            if denom_volume and denom_volume > 0:
                ratio = num_volume / denom_volume
            else:
                ratio = self.get_non_applicable_value()
            pairs[(str(num_idx), str(denom_idx))] = self._round_ratio(ratio)
        return pairs

    @staticmethod
    def _region_overlap_areas(
        numerator: CompositeStructure,
        denominator: CompositeStructure
    ) -> Dict[Tuple[str, str], float]:
        """Sum per-slice intersection areas between composite regions.

        Args:
            numerator: The numerator composite structure.
            denominator: The denominator composite structure.

        Returns:
            Dict mapping (numerator_region, denominator_region) to the
            total overlap area across all shared slices.
        """
        overlap: Dict[Tuple[str, str], float] = {}
        if numerator.region_table.empty or denominator.region_table.empty:
            return overlap
        slices = set(numerator.region_table['SliceIndex']) & set(
            denominator.region_table['SliceIndex'])
        for slice_index in slices:
            num_slice = numerator.get_slice(slice_index)
            denom_slice = denominator.get_slice(slice_index)
            if num_slice is None or denom_slice is None:
                continue
            for num_idx, num_poly in num_slice.regions.items():
                if num_poly is None or num_poly.is_empty:
                    continue
                for denom_idx, denom_poly in denom_slice.regions.items():
                    if denom_poly is None or denom_poly.is_empty:
                        continue
                    area = num_poly.intersection(denom_poly).area
                    if area > 0:
                        key = (num_idx, denom_idx)
                        overlap[key] = overlap.get(key, 0.0) + area
        return overlap

    @staticmethod
    def _best_overlap_match(
        num_idx: str,
        overlap_areas: Dict[Tuple[str, str], float]
    ) -> Optional[str]:
        """Find the denominator region with maximum overlap for a region.

        Args:
            num_idx: The numerator region index.
            overlap_areas: Per-pair overlap areas from
                _region_overlap_areas.

        Returns:
            The denominator region index with the largest overlap, or None
            when the numerator region overlaps nothing.
        """
        best_idx = None
        best_area = 0.0
        for (num, denom), area in overlap_areas.items():
            if num == num_idx and area > best_area:
                best_idx = denom
                best_area = area
        return best_idx

    def _round_volume(self, value: float) -> float:
        """Round a volume to the configured volume precision."""
        return round(value, self.config.volume_precision)

    def _round_ratio(self, value: Optional[float]) -> Optional[float]:
        """Round a ratio to the configured ratio precision, keeping NaN."""
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return value
        return round(value, self.config.ratio_precision)


@register_calculator
class OverlappingVolumeRatioCalculator(_VolumeRatioCalculator):
    """Intersection / Union volume ratio for shared-volume structures."""

    ratio_field = 'overlapping_ratio'
    per_region_ratio_field = 'per_region_overlapping_ratio'
    numerator_operator = 'INTERSECTION'
    numerator_volume_field = 'intersection_volume'
    numerator_regions_field = 'per_region_intersection_volumes'
    equal_ratio = 1.0

    def get_name(self) -> str:
        """Get calculator name."""
        return 'overlapping_volume_ratio'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'


@register_calculator
class NonOverlappingVolumeRatioCalculator(_VolumeRatioCalculator):
    """Difference (A - B) / Union volume ratio for shared-volume structures."""

    ratio_field = 'non_overlapping_ratio'
    per_region_ratio_field = 'per_region_non_overlapping_ratio'
    numerator_operator = 'DIFFERENCE'
    numerator_volume_field = 'difference_volume'
    numerator_regions_field = 'per_region_difference_volumes'
    equal_ratio = 0.0

    def get_name(self) -> str:
        """Get calculator name."""
        return 'non_overlapping_volume_ratio'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'
