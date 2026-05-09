"""Distance metric calculator for non-overlapping or touching structures.

Distance measures the minimum 3D separation between two structures.
Applicable to:
- DISJOINT: No intersection or contact (distance > 0)
- BORDERS: Touching at boundaries with no interior overlap (distance = 0)
- CONFINES: One structure confined within another with boundaries touching (distance = 0)
- SHELTERS: Structure within convex hull (distance to actual boundary)

NOT applicable to:
- EQUAL: N/A (identical structures have no separation)
"""

import math
import logging
from bisect import bisect_left
from typing import Dict, Tuple, Set, Optional

from shapely import distance as shapely_distance

from structures import StructureShape
from relationships import StructureRelationship
from types_and_classes import SliceIndexType, ContourIndex
from metrics.base import MetricCalculator, register_calculator
from metrics.data_structures import DistanceMetrics

logger = logging.getLogger(__name__)


@register_calculator
class MinimumDistanceCalculator(MetricCalculator):
    """Calculate minimum 3D distance between structures.

    For each region pair:
    1. Build candidate slice pairs using contour_lookup metadata
    2. Prioritize nearest-Z slice pairs
    3. Expand candidates without any fixed maximum Z-gap cutoff
    4. Prune only when Z-gap exceeds current best 3D distance
    2. For each slice pair, get boundary polygons
    3. Calculate 2D distance on slice
    4. Convert to 3D using slice height difference
    5. Track minimum across all region pairs and slice pairs

    Special cases:
    - BORDERS: Return 0 (structures are touching at boundaries)
    - CONFINES: Return 0 (structures confined with boundaries touching)
    - EQUAL: Return N/A (no separation)
    - DISJOINT: True 3D minimum distance
    - SHELTERS: Distance from contained to actual boundary (not hull)
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'minimum_distance'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if minimum distance applies to this relationship.

        Args:
            relationship: The spatial relationship

        Returns:
            True for DISJOINT, SHELTERS, BORDERS, CONFINES; False for EQUAL
        """
        rel_type = relationship.relationship_type.relation_type
        return rel_type in ['DISJOINT', 'SHELTERS', 'BORDERS', 'CONFINES']

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> DistanceMetrics:
        """Calculate minimum distance for structure pair.

        Args:
            structure_a: First structure
            structure_b: Second structure
            relationship: Relationship with type information

        Returns:
            DistanceMetrics with minimum_distance and per-region data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return DistanceMetrics(minimum_distance=na_value)

        # Identify regions in both structures
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        self.logger.debug(
            'Calculating minimum distance: %d regions in A, %d regions in B',
            len(regions_a), len(regions_b)
        )

        # Calculate per-region-pair distances
        per_region_distances = {}
        per_region_slice_distances = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)

                # Calculate minimum distance for this region pair
                min_dist, slice_dists = self._calculate_region_pair_distance(
                    structure_a, structure_b, contours_a, contours_b
                )

                per_region_distances[region_pair] = min_dist
                per_region_slice_distances[region_pair] = slice_dists

        # Aggregate to overall minimum
        if per_region_distances:
            overall_minimum = min(per_region_distances.values())
            closest_pair = min(per_region_distances.items(), key=lambda x: x[1])[0]

            # Find which slice has minimum distance
            closest_slice = None
            for region_pair, slice_dists in per_region_slice_distances.items():
                if region_pair == closest_pair and slice_dists:
                    closest_slice = min(slice_dists.items(), key=lambda x: x[1])[0]
                    break
        else:
            overall_minimum = float('inf')
            closest_pair = None
            closest_slice = None

        return DistanceMetrics(
            minimum_distance=overall_minimum,
            per_region_minimum_distance=per_region_distances,
            slice_distances=per_region_slice_distances,
            closest_region_pair=closest_pair,
            closest_slice=closest_slice,
        )

    def _calculate_region_pair_distance(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        contours_a: Set[ContourIndex],
        contours_b: Set[ContourIndex]
    ) -> Tuple[float, Dict[Tuple[SliceIndexType, SliceIndexType], float]]:
        """Calculate minimum distance for one region pair.

        Args:
            structure_a: First structure
            structure_b: Second structure
            contours_a: ContourIndex nodes for region A
            contours_b: ContourIndex nodes for region B

        Returns:
            Tuple of (minimum_distance, per_slice_pair_distances)
        """
        slices_a = self._get_region_slices(structure_a, contours_a)
        slices_b = self._get_region_slices(structure_b, contours_b)

        if not slices_a or not slices_b:
            return float('inf'), {}

        slice_pair_distances = {}
        best_distance = float('inf')

        # Seed the search with nearest-Z pairs to establish an initial bound.
        for slice_a in slices_a:
            for slice_b in self._nearest_target_slices(slice_a, slices_b):
                pair = (slice_a, slice_b)
                if pair in slice_pair_distances:
                    continue

                dist = self._calculate_slice_pair_distance(
                    structure_a, structure_b, slice_a, slice_b
                )

                if dist is None:
                    continue

                slice_pair_distances[pair] = dist
                if dist < best_distance:
                    best_distance = dist

        # Expand candidate pairs with no fixed Z-gap limit.
        # Safe pruning: if z_gap > best_distance, 3D distance cannot improve.
        for slice_a in slices_a:
            for slice_b in self._expand_candidate_slices(
                source_slice=slice_a,
                target_slices=slices_b,
                max_z_gap=best_distance,
            ):
                pair = (slice_a, slice_b)
                if pair in slice_pair_distances:
                    continue

                if abs(slice_a - slice_b) > best_distance:
                    continue

                # Calculate distance for this slice pair
                dist = self._calculate_slice_pair_distance(
                    structure_a, structure_b, slice_a, slice_b
                )

                if dist is not None:
                    slice_pair_distances[pair] = dist
                    if dist < best_distance:
                        best_distance = dist

        # Find minimum across all slice pairs
        if slice_pair_distances:
            min_distance = min(slice_pair_distances.values())
        else:
            min_distance = float('inf')

        return min_distance, slice_pair_distances

    def _get_region_slices(
        self,
        structure: StructureShape,
        contours: Set[ContourIndex],
    ) -> list[SliceIndexType]:
        """Get sorted slice indices for a region using contour lookup data."""
        contour_lookup = structure.contour_lookup

        if contour_lookup.empty:
            return sorted({contour[1] for contour in contours})

        labels = set(contours)
        region_rows = contour_lookup.loc[contour_lookup['Label'].isin(labels)]

        if region_rows.empty:
            return sorted({contour[1] for contour in contours})

        return sorted(region_rows['SliceIndex'].unique())

    def _nearest_target_slices(
        self,
        source_slice: SliceIndexType,
        target_slices: list[SliceIndexType],
    ) -> list[SliceIndexType]:
        """Return closest target slice(s) to source_slice in Z."""
        if not target_slices:
            return []

        insert_at = bisect_left(target_slices, source_slice)
        candidates = []

        if insert_at < len(target_slices):
            candidates.append(target_slices[insert_at])
        if insert_at > 0:
            candidates.append(target_slices[insert_at - 1])

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(candidates))

    def _expand_candidate_slices(
        self,
        source_slice: SliceIndexType,
        target_slices: list[SliceIndexType],
        max_z_gap: float,
    ) -> list[SliceIndexType]:
        """Expand target candidates from nearest to farthest in Z.

        Args:
            source_slice: Slice from structure A.
            target_slices: Sorted slices from structure B.
            max_z_gap: Maximum Z-gap to include. Use inf for no pruning.

        Returns:
            Ordered list of target slices, nearest-first.
        """
        if not target_slices:
            return []

        insert_at = bisect_left(target_slices, source_slice)
        left = insert_at - 1
        right = insert_at
        ordered = []

        while left >= 0 or right < len(target_slices):
            left_gap = (
                abs(source_slice - target_slices[left])
                if left >= 0 else float('inf')
            )
            right_gap = (
                abs(source_slice - target_slices[right])
                if right < len(target_slices) else float('inf')
            )

            use_left = left_gap <= right_gap
            candidate_gap = left_gap if use_left else right_gap

            if candidate_gap > max_z_gap:
                # target_slices are visited in non-decreasing Z-gap.
                break

            if use_left:
                ordered.append(target_slices[left])
                left -= 1
            else:
                ordered.append(target_slices[right])
                right += 1

        if math.isinf(max_z_gap):
            seen = set(ordered)
            for target_slice in target_slices:
                if target_slice not in seen:
                    ordered.append(target_slice)

        return ordered

    def _calculate_slice_pair_distance(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        slice_a: SliceIndexType,
        slice_b: SliceIndexType
    ) -> Optional[float]:
        """Calculate 3D distance between contours on two slices.

        Args:
            structure_a: First structure
            structure_b: Second structure
            slice_a: Slice index for structure A
            slice_b: Slice index for structure B

        Returns:
            3D distance, or None if calculation fails
        """
        # Get RegionSlice objects
        region_slice_a = structure_a.get_slice(slice_a)
        region_slice_b = structure_b.get_slice(slice_b)

        if region_slice_a is None or region_slice_b is None:
            return None

        # Get boundary polygons (including extrapolated boundaries)
        poly_a = region_slice_a.select('all')
        poly_b = region_slice_b.select('all')

        if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
            return None

        # Calculate 2D distance on slice plane
        # For solid structures, use filled polygons (not boundaries)
        # If polygons overlap in 2D projection, distance_2d = 0
        distance_2d = shapely_distance(poly_a, poly_b)

        # Calculate height difference between slices (in cm)
        height = abs(slice_a - slice_b)

        # Convert to 3D distance using Pythagorean theorem
        distance_3d = math.sqrt(height**2 + distance_2d**2)

        return round(distance_3d, self.config.distance_precision)
