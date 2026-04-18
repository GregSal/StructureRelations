"""Distance metric calculator for non-overlapping structures.

Distance measures the minimum 3D separation between two structures.
Applicable to:
- DISJOINT: No intersection or contact
- SHELTERS: Structure within convex hull (distance to actual boundary)

NOT applicable to:
- CONFINES (BORDERS_INTERIOR): N/A (distance within a structure is undefined)
- EQUAL: N/A (identical structures have no separation)
"""

import math
import logging
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
    1. Check all slice pairs (same slice and ±1 adjacent slices)
    2. For each slice pair, get boundary polygons
    3. Calculate 2D distance on slice
    4. Convert to 3D using slice height difference
    5. Track minimum across all region pairs and slice pairs

    Special cases:
    - CONFINES: Return N/A (distance undefined)
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
            True for DISJOINT, SHELTERS; False for CONFINES, EQUAL
        """
        rel_type = relationship.relationship_type.relation_type
        return rel_type in ['DISJOINT', 'SHELTERS']

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
        # Get all slice indices for each region
        slices_a = sorted(set(c[1] for c in contours_a))  # c[1] is slice_index
        slices_b = sorted(set(c[1] for c in contours_b))

        slice_pair_distances = {}

        # Check all slice pairs: same slice and adjacent slices (±1)
        for slice_a in slices_a:
            for slice_b in slices_b:
                # Only calculate if slices are same or adjacent
                slice_diff = abs(slice_a - slice_b)
                if slice_diff > structure_a.structure_set.slice_thickness * 1.5:
                    continue

                # Calculate distance for this slice pair
                dist = self._calculate_slice_pair_distance(
                    structure_a, structure_b, slice_a, slice_b
                )

                if dist is not None:
                    slice_pair_distances[(slice_a, slice_b)] = dist

        # Find minimum across all slice pairs
        if slice_pair_distances:
            min_distance = min(slice_pair_distances.values())
        else:
            min_distance = float('inf')

        return min_distance, slice_pair_distances

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
        region_slice_a = structure_a.get_region_slice(slice_a)
        region_slice_b = structure_b.get_region_slice(slice_b)

        if region_slice_a is None or region_slice_b is None:
            return None

        # Get boundary polygons
        poly_a = region_slice_a.select('contour')
        poly_b = region_slice_b.select('contour')

        if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
            return None

        # Calculate 2D distance on slice plane
        from shapely import boundary
        boundary_a = boundary(poly_a)
        boundary_b = boundary(poly_b)

        distance_2d = shapely_distance(boundary_a, boundary_b)

        # Calculate height difference between slices (in cm)
        height = abs(slice_a - slice_b)

        # Convert to 3D distance using Pythagorean theorem
        distance_3d = math.sqrt(height**2 + distance_2d**2)

        return round(distance_3d, self.config.distance_precision)
