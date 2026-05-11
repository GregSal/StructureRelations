"""Margin metric calculators for containment relationships.

Margins measure clearance distances inside containing structures. Applicable to:
- CONTAINS: Structure fully inside another
- SURROUNDS: Structure inside a hole of another
- SHELTERS: Structure within convex hull but not touching
- EQUAL: Special case, all margins are 0

This module implements two margin calculators:
- ContainmentMarginsCalculator: Both orthogonal and minimum clearance values together
  (name: 'minimum_margins', calculates both metrics in one pass)
- MaximumMarginCalculator: Largest clearance (Hausdorff-based)
"""

import logging
from typing import Dict, Tuple, Set

from shapely import Polygon, LineString
from shapely import centroid as shapely_centroid
from shapely import distance as shapely_distance
from shapely import hausdorff_distance as shapely_hausdorff

from structures import StructureShape
from relationships import StructureRelationship
from types_and_classes import SliceIndexType, ContourIndex
from metrics.base import MetricCalculator, register_calculator
from metrics.data_structures import MarginMetrics, MaximumMarginMetrics

logger = logging.getLogger(__name__)


@register_calculator
class ContainmentMarginsCalculator(MetricCalculator):
    """Calculate both orthogonal and minimum clearance distances together.

    Calculates:
    1. Orthogonal margins: Clearance in 6 directions (±X, ±Y, ±Z)
    2. Minimum margin: Single worst-case clearance across all directions

    These metrics are calculated together in one pass as they provide complementary
    information about containment relationships. Orthogonal margins show directional
    clearances, while minimum margin shows the bottleneck constraint.

    This is the recommended calculator for getting complete margin analysis.
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'minimum_margins'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if margins apply to this relationship."""
        rel_type = relationship.relationship_type.relation_type
        return rel_type in [
            'CONTAINS', 'SURROUNDS', 'SHELTERS', 'PARTITION', 'CONFINES',
            'EQUAL'
        ]

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> MarginMetrics:
        """Calculate both orthogonal and minimum margins for structure pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            relationship: Relationship with type information

        Returns:
            MarginMetrics with orthogonal_margins, minimum_margin, and per-region data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return MarginMetrics(
                orthogonal_margins={
                    'x_neg': na_value, 'x_pos': na_value,
                    'y_neg': na_value, 'y_pos': na_value,
                    'z_neg': na_value, 'z_pos': na_value,
                },
                minimum_margin=na_value,
            )

        # Special case: EQUAL relationship
        if relationship.relationship_type.relation_type == 'EQUAL':
            return MarginMetrics(
                orthogonal_margins={
                    'x_neg': 0.0, 'x_pos': 0.0,
                    'y_neg': 0.0, 'y_pos': 0.0,
                    'z_neg': 0.0, 'z_pos': 0.0,
                },
                minimum_margin=0.0,
            )

        # PARTITION and CONFINES have touching boundaries -> zero margins
        if relationship.relationship_type.relation_type in ['PARTITION', 'CONFINES']:
            return MarginMetrics(
                orthogonal_margins={
                    'x_neg': 0.0, 'x_pos': 0.0,
                    'y_neg': 0.0, 'y_pos': 0.0,
                    'z_neg': 0.0, 'z_pos': 0.0,
                },
                minimum_margin=0.0,
            )

        # Identify regions
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        self.logger.debug(
            'Calculating containment margins: %d regions in A, %d regions in B',
            len(regions_a), len(regions_b)
        )

        # Calculate per-region-pair, per-slice orthogonal margins
        per_region_slice_margins = {}
        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)
                per_region_slice_margins[region_pair] = (
                    self._calculate_region_pair_orthogonal_margins(
                        structure_a, structure_b,
                        contours_a, contours_b
                    )
                )

        # Calculate Z-direction margins
        z_margins = self._calculate_z_margins(
            structure_a, structure_b, regions_a, regions_b
        )

        # Aggregate orthogonal to per-region-pair summaries
        per_region_orthogonal = self._aggregate_orthogonal_to_per_region(
            per_region_slice_margins, z_margins
        )

        # Aggregate orthogonal to overall 3D summary (worst-case)
        aggregated_orthogonal, worst_case_ortho = self._aggregate_orthogonal_to_3d(
            per_region_orthogonal
        )

        # Calculate per-region-pair minimum margins
        per_region_minimum = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)

                # Calculate minimum margin for this region pair
                min_margin = self._calculate_region_pair_minimum_margin(
                    structure_a, structure_b, contours_a, contours_b
                )
                per_region_minimum[region_pair] = min_margin

        # Aggregate to overall minimum (worst-case across all region pairs)
        if per_region_minimum:
            overall_minimum = min(per_region_minimum.values())
            min_closest_pair = min(per_region_minimum.items(), key=lambda x: x[1])[0]
        else:
            overall_minimum = 0.0
            min_closest_pair = None

        # Return both metrics in single MarginMetrics object
        return MarginMetrics(
            orthogonal_margins=aggregated_orthogonal,
            minimum_margin=overall_minimum,
            per_region_orthogonal_margins=per_region_orthogonal,
            per_region_minimum_margin=per_region_minimum,
            slice_orthogonal_margins=per_region_slice_margins,
            worst_case_region_pair=min_closest_pair,
            worst_case_direction=worst_case_ortho['direction'],
            worst_case_slice=worst_case_ortho.get('slice'),
        )

    def _calculate_region_pair_orthogonal_margins(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        contours_a: Set[ContourIndex],
        contours_b: Set[ContourIndex]
    ) -> Dict[SliceIndexType, Dict[str, float]]:
        """Calculate per-slice orthogonal margins for one region pair."""
        slices_a = {contour[1] for contour in contours_a}
        slices_b = {contour[1] for contour in contours_b}
        common_slices = slices_a & slices_b

        per_slice_margins = {}

        for slice_idx in common_slices:
            region_slice_a = structure_a.get_slice(slice_idx)
            region_slice_b = structure_b.get_slice(slice_idx)

            if region_slice_a is None or region_slice_b is None:
                continue

            poly_a = region_slice_a.select('all')
            poly_b = region_slice_b.select('all')

            if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
                continue

            margins = self._calculate_slice_orthogonal_margins(poly_a, poly_b)
            per_slice_margins[slice_idx] = margins

        return per_slice_margins

    def _calculate_slice_orthogonal_margins(
        self,
        poly_a: Polygon,
        poly_b: Polygon
    ) -> Dict[str, float]:
        """Calculate orthogonal margins on a single slice."""
        centroid_b = shapely_centroid(poly_b)
        center_x, center_y = centroid_b.x, centroid_b.y

        minx, miny, maxx, maxy = poly_a.bounds
        margin = max(maxx - minx, maxy - miny) * 2

        lines = {
            'x_neg': LineString([(center_x - margin, center_y), (center_x, center_y)]),
            'x_pos': LineString([(center_x, center_y), (center_x + margin, center_y)]),
            'y_neg': LineString([(center_x, center_y - margin), (center_x, center_y)]),
            'y_pos': LineString([(center_x, center_y), (center_x, center_y + margin)]),
        }

        margins = {}
        for direction, line in lines.items():
            margins[direction] = self._calculate_directional_margin(
                line, poly_a, poly_b
            )

        return margins

    def _calculate_directional_margin(
        self,
        line: LineString,
        poly_a: Polygon,
        poly_b: Polygon
    ) -> float:
        """Calculate margin in one direction."""
        from utilities import make_solid
        exterior_a = make_solid(poly_a)
        exterior_b = make_solid(poly_b)

        try:
            line_outside_b = line.difference(exterior_b)
            line_between = line_outside_b.intersection(exterior_a)

            if line_between.is_empty:
                return 0.0

            return round(line_between.length, self.config.distance_precision)

        except (ValueError, AttributeError, TypeError) as e:
            self.logger.warning(
                'Error calculating directional margin: %s. Returning 0.', e
            )
            return 0.0

    def _calculate_z_margins(
        self,
        _structure_a: StructureShape,
        _structure_b: StructureShape,
        regions_a: Dict[int, Set],
        regions_b: Dict[int, Set]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Calculate Z-direction margins for each region pair."""
        z_margins = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                slices_a = sorted([c[1] for c in contours_a])
                slices_b = sorted([c[1] for c in contours_b])

                if not slices_a or not slices_b:
                    z_margins[(region_a_id, region_b_id)] = {'z_neg': 0.0, 'z_pos': 0.0}
                    continue

                z_neg_margin = slices_b[0] - slices_a[0]
                z_pos_margin = slices_a[-1] - slices_b[-1]

                z_margins[(region_a_id, region_b_id)] = {
                    'z_neg': round(abs(z_neg_margin), self.config.distance_precision),
                    'z_pos': round(abs(z_pos_margin), self.config.distance_precision),
                }

        return z_margins

    def _aggregate_orthogonal_to_per_region(
        self,
        per_region_slice_margins: Dict[Tuple[int, int], Dict[SliceIndexType, Dict[str, float]]],
        z_margins: Dict[Tuple[int, int], Dict[str, float]]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Aggregate per-slice orthogonal data to per-region-pair summaries."""
        per_region = {}

        for region_pair, slice_margins in per_region_slice_margins.items():
            if not slice_margins:
                continue

            directions = ['x_neg', 'x_pos', 'y_neg', 'y_pos']
            aggregated = {}

            for direction in directions:
                values = [
                    margins[direction]
                    for margins in slice_margins.values()
                    if direction in margins
                ]
                if values:
                    aggregated[direction] = min(values)
                else:
                    aggregated[direction] = 0.0

            if region_pair in z_margins:
                aggregated.update(z_margins[region_pair])
            else:
                aggregated['z_neg'] = 0.0
                aggregated['z_pos'] = 0.0

            per_region[region_pair] = aggregated

        return per_region

    def _aggregate_orthogonal_to_3d(
        self,
        per_region_orthogonal: Dict[Tuple[int, int], Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict]:
        """Aggregate per-region orthogonal data to overall 3D summary."""
        if not per_region_orthogonal:
            return {}, {'region_pair': None, 'direction': None}

        directions = ['x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos']
        aggregated = {}
        worst_case_region = None
        worst_case_direction = None
        worst_case_value = float('inf')

        for direction in directions:
            values = [
                (region_pair, margins[direction])
                for region_pair, margins in per_region_orthogonal.items()
                if direction in margins
            ]

            if values:
                min_pair = min(values, key=lambda x: x[1])
                aggregated[direction] = min_pair[1]

                if min_pair[1] < worst_case_value:
                    worst_case_value = min_pair[1]
                    worst_case_region = min_pair[0]
                    worst_case_direction = direction
            else:
                aggregated[direction] = 0.0

        worst_case_info = {
            'region_pair': worst_case_region,
            'direction': worst_case_direction,
        }

        return aggregated, worst_case_info

    def _calculate_region_pair_minimum_margin(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        contours_a: Set[ContourIndex],
        contours_b: Set[ContourIndex]
    ) -> float:
        """Calculate minimum margin for one region pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            contours_a: ContourIndex nodes for region A
            contours_b: ContourIndex nodes for region B

        Returns:
            Minimum margin distance
        """
        # Find common slices
        slices_a = {contour[1] for contour in contours_a}
        slices_b = {contour[1] for contour in contours_b}
        common_slices = slices_a & slices_b

        min_distances = []

        for slice_idx in common_slices:
            region_slice_a = structure_a.get_slice(slice_idx)
            region_slice_b = structure_b.get_slice(slice_idx)

            if region_slice_a is None or region_slice_b is None:
                continue

            poly_a = region_slice_a.select('all')
            poly_b = region_slice_b.select('all')

            if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
                continue

            # Distance from contained boundary to container boundary
            from shapely import boundary
            boundary_b = boundary(poly_b)
            boundary_a = boundary(poly_a)

            # Minimum distance between boundaries
            dist = shapely_distance(boundary_b, boundary_a)
            min_distances.append(dist)

        if min_distances:
            return round(min(min_distances), self.config.distance_precision)
        else:
            return 0.0


@register_calculator
class MaximumMarginCalculator(MetricCalculator):
    """Calculate largest clearance distance using Hausdorff distance.

    The maximum margin is the largest distance from any point on the contained
    structure to the nearest point on the container boundary.
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'maximum_margin'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if maximum margin applies to this relationship."""
        rel_type = relationship.relationship_type.relation_type
        return rel_type in [
            'CONTAINS', 'SURROUNDS', 'PARTITION', 'CONFINES', 'EQUAL'
        ]

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> MaximumMarginMetrics:
        """Calculate maximum margin for structure pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            relationship: Relationship with type information

        Returns:
            MaximumMarginMetrics with maximum_margin and per-region data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return MaximumMarginMetrics(maximum_margin=na_value)

        # Special case: EQUAL relationship
        if relationship.relationship_type.relation_type == 'EQUAL':
            return MaximumMarginMetrics(maximum_margin=0.0)

        # Identify regions
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        # Calculate per-region-pair maximum margins
        per_region_maximum = {}
        worst_region_pair = None
        max_overall = 0.0

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)

                max_margin = self._calculate_region_pair_maximum(
                    structure_a, structure_b, contours_a, contours_b
                )
                per_region_maximum[region_pair] = max_margin

                # Track the maximum
                if max_margin > max_overall:
                    max_overall = max_margin
                    worst_region_pair = region_pair

        return MaximumMarginMetrics(
            maximum_margin=max_overall,
            per_region_maximum_margin=per_region_maximum,
            worst_case_region_pair=worst_region_pair,
        )

    def _calculate_region_pair_maximum(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        contours_a: Set[ContourIndex],
        contours_b: Set[ContourIndex]
    ) -> float:
        """Calculate maximum margin for one region pair using Hausdorff distance.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            contours_a: ContourIndex nodes for region A
            contours_b: ContourIndex nodes for region B

        Returns:
            Maximum margin distance
        """
        # Find common slices
        slices_a = {contour[1] for contour in contours_a}
        slices_b = {contour[1] for contour in contours_b}
        common_slices = slices_a & slices_b

        max_distances = []

        for slice_idx in common_slices:
            region_slice_a = structure_a.get_slice(slice_idx)
            region_slice_b = structure_b.get_slice(slice_idx)

            if region_slice_a is None or region_slice_b is None:
                continue

            poly_a = region_slice_a.select('all')
            poly_b = region_slice_b.select('all')

            if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
                continue

            # Hausdorff distance between boundaries
            from shapely import boundary
            boundary_b = boundary(poly_b)
            boundary_a = boundary(poly_a)

            hausdorff_dist = shapely_hausdorff(boundary_b, boundary_a)
            max_distances.append(hausdorff_dist)

        if max_distances:
            return round(max(max_distances), self.config.distance_precision)
        else:
            return 0.0
