"""Margin metric calculators for containment relationships.

Margins measure clearance distances inside containing structures. Applicable to:
- CONTAINS: Structure fully inside another
- SURROUNDS: Structure inside a hole of another
- SHELTERS: Structure within convex hull but not touching
- EQUAL: Special case, all margins are 0

This module implements three margin calculators:
- OrthogonalMarginsCalculator: Clearance in 6 orthogonal directions (±X, ±Y, ±Z)
- MinimumMarginCalculator: Single worst-case clearance value
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
from metrics.data_structures import MarginMetrics

logger = logging.getLogger(__name__)


@register_calculator
class OrthogonalMarginsCalculator(MetricCalculator):
    """Calculate clearance in 6 orthogonal directions (±X, ±Y, ±Z).

    For each region pair on each slice:
    1. Get boundary polygons using RegionSlice.select('all')
    2. Find centroid of secondary (contained) region
    3. Generate orthogonal lines through centroid to container boundaries
    4. Calculate clearance distances in ±X and ±Y directions
    5. Store per-slice, per-region-pair results

    Z-direction margins are calculated from slice indices.
    Final aggregated margins represent worst-case across all region pairs.
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'orthogonal_margins'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if orthogonal margins apply to this relationship.

        Args:
            relationship: The spatial relationship

        Returns:
            True for CONTAINS, SURROUNDS, SHELTERS, EQUAL
        """
        rel_type = relationship.relationship_type.relation_type
        return rel_type in ['CONTAINS', 'SURROUNDS', 'SHELTERS', 'EQUAL']

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> MarginMetrics:
        """Calculate orthogonal margins for structure pair.

        Args:
            structure_a: Container/larger structure
            structure_b: Contained/smaller structure
            relationship: Relationship with type information

        Returns:
            MarginMetrics with per-region, per-slice, and aggregated data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            return self._create_non_applicable_result()

        # Special case: EQUAL relationship has all margins = 0
        if relationship.relationship_type.relation_type == 'EQUAL':
            return self._create_equal_result()

        # Identify regions in both structures
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        self.logger.debug(
            'Calculating orthogonal margins: %d regions in A, %d regions in B',
            len(regions_a), len(regions_b)
        )

        # Calculate per-region-pair, per-slice margins
        per_region_slice_margins = {}
        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)
                per_region_slice_margins[region_pair] = (
                    self._calculate_region_pair_margins(
                        structure_a, structure_b,
                        contours_a, contours_b
                    )
                )

        # Calculate Z-direction margins
        z_margins = self._calculate_z_margins(
            structure_a, structure_b, regions_a, regions_b
        )

        # Aggregate to per-region-pair summaries
        per_region_orthogonal = self._aggregate_to_per_region(
            per_region_slice_margins, z_margins
        )

        # Aggregate to overall 3D summary (worst-case)
        aggregated_margins, worst_case_info = self._aggregate_to_3d(
            per_region_orthogonal
        )

        return MarginMetrics(
            orthogonal_margins=aggregated_margins,
            per_region_orthogonal_margins=per_region_orthogonal,
            slice_orthogonal_margins=per_region_slice_margins,
            worst_case_region_pair=worst_case_info['region_pair'],
            worst_case_direction=worst_case_info['direction'],
            worst_case_slice=worst_case_info.get('slice'),
        )

    def _calculate_region_pair_margins(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        contours_a: Set[ContourIndex],
        contours_b: Set[ContourIndex]
    ) -> Dict[SliceIndexType, Dict[str, float]]:
        """Calculate per-slice margins for one region pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            contours_a: ContourIndex nodes for region A
            contours_b: ContourIndex nodes for region B

        Returns:
            Dict mapping slice_index -> {'x_neg', 'x_pos', 'y_neg', 'y_pos'}
        """
        # Find common slices where both regions exist
        slices_a = {contour[1] for contour in contours_a}  # contour[1] is slice_index
        slices_b = {contour[1] for contour in contours_b}
        common_slices = slices_a & slices_b

        per_slice_margins = {}

        for slice_idx in common_slices:
            # Get RegionSlice objects
            region_slice_a = structure_a.get_slice(slice_idx)
            region_slice_b = structure_b.get_slice(slice_idx)

            if region_slice_a is None or region_slice_b is None:
                continue

            # Get boundary polygons (including extrapolated boundaries)
            poly_a = region_slice_a.select('all')
            poly_b = region_slice_b.select('all')

            if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
                continue

            # Calculate orthogonal margins for this slice
            margins = self._calculate_slice_margins(poly_a, poly_b)
            per_slice_margins[slice_idx] = margins

        return per_slice_margins

    def _calculate_slice_margins(
        self,
        poly_a: Polygon,
        poly_b: Polygon
    ) -> Dict[str, float]:
        """Calculate orthogonal margins on a single slice.

        Args:
            poly_a: Container polygon
            poly_b: Contained polygon

        Returns:
            Dict with keys: 'x_neg', 'x_pos', 'y_neg', 'y_pos'
        """
        # Get centroid of contained structure
        centroid_b = shapely_centroid(poly_b)
        center_x, center_y = centroid_b.x, centroid_b.y

        # Get bounds of container
        minx, miny, maxx, maxy = poly_a.bounds

        # Create orthogonal lines through centroid to container boundaries
        # Lines extend well beyond container to ensure intersection
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
        """Calculate margin in one direction.

        The margin is the length of the line segment that lies:
        - Outside poly_b (exterior to contained structure)
        - Inside poly_a (interior to container)

        Args:
            line: Orthogonal line from centroid in one direction
            poly_a: Container polygon
            poly_b: Contained polygon

        Returns:
            Margin distance in this direction
        """
        # Get exterior-only polygons (no holes) for simpler calculation
        from utilities import make_solid
        exterior_a = make_solid(poly_a)
        exterior_b = make_solid(poly_b)

        # Find part of line outside poly_b
        try:
            line_outside_b = line.difference(exterior_b)

            # Find part of that which is inside poly_a
            line_between = line_outside_b.intersection(exterior_a)

            # Return length
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
        """Calculate Z-direction (through-plane) margins for each region pair.

        Args:
            _structure_a: Container structure (unused, signature for consistency)
            _structure_b: Contained structure (unused, signature for consistency)
            regions_a: Regions in structure A
            regions_b: Regions in structure B

        Returns:
            Dict mapping (region_a_id, region_b_id) -> {'z_neg', 'z_pos'}
        """
        z_margins = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                # Get slice indices for each region
                slices_a = sorted([c[1] for c in contours_a])  # c[1] is slice_index
                slices_b = sorted([c[1] for c in contours_b])

                if not slices_a or not slices_b:
                    z_margins[(region_a_id, region_b_id)] = {'z_neg': 0.0, 'z_pos': 0.0}
                    continue

                # Z margins: distance from extremes of B to extremes of A
                # z_neg (inferior): distance from bottom of B to bottom of A
                # z_pos (superior): distance from top of B to top of A
                z_neg_margin = slices_b[0] - slices_a[0]  # Both in cm
                z_pos_margin = slices_a[-1] - slices_b[-1]

                z_margins[(region_a_id, region_b_id)] = {
                    'z_neg': round(abs(z_neg_margin), self.config.distance_precision),
                    'z_pos': round(abs(z_pos_margin), self.config.distance_precision),
                }

        return z_margins

    def _aggregate_to_per_region(
        self,
        per_region_slice_margins: Dict[Tuple[int, int], Dict[SliceIndexType, Dict[str, float]]],
        z_margins: Dict[Tuple[int, int], Dict[str, float]]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Aggregate per-slice data to per-region-pair summaries.

        For each region pair, take minimum margin in each direction across all slices.

        Args:
            per_region_slice_margins: Per-slice margins for each region pair
            z_margins: Z-direction margins for each region pair

        Returns:
            Dict mapping (region_a_id, region_b_id) -> 6-direction margins dict
        """
        per_region = {}

        for region_pair, slice_margins in per_region_slice_margins.items():
            if not slice_margins:
                continue

            # Find minimum in each direction across all slices
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

            # Add Z-direction margins
            if region_pair in z_margins:
                aggregated.update(z_margins[region_pair])
            else:
                aggregated['z_neg'] = 0.0
                aggregated['z_pos'] = 0.0

            per_region[region_pair] = aggregated

        return per_region

    def _aggregate_to_3d(
        self,
        per_region_orthogonal: Dict[Tuple[int, int], Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict]:
        """Aggregate per-region data to overall 3D summary.

        Take worst-case (minimum) margin in each direction across all region pairs.

        Args:
            per_region_orthogonal: Per-region-pair margins

        Returns:
            Tuple of (aggregated_margins_dict, worst_case_info_dict)
        """
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
                # Find minimum (worst-case) for this direction
                min_pair = min(values, key=lambda x: x[1])
                aggregated[direction] = min_pair[1]

                # Track overall worst case
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

    def _create_non_applicable_result(self) -> MarginMetrics:
        """Create result for non-applicable relationship."""
        na_value = self.get_non_applicable_value()
        return MarginMetrics(
            orthogonal_margins={
                'x_neg': na_value, 'x_pos': na_value,
                'y_neg': na_value, 'y_pos': na_value,
                'z_neg': na_value, 'z_pos': na_value,
            },
            minimum_margin=na_value,
            maximum_margin=na_value,
        )

    def _create_equal_result(self) -> MarginMetrics:
        """Create result for EQUAL relationship (all margins are 0)."""
        return MarginMetrics(
            orthogonal_margins={
                'x_neg': 0.0, 'x_pos': 0.0,
                'y_neg': 0.0, 'y_pos': 0.0,
                'z_neg': 0.0, 'z_pos': 0.0,
            },
            minimum_margin=0.0,
            maximum_margin=0.0,
        )


@register_calculator
class MinimumMarginCalculator(MetricCalculator):
    """Calculate single worst-case clearance distance.

    Finds the minimum distance from any point on the contained structure
    to the boundary of the container. This is the smallest margin in any direction.
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'minimum_margin'

    def get_version(self) -> str:
        """Get calculator version."""
        return '1.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if minimum margin applies to this relationship."""
        rel_type = relationship.relationship_type.relation_type
        return rel_type in ['CONTAINS', 'SURROUNDS', 'SHELTERS', 'EQUAL']

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> MarginMetrics:
        """Calculate minimum margin for structure pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            relationship: Relationship with type information

        Returns:
            MarginMetrics with minimum_margin and per-region data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return MarginMetrics(minimum_margin=na_value)

        # Special case: EQUAL relationship
        if relationship.relationship_type.relation_type == 'EQUAL':
            return MarginMetrics(minimum_margin=0.0)

        # Identify regions
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        # Calculate per-region-pair minimum margins
        per_region_minimum = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)

                # Calculate minimum margin for this region pair
                min_margin = self._calculate_region_pair_minimum(
                    structure_a, structure_b, contours_a, contours_b
                )
                per_region_minimum[region_pair] = min_margin

        # Aggregate to overall minimum (worst-case across all region pairs)
        if per_region_minimum:
            overall_minimum = min(per_region_minimum.values())
            closest_pair = min(per_region_minimum.items(), key=lambda x: x[1])[0]
        else:
            overall_minimum = 0.0
            closest_pair = None

        return MarginMetrics(
            minimum_margin=overall_minimum,
            per_region_minimum_margin=per_region_minimum,
            worst_case_region_pair=closest_pair,
        )

    def _calculate_region_pair_minimum(
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
            # Get exterior of contained structure and boundary of container
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
        return rel_type in ['CONTAINS', 'SURROUNDS', 'EQUAL']

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> MarginMetrics:
        """Calculate maximum margin for structure pair.

        Args:
            structure_a: Container structure
            structure_b: Contained structure
            relationship: Relationship with type information

        Returns:
            MarginMetrics with maximum_margin and per-region data
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return MarginMetrics(maximum_margin=na_value)

        # Special case: EQUAL relationship
        if relationship.relationship_type.relation_type == 'EQUAL':
            return MarginMetrics(maximum_margin=0.0)

        # Identify regions
        regions_a = self._identify_regions(structure_a)
        regions_b = self._identify_regions(structure_b)

        # Calculate per-region-pair maximum margins
        per_region_maximum = {}

        for region_a_id, contours_a in regions_a.items():
            for region_b_id, contours_b in regions_b.items():
                region_pair = (region_a_id, region_b_id)

                max_margin = self._calculate_region_pair_maximum(
                    structure_a, structure_b, contours_a, contours_b
                )
                per_region_maximum[region_pair] = max_margin

        # Aggregate to overall maximum
        if per_region_maximum:
            overall_maximum = max(per_region_maximum.values())
        else:
            overall_maximum = 0.0

        return MarginMetrics(
            maximum_margin=overall_maximum,
            per_region_maximum_margin=per_region_maximum,
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
