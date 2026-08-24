"""Margin metric calculator for containment relationships.

Margins measure clearance distances inside containing structures. Applicable to:
- CONTAINS: Structure fully inside another
- PARTITIONED: Structure inside another, touching the boundary
- SURROUNDS: Structure inside a hole of another
- SHELTERS: Structure within convex hull but not touching
- CONFINES: Structure inside a hole of another, touching the boundary
- EQUAL: Special case, all margins are 0

Margins are not defined for OVERLAPS, BORDERS, or DISJOINT relationships
(NaN is returned).

This module implements the ContainmentMarginsCalculator (registered name:
'minimum_margins') which calculates the orthogonal margins (+/-X, +/-Y, +/-Z)
and the minimum margin in a single pass:

1. Orthogonal planar (slice based) margins: For each slice containing both
   regions of a region pair, orthogonal lines are cast from every coordinate
   point of the inner region's contour(s) to the extent of the outer region in
   the +/-X and +/-Y directions.  Each line is cut at its closest intersection
   with the outer region's boundary; the length of the resulting line is the
   margin for that point.  The length is NaN if the line does not intersect
   the outer region, which can only happen when the inner structure is
   SHELTERED by the outer structure.
2. Orthogonal Z margins: A slice walk is performed starting from each slice of
   the inner region (the reference slice, including pseudo-boundary slices)
   towards the slices of the outer region in the negative/positive Z
   directions (the test slices).  A Z margin is triggered when the 2D
   relationship between the outer region on the test slice and the inner
   region on the reference slice changes to one where the inner region is no
   longer fully contained.  The change is detected with DE-9IM mask tests that
   depend on the relationship class of the reference slice (see
   Z_MARGIN_CHANGE_TESTS).
3. Minimum margin: The minimum 3D distance between the contours of the two
   regions.  It is seeded with the planar minimum distances on slices
   containing both regions and updated during the Z-margin walk using
   sqrt(d_planar**2 + d_slice**2) for each reference/test slice pair.

In the Z (slice) direction, structures are assumed to extend 1/2 a slice
thickness beyond the last slice containing a contour.  This pseudo-boundary
contour is assumed to be half the size of the last original contour (linearly
scaled down).  Margins include these pseudo-boundary contours in their
calculations.
"""

import math
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import shapely
from shapely import LineString, Point
from shapely import boundary as shapely_boundary
from shapely import distance as shapely_distance
from shapely import get_coordinates as shapely_get_coordinates

from structures import StructureShape
from relationships import StructureRelationship
from relations import DE27IM, DE9IM, AmbiguousRelationshipError
from region_slice import RegionSlice
from utilities import make_multi
from types_and_classes import SliceIndexType, RegionIndex
from metrics.base import MetricCalculator, register_calculator
from metrics.data_structures import MarginMetrics

logger = logging.getLogger(__name__)

# Relationship types for which margins are defined (A is the outer structure).
MARGIN_RELATIONSHIP_TYPES = [
    'CONTAINS', 'PARTITIONED', 'SURROUNDS', 'SHELTERS', 'CONFINES', 'EQUAL'
]

# Relationship types where the inner structure sits in a hole or cavity of
# the outer structure.  For these, a Z margin is only defined in a direction
# if the Z-margin walk finds the cavity closing (a containment trigger);
# otherwise the direction opens to the exterior and the margin is NaN.
HOLE_RELATIONSHIP_TYPES = ['SURROUNDS', 'SHELTERS']

# Relationship types where the boundaries touch, so the minimum margin is 0.
ZERO_MINIMUM_RELATIONSHIP_TYPES = ['PARTITIONED', 'CONFINES']

PLANAR_DIRECTIONS = ['x_neg', 'x_pos', 'y_neg', 'y_pos']
Z_DIRECTIONS = ['z_neg', 'z_pos']
ALL_DIRECTIONS = PLANAR_DIRECTIONS + Z_DIRECTIONS

# DE-9IM masks used to detect a change in relationship on the Z-margin walk.
# The mask applied depends on the relationship class of the reference slice.
# A change is triggered when any masked cell of the cross-slice DE-9IM
# between the outer region on the test slice and the inner region on the
# reference slice is True (i.e. the inner region is no longer fully contained).
# Bit order matches shapely.relate: II, IB, IE, BI, BB, BE, EI, EB, EE with
# the II cell as the most significant bit of the 9-bit integer.
Z_MARGIN_CHANGE_TESTS = {
    # Contains: inner touches the boundary (B-B) or pokes out (E(A)-I(B)).
    'CONTAINS': 0b000010100,
    # Partitioned: inner pokes out (E(A)-I(B)).
    'PARTITIONED': 0b000000100,
    # Surrounds/Shelters: outer material enters the inner region (I-I) or the
    # cavity wall touches the inner region (B-B).
    'SURROUNDS': 0b100010000,
    'SHELTERS': 0b100010000,
    # Confines: outer material enters the inner region (I-I).
    'CONFINES': 0b100000000,
}

# Type alias for region pair keys: (region_index_a, region_index_b).
RegionPair = Tuple[RegionIndex, RegionIndex]


def nanmin(values) -> float:
    """Return the minimum of the values, ignoring NaN.

    Args:
        values: Iterable of float values, possibly containing NaN.

    Returns:
        float: The minimum of the non-NaN values, or NaN if there are none.
    """
    valid = [value for value in values
             if value is not None and not math.isnan(value)]
    if not valid:
        return math.nan
    return min(valid)


def safe_distance(geom_a, geom_b) -> float:
    """Distance between two geometries, returning NaN on GEOS failures.

    Near-degenerate polygons (e.g. the tiny tangent-slice contour at the tip
    of a sphere, with many coincident points) cause GEOS to emit a
    RuntimeWarning and return NaN.  The warning is suppressed here; the NaN
    result is handled by the callers (such pairs never set the minimum).

    Args:
        geom_a: First shapely geometry.
        geom_b: Second shapely geometry.

    Returns:
        float: The distance between the geometries, or NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return shapely_distance(geom_a, geom_b)


@register_calculator
class ContainmentMarginsCalculator(MetricCalculator):
    """Calculate orthogonal and minimum clearance distances together.

    Calculates:
    1. Orthogonal margins: Clearance in 6 directions (+/-X, +/-Y, +/-Z)
    2. Minimum margin: Single worst-case clearance across all directions

    These metrics are calculated together in one pass over the slice pairs as
    they provide complementary information about containment relationships and
    share the same slice iteration.  Orthogonal margins show directional
    clearances, while minimum margin shows the bottleneck constraint.

    This is the recommended calculator for getting complete margin analysis.
    """

    def get_name(self) -> str:
        """Get calculator name."""
        return 'minimum_margins'

    def get_version(self) -> str:
        """Get calculator version."""
        return '2.0.0'

    def is_applicable(self, relationship: StructureRelationship) -> bool:
        """Check if margins apply to this relationship."""
        rel_type = relationship.relationship_type.relation_type
        return rel_type in MARGIN_RELATIONSHIP_TYPES

    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship,
        tolerance: Optional[float] = None
    ) -> MarginMetrics:
        """Calculate orthogonal and minimum margins for a structure pair.

        Args:
            structure_a: Container (outer) structure.
            structure_b: Contained (inner) structure.
            relationship: Relationship with type information.
            tolerance: Structure set tolerance.  When supplied and positive,
                margin values are rounded to the number of decimals implied
                by the tolerance; otherwise config distance precision is used.

        Returns:
            MarginMetrics with orthogonal_margins, minimum_margin, per-region
            summaries, per-slice summaries and detailed value tables.
        """
        if not self.is_applicable(relationship):
            self._warn_non_applicable(relationship.relationship_type)
            na_value = self.get_non_applicable_value()
            return MarginMetrics(
                orthogonal_margins={
                    direction: na_value for direction in ALL_DIRECTIONS
                },
                minimum_margin=na_value,
            )

        # Special case: EQUAL relationship -> all margins are 0.
        if relationship.relationship_type.relation_type == 'EQUAL':
            return MarginMetrics(
                orthogonal_margins={
                    direction: 0.0 for direction in ALL_DIRECTIONS
                },
                minimum_margin=0.0,
            )

        decimals = self._rounding_decimals(tolerance)

        # Per-slice, per-region-pair relationships and merged pair relations.
        pair_relations, slice_relations = self._collect_region_relations(
            structure_a, structure_b, tolerance
        )

        # Classify each per-slice region-pair relation once.
        slice_classes = {
            key: self._classify(slice_de27im)
            for key, slice_de27im in slice_relations.items()
        }

        per_region_orthogonal: Dict[RegionPair, Dict[str, float]] = {}
        per_region_minimum: Dict[RegionPair, float] = {}
        slice_orthogonal = {}
        slice_minimum = {}

        for pair, pair_de27im in pair_relations.items():
            pair_type = self._classify(pair_de27im)
            if pair_type not in MARGIN_RELATIONSHIP_TYPES:
                continue
            if pair_type == 'EQUAL':
                per_region_orthogonal[pair] = {
                    direction: 0.0 for direction in ALL_DIRECTIONS
                }
                per_region_minimum[pair] = 0.0
                continue

            region_idx_a, region_idx_b = pair
            polys_a = self._region_slice_map(structure_a, region_idx_a)
            polys_b = self._region_slice_map(structure_b, region_idx_b)
            if not polys_a or not polys_b:
                continue

            # Orthogonal planar margins for slices containing both regions.
            common_slices = sorted(set(polys_a) & set(polys_b))
            planar_records = {}
            for slice_index in common_slices:
                planar_records[(slice_index, slice_index)] = (
                    self._planar_slice_margins(
                        polys_a[slice_index], polys_b[slice_index], decimals
                    )
                )

            # Seed the minimum margin with the planar distances on the
            # slices containing both regions.
            minimum, min_records = self._seed_minimum_margins(
                common_slices, polys_a, polys_b, decimals
            )

            # Z-margin walk (also updates the minimum margin).
            slices_a = self._valid_region_slices(structure_a, region_idx_a,
                                                 polys_a)
            slices_b = self._valid_region_slices(structure_b, region_idx_b,
                                                 polys_b)
            z_margins, minimum, z_records, walk_records = (
                self._z_margin_walk(
                    pair, pair_type, polys_a, polys_b, slices_a, slices_b,
                    slice_classes, minimum, decimals
                )
            )

            if pair_type in ZERO_MINIMUM_RELATIONSHIP_TYPES:
                # Boundaries touch -> minimum margin is 0.
                minimum = 0.0

            # Per-region-pair orthogonal summary.
            pair_orthogonal = {}
            for direction in PLANAR_DIRECTIONS:
                pair_orthogonal[direction] = self._round(
                    nanmin(margins[direction]
                           for margins in planar_records.values()),
                    decimals
                )
            pair_orthogonal.update(z_margins)

            per_region_orthogonal[pair] = pair_orthogonal
            per_region_minimum[pair] = self._round(minimum, decimals)

            pair_records = dict(planar_records)
            for slice_pair, z_values in z_records.items():
                pair_records.setdefault(slice_pair, {}).update(z_values)
            slice_orthogonal[pair] = pair_records
            min_records.update(walk_records)
            slice_minimum[pair] = min_records

        if not per_region_orthogonal:
            self.logger.warning(
                'No valid region pairs found for margin calculation.')
            na_value = self.get_non_applicable_value()
            return MarginMetrics(
                orthogonal_margins={
                    direction: na_value for direction in ALL_DIRECTIONS
                },
                minimum_margin=na_value,
            )

        # Per-reference-slice summaries (minimum across region pairs and
        # test slices).
        per_slice_orthogonal = self._summarize_slice_orthogonal(
            slice_orthogonal)
        per_slice_minimum = self._summarize_slice_minimum(slice_minimum)

        # Final (3D) aggregated values: minimum across all region pairs.
        orthogonal_margins = {
            direction: nanmin(margins[direction]
                              for margins in per_region_orthogonal.values())
            for direction in ALL_DIRECTIONS
        }
        minimum_margin = nanmin(list(per_region_minimum.values()))

        # Traceability metadata.
        closest_pair = None
        if per_region_minimum:
            closest_pair = min(
                per_region_minimum.items(),
                key=lambda item: (math.isnan(item[1]), item[1])
            )[0]
        worst_direction, worst_slice = self._find_worst_case_orthogonal(
            per_region_orthogonal, slice_orthogonal
        )

        return MarginMetrics(
            orthogonal_margins=orthogonal_margins,
            minimum_margin=minimum_margin,
            per_region_orthogonal_margins=per_region_orthogonal,
            per_region_minimum_margin=per_region_minimum,
            per_slice_orthogonal_margins=per_slice_orthogonal,
            per_slice_minimum_margin=per_slice_minimum,
            slice_orthogonal_margins=slice_orthogonal,
            slice_minimum_margins=slice_minimum,
            worst_case_region_pair=closest_pair,
            worst_case_direction=worst_direction,
            worst_case_slice=worst_slice,
        )

    # ------------------------------------------------------------------
    # Relationship collection helpers
    # ------------------------------------------------------------------

    def _collect_region_relations(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        tolerance: Optional[float]
    ) -> Tuple[Dict[RegionPair, DE27IM],
               Dict[Tuple[SliceIndexType, RegionPair], DE27IM]]:
        """Collect merged and per-slice region-pair DE-27IM relationships.

        Performs a single relate_to pass with a slice callback.  Per-slice
        region-pair relations are copied so that the merge into the per-pair
        totals does not mutate the per-slice records.

        Args:
            structure_a: Container (outer) structure.
            structure_b: Contained (inner) structure.
            tolerance: Geometric tolerance for the relationship calculation.

        Returns:
            Tuple of (pair_relations, slice_relations) where pair_relations
            maps each region pair to its merged DE27IM and slice_relations
            maps (slice_index, region_pair) to the per-slice DE27IM.
        """
        pair_relations: Dict[RegionPair, DE27IM] = {}
        slice_relations: Dict[Tuple[SliceIndexType, RegionPair], DE27IM] = {}

        def accumulate(slice_index, _relation, _region_self, _region_other,
                       region_relations):
            if not region_relations:
                return
            for pair, slice_de27im in region_relations.items():
                slice_relations[(slice_index, pair)] = DE27IM(
                    relation_int=slice_de27im.int)
                if pair in pair_relations:
                    pair_relations[pair].merge(slice_de27im)
                else:
                    pair_relations[pair] = DE27IM(
                        relation_int=slice_de27im.int)

        structure_a.relate_to(
            structure_b,
            tolerance=tolerance or 0.0,
            slice_relation_callback=accumulate,
        )
        return pair_relations, slice_relations

    @staticmethod
    def _classify(de27im: DE27IM) -> Optional[str]:
        """Identify the relationship type string of a DE-27IM relation.

        Args:
            de27im: The DE27IM relation to classify.

        Returns:
            The relationship type string (e.g. 'CONTAINS'), or None if the
            relation is ambiguous or unknown.
        """
        try:
            relation_type = de27im.identify_relation()
        except AmbiguousRelationshipError:
            return None
        if relation_type is None:
            return None
        return relation_type.relation_type

    # ------------------------------------------------------------------
    # Region geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _region_polygon(region_slice: RegionSlice,
                        region_index: RegionIndex
                        ) -> Optional[shapely.MultiPolygon]:
        """Get the polygon of one region on a slice.

        Combines the region geometry with any pseudo-boundary geometry for
        the region on the slice.

        Args:
            region_slice: The RegionSlice for the slice.
            region_index: The RegionIndex of the region.

        Returns:
            The region polygon (Polygon or MultiPolygon), or None if the
            region has no geometry on the slice.
        """
        if region_slice is None or not hasattr(region_slice, 'regions'):
            return None
        parts = []
        region = region_slice.regions.get(region_index)
        if region is not None and not region.is_empty:
            parts.append(region)
        boundary = region_slice.boundaries.get(region_index)
        if boundary is not None and not boundary.is_empty:
            parts.append(boundary)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return make_multi(shapely.union_all(parts))

    def _region_slice_map(
        self,
        structure: StructureShape,
        region_index: RegionIndex
    ) -> Dict[SliceIndexType, shapely.MultiPolygon]:
        """Map slice indexes to region polygons for all slices with geometry.

        Args:
            structure: The structure to analyze.
            region_index: The RegionIndex of the region.

        Returns:
            Dict mapping slice index to the region's polygon on that slice.
        """
        slice_map = {}
        for _, row in structure.region_table.iterrows():
            if row['Empty']:
                continue
            poly = self._region_polygon(row['RegionSlice'], region_index)
            if poly is None or poly.is_empty or not poly.area > 0:
                # Skip empty and degenerate (zero-area or NaN) slices.
                continue
            slice_map[row['SliceIndex']] = poly
        return slice_map

    @staticmethod
    def _valid_region_slices(
        structure: StructureShape,
        region_index: RegionIndex,
        slice_map: Dict[SliceIndexType, shapely.MultiPolygon]
    ) -> List[SliceIndexType]:
        """Get the sorted valid slice indexes containing a region.

        Valid slices are non-interpolated (original) slices and
        pseudo-boundary slices, matching the slice positions used for
        relationship calculation.  Interior fill-in interpolated slices are
        excluded because their geometry may be incorrect.

        Args:
            structure: The structure to analyze.
            region_index: The RegionIndex of the region.
            slice_map: Slice index to polygon map for the region.

        Returns:
            Sorted list of valid slice indexes for the region.
        """
        table = structure.region_table
        valid_mask = (~table.Interpolated | table.IsBoundary) & ~table.Empty
        valid_slices = table.loc[valid_mask, 'SliceIndex']
        return sorted(slice_index for slice_index in valid_slices
                      if slice_index in slice_map)

    # ------------------------------------------------------------------
    # Orthogonal planar (slice based) margins
    # ------------------------------------------------------------------

    def _planar_slice_margins(
        self,
        poly_a: shapely.MultiPolygon,
        poly_b: shapely.MultiPolygon,
        decimals: int
    ) -> Dict[str, float]:
        """Calculate orthogonal planar margins on a single slice.

        Casts orthogonal lines from every coordinate point of the inner
        region's contour(s) to the extent of the outer region in the +/-X and
        +/-Y directions.  Each line is cut at its closest intersection with
        the outer region's boundary; the length of the resulting line is the
        margin for that point (NaN if the line does not intersect the outer
        region).

        Args:
            poly_a: The outer region's polygon on the slice.
            poly_b: The inner region's polygon on the slice.
            decimals: Number of decimals used to round the slice margins.

        Returns:
            Dict mapping direction ('x_neg', 'x_pos', 'y_neg', 'y_pos') to the
            minimum margin in that direction for the slice.
        """
        boundary_a = shapely_boundary(poly_a)

        minx, miny, maxx, maxy = poly_a.bounds
        bxmin, bymin, bxmax, bymax = poly_b.bounds
        minx, maxx = min(minx, bxmin), max(maxx, bxmax)
        miny, maxy = min(miny, bymin), max(maxy, bymax)
        # Ray length is the diagonal of the combined extent, so every
        # boundary point of the outer region is reachable from every point.
        extent = 2 * (math.hypot(maxx - minx, maxy - miny) or 1.0)

        direction_values: Dict[str, List[float]] = {
            direction: [] for direction in PLANAR_DIRECTIONS
        }
        for x_coord, y_coord in shapely_get_coordinates(poly_b):
            origin = Point(x_coord, y_coord)
            rays = {
                'x_neg': LineString([(x_coord, y_coord),
                                     (x_coord - extent, y_coord)]),
                'x_pos': LineString([(x_coord, y_coord),
                                     (x_coord + extent, y_coord)]),
                'y_neg': LineString([(x_coord, y_coord),
                                     (x_coord, y_coord - extent)]),
                'y_pos': LineString([(x_coord, y_coord),
                                     (x_coord, y_coord + extent)]),
            }
            for direction, ray in rays.items():
                crossings = shapely.intersection(ray, boundary_a)
                if crossings.is_empty:
                    # No intersection with the outer structure in this
                    # direction (only possible for SHELTERS).
                    continue
                direction_values[direction].append(
                    safe_distance(origin, crossings))

        return {
            direction: self._round(nanmin(values), decimals)
            for direction, values in direction_values.items()
        }

    # ------------------------------------------------------------------
    # Minimum margin seeding
    # ------------------------------------------------------------------

    def _seed_minimum_margins(
        self,
        common_slices: List[SliceIndexType],
        polys_a: Dict[SliceIndexType, shapely.MultiPolygon],
        polys_b: Dict[SliceIndexType, shapely.MultiPolygon],
        decimals: int
    ) -> Tuple[float, Dict[Tuple[SliceIndexType, SliceIndexType, str], float]]:
        """Seed the minimum margin from planar distances on common slices.

        Args:
            common_slices: Slices containing both regions of the pair.
            polys_a: Slice index to polygon map for the outer region.
            polys_b: Slice index to polygon map for the inner region.
            decimals: Number of decimals used to round the stored values.

        Returns:
            Tuple of (minimum, records) where minimum is the smallest planar
            distance and records maps (slice, slice, 'planar') to the planar
            minimum distance on that slice.
        """
        minimum = math.inf
        records = {}
        for slice_index in common_slices:
            boundary_a = shapely_boundary(polys_a[slice_index])
            if boundary_a.is_empty:
                continue
            distance = safe_distance(polys_b[slice_index], boundary_a)
            records[(slice_index, slice_index, 'planar')] = self._round(
                distance, decimals)
            if not math.isnan(distance):
                minimum = min(minimum, distance)
        return minimum, records

    # ------------------------------------------------------------------
    # Orthogonal Z margins and cross-slice minimum margin walk
    # ------------------------------------------------------------------

    def _z_margin_walk(
        self,
        pair: RegionPair,
        pair_type: str,
        polys_a: Dict[SliceIndexType, shapely.MultiPolygon],
        polys_b: Dict[SliceIndexType, shapely.MultiPolygon],
        slices_a: List[SliceIndexType],
        slices_b: List[SliceIndexType],
        slice_classes: Dict[Tuple[SliceIndexType, RegionPair],
                            Optional[str]],
        minimum: float,
        decimals: int
    ) -> Tuple[Dict[str, float], float, Dict, Dict]:
        """Calculate the orthogonal Z margins and update the minimum margin.

        Walks from each slice of the inner region (the reference slice)
        through the slices of the outer region in the negative and positive Z
        directions (the test slices).  A Z margin is triggered when the 2D
        relationship between the outer region on the test slice and the inner
        region on the reference slice changes to one where the inner region
        is no longer fully contained.  The 3D distance for each visited slice
        pair is used to update the minimum margin.

        Args:
            pair: The (region_a, region_b) pair being processed.
            pair_type: The relationship type string for the pair.
            polys_a: Slice index to polygon map for the outer region.
            polys_b: Slice index to polygon map for the inner region.
            slices_a: Sorted valid slice indexes of the outer region.
            slices_b: Sorted valid slice indexes of the inner region.
            slice_classes: (slice, pair) to relationship type string map.
            minimum: The seeded minimum margin for the pair.
            decimals: Number of decimals used to round the stored values.

        Returns:
            Tuple of (z_margins, minimum, z_records, min_records) where
            z_margins maps 'z_neg'/'z_pos' to the margin, z_records maps
            (reference_slice, test_slice) to triggered Z margins, and
            min_records maps (reference_slice, test_slice, direction) to the
            3D distance for each visited slice pair.
        """
        z_records: Dict[Tuple[SliceIndexType, SliceIndexType],
                        Dict[str, float]] = {}
        min_records: Dict[Tuple[SliceIndexType, SliceIndexType, str],
                          float] = {}

        # Initial Z margins from the starting and ending slice indexes.
        z_current = {
            'z_neg': slices_b[0] - slices_a[0],
            'z_pos': slices_a[-1] - slices_b[-1],
        }
        if z_current['z_neg'] < 0 or z_current['z_pos'] < 0:
            # Should not happen if the relationships were correctly
            # identified; indicates a problem with relationship identification.
            self.logger.warning(
                'Negative initial Z margin for region pair %s; setting both '
                'Z margins to NaN.', pair)
            return {'z_neg': math.nan, 'z_pos': math.nan}, minimum, {}, {}

        triggered = {'z_neg': False, 'z_pos': False}

        passes = (
            # (direction, tag, reference slices, test slices)
            ('z_neg', 'neg', slices_b, list(reversed(slices_a))),
            ('z_pos', 'pos', list(reversed(slices_b)), slices_a),
        )
        for direction, tag, ref_list, test_list in passes:
            for ref in ref_list:
                poly_b_ref = polys_b.get(ref)
                if poly_b_ref is None:
                    continue
                mask = Z_MARGIN_CHANGE_TESTS.get(
                    slice_classes.get((ref, pair)))
                for test in test_list:
                    gap = ref - test if direction == 'z_neg' else test - ref
                    if gap <= 0:
                        continue
                    if gap >= z_current[direction]:
                        # The difference between the two slice indexes is
                        # greater than or equal to the current margin: move
                        # on to the next reference slice.
                        break
                    poly_a_test = polys_a.get(test)
                    if poly_a_test is None:
                        continue

                    # Cross-slice 3D minimum distance.
                    boundary_a = shapely_boundary(poly_a_test)
                    if boundary_a.is_empty:
                        continue
                    d_planar = safe_distance(poly_b_ref, boundary_a)
                    if math.isnan(d_planar):
                        continue
                    distance_3d = math.hypot(d_planar, gap)
                    min_records[(ref, test, tag)] = self._round(
                        distance_3d, decimals)
                    if distance_3d < minimum:
                        minimum = distance_3d

                    # Z-margin change test.
                    if mask is None:
                        continue
                    relation_int = DE9IM(poly_a_test, poly_b_ref).int
                    if relation_int & mask:
                        # The inner region is no longer fully contained.
                        z_current[direction] = gap
                        triggered[direction] = True
                        z_records.setdefault((ref, test), {})[direction] = (
                            self._round(gap, decimals))
                        # Further test slices only increase the gap, which is
                        # now >= the updated margin.
                        break

        z_margins = {}
        for direction in Z_DIRECTIONS:
            if pair_type in HOLE_RELATIONSHIP_TYPES and not triggered[direction]:
                # The cavity does not close in this direction (the inner
                # region is open to the exterior): the margin is undefined.
                z_margins[direction] = math.nan
            else:
                z_margins[direction] = self._round(z_current[direction],
                                                   decimals)
        return z_margins, minimum, z_records, min_records

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_slice_orthogonal(
        slice_orthogonal: Dict[RegionPair,
                               Dict[Tuple[SliceIndexType, SliceIndexType],
                                    Dict[str, float]]]
    ) -> Dict[SliceIndexType, Dict[str, float]]:
        """Summarize orthogonal margins per reference slice.

        Args:
            slice_orthogonal: Detailed orthogonal margin table.

        Returns:
            Dict mapping each reference slice to the minimum margin per
            direction across all region pairs and test slices.
        """
        per_slice: Dict[SliceIndexType, Dict[str, float]] = {}
        for records in slice_orthogonal.values():
            for (ref, _test), margins in records.items():
                slot = per_slice.setdefault(ref, {})
                for direction, value in margins.items():
                    if value is None or math.isnan(value):
                        continue
                    slot[direction] = min(slot.get(direction, math.inf),
                                          value)
        return per_slice

    @staticmethod
    def _summarize_slice_minimum(
        slice_minimum: Dict[RegionPair,
                            Dict[Tuple[SliceIndexType, SliceIndexType, str],
                                 float]]
    ) -> Dict[SliceIndexType, float]:
        """Summarize minimum margins per reference slice.

        Args:
            slice_minimum: Detailed minimum margin table.

        Returns:
            Dict mapping each reference slice to the minimum margin across
            all region pairs and test slices.
        """
        per_slice: Dict[SliceIndexType, float] = {}
        for records in slice_minimum.values():
            for (ref, _test, _tag), value in records.items():
                if value is None or math.isnan(value):
                    continue
                per_slice[ref] = min(per_slice.get(ref, math.inf), value)
        return per_slice

    @staticmethod
    def _find_worst_case_orthogonal(
        per_region_orthogonal: Dict[RegionPair, Dict[str, float]],
        slice_orthogonal: Dict[RegionPair,
                               Dict[Tuple[SliceIndexType, SliceIndexType],
                                    Dict[str, float]]]
    ) -> Tuple[Optional[str], Optional[SliceIndexType]]:
        """Find the direction and reference slice of the smallest margin.

        Args:
            per_region_orthogonal: Per-region-pair orthogonal summaries.
            slice_orthogonal: Detailed orthogonal margin table.

        Returns:
            Tuple of (direction, reference slice) for the smallest margin,
            or (None, None) if there are no valid margins.
        """
        worst_pair = None
        worst_direction = None
        worst_value = math.inf
        for pair, margins in per_region_orthogonal.items():
            for direction, value in margins.items():
                if value is None or math.isnan(value):
                    continue
                if value < worst_value:
                    worst_pair = pair
                    worst_direction = direction
                    worst_value = value
        if worst_direction is None:
            return None, None

        worst_slice = None
        for (ref, _test), margins in slice_orthogonal.get(worst_pair,
                                                          {}).items():
            value = margins.get(worst_direction)
            if value is None or math.isnan(value):
                continue
            if value <= worst_value:
                worst_slice = ref
        return worst_direction, worst_slice

    # ------------------------------------------------------------------
    # Rounding helpers
    # ------------------------------------------------------------------

    def _rounding_decimals(self, tolerance: Optional[float]) -> int:
        """Get the number of decimals implied by the structure set tolerance.

        Args:
            tolerance: The structure set tolerance.  When not supplied or not
                positive, the config distance precision is used.

        Returns:
            Number of decimals to round margin values to.
        """
        if tolerance and tolerance > 0:
            return max(0, int(math.ceil(-math.log10(tolerance))))
        return self.config.distance_precision

    @staticmethod
    def _round(value: float, decimals: int) -> float:
        """Round a margin value, preserving NaN.

        Args:
            value: The value to round.
            decimals: Number of decimals to round to.

        Returns:
            The rounded value, or NaN if the value is NaN.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return value
        return round(value, decimals)
