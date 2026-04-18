"""Data structures for storing spatial relationship metrics.

This module defines typed dataclasses for organizing metric results with:
- Slice-oriented architecture: Per-slice data is PRIMARY, 3D summaries are derived
- Region-aware design: Separate metrics for each region pair in multi-region structures
- Clinical focus: Uses actual contour boundaries, not exterior/hull
- Clear non-applicability: Returns NaN for metrics that don't apply to relationship type
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math

from types_and_classes import SliceIndexType, ROI_Type


@dataclass
class MarginMetrics:
    """Clearance distances inside containing structure.

    Applicable to: CONTAINS, SURROUNDS, SHELTERS relationships.
    Special case: EQUAL returns 0 for all margins.

    Margins measure the minimum distance from the contained structure to the
    boundary of the containing structure in each direction. This is different
    from minimum_distance which measures gaps between disjoint structures.
    """
    # 3D aggregated summaries (derived from per-region data)
    # For multi-region structures, these represent the worst-case (minimum) across all region pairs
    orthogonal_margins: Optional[Dict[str, float]] = None  # {'x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos'}
    minimum_margin: Optional[float] = None  # Single worst-case clearance across all directions and regions
    maximum_margin: Optional[float] = None  # Largest clearance (Hausdorff-based)

    # Per-region-pair data (region IDs from contour graph connected components)
    # Keys: (region_a_id, region_b_id) tuples
    # Values: metric results for that specific region pair
    per_region_orthogonal_margins: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None
    per_region_minimum_margin: Optional[Dict[Tuple[int, int], float]] = None
    per_region_maximum_margin: Optional[Dict[Tuple[int, int], float]] = None

    # Per-slice data for each region pair
    # Keys: (region_a_id, region_b_id) tuples
    # Values: {slice_index: {'x_neg': ..., 'x_pos': ..., ...}}
    slice_orthogonal_margins: Optional[Dict[Tuple[int, int], Dict[SliceIndexType, Dict[str, float]]]] = None

    # Metadata for traceability
    worst_case_region_pair: Optional[Tuple[int, int]] = None  # Which region pair has minimum margin
    worst_case_direction: Optional[str] = None  # Which direction has minimum margin
    worst_case_slice: Optional[SliceIndexType] = None  # Which slice has minimum margin

    def __post_init__(self):
        """Validate that margin values are non-negative or NaN."""
        if self.minimum_margin is not None and not math.isnan(self.minimum_margin):
            if self.minimum_margin < 0:
                raise ValueError(f'Minimum margin cannot be negative: {self.minimum_margin}')
        if self.maximum_margin is not None and not math.isnan(self.maximum_margin):
            if self.maximum_margin < 0:
                raise ValueError(f'Maximum margin cannot be negative: {self.maximum_margin}')


@dataclass
class DistanceMetrics:
    """Gap between disjoint structures.

    Applicable to: DISJOINT, SHELTERS relationships.
    N/A for: BORDERS, CONFINES (BORDERS_INTERIOR), EQUAL (structures touch or are identical).

    Distance measures the gap between structures that don't touch. This is different
    from margins which measure clearance inside a containing structure.
    """
    # 3D aggregated summaries (derived from per-region data)
    # For multi-region structures, this is the minimum across all region pairs
    minimum_distance: Optional[float] = None

    # Per-region-pair data (region IDs from contour graph)
    # Keys: (region_a_id, region_b_id) tuples
    per_region_minimum_distance: Optional[Dict[Tuple[int, int], float]] = None

    # Per-slice data for each region pair
    # Keys: (region_a_id, region_b_id) tuples
    # Values: {slice_index: distance_value}
    slice_distances: Optional[Dict[Tuple[int, int], Dict[SliceIndexType, float]]] = None

    # Z-direction (through-plane) distance component
    z_distance: Optional[float] = None

    # Metadata for traceability
    closest_region_pair: Optional[Tuple[int, int]] = None  # Which region pair has minimum distance
    closest_slice: Optional[SliceIndexType] = None  # Which slice has minimum distance

    def __post_init__(self):
        """Validate that distance values are non-negative or NaN."""
        if self.minimum_distance is not None and not math.isnan(self.minimum_distance):
            if self.minimum_distance < 0:
                raise ValueError(f'Minimum distance cannot be negative: {self.minimum_distance}')


@dataclass
class VolumeMetrics:
    """Volume overlap metrics for intersecting structures.

    Applicable to: OVERLAPS, PARTITION, CONTAINS, EQUAL relationships.
    Special case: EQUAL has overlap_ratio=1.0, dice_coefficient=1.0.

    Volume calculations sum all regions on each slice before aggregating to 3D.
    """
    # 3D aggregated summaries (derived from per-slice data)
    overlap_ratio: Optional[float] = None  # Ratio of overlap volume to reference volume (see ratio_basis)
    dice_coefficient: Optional[float] = None  # 2 * |A ∩ B| / (|A| + |B|)

    volume_a: Optional[float] = None  # Total volume of structure A
    volume_b: Optional[float] = None  # Total volume of structure B
    overlap_volume: Optional[float] = None  # Volume of intersection

    # Basis for ratio calculation
    ratio_basis: str = 'auto'  # 'larger', 'smaller', 'average', 'auto' (relationship-dependent)

    # Per-slice area data (PRIMARY data for slice-oriented architecture)
    # All regions on a slice are summed before storing
    # Keys: slice_index
    # Values: {'area_a': ..., 'area_b': ..., 'overlap_area': ...}
    slice_areas: Optional[Dict[SliceIndexType, Dict[str, float]]] = None
    slice_dice: Optional[Dict[SliceIndexType, float]] = None  # Per-slice Dice coefficients

    def __post_init__(self):
        """Validate volume and ratio values."""
        if self.overlap_ratio is not None and not math.isnan(self.overlap_ratio):
            if not 0 <= self.overlap_ratio <= 1:
                raise ValueError(f'Overlap ratio must be between 0 and 1: {self.overlap_ratio}')
        if self.dice_coefficient is not None and not math.isnan(self.dice_coefficient):
            if not 0 <= self.dice_coefficient <= 1:
                raise ValueError(f'Dice coefficient must be between 0 and 1: {self.dice_coefficient}')


@dataclass
class SurfaceMetrics:
    """Surface boundary overlap metrics for touching structures.

    Applicable to: BORDERS, CONFINES (BORDERS_INTERIOR) relationships.

    Surface calculations use Shapely's shared_paths() to find touching boundaries.
    For multi-region structures, boundaries from all regions are summed.
    """
    # 3D aggregated summaries (derived from per-slice data)
    overlap_ratio: Optional[float] = None  # Ratio of overlap to reference perimeter

    surface_area_a: Optional[float] = None  # Total surface area of structure A
    surface_area_b: Optional[float] = None  # Total surface area of structure B
    overlap_surface_area: Optional[float] = None  # Shared surface area

    # Basis for ratio calculation
    ratio_basis: str = 'larger'  # 'larger', 'smaller', 'average', 'exterior', 'interior'

    # Per-slice perimeter data (PRIMARY data)
    # All regions on a slice are summed before storing
    # Keys: slice_index
    # Values: {'perimeter_a': ..., 'perimeter_b': ..., 'overlap_length': ...}
    slice_perimeters: Optional[Dict[SliceIndexType, Dict[str, float]]] = None

    def __post_init__(self):
        """Validate surface ratio values."""
        if self.overlap_ratio is not None and not math.isnan(self.overlap_ratio):
            if not 0 <= self.overlap_ratio <= 1:
                raise ValueError(f'Overlap ratio must be between 0 and 1: {self.overlap_ratio}')


@dataclass
class GeometryMetrics:
    """Geometric properties of structures and their relationships.

    For multi-region structures, separate centroids are calculated for each region.
    """
    # Centroids for each region in structure A
    # List of (region_id, (x, y, z)) tuples
    centroids_a: Optional[List[Tuple[int, Tuple[float, float, float]]]] = None

    # Centroids for each region in structure B
    centroids_b: Optional[List[Tuple[int, Tuple[float, float, float]]]] = None

    # Distance between primary centroids (first region of each structure)
    centroid_distance: Optional[float] = None

    # For multi-region structures: min/max distances between all centroid pairs
    min_centroid_distance: Optional[float] = None
    max_centroid_distance: Optional[float] = None


@dataclass
class RelationshipMetrics:
    """Complete metrics container for a spatial relationship between two structures.

    This is the top-level metrics object stored in StructureRelationship.metrics field.
    Only applicable metrics will be populated based on relationship type.
    Non-applicable metrics will be None or contain NaN values.
    """
    # Metric categories
    margin: Optional[MarginMetrics] = None
    distance: Optional[DistanceMetrics] = None
    volume: Optional[VolumeMetrics] = None
    surface: Optional[SurfaceMetrics] = None
    geometry: Optional[GeometryMetrics] = None

    # Structure metadata
    structure_a_name: str = ''
    structure_b_name: str = ''
    structure_a_roi: Optional[ROI_Type] = None
    structure_b_roi: Optional[ROI_Type] = None
    region_count_a: int = 0  # Number of disconnected regions in structure A
    region_count_b: int = 0  # Number of disconnected regions in structure B

    # Measurement metadata
    unit: str = 'cm'  # Measurement unit for distances/volumes
    precision: int = 2  # Decimal places for display
    slice_thickness: Optional[float] = None  # For volume/surface area calculations

    # Configuration snapshot
    config_snapshot: Optional[Dict] = None  # Record of config used for calculation

    # Calculation metadata
    calculation_time: Optional[float] = None  # Time to calculate metrics (seconds)
    calculator_versions: Optional[Dict[str, str]] = None  # Version of each calculator used

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for webapp API responses
        """
        result = {
            'structure_a': self.structure_a_name,
            'structure_b': self.structure_b_name,
            'unit': self.unit,
            'region_count_a': self.region_count_a,
            'region_count_b': self.region_count_b,
        }

        # Add non-None metric categories
        if self.margin is not None:
            result['margin'] = self._margin_to_dict()
        if self.distance is not None:
            result['distance'] = self._distance_to_dict()
        if self.volume is not None:
            result['volume'] = self._volume_to_dict()
        if self.surface is not None:
            result['surface'] = self._surface_to_dict()
        if self.geometry is not None:
            result['geometry'] = self._geometry_to_dict()

        return result

    def _margin_to_dict(self) -> Dict:
        """Convert MarginMetrics to dictionary."""
        if self.margin is None:
            return {}
        return {
            'orthogonal_margins': self.margin.orthogonal_margins,
            'minimum_margin': self.margin.minimum_margin,
            'maximum_margin': self.margin.maximum_margin,
            'worst_case_region': self.margin.worst_case_region_pair,
            'worst_case_direction': self.margin.worst_case_direction,
        }

    def _distance_to_dict(self) -> Dict:
        """Convert DistanceMetrics to dictionary."""
        if self.distance is None:
            return {}
        return {
            'minimum_distance': self.distance.minimum_distance,
            'z_distance': self.distance.z_distance,
            'closest_region': self.distance.closest_region_pair,
        }

    def _volume_to_dict(self) -> Dict:
        """Convert VolumeMetrics to dictionary."""
        if self.volume is None:
            return {}
        return {
            'overlap_ratio': self.volume.overlap_ratio,
            'dice_coefficient': self.volume.dice_coefficient,
            'volume_a': self.volume.volume_a,
            'volume_b': self.volume.volume_b,
            'overlap_volume': self.volume.overlap_volume,
            'ratio_basis': self.volume.ratio_basis,
        }

    def _surface_to_dict(self) -> Dict:
        """Convert SurfaceMetrics to dictionary."""
        if self.surface is None:
            return {}
        return {
            'overlap_ratio': self.surface.overlap_ratio,
            'surface_area_a': self.surface.surface_area_a,
            'surface_area_b': self.surface.surface_area_b,
            'overlap_surface_area': self.surface.overlap_surface_area,
            'ratio_basis': self.surface.ratio_basis,
        }

    def _geometry_to_dict(self) -> Dict:
        """Convert GeometryMetrics to dictionary."""
        if self.geometry is None:
            return {}
        return {
            'centroids_a': self.geometry.centroids_a,
            'centroids_b': self.geometry.centroids_b,
            'centroid_distance': self.geometry.centroid_distance,
            'min_centroid_distance': self.geometry.min_centroid_distance,
            'max_centroid_distance': self.geometry.max_centroid_distance,
        }
