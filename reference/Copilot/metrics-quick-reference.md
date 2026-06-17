# Metrics Package - Quick Reference

## Data Structures (from data_structures.py)

```python
@dataclass
class DistanceMetrics:
    # 3D aggregated summaries (derived from slice data)
    orthogonal_margins: Optional[Dict[str, float]] = None  # {'x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos'}
    minimum_distance: Optional[float] = None
    hausdorff_distance: Optional[float] = None
    
    # PRIMARY: Per-slice data (keyed by SliceIndexType)
    slice_orthogonal_margins: Optional[Dict[SliceIndexType, Dict[str, float]]] = None
    slice_distances: Optional[Dict[SliceIndexType, float]] = None
    slice_hausdorff: Optional[Dict[SliceIndexType, float]] = None

@dataclass
class VolumeMetrics:
    # 3D aggregated summaries
    overlap_volume: Optional[float] = None
    overlap_ratio: Optional[float] = None
    dice_coefficient: Optional[float] = None
    volume_a: Optional[float] = None
    volume_b: Optional[float] = None
    
    # PRIMARY: Per-slice data
    slice_areas: Optional[Dict[SliceIndexType, Dict[str, float]]] = None
    slice_dice: Optional[Dict[SliceIndexType, float]] = None

@dataclass
class SurfaceMetrics:
    # 3D aggregated summaries
    surface_overlap_area: Optional[float] = None
    surface_overlap_ratio: Optional[float] = None
    
    # PRIMARY: Per-slice data
    slice_perimeters: Optional[Dict[SliceIndexType, Dict[str, float]]] = None

@dataclass
class RelationshipMetrics:
    distance: Optional[DistanceMetrics] = None
    volume: Optional[VolumeMetrics] = None
    surface: Optional[SurfaceMetrics] = None
    centroids_a: Optional[List[Tuple[int, Tuple[float, float, float]]]] = None
    centroids_b: Optional[List[Tuple[int, Tuple[float, float, float]]]] = None
    unit: str = 'cm'
    precision: int = 2
    slice_thickness: Optional[float] = None
    config_snapshot: Optional[Dict] = None
```

## Calculator Interface (from base.py)

```python
class MetricCalculator(ABC):
    @abstractmethod
    def is_applicable(self, relationship_type: RelationshipType) -> bool:
        """Check if this metric applies to the given relationship type."""
        pass
    
    @abstractmethod
    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship,
        config: Dict
    ) -> Any:
        """Calculate the metric and return appropriate data structure."""
        pass
```

## Slice Iteration Template

```python
def calculate_metric(structure_a, structure_b):
    # Find common slices
    common_slices = set(structure_a.region_table.index) & set(structure_b.region_table.index)
    
    # Per-slice calculation
    slice_results = {}
    for slice_index in sorted(common_slices):
        region_a = structure_a.region_table.loc[slice_index]
        region_b = structure_b.region_table.loc[slice_index]
        
        # Get appropriate contours
        contours_a = region_a.select('contour')  # or 'exterior' or 'hull'
        contours_b = region_b.select('contour')
        
        # Calculate metric for this slice
        slice_results[slice_index] = compute_slice_metric(contours_a, contours_b)
    
    # Aggregate to 3D summary
    aggregated = aggregate_results(slice_results)
    
    return {
        'slice_data': slice_results,
        'aggregated': aggregated
    }
```

## Common Shapely Operations for Metrics

```python
# Distance between polygons
distance = polygon_a.distance(polygon_b)

# Hausdorff distance (maximum distance)
hausdorff = polygon_a.hausdorff_distance(polygon_b)

# Intersection (for volume overlap)
overlap = shapely.intersection(polygon_a, polygon_b)
overlap_area = overlap.area

# Shared boundaries (for surface overlap)
shared = shapely.shared_paths(polygon_a.exterior, polygon_b.exterior)
shared_length = shapely.length(shared)

# Centroid
centroid = shapely.centroid(polygon)
coords = shapely.get_coordinates(centroid)[0]  # (x, y)

# Bounds for orthogonal lines
bounds = polygon.bounds  # (minx, miny, maxx, maxy)
```

## Import Statements

```python
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import shapely

from types_and_classes import (
    ROI_Type, SliceIndexType, PolygonType,
    DEFAULT_TRANSVERSE_TOLERANCE, SLICE_INDEX_PRECISION
)
from region_slice import RegionSlice
from structures import StructureShape
from relationships import StructureRelationship
from relations import RelationshipType
```

## Key Constants

```python
# From types_and_classes.py
DEFAULT_TRANSVERSE_TOLERANCE = 0.01  # cm
SLICE_INDEX_PRECISION = 0.01  # cm

# Default units and precision for metrics
DEFAULT_UNIT = 'cm'
DEFAULT_PRECISION = 2
```

## Webapp Integration Pattern

```python
# Backend: Add metrics to relationship data
relationship_data = {
    'de27im': relationship.de27im.to_dict(),
    'relation_type': relationship.relationship_type.relation_type,
    'metrics': serialize_metrics(relationship.metrics) if relationship.metrics else None
}

# Frontend: Display in tooltip
function buildEdgeTooltip(edge) {
    let lines = [/* ... relationship info ... */];
    
    if (edge.metrics) {
        const metricsToShow = selectRelevantMetrics(edge.relation_type, edge.metrics);
        for (const [name, value] of Object.entries(metricsToShow)) {
            lines.push(`${name}: ${formatMetricValue(value, edge.metrics.unit)}`);
        }
    }
    
    return lines.join('\n');
}
```
