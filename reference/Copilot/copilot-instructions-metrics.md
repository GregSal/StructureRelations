# GitHub Copilot Instructions - Metrics Package

## Project Context
The StructureRelations project analyzes spatial relationships between DICOM RT structures using DE-27IM. The metrics sub-package (`src/metrics/`) adds quantitative measurements (distances, volumes, surface overlaps) for clinical validation.

## Critical Architecture Rule
⚠️ **SLICE-ORIENTED CALCULATION IS MANDATORY** ⚠️

All metrics MUST be calculated slice-by-slice, then aggregated to 3D summaries:
- Calculate for each slice where both structures exist
- Store complete per-slice results keyed by `SliceIndexType`
- Derive 3D summaries (min/max/mean/sum) from slice data
- Never calculate 3D metrics directly

## Quick Implementation Checklist

When implementing a new metric calculator:
- [ ] Inherit from `MetricCalculator` ABC in `base.py`
- [ ] Implement `is_applicable(relationship_type)` to check validity
- [ ] Implement `calculate()` with slice iteration pattern
- [ ] Store per-slice results in appropriate dataclass field
- [ ] Aggregate to 3D summary
- [ ] Return NaN for non-applicable cases
- [ ] Add unit tests using `debug_tools.py` geometries

## File Structure Reference

```
src/metrics/
├── __init__.py          # Export all calculators, load config
├── metrics_config.json  # Default units, precision, enabled metrics
├── config.py           # Configuration loading/validation
├── base.py             # MetricCalculator ABC, calculator registry
├── data_structures.py  # DistanceMetrics, VolumeMetrics, SurfaceMetrics, RelationshipMetrics
├── distance.py         # OrthogonalDistanceCalculator, MinimumDistanceCalculator, HausdorffDistanceCalculator
├── volume.py           # VolumeOverlapCalculator, DiceCalculator
├── surface.py          # SurfaceOverlapCalculator
├── geometry.py         # CentroidCalculator
└── orchestrator.py     # MetricOrchestrator workflow manager
```

## Essential Imports

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shapely
from types_and_classes import ROI_Type, SliceIndexType
from region_slice import RegionSlice
from structures import StructureShape
from relationships import StructureRelationship
from relations import RelationshipType
```

## Slice Iteration Pattern (MANDATORY)

```python
def calculate_metric(structure_a: StructureShape, structure_b: StructureShape) -> Dict:
    common_slices = set(structure_a.region_table.index) & set(structure_b.region_table.index)
    slice_results = {}
    
    for slice_index in sorted(common_slices):
        region_a = structure_a.region_table.loc[slice_index]
        region_b = structure_b.region_table.loc[slice_index]
        contours_a = region_a.select('contour')
        contours_b = region_b.select('contour')
        slice_results[slice_index] = compute_for_slice(contours_a, contours_b)
    
    aggregated = aggregate_function(slice_results.values())
    return {'slice_data': slice_results, 'aggregated': aggregated}
```

## Metric Applicability Matrix

| Relationship      | Orthogonal Margins | Min Distance | Volume Overlap | Surface Overlap |
|-------------------|-------------------|--------------|----------------|-----------------|
| CONTAINS          | ✓                 | ✗            | ✓              | ✗               |
| SURROUNDS         | ✓                 | ✗            | ✗              | ✗               |
| SHELTERS          | ✓ (no max)        | ✓            | ✗              | ✗               |
| OVERLAPS          | ✗                 | ✗            | ✓              | ✗               |
| PARTITION         | ✗                 | ✗            | ✓              | ✗               |
| BORDERS           | ✗                 | 0            | ✗              | ✓               |
| BORDERS_INTERIOR  | ✗                 | 0            | ✗              | ✓               |
| DISJOINT          | ✗                 | ✓            | ✗              | ✗               |

## Data Structure Pattern

All metric dataclasses follow this pattern:
```python
@dataclass
class XxxMetrics:
    # 3D aggregated summaries (DERIVED)
    summary_field: Optional[float] = None
    
    # PRIMARY: Per-slice data (keyed by SliceIndexType)
    slice_field: Optional[Dict[SliceIndexType, Any]] = None
```

## Common Shapely Operations

```python
# Distance
distance = polygon_a.distance(polygon_b)

# Hausdorff (max distance)
hausdorff = polygon_a.hausdorff_distance(polygon_b)

# Intersection (volume overlap)
overlap = shapely.intersection(polygon_a, polygon_b)
overlap_area = overlap.area

# Shared boundaries (surface overlap)
shared = shapely.shared_paths(polygon_a.exterior, polygon_b.exterior)
shared_length = shapely.length(shared)

# Centroid
centroid = shapely.centroid(polygon)
```

## Testing Requirements

Use geometries from `debug_tools.py`:
```python
from debug_tools import make_sphere, make_box, make_vertical_cylinder

# Example test
def test_orthogonal_margins():
    outer = make_sphere(radius=5, center=(0, 0, 0))
    inner = make_sphere(radius=3, center=(0, 0, 0))
    metrics = calculate_orthogonal_margins(outer, inner)
    assert abs(metrics['x_neg'] - 2.0) < 0.1  # Expected 5-3=2cm margin
```

## Key Design Decisions

1. **Slice-oriented**: Calculate per-slice first, aggregate to 3D
2. **Contour boundaries**: Use actual contours, not exterior/hull (clinical relevance)
3. **Non-applicable metrics**: Return NaN, not zero
4. **Configuration**: Load from `metrics_config.json`, allow user override
5. **Units**: Default to cm (matches DICOM coordinates), precision=2 decimals
6. **Storage**: Store in `StructureRelationship.metrics` field (already exists)

## Reference Materials

- **Implementation guide**: `/memories/repo/metrics-implementation-guide.md`
- **Quick reference**: `/memories/repo/metrics-quick-reference.md`
- **Old code (reference only)**: `reference/Reference_Code/metrics/` - DO NOT COPY
- **Test examples**: `reference/Reference_Code/metrics/Metrics Notebooks/ContourMetricTests.ipynb`
- **Design doc**: `StructureMetrics.md`

## Common Pitfalls to Avoid

❌ Calculating 3D metrics directly  
❌ Copying old code verbatim  
❌ Returning zero for non-applicable metrics  
❌ Ignoring multi-region structures  
❌ Forgetting slice thickness for volumes  
❌ Not validating applicability first  

✅ Calculate per-slice, then aggregate  
✅ Reference old code for logic, implement slice-oriented  
✅ Return NaN for non-applicable  
✅ Handle multiple polygons per slice  
✅ Include slice thickness in calculations  
✅ Check relationship type before calculating  
