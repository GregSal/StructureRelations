# Metrics Sub-Package Implementation Guide

## Purpose
This document provides implementation guidance for the StructureRelations metrics sub-package. The metrics package calculates quantitative spatial relationships (distances, volumes, surface overlaps) between RT structures for clinical validation in radiotherapy planning.

## Core Architecture Principles

### Slice-Oriented Calculation (CRITICAL)
- **Slice-by-slice calculation is PRIMARY**, 3D summaries are DERIVED
- Matches existing DE-27IM relationship calculation approach
- Medical structures are inherently slice-based (DICOM RT contours stored as 2D slices)
- Per-slice results keyed by `SliceIndexType` from `types_and_classes.py`
- Enables investigation of slice-level discrepancies via Contour Plotting tab
- Mirrors `region_table` structure in `StructureShape` and `slice_relationship_records` in `StructureSet`

### Custom Implementation Stack
- **Shapely** for 2D geometry operations (distance, intersection, shared_paths)
- **NetworkX** for contour graph traversal (already used for relationships)
- **NumPy/Pandas** for numerical operations and aggregation
- **Optional**: `surface-distance` package for validation only (requires voxel conversion)

### Data Storage Pattern
- Store complete per-slice results in dataclass fields (e.g., `slice_orthogonal_margins`)
- Derive 3D aggregated summaries (min/max/mean/sum) from slice data
- Use typed dataclasses for type safety and JSON serialization
- Store in `StructureRelationship.metrics` field (already exists as `Optional[Any]`)

## File Organization

### New Files to Create
```
src/metrics/
├── __init__.py (replace existing)
├── metrics_config.json (configuration defaults)
├── config.py (config loading/validation)
├── base.py (MetricCalculator ABC, registry)
├── data_structures.py (dataclass definitions)
├── distance.py (distance calculators)
├── volume.py (volume calculators)
├── surface.py (surface calculators)
├── geometry.py (centroid calculator)
└── orchestrator.py (workflow management)

tests/
├── test_metrics.py (unit tests)
└── test_metrics_integration.py (e2e tests)
```

### Files to Remove
- `src/metrics/margins.py` — outdated, non-working code

### Reference Only (Do Not Copy)
- `reference/Reference_Code/metrics/` — old implementations for design reference
- `reference/Reference_Code/metrics/Metrics Notebooks/ContourMetricTests.ipynb` — test case examples

## Key Implementation Patterns

### Slice Iteration Pattern
```python
# Iterate through slices where both structures exist
common_slices = set(structure_a.region_table.index) & set(structure_b.region_table.index)

slice_results = {}
for slice_index in sorted(common_slices):
    region_a = structure_a.region_table.loc[slice_index]
    region_b = structure_b.region_table.loc[slice_index]
    
    # Calculate metric for this slice
    metric_value = calculate_slice_metric(region_a, region_b)
    slice_results[slice_index] = metric_value

# Aggregate to 3D summary
summary_metric = aggregate_function(slice_results.values())
```

### Accessing Contour Types
```python
# Use RegionSlice.select() to get different boundary types
contour_polygons = region_slice.select('contour')  # Actual boundaries
exterior_polygons = region_slice.select('exterior')  # Holes filled
hull_polygons = region_slice.select('hull')  # Convex hull
```

### Configuration Loading
```python
# Load from metrics_config.json, allow user overrides
config = load_metrics_config()
unit = config.get('unit', 'cm')
precision = config.get('precision', 2)
```

## Metric Applicability Matrix

| Relationship Type | Orthogonal Margins | Min Distance | Volume Overlap | Surface Overlap |
|-------------------|-------------------|--------------|----------------|-----------------|
| CONTAINS          | ✓                 | N/A          | ✓              | N/A             |
| SURROUNDS         | ✓                 | N/A          | N/A            | N/A             |
| SHELTERS          | ✓ (no max)        | ✓            | N/A            | N/A             |
| OVERLAPS          | N/A               | N/A          | ✓              | N/A             |
| PARTITION         | N/A               | N/A          | ✓              | N/A             |
| BORDERS           | N/A               | 0            | N/A            | ✓               |
| BORDERS_INTERIOR  | N/A               | 0            | N/A            | ✓               |
| DISJOINT          | N/A               | ✓            | N/A            | N/A             |
| EQUAL             | N/A               | 0            | ✓ (1.0)        | ✓ (1.0)         |

## Testing Requirements

### Unit Test Geometries
Use `debug_tools.py` functions:
- `make_sphere(radius, center)` — concentric spheres for CONTAINS
- `make_vertical_cylinder(radius, height, center)` — for OVERLAPS
- `make_box(width, height, depth)` — for orthogonal margin tests

### Test Case Pattern
```python
def test_orthogonal_margins_contains():
    # Create test geometry
    outer = make_sphere(radius=5, center=(0, 0, 0))
    inner = make_sphere(radius=3, center=(0, 0, 0))
    
    # Calculate metrics
    metrics = calculate_orthogonal_margins(outer, inner)
    
    # Verify expected values
    expected_margin = 2.0  # 5 - 3 = 2 cm in all directions
    assert abs(metrics['x_neg'] - expected_margin) < 0.1
    # ... etc
```

### Validation Against Reference Notebook
Reproduce test cases from `ContourMetricTests.ipynb` to ensure correctness.

## Configuration Schema

### metrics_config.json Structure
```json
{
    "units": {
        "default": "cm",
        "allowed": ["cm", "mm"]
    },
    "precision": {
        "default": 2,
        "tolerance_based": false
    },
    "enabled_metrics": {
        "orthogonal_margins": true,
        "minimum_distance": true,
        "volume_overlap": true,
        "dice_coefficient": true,
        "hausdorff_distance": false,
        "surface_overlap": true
    },
    "calculation": {
        "use_interpolated_contours": false,
        "slice_thickness_source": "auto"
    }
}
```

## Integration Points

### StructureSet Integration
```python
# Add to StructureSet.calculate_relationships()
def calculate_relationships(self, calculate_metrics=False, requested_metrics='all'):
    # ... existing DE27IM calculation ...
    
    if calculate_metrics:
        orchestrator = MetricOrchestrator(config)
        metrics = orchestrator.calculate_metrics(
            structure_a, structure_b, relationship, requested_metrics
        )
        relationship_obj.metrics = metrics
```

### Webapp Tooltip Integration
```javascript
// In buildEdgeTooltip() in app.js
if (edge.metrics) {
    const relevantMetrics = selectMetricsForRelationType(edge.relation_type, edge.metrics);
    lines.push(...formatMetrics(relevantMetrics));
}
```

## Common Pitfalls to Avoid

1. **Don't calculate 3D metrics directly** — always calculate per-slice first
2. **Don't copy old code verbatim** — reference for logic, but use slice-oriented approach
3. **Don't return zero for non-applicable metrics** — use NaN
4. **Don't forget multi-region structures** — handle multiple polygons per slice
5. **Don't ignore slice thickness** — needed for volume/surface area calculations
6. **Don't forget to validate applicability** — check relationship type before calculating

## Clinical Context Notes

- **Margins** measure clearance between structures (critical for radiation planning)
- **PTV margin** typically 0.3-1.0 cm around GTV/CTV in radiotherapy
- **Orthogonal directions** correspond to anatomical axes: ±X (L/R), ±Y (A/P), ±Z (S/I)
- **Dice coefficient** standard metric for contour comparison (range 0-1, higher = better overlap)
- **Contour boundary** must be used (not exterior/hull) for clinical accuracy
