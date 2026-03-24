# StructureRelations — Implementation Plan

## Context

The current DE27IM bands are asymmetric:
- Band 1: region_a vs region_b
- Band 2: exterior_a vs region_b  ← only A's derived shape
- Band 3: hull_a vs region_b      ← only A's derived shape

Target semantic model:
- Band 1: region_a vs region_b
- Band 2: exterior_a vs exterior_b
- Band 3: hull_a vs hull_b

This single change unblocks all other priorities:
complementary relations become auto-derivable via transpose,
the A > B size assumption is removed, and
caching both sides becomes worthwhile.

**Implementation order is strict:** Stage 1 must complete before Stage 2.
Stage 3 must complete after Stage 1. Stage 4 is independent.

---

## Stage 1 — Redesign DE27IM comparison bands

### Architectural additions from Quick Wins (fold in at zero extra cost)
- Add `AmbiguousRelationshipError` exception while `identify_relation` is open
- Add `RELATION_SCHEMA_VERSION = 2` module-level constant to relations.py
- Include schema version in debug log entries in `relate_contours`

### Plan agent prompt

```
I need to change the DE27IM relationship bands in src/relations.py.
Current bands: (region_a vs region_b), (exterior_a vs region_b), (hull_a vs region_b).
Target bands:  (region_a vs region_b), (exterior_a vs exterior_b), (hull_a vs hull_b).

Step 1 — Update relate_poly signature:
- Rename parameter `external_polygon` to `external_polygon_a`.
- Rename parameter `hull_polygon` to `hull_polygon_a`.
- Add optional parameters `external_polygon_b=None` and `hull_polygon_b=None`.
- When `external_polygon_b` is supplied, use it as the second argument to the
  DE9IM constructor for band 2; otherwise fall back to `poly_b`.
- When `hull_polygon_b` is supplied, use it as the second argument to the
  DE9IM constructor for band 3; otherwise fall back to `poly_b`.
- Keep all defaults None so existing callers are unaffected.

Step 2 — Update relate_contours, RegionSlice path:
- Compute `exterior_b = shapely.union_all(list(region_b.exterior.values()))`.
- Compute `hull_b = shapely.union_all(list(region_b.hull.values()))`.
- Pass both to relate_poly as external_polygon_b and hull_polygon_b.
- When region_b is a boundary (boundary_b in adjustments), do NOT pass
  exterior_polygon_b or hull_polygon_b — preserve the existing skip logic.

Step 3 — Update relate_contours, non-RegionSlice path:
- For Contour region_b: set external_polygon_b = make_solid(region_b.polygon),
  hull_polygon_b = region_b.hull.
- For shapely.Polygon / shapely.MultiPolygon region_b: set
  external_polygon_b = make_solid(region_b),
  hull_polygon_b = region_b.convex_hull.

Step 4 — Update relationship_definitions.json:
- Recompute and replace pattern, mask, value, mask_decimal, value_decimal
  for all primary relationships (reversed_arrow: false or missing).
- Do NOT add patterns to reversed_arrow: true entries yet.
- New logical hierarchy:
    DISJOINT:  all three bands disjoint
    SHELTERS:  band 1 disjoint, band 2 disjoint, band 3 hull_a contains hull_b
    SURROUNDS: band 1 disjoint, band 2 exterior_a contains exterior_b,
               band 3 hull_a contains hull_b
    CONTAINS:  all three bands containment
    BORDERS:   band 1 boundary-touch, band 2 boundary-touch
    OVERLAPS:  band 1 overlap, band 2 overlap

Step 5 — Add schema version and error class:
- Add `RELATION_SCHEMA_VERSION = 2` at module level in relations.py.
- Add `AmbiguousRelationshipError(ValueError)` exception class.
- In identify_relation, if more than one test_binary matches, raise
  AmbiguousRelationshipError listing the matching relation types.
- Include schema version in debug log messages in relate_contours.

Step 6 — Run tests:
- Run pytest tests/. Fix failing tests by updating expected RelationshipType
  values only. Do not change geometric primitives in debug_tools.py.
```

### Acceptance criteria
- [ ] `pytest tests/` passes with zero failures
- [ ] `DE27IM.relate_poly` has parameters `external_polygon_a`, `hull_polygon_a`,
      `external_polygon_b`, `hull_polygon_b` all defaulting to None
- [ ] `RELATION_SCHEMA_VERSION = 2` exists in relations.py
- [ ] `AmbiguousRelationshipError` class exists and is raised by `identify_relation`
      when multiple tests match
- [ ] All primary relationships in relationship_definitions.json have non-empty
      mask, value, mask_decimal, value_decimal fields
- [ ] The reversed_arrow: true entries (WITHIN, SHELTERED, ENCLOSED, CONFINED,
      PARTITIONS) still have no mask/value fields
- [ ] No changes to debug_tools.py, DE9IM, merge, to_str, or to_int

---

## Stage 2 — Complementary relations and order-invariance

### Architectural additions (fold in at zero extra cost)
- Convert adjustment string literals ('boundary_a', 'hole_a', etc.) to a
  `HoleType` Enum in types_and_classes.py while apply_adjustments is open

### Plan agent prompt

```
Following the Stage 1 DE27IM band redesign (schema version 2), implement
full complementary relation support in relations.py and
src/relationship_definitions.json.

Step 1 — Implement RelationshipTest.transpose():
- The 27-bit integer contains three consecutive 9-bit DE-9IM segments.
- For each 9-bit segment, the DE-9IM matrix transpose swaps A and B roles.
  Positions within each segment (0-indexed within that segment):
    [0]=II [1]=IB [2]=IE [3]=BI [4]=BB [5]=BE [6]=EI [7]=EB [8]=EE
  Swaps required: 1↔3, 2↔6, 5↔7 (within each segment).
- Apply these swaps to both the mask integer and the value integer.
- Return a new RelationshipTest with transposed mask and value, linked to
  the supplied relation_type argument.

Step 2 — Auto-generate reversed-arrow tests in _initialize_relationships:
- After all primary RelationshipTest objects are appended to cls.test_binaries,
  iterate over cls.relationship_definitions.values() where reversed_arrow is True.
- For each, look up the primary test using complementary_relation as key.
- Call primary_test.transpose(relation_type=reversed_rel_type).
- Append the result to cls.test_binaries.

Step 3 — Update relationship_definitions.json:
- For each reversed_arrow: true entry, manually derive and add pattern, mask,
  value, mask_decimal, value_decimal by applying the transpose logic to the
  complementary relation's fields. These fields are informational only;
  the runtime derives them via transpose in Step 2.

Step 4 — Add HoleType Enum:
- In src/types_and_classes.py, add:
    class HoleType(str, Enum):
        NONE = 'None'
        OPEN = 'Open'
        CLOSED = 'Closed'
        BOUNDARY = 'Boundary'
- In relations.py apply_adjustments and relate_contours, replace all
  string literal comparisons ('boundary_a', 'hole_a', etc.) with HoleType
  enum values. Keep the string values identical so JSON/dict serialization
  is unaffected.

Step 5 — Add complementary relation tests:
- In test_2D_relations.py and tests/test_3D_relations.py, add test
  classes for: WITHIN, SHELTERED, ENCLOSED, CONFINED, PARTITIONS.
- Each test must use a configuration where structure A is explicitly smaller
  than structure B. Use existing helpers from debug_tools.py.
- Add at least one order-invariance test per symmetric relationship verifying
  that relate(A, B) and relate(B, A) return equivalent classifications.

Step 6 — Run tests:
- Run pytest tests/. All tests must pass including the new ones.
```

### Acceptance criteria
- [ ] `RelationshipTest.transpose()` exists and returns a correctly transposed
      `RelationshipTest`
- [ ] `_initialize_relationships` appends transposed tests for all five
      reversed-arrow relationships
- [ ] All five reversed-arrow JSON entries have mask/value/pattern fields
- [ ] `HoleType` Enum exists in types_and_classes.py; no bare string literals
      remain in apply_adjustments
- [ ] Test classes exist for WITHIN, SHELTERED, ENCLOSED, CONFINED, PARTITIONS
      with A smaller than B
- [ ] Order-invariance tests exist for all symmetric relationships
- [ ] `pytest tests/` passes with zero failures

---

## Stage 3 — Processing efficiency and web-native progress

### Primary goal: web-app job model
Progress must be observable via API, not just terminal output.
tqdm integration is optional for local developer use only.

### Architectural additions (fold in at zero extra cost)
- Add `VolumeMetrics` dataclass while StructureShape is open
- Add `__enter__`/`__exit__` to DicomStructureFile for safe memory release
  in web worker processes

### Plan agent prompt

```
Improve processing efficiency and add web-safe progress reporting across
src/region_slice.py, src/relations.py, src/structures.py, and
src/structure_set.py.

Step 1 — Cache geometry in RegionSlice (src/region_slice.py):
- Import functools.
- Verify that regions, boundaries, and open_holes are not mutated after
  __init__ completes. If any mutation is found, document it and use an
  explicit _cache dict instead of cached_property for affected properties.
- Convert exterior and hull from @property to @functools.cached_property.
- Add cached properties:
    merged_region   → shapely.union_all(list(self.regions.values()))
    merged_exterior → shapely.union_all(list(self.exterior.values()))
    merged_hull     → shapely.union_all(list(self.hull.values()))
    merged_boundary → shapely.union_all(list(self.boundaries.values()))
      (return empty MultiPolygon if has_boundaries() is False)

Step 2 — Simplify relate_contours (src/relations.py):
- Replace all inline shapely.union_all(list(region_x.y.values())) calls with
  the corresponding cached property: region_a.merged_region,
  region_a.merged_exterior, region_a.merged_hull,
  region_b.merged_region, region_b.merged_exterior, region_b.merged_hull,
  region_b.merged_boundary. Logic must not change.

Step 3 — Add VolumeMetrics dataclass (src/structures.py):
- Add:
    @dataclass
    class VolumeMetrics:
        physical: float = 0.0
        exterior: float = 0.0
        hull: float = 0.0
- Replace the three volume float attributes on StructureShape with a single
  volume_metrics: VolumeMetrics attribute. Update all read/write sites.

Step 4 — Add DicomStructureFile context manager (src/dicom.py):
- Add __enter__ returning self after loading the dataset.
- Add __exit__ that deletes self.dataset if present, to free memory.
- Do not change constructor behavior for callers that do not use with.

Step 5 — Add progress callback to StructureShape.relate_to (src/structures.py):
- Add optional parameter:
    progress_callback: Optional[Callable[[int, int], None]] = None
  The callable receives (current_slice_index, total_slices).
- Call it after each slice iteration if not None.
- Default None; existing callers unaffected.

Step 6 — Add job-level progress to StructureSet (src/structure_set.py):
- Add optional parameter to calculate_relationships:
    progress_callback: Optional[Callable[[int, int], None]] = None
  The callable receives (completed_pairs, total_pairs).
- Call it after each pair completes.
- Pass a slice-level callback from calculate_relationships down into
  StructureShape.relate_to that updates a shared progress state dict
  containing keys: current_pair, total_pairs, current_slice, total_slices,
  status (string), percent_complete (float 0-100).
- Add a separate convenience method calculate_relationships_with_progress()
  that creates a tqdm progress bar if tqdm is importable, otherwise logs
  at INFO level. Do not make tqdm a hard dependency; wrap the import in
  a try/except ImportError.

Step 7 — Run tests:
- Run pytest tests/. All tests must pass.
```

### Acceptance criteria
- [ ] `RegionSlice.exterior` and `RegionSlice.hull` are `cached_property`
- [ ] `RegionSlice` has `merged_region`, `merged_exterior`, `merged_hull`,
      `merged_boundary` cached properties
- [ ] `relate_contours` contains no inline `shapely.union_all(list(...))` calls
      for region/exterior/hull values
- [ ] `VolumeMetrics` dataclass exists; `StructureShape` uses it
- [ ] `DicomStructureFile` supports the `with` statement
- [ ] `StructureShape.relate_to` accepts `progress_callback`
- [ ] `StructureSet.calculate_relationships` accepts `progress_callback` and
      maintains a shared progress state dict
- [ ] `calculate_relationships_with_progress` exists and does not crash when
      tqdm is absent
- [ ] `pytest tests/` passes with zero failures

---

## Stage 4 — API contract hardening and structural quick-wins

### Plan agent prompt

```
Harden the web API contracts and apply architectural quick-wins identified
as low-cost additions.

Step 1 — Pydantic response models:
- Define Pydantic models for: JobSubmitResponse, JobStatusResponse,
  RelationshipResultResponse, StructurePairResult.
- Include fields: relation_type, label, symbol, schema_version (from
  RELATION_SCHEMA_VERSION), tolerance, computed_at, provenance (dict).
- Use these models in all relationship-related API endpoints.

Step 2 — StructureSetBuilder fluent API:
- Add class StructureSetBuilder in structure_set.py with methods:
    with_dicom_file(path) → self
    with_tolerance(tol)   → self
    build()               → StructureSet
- Validate that dicom_file is set before build(); raise ValueError if not.
- Do not change StructureSet constructor.

Step 3 — Frozen dataclass ContourIndex:
- In types_and_classes.py, convert the ContourIndex tuple alias (or NewType)
  to a frozen dataclass:
    @dataclass(frozen=True)
    class ContourIndex:
        roi: ROI_Type
        slice_index: SliceIndexType
        uniqueness_int: int
- Update all sites that construct or unpack ContourIndex to use the
  dataclass. Maintain sort compatibility where ContourIndex is used as a
  dict key or in sets.

Step 4 — Integration tests:
- Add integration tests for: job submission, status polling, result retrieval,
  cancellation, and error recovery.
- Add a test verifying schema_version appears in relationship result responses.

Step 5 — Run tests:
- Run pytest tests/. All tests must pass.
```

### Acceptance criteria
- [ ] Pydantic models exist for all relationship API responses
- [ ] `schema_version` field present in all relationship result responses
- [ ] `StructureSetBuilder` exists with `with_dicom_file`, `with_tolerance`,
      `build` methods
- [ ] `ContourIndex` is a frozen dataclass
- [ ] Integration tests cover submit, poll, fetch, cancel, error paths
- [ ] `pytest tests/` passes with zero failures

---

## Cross-Cutting Notes (apply to all stages)

- Keep geometry computation pure and side-effect-free; progress updates
  belong in orchestration layers only.
- When boundary_b is in adjustments, do not pass exterior_polygon_b or
  hull_polygon_b to relate_poly (preserve existing skip logic from Stage 1).
- Cache keys for any shared/persistent cache must include structure
  fingerprint, tolerance, and RELATION_SCHEMA_VERSION.
- `RelationshipType.complementary` already resolves by name from
  RELATIONSHIP_TYPES — no changes needed to that property.
- tqdm is never a hard dependency; always guard with try/except ImportError.
