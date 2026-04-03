## Data Processing Pipeline Summary

### **Phase 1: DICOM Parsing & Contour Table Creation**

**Starting Point:** `DicomStructureFile` reads contour points from DICOM RT Structure Set file

**Module:** contours.py
- **Function:** `build_contour_table(slice_data: List[ContourPoints])`
- **Input:** List of `ContourPoints` dictionaries (from DICOM file, containing ROI, Slice, Points)
- **Process:**
  1. Creates pandas DataFrame from contour points
  2. Converts point lists to `shapely.Polygon` objects via `points_to_polygon()`
  3. Calculates polygon area for each contour
  4. Sorts by ROI, Slice, Area (descending)
- **Output:**
  - `contour_table` (DataFrame with columns: ROI, Slice, Points, Polygon, Area)
  - `SliceSequence` object tracking all unique Z-indices and their neighbors

---

### **Phase 2: Contour Graph Construction (Per-ROI)**

**Module:** contour_graph.py
- **Function:** `build_contour_graph(contour_table, slice_sequence, roi)`

**Step 2.1 - Create Contour Objects**
- **Function:** `build_contours()`
- Creates individual `Contour` objects from table rows
- Each `Contour` has:
  - `roi`, `slice_index`, `polygon`, `contour_index` (unique ID tuple)
  - `hole_type` (Open/Closed/Unknown/None)
  - `is_boundary`, `is_interpolated` flags
  - `region_index` (assigned later)

**Step 2.2 - Link Contours Across Slices**
- **Function:** `add_graph_edges()`
- Creates directed edges between `Contour` nodes on adjacent slices
- Edge criteria: same ROI, same hole type, intersecting convex hulls
- Results in `ContourGraph` (NetworkX DiGraph) with directed edges pointing from lower → higher Z-indices
- Edge attributes: `ContourMatch` object with overlap metrics

**Step 2.3 - Add Boundary Contours**
- **Function:** `add_boundary_contours()`
- Extends slices to include interpolated boundary contours beyond actual data
- Uses `interpolate_polygon()` for intermediate slices

**Step 2.4 - Identify Regions**
- **Function:** `set_enclosed_regions()`
- Uses NetworkX connected components analysis to identify distinct 3D regions
- Assigns `RegionIndex` to each contour

**Step 2.5 - Classify Hole Types**
- **Function:** `set_hole_type()`
- Determines if regions are holes (interior voids) or exterior boundaries
- Based on containment relationships within each slice

**Output:** structures.py → `StructureShape` object with:
- `contour_graph`: NetworkX DiGraph containing all Contour nodes
- `contour_lookup`: DataFrame index (ROI, SliceIndex, HoleType, RegionIndex, etc.)

---

### **Phase 3: Interpolation & Region Slicing**

**Module:** structures.py

**Step 3.1 - Add Interpolated Contours**
- **Method:** `StructureShape.add_interpolated_contours()`
- Fills gaps between contours using `interpolate_polygon()`
- Creates new `Contour` objects with `is_interpolated=True` flag
- Updates `contour_lookup` table

**Step 3.2 - Build Region Table**
- **Method:** `StructureShape.build_region_table()`
- For each unique SliceIndex, creates a `RegionSlice` object
- **Module:** region_slice.py

**RegionSlice Structure:** Contains all regions on a single slice as MultiPolygon dictionaries:
- `regions`: Main region polygons (contours with holes filled per region)
- `boundaries`: Contours that are region boundaries
- `open_holes`: Holes (voids) in the region
- `exterior`, `hull` properties (computed via `make_solid()` and convex hull)

**Output:** `region_table` DataFrame with columns:
- SliceIndex, RegionSlice object, Empty flag, Interpolated flag

---

### **Phase 4: Volume Calculations**

**Module:** structures.py

Three volume types calculated by iterating through contour graph edges:
- **`physical_volume`**: Sum of actual region volumes (subtracting holes)
- **`exterior_volume`**: Sum with all closed holes filled
- **`hull_volume`**: Sum of convex hull volumes

---

### **Phase 5: Structure Set Assembly**

**Module:** structure_set.py

**StructureSet.__init__() → build_from_dicom_file() → build_from_slice_data():**

1. Calls `build_contour_table()` once for all structures (shared SliceSequence)
2. For each unique ROI:
   - Create `StructureShape` object
   - Call `add_contour_graph()` to build contour graph
   - Store in `self.structures` dict keyed by ROI
3. For each structure:
   - Call `finalize()` to:
     - Add interpolated contours
     - Calculate volumes
     - Build region table
4. Call `self.finalize()` which triggers relationship calculation

---

### **Phase 6: Relationship Identification**

**Module:** relations.py

**StructureSet.calculate_relationships():**

For each pair of structures (ROI_A, ROI_B):

1. **Call:** `StructureShape_A.relate(StructureShape_B, tolerance)`
   - Iterates through all common slices
   - Creates `DE27IM` object per slice (see below)
   - Merges slice relationships using logical OR

2. **DE27IM Calculation** (per-slice):
   - Creates three `DE9IM` relationships:
     - **Contour vs Contour:** Actual region boundaries
     - **Exterior vs Contour:** Filled region (holes) vs boundary
     - **Hull vs Contour:** Convex hull vs boundary
   - Uses `shapely.relate()` for geometric DE-9IM computation
   - Concatenates into 27-bit binary pattern

3. **Pattern Matching:**
   - **Method:** `DE27IM.identify_relation()`
   - Tests 27-bit pattern against `RelationshipTest` definitions
   - Each test has mask and value for bit-pattern matching
   - Returns one of: CONTAINS, OVERLAPS, SURROUNDS, SHELTERS, CONFINES, BORDERS, PARTITIONED, DISJOINT, EQUAL, or UNKNOWN

4. **Store Result:**
   - Creates `StructureRelationship` object
   - Adds edge to `relationship_graph` (NetworkX DiGraph)
   - Edge attributes: RelationshipType, DE27IM data, logical flags

---

## Key Data Types Throughout Pipeline

| **Stage** | **Primary Types** |
|-----------|-------------------|
| DICOM Input | `ContourPoints` (dict), `DicomStructureFile` |
| Contour Table | pandas `DataFrame`, `Polygon`, `SliceSequence` |
| Contour Graph | `Contour` objects, `ContourGraph` (NetworkX DiGraph), `ContourMatch` |
| Regions | `RegionSlice`, `region_table` (DataFrame) |
| Relationships | `DE9IM`, `DE27IM`, `RelationshipType`, `StructureRelationship` |
| Final Output | `StructureSet.relationship_graph` (NetworkX DiGraph with relationship edges) |

---

## Critical Interfaces

**DICOM → Processing:**
```python
StructureSet(dicom_structure_file=dicom_file)
  ↓ Calls build_from_dicom_file()
  ↓ Passes dicom_file.contour_points to build_from_slice_data()
```

**Graph Construction → Relationship:**
```python
structure.finalize(slice_sequence)  # Finalizes structure data
structure_set.calculate_relationships()  # Uses finalized structures
  ↓ Calls structure_a.relate(structure_b)  # Returns DE27IM
  ↓ de27im.identify_relation()  # Returns RelationshipType
```
