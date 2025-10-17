# StructureRelations Project - AI Coding Agent Instructions

## Project Overview
This project analyzes spatial relationships between DICOM RT (Radiotherapy) structures using computational geometry. The core algorithm builds 3D structure representations from 2D contour slices and classifies relationships using DE-9IM (Dimensionally Extended 9-Intersection Model) extended to 27 dimensions for 3D structures.

## Architecture

### Core Data Flow
1. **DICOM Input** (`dicom.py`) → Parses RT Structure Set files
2. **Contour Building** (`contours.py`) → Creates 2D polygons from contour points, builds slice sequence
3. **Graph Construction** (`contour_graph.py`) → Links contours across slices using NetworkX graphs
4. **3D Regions** (`structures.py`, `region_slice.py`) → Aggregates contours into 3D structures
5. **Relationship Analysis** (`relations.py`) → Computes DE-27IM relationships and classifies them

### Key Classes & Their Roles
- **`StructureSet`** (`structure_set.py`): Top-level container managing multiple structures and their relationships
- **`StructureShape`** (`structures.py`): Represents a single 3D structure with its contour graph and region slices
- **`Contour`** (`contours.py`): Individual 2D polygon on a slice with ROI, slice_index, and shapely.Polygon
- **`RegionSlice`** (`region_slice.py`): All contours from one structure on a single slice (handles multi-region structures)
- **`DE27IM`** (`relations.py`): Extended DE-9IM relationship encoding (hull, exterior, contour relationships)
- **`ContourGraph`**: NetworkX graph where nodes are contours, edges connect matching contours on adjacent slices

### Type System (`types_and_classes.py`)
- `ROI_Type`: Structure identifier (int)
- `SliceIndexType`: Z-axis position in cm (float)
- `ContourIndex`: Unique contour identifier `(ROI, SliceIndex, uniqueness_int)`
- `PolygonType`: `shapely.Polygon | shapely.MultiPolygon`
- Global precision constants: `TRANSVERSE_PRECISION = 0.01`, `SLICE_INDEX_PRECISION = 0.01`, `PRECISION = 3`

## Critical Conventions

### Coordinate Systems & Precision
- **DICOM coordinates**: Right-handed 3D system, Z-axis perpendicular to slices
- **Floating-point handling**: Coordinates rounded to `TRANSVERSE_PRECISION` (0.01 cm) to avoid geometry errors
- **Resolution issues**: Shapely relationship computations are sensitive to rounding; see `src/notebooks/relations/ResolutionAnalysis.ipynb` for ongoing work on `simplify()`, `snap()`, and `make_valid()` approaches

### Relationship Classification
Relationships are identified by pattern-matching against DE-27IM bit patterns (see `RelationshipTest` dataclass):
- **CONTAINS**: One structure fully inside another
- **SURROUNDS**: Structure inside a hole of another
- **SHELTERS**: Structure within convex hull but not touching (see `docs/Shelters Issues.md` for known complexities)
- **OVERLAPS**, **PARTITION**, **CONFINES**, **BORDERS**, **DISJOINT**, **EQUALS**

**Important**: The convex hull definition affects SHELTERS detection. Current implementation uses per-region hulls, not a single hull encompassing all regions (documented issue in Shelters Issues.md).

### Contour Graph Linking
Contours on adjacent slices are connected if:
1. Same ROI number
2. Same hole type (exterior vs. hole)
3. Convex hulls intersect on slice plane

## Development Workflows

### Environment Setup
- **Conda environment**: `StructureRelations` (see `environment.yml`)
- **Python interpreter**: `D:\.conda\envs\StructureRelations\python.exe`
- **PYTHONPATH**: Auto-configured to include `src/`, `tests/`, `examples/`
- **Activate**: Use VS Code's Python extension (already configured in workspace settings)

### Running Tests
- **Framework**: pytest (configured in workspace settings: `"python.testing.pytestEnabled": true`)
- **Test location**: `tests/` directory
- **Run tests**: Use VS Code Test Explorer or `pytest` command
- **Test files**: `test_2D_relations.py`, `test_3D_relations.py`, etc.
- **Test pattern**: Classes group related tests (e.g., `TestContains`, `TestSurrounds`)
- **Helper functions**: `debug_tools.py` provides geometric primitives (`make_sphere`, `make_box`, `make_vertical_cylinder`, etc.)

**Known Issue**: Debug mode for tests may fail with `[WinError 5] Access is denied` on Windows. Run mode works correctly. This is a permissions issue with the debugger launcher, not the code.

### Debugging & Visualization
- **Notebooks**: Jupyter notebooks in `src/notebooks/` for exploratory analysis
- **Visualization**: `debug_tools.plot_ab()` shows polygon differences (blue=only A, green=only B, orange=intersection)
- **Logging**: Configure via `logging.basicConfig(level=logging.DEBUG)` for detailed contour/relationship tracing

### Common Tasks
**Add new relationship type**:
1. Add enum to `RelationshipType` in `relations.py`
2. Define `RelationshipTest` with mask/value patterns
3. Add to `RELATIONSHIP_TESTS` list
4. Create test cases in `tests/test_2D_relations.py` and `tests/test_3D_relations.py`

**Process new DICOM file**:
```python
from dicom import DicomStructureFile
from structure_set import StructureSet

dicom_file = DicomStructureFile('path/to/RS.dcm')
structure_set = StructureSet(dicom_structure_file=dicom_file)
structure_set.finalize()
structure_set.calculate_relationships()
```

## Project-Specific Patterns

### Error Handling
- Custom exceptions: `InvalidSlice`, `InvalidContour`, `InvalidContourRelation` (all inherit from `StructuresException`)
- Validation happens in constructors/setters (e.g., `ContourPoints.validate_points()`)

### NetworkX Integration
- Contour graphs use node attributes: `contour` (Contour object), `roi`, `slice_index`
- Edge attributes: `match` (ContourMatch object with overlap metrics)
- Region identification uses `nx.connected_components()` to find contiguous 3D volumes

### Shapely Geometry Utilities
- `utilities.make_solid()`: Fills holes in polygons (creates exterior-only version)
- `utilities.poly_round()`: Rounds polygon coordinates to precision
- `utilities.points_to_polygon()`: Converts point list to shapely.Polygon with validation

## Files to Reference
- **`types_and_classes.py`**: Type definitions, global constants, custom exceptions
- **`relations.py`**: DE-27IM implementation, relationship classification logic
- **`structure_set.py`**: Entry point for processing multiple structures
- **`debug_tools.py`**: Test geometry generators and visualization
- **`docs/Shelters Issues.md`**: Known geometric challenges with SHELTERS relationship
- **`StructureRelations.code-workspace`**: VS Code configuration (Python paths, test settings)

## Testing Strategy
- **2D tests**: Verify relationship detection on single slice pairs
- **3D tests**: Multi-slice scenarios (cylinders, spheres, nested structures)
- **Test data**: DICOM files in `tests/` directory (e.g., `RS.GJS_Struct_Tests.Relations.dcm`)
- **Assertion pattern**: `assert relation_type == RelationshipType.EXPECTED_TYPE`
