# StructureRelations Project - AI Coding Agent Instructions

## Project Overview
This project analyzes spatial relationships between DICOM RT (Radiotherapy) structures using computational geometry. The core algorithm builds 3D structure representations from 2D contour slices and classifies relationships using DE-9IM (Dimensionally Extended 9-Intersection Model) extended to 27 dimensions for 3D structures.

## Formatting conventions
- Use 4 spaces for indentation
- Prefer limiting lines to 80 characters, accept lines up to 100 characters
- use single quotes for strings unless double quotes are needed for interpolation or nested quotes
- use google style docstring for functions and classes
- Don't add a newline at the beginning of a doc string
- use type hints for function arguments and return values
- Use f-string for string formatting except for logging statements where lazy formatting is preferred
- Use str.join() for string concatenation instead of + operator

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
- Global precision constants: `DEFAULT_TRANSVERSE_TOLERANCE = 0.01`, `SLICE_INDEX_PRECISION = 0.01`

## Critical Conventions

### Coordinate Systems & Precision
- **DICOM coordinates**: Right-handed 3D system, Z-axis perpendicular to slices
- **Floating-point handling**: Coordinates rounded to `DEFAULT_TRANSVERSE_TOLERANCE` (0.01 cm) to avoid geometry errors


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
- Code and documents in the reference and the FreeCAD_Scripts folders are not part of the main project and should not be referenced or modified.

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
- **Visualization**: `contour_plotting.plot_ab()` shows polygon differences (blue=only A, green=only B, orange=intersection)
- `contour_plotting.plot_roi_slice()` calls plot_ab to visualize all contours on a slice. Takes a StructureSet, slice index and one or two ROI numbers or names.
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
- `utilities.poly_round()`: Rounds polygon coordinates to specified tolerance.
- `utilities.points_to_polygon()`: Converts point list to shapely.Polygon with validation

## Files to Reference
- **`types_and_classes.py`**: Type definitions, global constants, custom exceptions
- **`utilities.py`**: Shared geometry utilities and helpers
- **`contours.py`**: 2D contour creation and validation
- **`dicom.py`**: DICOM RT Structure Set parsing- **`contour_graph.py`**: Contour graph construction and linking logic
- **`region_slice.py`**: 2D slice of contours for one structure, multi-region handling
- **`contour_plotting.py`**: Visualization functions for contours and relationships
- **`structures.py`**: 3D structure representation and region slicing
- **`structure_set.py`**: Entry point for processing multiple structures
- **`relations.py`**: DE-27IM implementation, relationship classification logic
- **`debug_tools.py`**: Test geometry generators and visualization
- **`docs/Shelters Issues.md`**: Known geometric challenges with SHELTERS relationship
- **`StructureRelations.code-workspace`**: VS Code configuration (Python paths, test settings)

## Testing Strategy
- **2D tests**: Verify relationship detection on single slice pairs
- **3D tests**: Multi-slice scenarios (cylinders, spheres, nested structures)
- **Test data**: DICOM files in `tests/` directory (e.g., `RS.GJS_Struct_Tests.Relations.dcm`)
- **Assertion pattern**: `assert relation_type == RelationshipType.EXPECTED_TYPE`
