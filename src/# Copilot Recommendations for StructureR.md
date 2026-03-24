# Copilot Recommendations for StructureRelations Architecture

Based on my analysis of the StructureRelations codebase compared to established projects like **Shapely**, **GeoPandas**, **GDAL**, **Open3D**, and **NetworkX**, I've identified both strengths and areas for architectural improvement. Here are my recommendations:

---

## **1. DATA FLOW ARCHITECTURE**

### **Current Issues:**

- **Pipeline re-materialization**: Data gets serialized/deserialized multiple times (ContourPoints → Contour objects → ContourGraph nodes → RegionSlice → DE9IM/DE27IM)
- **Lazy vs. eager evaluation**: Some computations happen at multiple points; unclear when/where calculations occur
- **No streaming capability**: Entire dataset must fit in memory; can't process large DICOM sets incrementally
- **Mutable graph modification**: Contour graphs are modified in-place across multiple functions, making it hard to track state changes

### **Recommendations (following GeoPandas/GDAL patterns):**

```python
# Pattern 1: Use lazy evaluation with explicit materialization points
class StructureSet:
    def __init__(self):
        self._structures = {}  # Lazy-loaded structures
        self._relationships = None  # Computed on demand

    @property
    def relationships(self):
        """Lazy relationship computation"""
        if self._relationships is None:
            self._relationships = self.calculate_relationships()
        return self._relationships

# Pattern 2: Immutable data transformations (Shapely-style)
def build_contour_graph(contour_table: DataFrame,
                       roi: ROI_Type) -> ContourGraph:
    """Returns new graph; doesn't mutate inputs"""
    graph = nx.DiGraph()
    # ... operations create new graph
    return graph  # Return immutable copy

# Pattern 3: Pipeline with clear intermediate outputs
class ROIProcessingPipeline:
    """Explicit data transformation stages"""
    def __init__(self, dicom_file: DicomStructureFile):
        self.stage_1_raw_contours = self._extract_raw_contours()
        self.stage_2_contour_objects = self._create_contours()
        self.stage_3_contour_graph = self._build_graph()
        self.stage_4_regions = self._build_regions()
        self.stage_5_relationships = self._calculate_relationships()
```

---

## **2. CODE STRUCTURE & MODULARITY**

### **Current Issues:**

- **God classes**: `StructureSet` has 50+ methods mixing concerns (I/O, graph building, relationship calculation, export)
- **Circular imports**: Contour ↔ ContourGraph ↔ RegionSlice have tight coupling
- **No clear interfaces**: Multiple classes do similar things (SliceSequence, RegionSlice both manage slices)
- **Utility bloat**: utilities.py mixes geometry, rounding, type handling
- **Mixed responsibilities**: `StructureShape` does: graph building, volume calculation, relationship computation

### **Recommendations (following GDAL/Fiona patterns):**

```python
# Pattern 1: Single Responsibility Principle
# ❌ Current: class StructureShape(graph_builder, volume_calculator, relationship_computer)
# ✅ Recommended:

class Structure(ABC):
    """Core data model"""
    roi: ROI_Type
    name: str
    @property
    def volume(self) -> float: ...

class StructureGraphBuilder:
    """Graph construction only"""
    def build(self, contours: List[Contour]) -> ContourGraph: ...

class StructureVolumeCalculator:
    """Volume computation only"""
    def calculate(self, graph: ContourGraph) -> VolumeMetrics: ...

class StructureRelationshipAnalyzer:
    """Relationship logic only"""
    def relate(self, structure_a: Structure,
               structure_b: Structure) -> DE27IM: ...

# Pattern 2: Strategy pattern for operations (Open3D/scikit-image style)
class ContourGraphProcessor:
    """Abstract interface for graph processing"""
    @abstractmethod
    def process(self, graph: ContourGraph) -> ContourGraph: ...

class InterpolationProcessor(ContourGraphProcessor):
    """Add interpolated contours"""
    def process(self, graph: ContourGraph) -> ContourGraph: ...

class HoleTypeProcessor(ContourGraphProcessor):
    """Set hole types"""
    def process(self, graph: ContourGraph) -> ContourGraph: ...

# Usage: Chain of responsibility
pipeline = (InterpolationProcessor()
           | BoundaryProcessor()
           | HoleTypeProcessor())
final_graph = pipeline.process(initial_graph)

# Pattern 3: Separate I/O layer (Fiona-style)
class DicomReader:
    """Handles all DICOM reading"""
    def read(self, path: Path) -> DicomStructureFile: ...
    def to_contour_points(self, dataset) -> List[ContourPoints]: ...

class StructureWriter:
    """Handles all export"""
    def write_to_json(self, structure_set, path): ...
    def write_to_hdf5(self, structure_set, path): ...
```

---

## **3. DATA STRUCTURES**

### **Current Issues:**

- **NewType over TypedDict**: ROI_Type as NewType is not enforced at runtime; TypedDict provides better IDE support
- **Mixed homogeneity**: DataFrames mixed with custom classes (RegionSlice, Contour) make data handling inconsistent
- **No clear schemas**: No validation of DataFrame column types or required fields
- **Tuple-based indexes**: `ContourIndex = (ROI, SliceIndex, int)` is error-prone; should be a dataclass
- **Wide DataFrames**: region_table and contour_lookup have many columns; fragile to refactoring

### **Recommendations (following GeoPandas/Shapely patterns):**

```python
# Pattern 1: Use TypedDict + dataclasses for clarity
from typing import TypedDict
from dataclasses import dataclass

class ContourRecord(TypedDict):
    """Strongly-typed contour record"""
    roi: ROI_Type
    slice_index: SliceIndexType
    polygon: Polygon
    area: float
    hole_type: str
    region_index: RegionIndex

@dataclass(frozen=True)  # Immutable for graph nodes
class ContourIndex:
    roi: ROI_Type
    slice_index: SliceIndexType
    uniqueness_int: int

# Pattern 2: Use pandas accessor pattern (GeoPandas style)
@pd.api.extensions.register_dataframe_accessor("contour")
class ContourAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_polygon_series(self) -> pd.Series:
        """Convert Points column to Polygon series"""
        return self._obj['Points'].apply(points_to_polygon)

# Usage: df.contour.to_polygon_series()

# Pattern 3: Nested dataclass for complex objects
@dataclass
class VolumeMetrics:
    physical: float
    exterior: float
    hull: float

@dataclass
class StructureMetadata:
    roi: ROI_Type
    name: str
    volume_metrics: VolumeMetrics
    contour_count: int
    slice_range: Tuple[SliceIndexType, SliceIndexType]

# Pattern 4: Use Enum for restricted values (SQLAlchemy pattern)
class HoleType(Enum):
    EXTERIOR = "Exterior"
    OPEN_HOLE = "Open"
    CLOSED_HOLE = "Closed"
    UNKNOWN = "Unknown"

    # With validation
    def validate(self, value: str) -> 'HoleType':
        try:
            return HoleType(value)
        except ValueError as e:
            raise InvalidContour(f"Invalid hole type: {value}") from e
```

---

## **4. INTERFACES & API DESIGN**

### **Current Issues:**

- **Asymmetric APIs**: `relate()` requires both structures; no caching mechanism
- **Brittle constructors**: Many optional parameters with unclear defaults (tolerance, is_boundary, is_hole)
- **Silent failures**: Invalid inputs don't always raise; sometimes return empty structures
- **Inconsistent naming**: Mix of `add_*`, `build_*`, `create_*`, `calculate_*` patterns
- **No versioning**: No API stability guarantees for library consumers

### **Recommendations (following NetworkX/Shapely patterns):**

```python
# Pattern 1: Factory pattern with builder
class StructureSetBuilder:
    """Fluent builder API (scikit-learn style)"""
    def __init__(self):
        self._dicom_file = None
        self._tolerance = 0.0
        self._auto_interpolate = True

    def with_dicom_file(self, path: Path) -> 'StructureSetBuilder':
        self._dicom_file = DicomStructureFile(path)
        return self

    def with_tolerance(self, tol: float) -> 'StructureSetBuilder':
        self._tolerance = tol
        return self

    def with_auto_interpolation(self, enabled: bool) -> 'StructureSetBuilder':
        self._auto_interpolate = enabled
        return self

    def build(self) -> StructureSet:
        """Validates and constructs"""
        if not self._dicom_file:
            raise ValueError("DICOM file required")
        return StructureSet(
            dicom_file=self._dicom_file,
            tolerance=self._tolerance,
            auto_interpolate=self._auto_interpolate
        )

# Usage:
structure_set = (StructureSetBuilder()
                 .with_dicom_file('RS.dcm')
                 .with_tolerance(0.01)
                 .build())

# Pattern 2: Cache relationships (Redis/LRU pattern)
from functools import lru_cache

class StructureSet:
    @lru_cache(maxsize=1024)
    def get_relationship(self, roi_a: ROI_Type,
                        roi_b: ROI_Type) -> Optional[DE27IM]:
        """Cached relationship computation"""
        # ... expensive calculation
        return relationship

# Pattern 3: Context managers for resource management
class DicomStructureFile:
    def __enter__(self):
        self._load_dataset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup large datasets
        if hasattr(self, 'dataset'):
            del self.dataset

# Usage:
with DicomStructureFile('RS.dcm') as dicom_file:
    structure_set = StructureSet(dicom_structure_file=dicom_file)

# Pattern 4: Explicit error handling (Shapely style)
class RelationshipError(Exception):
    """Base relationship error"""
    pass

class AmbiguousRelationshipError(RelationshipError):
    """Multiple relationships match"""
    pass

class InvalidOperationError(RelationshipError):
    """Operation not supported for structure type"""
    pass

# Usage:
try:
    relation = structure_a.relate(structure_b)
except AmbiguousRelationshipError as e:
    logger.warning("Ambiguous relationship: %s", e)
```

---

## **5. SPECIFIC ARCHITECTURAL PATTERNS TO ADOPT**

| Pattern | Current | Recommended | Example Project |
|---------|---------|-------------|-----------------|
| **Graph mutation** | In-place modification | Immutable with new graphs | NetworkX 3.0+ |
| **Data access** | Direct attribute access | Property + descriptor protocol | GeoPandas (`.bounds`, `.area`) |
| **Validation** | Scattered checks | Central validator classes | Pydantic / SQLAlchemy |
| **I/O** | Integrated in classes | Separate reader/writer layer | Fiona / GDAL |
| **Relationship caching** | Recalculated each time | LRU cache + memoization | Shapely (rtree) |
| **Configuration** | Constructor parameters | Config object / constants module | GeoPandas (`options.py`) |
| **Logging** | Basic logging | Structured logging + metrics | Rich / Loguru |

---

## **6. QUICK WINS (Highest ROI Changes)**

1. **Extract I/O layer** (2-3 hours)
   - Move all DICOM reading to `DicomReader` class
   - Move all export to separate module
   - Benefit: Code reusability, testability

2. **Replace tuple-based ContourIndex** (1 hour)
   - Change to frozen dataclass
   - Benefit: Type safety, IDE support, debuggability

3. **Separate graph building from StructureShape** (3-4 hours)
   - Create `ContourGraphBuilder` class
   - Benefit: Testability, composition over inheritance

4. **Add strict validation** (2 hours)
   - Use Pydantic for DICOM inputs
   - Use dataclass field validators
   - Benefit: Earlier error detection, better error messages

5. **Implement relationship caching** (1 hour)
   - Add `@lru_cache` to relationship methods
   - Benefit: 10-100x speedup for repeated queries

---

## **7. LONG-TERM IMPROVEMENTS**

| Item | Effort | Benefit | Priority |
|------|--------|---------|----------|
| Async relationship computation (multiprocessing) | Medium | 3-5x speedup on large datasets | High |
| Stream processing for huge DICOM files | Medium | Support 1GB+ DICOM files | Medium |
| Pluggable relationship definitions (JSON → strategy) | Low | Dynamic relationship types | Low |
| Arrow/Parquet export for large results | Low | Better interoperability | Medium |
| Type stub (.pyi) files | Low | Better IDE support | Low |

---

This architecture aligns StructureRelations with proven patterns from projects like **GeoPandas** (pandas + spatial), **Shapely** (immutable geometry), and **GDAL** (plugin I/O layers), making it more maintainable and extensible for medical imaging workflows.
