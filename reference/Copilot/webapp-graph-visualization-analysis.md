# Webapp Relationship Graph Visualization Analysis

## Frontend Graph Rendering
- **Library**: vis-network (UMD CDN: `https://unpkg.com/vis-network/standalone/umd/vis-network.min.js`)
- **Container**: `<div id="networkDiagram" class="network-diagram"></div>` (600px height)
- **JS Class**: `WebAppClient.renderDiagram(data)` at [app.js#L1315-L1420](app.js#L1315-L1420)
- **Node Creation**: Maps data.nodes → vis-network nodes with id, label, color, shape, title, font
- **Edge Creation**: Maps data.edges → vis-network edges with from, to, label, color, width, dashes, arrows
- **Physics**: Barnes-Hut with gravitationalConstant=-2000, springConstant=0.04, springLength=150

## Current Tooltip Support
- **Nodes**: Have `title` property populated with structure info (name, ROI, type, volume, regions)
- **Edges**: NO tooltip property exists in DiagramEdge model  
- **CSS**: `.network-diagram` is 600px height, no special tooltip styling

## Existing Legend
- **Location**: Matrix tab only (`#tab-matrix`), NOT diagram tab
- **Type**: Hardcoded HTML static legend with relationship symbols
- **Structure**: `.legend-grid` (grid layout) → `.legend-item` (flex rows with symbol, label, description)
- **Content**: 9 relationship symbols (⊂, ∩, |, ○, △, ⊕, ⊏, ∅, =)
- **NOT for diagram**: Legend doesn't reflect edge styles, directions, arrows, or node shapes

## Backend Data Contract

### DiagramEdge Model (lines 125-132)
```python
class DiagramEdge(BaseModel):
    from_node: int
    to_node: int
    label: str          # Relationship type (e.g., "CONTAINS", "[CONTAINS]" for logical)
    color: str          # Hex color from diagram_settings.json
    width: int
    dashes: bool        # Edge style
    arrows: Optional[str] = None  # "to", "to;from", or None
```
**GAP**: No `tooltip` or `title` field for edge hover text

### DiagramNode Model (lines 117-123)
```python
class DiagramNode(BaseModel):
    id: int
    label: str
    color: str          # From DICOM ROI display color
    shape: str          # From diagram_settings.json shape_map
    title: str          # Tooltip with ROI info
```

### Diagram Endpoint
- **Path**: POST `/api/diagram`
- **Request**: MatrixRequest (session_id, row_rois, col_rois, show_disjoint, logical_relations_mode)
- **Response**: DiagramResponse (nodes[], edges[])
- **Edge Styling**: Loaded from config/diagram_settings.json `relationship_styles` dict

## Configuration Files

### diagram_settings.json
- **Path**: [src/webapp/config/diagram_settings.json](src/webapp/config/diagram_settings.json)
- **content**: Node shapes (GTV→ellipse, EXTERNAL→box, etc.) + relationship_styles
- **Relationship styles for edges**:
  - Keys: CONTAINS, WITHIN, OVERLAPS, BORDERS, CONFINES, CONFINED, SURROUNDS, ENCLOSED, SHELTERS, SHELTERED, PARTITIONED, PARTITIONS, EQUAL, DISJOINT, UNKNOWN
  - Fields: color (hex), width (int), dashes (bool), arrows ("to", "to;from", null)

### relationship_definitions.json
- **Path**: [src/relationship_definitions.json](src/relationship_definitions.json)
- **Contains**: symbol (⊂, ∩, etc.), label, description, symmetric flag, reversed_arrow flag
- **Used by**: Backend /api/config/symbols endpoint

## Frontend Event Handling
- **Node click**: `this.network.on('click')` logs clicked node
- **Node doubleClick**: Removes structure from diagram
- **Edge hover**: NOT IMPLEMENTED - would require network.on('hoverEdge')
- **Legend toggle**: `toggleLegend()` shows/hides hardcoded legend (toggles #legendContent display)

## CSS Styling
- **Custom properties**: --rel-{type}-color (CSS vars for colors)
- **Legend classes**: 
  - `.legend-grid`: grid layout
  - `.legend-item`: flex container with symbol/label/description
  - `.legend-symbol`: monospace 1.25rem font
  - `.legend-label`: 0.9rem bold
  - `.legend-description`: 0.8rem secondary text

## UI Tests
- **Framework**: Selenium 4.38.0
- **Test files**: [test_webapp_selenium.py](test_webapp_selenium.py), [test_webapp_api_integration.py](test_webapp_api_integration.py)
- **WebAppTestHelper**: Helper class for upload, selection, processing
- **Current diagram tests**: Minimal - only checks if server starts; no test for diagram rendering, edges, tooltips, or legend

## Gaps for Edge Tooltips + Legend

1. **DiagramEdge model**: Missing `tooltip` or `title` field
2. **Backend**: No field construction for edge tooltips in get_diagram_data()
3. **Frontend**: No edge hover listener or tooltip display logic
4. **Legend placement**: Currently on Matrix tab; diagram tab needs separate or combined legend
5. **Legend content**: Must include edge styles (color, arrows, dashes), node shapes, direction indicators
6. **Tests**: Need Selenium tests for:
   - Edge label visibility/visibility toggle
   - Edge tooltip on hover
   - Legend rendering on diagram tab
   - Legend updates when diagram changes

## Dependencies
- Backend: FastAPI, Pydantic, NetworkX, Pandas
- Frontend: vis-network (UMD), SortableJS
- Testing: Selenium, pytest, requests
