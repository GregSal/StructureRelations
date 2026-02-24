# Relationship Symbols Configuration

This directory documents the configuration for relationship visualization in the StructureRelations webapp.

## Configuration Source

Relationship symbols and colors are now defined in **`src/relationship_definitions.json`** (consolidated with relationship type definitions).

The `/api/config/symbols` endpoint dynamically extracts symbol, color, and label information from this file.

## Files

- `relationship_definitions.json` (in parent src/ directory) - Main configuration file for all relationship properties including symbols, colors, and styling
- `diagram_settings.json` - Configuration for network diagram node shapes and visualization settings

## Customization Guide

### Configuration Structure

The `relationship_definitions.json` file defines all properties for each relationship type:

```json
{
  "Introduction": "Description of relationships",
  "logical_relationships": {
    "transparency": 20,
    "description": "Settings for logical relationship display"
  },
  "Relationships": [
    {
      "relation_type": "RELATIONSHIP_TYPE",
      "symbol": "Unicode character or ASCII",
      "label": "Display name",
      "description": "Explanation shown in tooltip",
      "color": "#RRGGBB hex color code",
      "relation_style": {
        "color": "#RRGGBB",
        "width": 2,
        "dashes": false,
        "arrows": "to"
      }
    }
  ]
}
```

### Available Relationship Types

- **CONTAINS** - Structure A fully encloses structure B
- **OVERLAPS** - Structures share common volume
- **BORDERS** - Structures touch at boundaries
- **SURROUNDS** - Structure B is within a hole in A
- **SHELTERS** - B within convex hull of A, not touching
- **PARTITION** - Structures partition space between them
- **CONFINES** - B contacts inner surface of A
- **DISJOINT** - Structures are completely separated
- **EQUALS** - Same structure (diagonal of matrix)
- **UNKNOWN** - Relationship not determined

### Customizing Symbols

You can change the symbol for any relationship type. Common Unicode symbols:

- Mathematical: ⊂ ⊃ ∩ ∪ ⊕ ⊗ ⊏ ⊐ ∅ ∈ ∉
- Geometric: ○ ● △ ▽ ◊ □ ■
- Arrows: ← → ↑ ↓ ↔ ⇐ ⇒ ⇔
- Others: | ∥ ⊥ ≈ ≠ ≡ ≤ ≥

### Customizing Colors

Colors are specified in hex format (#RRGGBB). Recommended color schemes:

**Semantic (Default)**:
- Green (#10b981) - Positive/contains
- Red (#ef4444) - Conflict/overlaps
- Blue (#3b82f6) - Adjacent/borders
- Purple (#8b5cf6) - Special/surrounds
- Amber (#f59e0b) - Warning/shelters

**High Contrast**:
- Use darker colors for better visibility
- Ensure sufficient contrast against white background
- Test with color blindness simulators

**Custom Themes**:
- Monochrome: Use grayscale values
- Colorblind-friendly: Use patterns that work for common color blindness types

### Example Modifications

#### Change CONTAINS to use a different symbol and color
```json
{
  "relation_type": "CONTAINS",
  "symbol": "⊃",
  "label": "Contains",
  "description": "Structure A fully encloses structure B",
  "color": "#22c55e",
  "relation_style": {
    "color": "#22c55e",
    "width": 4,
    "dashes": false,
    "arrows": "to"
  }
}
```

#### Use ASCII characters instead of Unicode
```json
{
  "relation_type": "OVERLAPS",
  "symbol": "X",
  "label": "Overlaps",
  "description": "Structures share common volume",
  "color": "#dc2626",
  "relation_style": {
    "color": "#dc2626",
    "width": 5,
    "dashes": false,
    "arrows": null
  }
}
```

### Diagram Settings Configuration

The `diagram_settings.json` file defines node shapes for the network diagram visualization:

```json
{
  "description": "Configuration for network diagram visualization",
  "version": "1.0.0",
  "node_shapes": {
    "description": "Node shapes mapped to DICOM structure types",
    "shape_map": {
      "GTV": "star",
      "CTV": "hexagon",
      "PTV": "diamond",
      "EXTERNAL": "box",
      "ORGAN": "ellipse",
      "AVOIDANCE": "triangle",
      "BOLUS": "dot",
      "SUPPORT": "square",
      "FIXATION": "triangleDown"
    },
    "default_shape": "ellipse"
  }
}
```

**Available Shapes**:
- `star` - Star shape
- `hexagon` - Six-sided polygon
- `diamond` - Diamond/rhombus shape
- `box` - Rectangle/square
- `ellipse` - Oval/circle (default)
- `triangle` - Upward pointing triangle
- `triangleDown` - Downward pointing triangle
- `dot` - Small circle
- `square` - Square with rounded corners

**Customization Example**:
To change all ORGAN structures to use a triangle shape:
```json
{
  "node_shapes": {
    "shape_map": {
      "ORGAN": "triangle"
    },
    "default_shape": "ellipse"
  }
}
```

### Applying Changes

1. Edit `src/relationship_definitions.json`
2. Save the file
3. Refresh the webapp in your browser
4. The new symbols, colors, and styles will be loaded automatically

**Note**: If the configuration file is invalid JSON or missing, the webapp will fall back to default symbols and colors.

### Validation

To validate your JSON configuration:
- Use a JSON validator (e.g., jsonlint.com)
- Ensure all required fields are present
- Use valid hex color codes
- Unicode symbols should be properly encoded in UTF-8

### Troubleshooting

**Symbols not displaying correctly**:
- Ensure your font supports the Unicode characters
- Try using ASCII alternatives
- Check browser console for errors

**Colors not applying**:
- Verify hex color format (#RRGGBB)
- Check browser console for CSS errors
- Clear browser cache

**Config not loading**:
- Verify JSON syntax is valid
- Check file permissions
- Review server logs for errors
