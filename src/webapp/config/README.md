# Relationship Symbols Configuration

This directory documents the configuration for relationship visualization in the StructureRelations webapp.

## Configuration Source

Relationship definitions are split between two files:
- **`src/relationship_definitions.json`** - Core relationship types, symbols, labels, and logical definitions
- **`src/webapp/config/diagram_settings.json`** - Visual styling including colors and edge styles for network diagrams

The `/api/config/symbols` endpoint dynamically extracts symbol and label
information from relationship_definitions.json, and colors from
diagram_settings.json.

## Files

- `relationship_definitions.json` (in parent src/ directory) - Main configuration file for all relationship type definitions
- `diagram_settings.json` - Visual styling for diagrams including colors, edge styles, and node shapes

## Customization Guide

### Configuration Structure

The `relationship_definitions.json` file defines the core properties for each relationship type:

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
      "pattern": "DE-9IM pattern",
      "mask": "Binary mask",
      "value": "Binary value"
    }
  ]
}
```

The `diagram_settings.json` file contains visual styling:

```json
{
  "relationship_styles": {
    "RELATIONSHIP_TYPE": {
      "color": "#RRGGBB",
      "width": 2,
      "dashes": false,
      "arrows": "to"
    }
  }
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
- **EQUAL** - Same structure (diagonal of matrix)
- **UNKNOWN** - Relationship not determined

### Customizing Symbols

You can change the symbol for any relationship type. Common Unicode symbols:

- Mathematical: ⊂ ⊃ ∩ ∪ ⊕ ⊗ ⊏ ⊐ ∅ ∈ ∉
- Geometric: ○ ● △ ▽ ◊ □ ■
- Arrows: ← → ↑ ↓ ↔ ⇐ ⇒ ⇔
- Others: | ∥ ⊥ ≈ ≠ ≡ ≤ ≥

### Customizing Colors

Colors for matrix display are specified in `diagram_settings.json` under
`relationship_styles.*.color` in hex format (#RRGGBB). Edge styles for network
diagrams are defined in `relationship_styles`. Recommended color schemes:

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

#### Change CONTAINS to use a different symbol
In `src/relationship_definitions.json`:
```json
{
  "relation_type": "CONTAINS",
  "symbol": "⊃",
  "label": "Contains",
  "description": "Structure A fully encloses structure B"
}
```

#### Change CONTAINS color and edge style
In `src/webapp/config/diagram_settings.json`:
```json
{
  "relationship_styles": {
    "CONTAINS": {
      "color": "#22c55e",
      "width": 4,
      "dashes": false,
      "arrows": "to"
    }
  }
}
```

#### Use ASCII characters instead of Unicode
In `src/relationship_definitions.json`:
```json
{
  "relation_type": "OVERLAPS",
  "symbol": "X",
  "label": "Overlaps",
  "description": "Structures share common volume"
}
```

Then update the style in `src/webapp/config/diagram_settings.json`:
```json
{
  "relationship_styles": {
    "OVERLAPS": {
      "color": "#dc2626",
      "width": 5,
      "dashes": false,
      "arrows": null
    }
  }
}
```

### Diagram Settings Configuration

The `diagram_settings.json` file defines colors and edge styles for relationships:

```json
{
  "description": "Configuration for network diagram visualization",
  "version": "1.0.0",
  "relationship_styles": {
    "CONTAINS": {
      "color": "#00CED1",
      "width": 4,
      "dashes": false,
      "arrows": "to"
    }
  },
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

1. Edit `src/relationship_definitions.json` for symbol, label, and description changes
2. Edit `src/webapp/config/diagram_settings.json` for color and style changes
3. Save both files
4. Refresh the webapp in your browser
5. The new symbols, colors, and styles will be loaded automatically

**Note**: If the configuration files are invalid JSON or missing, the webapp will fall back to default symbols and colors.

### Validation

To validate your JSON configurations:
- Use a JSON validator (e.g., jsonlint.com)
- Ensure all required fields are present in `relationship_definitions.json`
- For visual customizations, edit `diagram_settings.json`
- Use valid hex color codes (#RRGGBB format)
- Unicode symbols should be properly encoded in UTF-8
- Verify relationship type names match exactly between both files

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
