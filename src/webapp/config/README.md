# Relationship Symbols Configuration

This directory contains the configuration file for customizing relationship symbols and colors in the StructureRelations webapp.

## Files

- `relationship_symbols.json` - Main configuration file for relationship visualization

## Customization Guide

### Configuration Structure

The `relationship_symbols.json` file defines how each relationship type is displayed in the matrix:

```json
{
  "description": "Customizable symbols and colors for relationship types",
  "version": "1.0",
  "relationships": {
    "RELATIONSHIP_TYPE": {
      "symbol": "Unicode character or ASCII",
      "label": "Display name",
      "description": "Explanation shown in tooltip",
      "color": "#RRGGBB hex color code"
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
"CONTAINS": {
  "symbol": "⊃",
  "label": "Contains",
  "description": "Structure A fully encloses structure B",
  "color": "#22c55e"
}
```

#### Use ASCII characters instead of Unicode
```json
"OVERLAPS": {
  "symbol": "X",
  "label": "Overlaps",
  "description": "Structures share common volume",
  "color": "#dc2626"
}
```

### Applying Changes

1. Edit `relationship_symbols.json`
2. Save the file
3. Refresh the webapp in your browser
4. The new symbols and colors will be loaded automatically

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
