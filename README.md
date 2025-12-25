# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Improve Structure Selection table
- ~~It appears that the relationships are being calculated more than once.  This should be optimized to only calculate once per structure set load.~~ ✅ **COMPLETED** - Added check to skip recalculation if relationships already exist
- ~~Structure Selection should include Structure ID, DICOM Type, and Code Meaning (label)~~ ✅ **COMPLETED** - Added to both Structure Selection and Structure Summary tables

3. Enhance Relationship symbol visualization:
- The symbol font should be about 1.5 times larger
- The symbols should also have a color coding scheme (e.g. green for 'contains', red for 'overlaps', blue for 'adjacent', etc.)
- It should be possible to customize the relation symbols and colors via a config file
- There should be a tooltip or legend explaining the relation symbols
- Mouse hover over symbol should give relation label
- There should be an option to select which relations to display in the Relationship Matrix

4. Enhance Relationship Matrix layout:
- ~~Relationship Matrix Configuration should be collapsible~~ ✅ **COMPLETED** - Added collapsible card with toggle button
- ~~Relationship Matrix Configuration should include filters for DICOM Type~~ ✅ **COMPLETED** - Added dropdown filters for rows and columns
- ~~Relationship Matrix Configuration should be sortable by DICOM Type as default~~ ✅ **COMPLETED** - Structures now sorted by DICOM Type, then by name

5. Contour plotting display
- Add an additional section that allows plotting of individual slices with structure contours overlaid

6. Relationship Diagram
- ~~Add a section that generates a graphical diagram (e.g., network graph) showing relationships between structures.~~ ✅ **COMPLETED** - Added interactive network diagram using vis-network with:
  - Collapsible diagram card
  - Color-coded nodes by structure color
  - Shape-coded nodes by DICOM type (star for GTV, diamond for PTV, etc.)
  - Color-coded edges by relationship type
  - Toggle for showing/hiding edge labels
  - Interactive navigation (pan, zoom, click)
  - Tooltips showing structure details
  - Auto-refresh when matrix is updated
