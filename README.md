# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Improve Structure Selection table
- It appears that the relationships are being calculated more than once.  This should be optimized to only calculate once per structure set load.
- Structure Selection should include Structure ID, DICOM Type, and Code Meaning (label)

3. Enhance Relationship symbol visualization:
- The symbol font should be about 1.5 times larger
- The symbols should also have a color coding scheme (e.g. green for 'contains', red for 'overlaps', blue for 'adjacent', etc.)
- It should be possible to customize the relation symbols and colors via a config file
- There should be a tooltip or legend explaining the relation symbols
- Mouse hover over symbol should give relation label
- There should be an option to select which relations to display in the Relationship Matrix

4. Enhance Relationship Matrix layout:
- Relationship Matrix Configuration should be collapsible
- Relationship Matrix Configuration should include filters for DICOM Type
- Relationship Matrix Configuration should be sortable by DICOM Type as default

5. Contour plotting display
- Add an additional section that allows plotting of individual slices with structure contours overlaid

6. Relationship Diagram
- Add a section that generates a graphical diagram (e.g., network graph) showing relationships between structures.  See the code in the src/diagram folder for a starting point.
