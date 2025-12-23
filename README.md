# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Improve Structure Selection table
- Structure Selection should include Structure ID, DICOM Type, and Code Meaning (label)
- Structure Selection should take up only a single line per structure and be more compact
- Structure with no contours should not be displayed in Structure Selection

2a Fix StructureSet.Summary() method
- Round volumes to 2 decimal places
- Add to include DICOM Type, and Code Meaning
- Add slice Max and Slice Min
- Add count of regions per structure

2. Improve Structure Summary table
- Structure Summary should include DICOM Type, Code Meaning (label), number of regions, total volume, slice range (in the form f'{min_slice} to {max_slice}')
- Structure Summary should take up less vertical space
- Structure Summary should be a collapsible block
- Structure Summary should be sortable by DICOM Type, Label, Volume, etc.
- Structure Summary should include tick boxes to include or exclude them from the Relationship Matrix row and column selection

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
