# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Clean up layout of webapp
- Structures that are not in either the From or To list should not be shown in the diagram
- Structures in either the "From" or "To" list should be displayed in the diagram
- All Edges representing symmetric relationships should be displayed for all structures in either the "From" or "To" list
- For directional relationships (non symmetric), edges should be displayed only if the "From" structure is in the From list and the "To" structure is in the To list
- Currently "disjoint" edges are not visible in the diagram even when "Show Disjoint" is selected.
-Double-clicking a structure in the diagram should remove it from the diagram (and from either/both the From or To list as applicable)

- Double-clicking a structure in either the From or To list should remove it from the list and from the diagram
- Double-clicking a structure in the Available list should add it to the bottom of either the From or To list (depending on which list was last clicked) and add it to the diagram

- Add Zoom & Pan options to the Contour Plotting area
- If CT images are available, add option to show/hide CT background in Contour Plotting area

2. Begin work on Metric Calculations

3. Identify "Logical" relations

4. Define complimentary relations
