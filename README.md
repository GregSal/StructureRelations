# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Identify "Logical" relations

2. Define complimentary relations

3. restructure relationships definitions to to obtain definitions from Jason file along with descriptions and display parameters

4. Begin work on Metric Calculations



## Future work

- Add a status bar to the webapp to show current status, messages, etc.  put logging.info() messages there.

- Make the progress bar update during long operations (loading structures, calculating relationships, etc.)

- If CT images are available, add option to show/hide CT background in Contour Plotting area

- I will likely want to display some of the additional information in the relationship matrix within the webapp.  What will need to change to make it easy to customize what is displayed?  (this may become something in the configuration, settings or selectable within the web page - I haven't decided what to do here yet)

- Webapp edge cases: Should verify that empty matrices, single-structure sets, and filtered matrices still serialize correctly to JSON through to_dict()—the string conversion needs to handle StructureRelationship objects with de27im=None gracefully.
