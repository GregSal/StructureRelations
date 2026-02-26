# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Web App Updates
    1. Cosmetic Changes
        - Tool Tips need to use `<br>` instead of /n
        - Need tooltips for edges
        - Put Structure Set info at the top of the form
        - Make Diagram the first tab and open to it by default
    2. Relation (Edge) Updates
        - Move shape definitions to a json file
        - Move display options to a separate definitions file from the relationship definitions
        - implement expanded options for line types
            - Different dash types
            - double lines
            - glow
    3. General Diagram Updates
        - Make separate selectors for matrix and diagram
            - Include "Copy from matrix" and "Copy from diagram" buttons
        - More flexibility in diagram layout so that nodes can be dragged without pulling the entire diagram
            - Move parameters for layout algorithm to json file.
    4. Processing Updates
        - Add a status bar to the webapp to show current status, messages, etc. put logging.info() messages there.
        - Make the progress bar update during long operations (loading structures, calculating relationships, etc.)
    5. Plot Contours update
        - Add a toggle-able legend to Plot Contours
        - Allow Plot contours to select more than two structures in an orderable list
        - Add option to use 3 contours for relationship mode to show the relationship between three structures (e.g. A, B, and A-B)
        - Add option to enable or disable plotting margins (tolerance) around contours
        - Allow Plot contours to toggle between contour mode and relationship mode
            - In contour mode structures are displayed as outlines in their assigned colours (or from a default list if no colours assigned).
            - In relationship mode contours are plotted filled with colours assigned based on the relationship between the structures.
            - In relationship mode, only the first two or three structures are used
        - provide a dropdown list to select slice
            - this complements the slider bar
            - The drop down includes what structures are present and the relationships between the structures on that slice. This allows users to quickly jump to slices of interest.
            - Keep a record of the relationships calculated for every slice and for boundary slices

5. Add a unit attribute to StructureSet
    - Populate it from DICOM data
    - For manually entered contours accept a unit parameter
    - set a default value

6. Begin work on Metric Calculations


## Metrics

### Metrics Types

- 4 kinds of Metrics
  - Distance
    - Units of cm or mm
    - Distance between contours of two structures
    - Direction any of 3D (perpendicular to contour), or orthogonal directions (L, R, Ant, Post, Sup, Inf)
    - Aggregate methods Max, Min, Ave
    - Not defined for a single structure
  - Volume
    - Units of $cm^3$, cc, ml
    - Volume of overlap ($A \cap B$), or difference ($A - B$) between two structures
    - For a single structure this is teh volume of that structure
  - Surface Area
    - Units of $cm^2$, $mm^2$
    - Overlapping surfaces ($A_{boundary} \cap B_{boundary}$) Surface of B within A ($[A - B]_{boundary}$)
    - For a single structure this is the surface area of that structure.
  - Ratio
    - Ratio of two other metrics
    - The two metrics must be of the same type and units
    - The two metrics can be the same of from different structures
    - Ratios have no units (None) or \%

### Classes

#### Metric abstract class

##### Attributes

###### Class Attributes

- Class Attribute: default_tolerance (float) Used if tolerance is not supplied or derived for individual objects
- Abstract Class Attribute: metric_type (str) valid values are: ['Distance', 'Volume', 'Area', 'Ratio']
- Abstract Class Attribute: metric_arg_list (List[Tuple]) Lists of possible valid arguments
  - e.g. [(ROI,ROI), (ROI)] *(one or two structure IDs)*
  - Unspecified number of arguments: [(ROI, ROI, ...)] $\to$ Minimum 2 structure IDs with any number of additional ROIs
- Abstract Class Attribute: name (str) metric name
- Abstract Class Attribute: description (str, optional) Description of metric

###### Instance Attributes

- value (float)
- unit (str)
- tolerance (float) minimum increment of value (used for rounding)
- structure (StructureShape) Used when construction method needs to create a temporary structure from two or more supplied structures. Default None.

##### Methods

- Abstract Method: calculate_metric takes one or two structures, and returns value and unit
- \_\_init\_\_
  - Takes StructureSet and one or two of structure IDs or Metrics according to   type, optional tolerance, optional units
  - calls validate_args
  - if unit supplied, calls validate_units
  - sets tolerance from argument, StructureSet or from default tolerance
  - calls calculate_metric and sets value and unit
  - if unit supplied, calls convert_unit
  - rounds value by tolerance

- validate_args: Checks supplied arguments against valid list of arguments for the metric
- validate_units: checks whether optional supplied unit is valid for the metric type.

- convert_unit
  - takes a unit and modifies value and unit accordingly.
  - Raises error if the unit and type are not compatible
- default method \_\_str\_\_ returns a formatted string with `f'{name}: {value} {unit}'`
