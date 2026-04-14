# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Processing Updates

    1. remove the box showing "true" that appears just after uploading the DICOM data

    2. When processing structures replace the messages like: "Computing running (slice 92/92)" with messages like: "Calculated relationships between OralCavity (ROI 107) and opt Larynx (ROI 27) as: is Disjoint from"

    3. The progress bar is not scaled correctly. When it starts calculating relationships, it should be at about 30%, when it finishes calculating relationships it should be at 70% During the time that it is rendering the diagram it should be progressing from about 80% to 100%.

2. Contour Plot Updates

    1. The y axis of the contour plot should be flipped.

    2. The Relationship overlay options should change depending on how many
        structures are selected.
        1. If only one structure is selected, the only option should be: Structure A
        2. If two structures are selected, the only options should be: A vs B
        3. If three structures are selected, the options should be:
            - A vs B, C outline
            - (A AND B) vs C
            - (A OR B) vs C
            - (A XOR B) vs C
            - (A - B) vs C
        4. If more than three structures are selected, the only options should
            be: A vs B , outlines of all other structures

    with the "A vs B, C outline" and "A vs B , outlines" (Structure A and, B are displayed in as they are with only two structures selected and all other selected structures are drawn as outlines)

    3. Add an option to plot an axis

    4. The tooltip for the structure sets should have the upload prefix dropped from the file name.  This should also apply internally to the file name attribute in the structure set.
    ![alt text](image-2.png)

    5. Try to improve the speed of rendering contour plots, especially when switching slices.

    3. Relation (Edge) Updates
        - implement expanded options for line types
            - Different dash types
            - double lines
            - glow

5. Add a unit attribute to StructureSet
    - Populate it from DICOM data
    - For manually entered contours accept a unit parameter
    - set a default value

6. Add an optional origin/ isocentre attribute to StructureSet
    - The structure Set DICOM data does not contain an origin or isocentre, but
      it is included in the DICOM RT Plan. In teh future, when we add support
      for DICOM RT Plan, we can populate this attribute from the DICOM data.
      For now, it should be possible to set is manually for the sake of testing.


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
