# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next
1. The bit order of the DE27IM is not consistant in the documentation.
The bit order should be Region (contour), External, Hull.  In some cased the
documentation has it reversed as Hull, External, Region.  This is a critical
detail for the implementation of the relationship calculations, so I need to
verify that the bit order is correct and consistent in the documentation and in
the code.
I need to search through the notebooks and markdown files to identify instances
where the bit order is specified, and check that it is correct and consistent
with the implementation in the code.

2. Begin work on Metric Calculations
    1. Design tests for each metric type for each relevant relationship type.

## Metrics

### Metrics Types

- 4 kinds of Metrics
  - Length
    - Units of cm or mm
    - Distance between contours of two structures
    - Direction any of 3D (perpendicular to contour), or orthogonal directions (L, R, Ant, Post, Sup, Inf)
    - Aggregate methods Max, Min, Ave
    - Not defined for a single structure
  - Volume
    - Units of $cm^3$, cc, ml
    - Volume of overlap ($A \cap B$), or difference ($A - B$) between two structures
    - For a single structure this is the volume of that structure
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



## Future work

1. Exports do not respect selection, order, or symbol option in matrix display
2. Include a legend in the spreadsheet / Json file if using symbols
3. Include a description section in the spreadsheet / Json file with descriptions of relationships.  This should be expanded on as Metrics are added

1. Add an optional origin/ isocentre attribute to StructureSet
    - The structure Set DICOM data does not contain an origin or isocentre, but
      it is included in the DICOM RT Plan. In teh future, when we add support
      for DICOM RT Plan, we can populate this attribute from the DICOM data.
      For now, it should be possible to set is manually for the sake of testing.

2. Relation (Edge) Styling
    - Explore expanded options for line types
        - Different dash types
        - double lines
        - glow

3. Create a new relationship type for SHELTERED structures that
    extend beyond the hull of the other structure.
    e.g ![Cylinder extending beyond sheltering ring](./src/Images/FreeCAD%20Images/Extended%20Inner%20cylinder.png)
