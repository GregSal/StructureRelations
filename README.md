# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Fix issue that an open hole that is open only on one end will be treated as
open on both ends for the sake of boundary testing.

4. Investigate Issue that Warnings are being returned when calculating some
relationships:

  ``` cmd
  d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171:
  RuntimeWarning: divide by zero encountered in relate
  return lib.relate(a, b, **kwargs)
  DEBUG:structure_set:Calculated relationship between ROI Lung B and ROI Lung R:
  Relationship: Partition
  d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171:
  RuntimeWarning: invalid value encountered in relate
  return lib.relate(a, b, **kwargs)
  d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171:
  RuntimeWarning: divide by zero encountered in relate
  return lib.relate(a, b, **kwargs)
  DEBUG:structure_set:Calculated relationship between ROI Lung B and ROI Lung L:
  Relationship: Overlaps
  d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171:
  RuntimeWarning: divide by zero encountered in relate
  return lib.relate(a, b, **kwargs)
  DEBUG:structure_set:Calculated relationship between ROI PTV Cavity and ROI eval PTV:
  Relationship: Partition
  ```

5. Add a unit attribute to StructureSet
    - Populate it from DICOM data
    - For manually entered contours accept a unit parameter
    - set a default value

6. Begin work on Metric Calculations

## Web App Updates

- Identify or Hide logical relations

  - If the relationship is logical and "Hide Logical Relations" is selected
    in the webapp, then this relationship will not be displayed if all of the
    intermediate structures are shown. If any one of the intermediate structures
    is not shown, then the logical relationship will be displayed because the
    relationship is not logical based on the displayed relations.

  - Logical relations include brackets around symbol or label

- Expand options for formatting edge lines

- More flexibility in diagram layout so that nodes can be dragged without
pulling the entire diagram

- Add a status bar to the webapp to show current status, messages, etc.
put logging.info() messages there.

- Make the progress bar update during long operations (loading structures,
calculating relationships, etc.)

- If CT images are available, add option to show/hide CT background in Contour
Plotting area

- I will likely want to display some of the additional information in the
relationship matrix within the webapp.  What will need to change to make it
easy to customize what is displayed?  (this may become something in the
configuration, settings or selectable within the web page - I haven't decided
what to do here yet)

- Webapp edge cases: Should verify that empty matrices, single-structure sets,
and filtered matrices still serialize correctly to JSON through to_dict()

— The string conversion needs to handle StructureRelationship objects with
de27im=None gracefully.

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
