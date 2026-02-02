# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. Identify "Logical" relations

2. Fix issue that an open hole that is open only on one end will be treated as open on both ends for the sake of boundary testing

3. Investigate Issue that Warnings are being returned when calculating some relationships
    ```
    d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171: RuntimeWarning: divide by zero encountered in relate
    return lib.relate(a, b, **kwargs)
    DEBUG:structure_set:Calculated relationship between ROI Lung B and ROI Lung R:
    Relationship: Partition
    d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171: RuntimeWarning: invalid value encountered in relate
    return lib.relate(a, b, **kwargs)
    d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171: RuntimeWarning: divide by zero encountered in relate
    return lib.relate(a, b, **kwargs)
    DEBUG:structure_set:Calculated relationship between ROI Lung B and ROI Lung L:
    Relationship: Overlaps
    d:\.conda\envs\StructureRelations\Lib\site-packages\shapely\predicates.py:1171: RuntimeWarning: divide by zero encountered in relate
    return lib.relate(a, b, **kwargs)
    DEBUG:structure_set:Calculated relationship between ROI PTV Cavity and ROI eval PTV:
    Relationship: Partition
    ```

4. Add a unit attribute to StructureSet
    - Populate it from DICOM data
    - For manually entered contours accept a unit parameter
    - set a default value

5. Begin work on Metric Calculations

## Web App Updates

- Add a status bar to the webapp to show current status, messages, etc.  put logging.info() messages there.

- Make the progress bar update during long operations (loading structures, calculating relationships, etc.)

- If CT images are available, add option to show/hide CT background in Contour Plotting area

- What other options are available for formatting edge lines?

- More flexibility in diagram layout so that nodes can be dragged without pulling the entire diagram

- Hide or shade logical relations

- Logical relations include brackets around symbol or label

- I will likely want to display some of the additional information in the relationship matrix within the webapp.  What will need to change to make it easy to customize what is displayed?  (this may become something in the configuration, settings or selectable within the web page - I haven't decided what to do here yet)

- Webapp edge cases: Should verify that empty matrices, single-structure sets, and filtered matrices still serialize correctly to JSON through to_dict()—the string conversion needs to handle StructureRelationship objects with de27im=None gracefully.

## Logical Relationships

Within a set of structures, transitive relationships can result in **Logical** relationships, which exists out of necessity due to other relationships in the structure set. The simplest example is one where A *Contains* B and B *Contains* C, therefore the relationship A *Contains* C is a **Logical** one since it is a requirement of the other two relationships.  **Logical** relationships can also be chained further. For example, if C *Contains* D, the relationships A *Contains* D and B *Contains* D are both **Logical**.

**Implied relationships**
Identifying **Logical** relationships is complicated by the fact that some relationships **Imply** other ones.  For example, *Partitioned* is not transitive, but it **Implies** the *Contains* relationship, so the following scenario is possible: If A *Is Partitioned by* B and B *Contains* C the relationship A *Contains* C is **Logical**.  However, If A *Is Partitioned by* B and B *Is Partitioned by* C either the relationship A *Contains* C or the relationship A *Is Partitioned by* C are possible, so a **Logical** relationship
does not exist.

To identify **Logical** *Contains* relationships, one must construct a directed graph of the structure relations where the edges are all the *Contains* relationships in the structure set and all **Implied** *Contains* relationships (*Partitioned*) in the structure set. Next, for each *Contains* relationship, check for an alternate path between the two structures. Next, eliminate any path that is not composed *entirely* of **Implied** *Contains* relationships. If an alternate path exists, then the relationship is **Logical**.

### The special case of Equals
The *Equals* relationship is a special case since it is both symmetric and
transitive. Therefore, if A *Equals* B and B *Contains* C, then A *Contains* C is
**Logical**. Similarly, if A *Equals* B and B *Is Partitioned by* C, then A
*Is Partitioned by* C is **Logical**. This can be extended to chains of
*Equals* relationships of any length.

### Implementation Plan
I want to use graph analysis make the structure_set.calculate_logical_flags method apply this logic to identify **Logical** relationships within a structure set.
This may involve:

- Transitivity analysis (e.g., A contains B, B contains C)
- Connected component analysis
- Path analysis through relationship graph
- Pattern matching for specific relationship combinations

## Metrics

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
