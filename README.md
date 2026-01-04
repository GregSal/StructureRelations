# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

1. update `generate_interpolated_polygon` function to make use of directional graph.

1. Define complimentary relations

2. restructure relationships definitions to to obtain definitions from Jason file along with descriptions and display parameters

3. Identify "Logical" relations

4. Begin work on Metric Calculations



## Future work

- Add a status bar to the webapp to show current status, messages, etc.  put logging.info() messages there.

- Make the progress bar update during long operations (loading structures, calculating relationships, etc.)

- If CT images are available, add option to show/hide CT background in Contour Plotting area

- I will likely want to display some of the additional information in the relationship matrix within the webapp.  What will need to change to make it easy to customize what is displayed?  (this may become something in the configuration, settings or selectable within the web page - I haven't decided what to do here yet)

- Webapp edge cases: Should verify that empty matrices, single-structure sets, and filtered matrices still serialize correctly to JSON through to_dict()—the string conversion needs to handle StructureRelationship objects with de27im=None gracefully.


## Logical Relationships
Within a set of structures, transitive relationships can result in **Logical**
relationships, which exists out of necessity due to other relationships in the
structure set. The simplest example is one where A *Contains* B and B
*Contains* C, therefore the relationship A *Contains* C is a **Logical** one
since it is a requirement of the other two relationships.  **Logical**
relationships can also be chained further. For example, if C *Contains* D, the
relationships A *Contains* D and B *Contains* D are both **Logical**.

**Implied relationships**
Identifying **Logical** relationships is complicated by the fact that some
relationships **Imply** other ones.  For example, *Partitioned* is not transitive,
but it **Implies** the *Contains* relationship, so the following scenario is
possible: If A *Is Partitioned by* B and B *Contains* C the relationship
A *Contains* C is **Logical**.  However, If A *Is Partitioned by* B and
B *Is Partitioned by* C either the relationship A *Contains* C or the
relationship A *Is Partitioned by* C are possible, so a **Logical** relationship
does not exist.

To identify **Logical** *Contains* relationships, one must construct a directed
graph of the structure relations where the edges are all the *Contains*
relationships in the structure set and all **Implied** *Contains* relationships
(*Partitioned*) in the structure set. Next, for each *Contains* relationship,
check for an alternate path between the two structures. Next, eliminate any path
that is not composed *entirely* of **Implied** *Contains* relationships.
If an alternate path exists, then the relationship is **Logical**.
