# Structure Relations

Test and reports on relationships between DICOM RT Structures

## TO Do Next

2. restructure relationships definitions to to obtain definitions from Jason file along with descriptions and display parameters

3. Identify "Logical" relations

3. Fix issue that an open hole that is open only on one end will be treated as open on both ends for the sake of boundary testing

4. Investigate Issue that Warnings are being returned when calculating some relationships

4. Begin work on Metric Calculations



## Future work

- Add a status bar to the webapp to show current status, messages, etc.  put logging.info() messages there.

- Make the progress bar update during long operations (loading structures, calculating relationships, etc.)

- If CT images are available, add option to show/hide CT background in Contour Plotting area

- I will likely want to display some of the additional information in the relationship matrix within the webapp.  What will need to change to make it easy to customize what is displayed?  (this may become something in the configuration, settings or selectable within the web page - I haven't decided what to do here yet)

- Webapp edge cases: Should verify that empty matrices, single-structure sets, and filtered matrices still serialize correctly to JSON through to_dict()—the string conversion needs to handle StructureRelationship objects with de27im=None gracefully.

## Relationship definitions
Currently there are three classes used together to define 3D relationships between structures:
- RelationshipType
- RelationshipDefinition
- DE27IM

I have created a JSON file (structure_relationships.json) that contains the definitions of the relationships used in this package.  I would like to restructure the code to load the definitions from this file rather than hard-coding them in the RelationshipDefinition class.  This will make it easier to add new relationships in the future without needing to modify the code.

1. The RelationshipType(Enum) class defines the names of the relationships used in the package.  This will need to be modified to load the names from the JSON file.  It will also need to be converted from an Enum to a simple class, so that it can contain other attributes such as description, symbol, etc.
The is_symmetric and is_transitive properties will need to be modified to load their values from the JSON file as well.
New methods will need to be added to link complementary and implied relationships.
2. The RelationshipTest just supplies a container for the three DE-9IM tests that make up a DE-27IM relationship.  This class does not need to be modified.
3. The DE27IM class contains the hard-coded definitions of the DE-27IM relationship strings used to define each relationship.  This will need to be modified to load the definitions from the JSON file at run time
4. The structure_relationships.json file should also replace the relationship_symbols,json file, used by the webapp.

## Logical Relationships
Within a set of structures, transitive relationships can result in **Logical** relationships, which exists out of necessity due to other relationships in the structure set. The simplest example is one where A *Contains* B and B *Contains* C, therefore the relationship A *Contains* C is a **Logical** one since it is a requirement of the other two relationships.  **Logical** relationships can also be chained further. For example, if C *Contains* D, the relationships A *Contains* D and B *Contains* D are both **Logical**.

**Implied relationships**
Identifying **Logical** relationships is complicated by the fact that some relationships **Imply** other ones.  For example, *Partitioned* is not transitive, but it **Implies** the *Contains* relationship, so the following scenario is possible: If A *Is Partitioned by* B and B *Contains* C the relationship A *Contains* C is **Logical**.  However, If A *Is Partitioned by* B and B *Is Partitioned by* C either the relationship A *Contains* C or the relationship A *Is Partitioned by* C are possible, so a **Logical** relationship
does not exist.

To identify **Logical** *Contains* relationships, one must construct a directed graph of the structure relations where the edges are all the *Contains* relationships in the structure set and all **Implied** *Contains* relationships (*Partitioned*) in the structure set. Next, for each *Contains* relationship, check for an alternate path between the two structures. Next, eliminate any path that is not composed *entirely* of **Implied** *Contains* relationships. If an alternate path exists, then the relationship is **Logical**.

I want to use graph analysis make the structure_set.calculate_logical_flags method apply this logic to identify **Logical** relationships within a structure set.
This may involve:
- Transitivity analysis (e.g., A contains B, B contains C)
- Connected component analysis
- Path analysis through relationship graph
- Pattern matching for specific relationship combinations
