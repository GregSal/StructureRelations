# Structure Relationships for Multiple Regions

## Handling Different Relationships for Different Regions

Currently The only scenario that has been explicitly addressed is where there
is a structure that is both *Surrounded* and *Disjoint*.  We declared that the
over all relationship wad *Disjoint*, which overrode the surround portion.
Here we have a *Confines* and *Shelters* combination.  In this case it would be
useful to recognize the distinct relationships for the two regions.

First we will summarize the ways in which relationships can be combined and
transformed when there is only one region in each structure.  Then we will
extend the analysis to consider how to handle multiple regions in each
structure.
## Relationship Combinations and Transformations

### Relationships that cannot co-exist
What are impossible combinations when both structures have only one region?
- DISJOINT + SURROUNDS, or CONFINES
- SHELTERS + SURROUNDS, or CONFINES

A continuous region cannot have a portion that is disjoint or sheltering and a
portion that is surrounding or confined.  A continuous region would have to be
present in the volume that defines the hole surrounding the inner part.

All other relationships that occur on different regions of the same structure
can be consolidated using the DE27IM bitwise OR operation.  However the above
relationship combinations cannot be consolidated because they cannot be present
in the same volume.  The results is an UNKNOWN relationship.

- Any combination of relationships is possible when one of the structures has
more than one distinct region.

### Relationships that override others
As individual slice relationships are combined, one relationship will override
another.

- OVERLAPS + ALL OTHERS -> OVERLAPS,
- DISJOINT + SHELTERS -> DISJOINT,
- BORDERS + SHELTERS, or DISJOINT -> BORDERS,
- CONFINES + SURROUNDS -> CONFINES,
- PARTITIONED + CONTAINS, or EQUAL -> PARTITIONED,

There are also situations where relationship combinations can result in a third
type of relationship:
- DISJOINT + PARTITIONED, CONTAINS, or EQUAL -> OVERLAPS,
- SURROUNDS + PARTITIONED, CONTAINS, or EQUAL -> OVERLAPS,
- EQUAL + CONTAINS -> PARTITIONED,

### Boundary Effects
On boundary slices the relationship will always be UNKNOWN because they only
include boundary interactions.
These relationships can result in the following conversions:

If a boundary interaction is present, the relationship will be changed as
follows:
- SHELTERS or DISJOINT -> BORDERS,
- CONTAINS -> PARTITIONED,
- SURROUNDS -> CONFINES,

In this case if a boundary interaction is NOT present, the relationship will be
changed as follows:
- EQUAL -> CONTAINS

### Complementary Relationships
For all of the above patterns, changing the relationships to their complements
will also result in a valid pattern.
For example:
- CONTAINS + EQUAL -> PARTITIONED
- WITHIN + EQUAL -> PARTITIONS

## Multiple Regions in Each Structure

As noted above, in most cases it will make sense to consolidate the
relationships between multiple regions into a single relationship between the
two structures using the same rules that are used to consolidate the
relationships on each slice into a relationship between regions.  However, there
are a few exceptions to this. When one region is disjoint or shelters and the
other region is surrounding or confined, the overall relationship cannot be
described using the DE27IM bitwise OR operation.

### Reporting Relationships

- Continue to use the DE27IM bitwise OR operation to consolidate relationships
between multiple regions into a single relationship between the two structures.

- When all of the relationships between the different regions are identical,
treat the overall relationship as if it were a single region with that
relationship.

- When one or both of the structures has multiple regions, the relationships
between the regions of the two structures are not the same, and the the DE27IM
of the consolidated relationship can be associated with a named relationship
(i.e. is not UNKNOWN), then add '{}' brackets around the relationship label
to indicate that there are multiple relationships between the two structures.

- If the DE27IM of the consolidated relationship is UNKNOWN, then report the
overall relationship by combining the labels of the different relationships
with a '&' symbol and add '{}' brackets around the combined label to indicate
that there are multiple relationships between the two structures.

- Add a toggle to show a single relationship per structure, or an expanded
    reporting of the relationships for multiple regions.


### Reporting Relationships for Multiple Regions
When there are multiple regions for a given structure there will be multiple
relationships stored for that structure and another structure.  If the other
structure also has multiple regions, the number of relationships can quickly
become unmanageable for display purposes.  We will need to decide on some rules
for how to consolidate these relationships for reporting purposes.

1. If all relationships are the same, report that relationship
2. Otherwise:
    1. Count the number of relationships of each type.
    3. Sort the relationships by rank and then by count.
    4. Report the relationships in the following format:
        > {n_1} {relationship_1} Regions<br>
        > {n_2} {relationship_2} Regions

        **Rank**
        1. EQUAL
        2. PARTITIONED
        3. PARTITIONS
        4. CONTAINS
        5. WITHIN
        6. CONFINES
        7. CONFINED
        8. SURROUNDS
        9. ENCLOSED
        10. BORDERS
        11. SHELTERS
        12. SHELTERED
        13. OVERLAPS
        14. DISJOINT

5. Include an option to drop any DISJOINT relationships.
6. Include an option to drop any OVERLAPS relationships.


### Future Considerations
- How to we take future metrics into account?
- A given metric might be meaningful for one relationship and meaningless for
    another.
- If there are two relationships which is the most important one to report?
    i.e. should the case of *Surrounds* and *Disjoint* actually be reported as
    *Surrounds* rather than *Disjoint*?
- Consider the possibility of created display nodes for each region in the
    structure and then grouping them with an oval.  There would then be a toggle
    to switch between displaying individual regions and the entire structure as
    one node?
- For now, continue with the current incremental labeling of regions. In the
    future, explore using relational designators for different regions.
    For example, Left and Right Lung, or Superior and Inferior PTV.  This would
    be more meaningful to users and would also be more consistent with the way
    that structures are currently named.  It would also allow for more
    meaningful reporting of relationships and metrics.
## TODO:
- Currently the code calculates the relationship for each slice based on the
    union of all regions in each structure. In order to accurately deal with
    relationships between structures where there are multiple regions in one or
    both structures it will be necessary to modify this so that the relationship
    is calculated and stored for each region on each slice.  The slice-by-slice
    relationships for each region can then be consolidated to identify distinct
    relationships for distinct regions.  It will also be necessary to modify
    the relationship graph so that there are edges for each region's
    relationship rather than just one edge for the overall relationship between
    the two structures.

- This code modification will also require a change in the data structure.
    Before making any cde modifications, Update the previous review of the data
    structure and process to decide if it is still appropriate.

- The following things need to be considered:
    - The relationship graph type needs to be changed to allow for multiple
        edges (relationships) between the same two nodes (structures).
    - Is the Region table still a useful construct? Should it be modified or
        replaced?
    - We currently store the relationship between structures for each slice. We
        will now need to store the relationship between each region of each
        structure for each slice.  What type of data structure will be best for
        storing and accessing this data?
    - We will need a new method / function to summarize and display the
        relationship(s) between structures.  I will want this accessible to the
        future API, not just in the webapp.  Where should it be stored? In
        structure_set.py in relations.py or in a new file?

    - In addition to current goals, future plans also need to be considered.

- Build some test cases that will cover the different possibilities.
    - The tests do not have to pass initially and the expected result can
        change as the plan develops.
    - Start with a notebook so that the thinking process can be placed
        alongside the tests

- Allow the Metrics development to proceed in parallel because the two will be
    related.
    - Clearly document what metrics apply to what relationships
    - Look for cases where a change in relationship in the same region might
        make the metric definition ambiguous.
    - Consider when and how metrics for different regions can be practically
        consolidated.
    - Asses the impact of any change in relationship definitions on metric
        definitions.
