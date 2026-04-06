# Logical Relationships

Within a set of structures, transitive relationships can result in **Logical**
relationships, which exists out of necessity due to other relationships in the
structure set. The simplest example is one where:<br>
- **A** *Contains* **B** and
- **B** *Contains* **C**

therefore the relationship
- **A** *Contains* **C**

is **Logical** since it is a requirement of the other two relationships.

**Logical** relationships can also be chained further. For example, if
- **C** *Contains* **D**,

the relationships
- **A** *Contains* **D** and
- **B** *Contains* **D**

are both **Logical**.

## Implied relationships
Identifying **Logical** relationships is complicated by the fact that some
relationships **Imply** other ones.  For example, *Partitioned* is not
transitive, but it **Implies** the *Contains* relationship, so the following
scenario is possible:
- If **A** *Is Partitioned by* **B** and
- **B** *Contains* **C**

the relationship
- **A** *Contains* **C** is **Logical**.

However, if
- **A** *Is Partitioned by* **B**
    - and
- **B** *Is Partitioned by* **C**

either the relationship
- **A** *Contains* **C**

or the relationship
- **A** *Is Partitioned by* **C**

are possible, so a **Logical** relationship does not exist.

### Example: Identifying Logical *Contains* Relationships

To identify **Logical** *Contains* relationships, one must construct a directed
graph of the structure relations where the edges are all the *Contains*
relationships in the structure set and all **Implied** *Contains*
relationships (*Partitioned*) in the structure set. Next, for each *Contains*
relationship, check for alternate paths between the two structures. Eliminate
any path that contains **no** direct *Contains* edge anywhere along it — a
path composed entirely of **Implied** *Contains* segments (e.g., all
*Partitioned* edges) is rejected because it is ambiguous: the result could
equally be *Contains* or *Partitioned*. A path is valid as long as at least
one of its segments — at any position, **including the final one** — is a
direct *Contains* edge. If at least one valid alternate path exists, the
relationship is **Logical**.

## The special case of Equals
The *Equals* relationship is a special case since it is both symmetric and
transitive. Therefore, if:
- A *Equals* B
    - and
- B *Contains* C,
    - then
- A *Contains* C

is **Logical**. Similarly, if
- A *Equals* B
    - and<
- B *Is Partitioned by* C,
    - then
- A *Is Partitioned by* C

is **Logical**. This can be extended to chains of *Equals* relationships of any
length.

## Logical Relationship Identification
Use graph analysis to identify **Logical** relationships within a
structure set.
This will involve:

- Transitivity analysis (e.g., A contains B, B contains C)
- Connected component analysis
- Path analysis through relationship graph
- Pattern matching for specific relationship combinations

If a relationship is identified as Logical, then Intermediate Structures are
identified as the ROIs of the structures that form the longest path between the
Starting and ending structures of the relationship.

If the relationship is logical and "Hide Logical Relations" is selected in the
webapp, then this relationship will not be displayed if all of the intermediate
structures are shown. If any one of the intermediate structures is not shown,
then the logical relationship will be displayed because the relationship is not
logical based on the displayed relations.

## Relationship Properties

The spatial relationship have the following properties:

**Symmetric**
> A relation is **symmetric** if $aRb \iff bRa$ for all $a$ and $b$.
>
> - For example, if A *Equals* B then B *Equals* A.
>
> The **symmetric** relations are:
>
> - Equals, Overlaps, Disjoint, Borders

**Transitive**

> A relation is **transitive** if whenever, $aRb$ and $bRc$ then $aRc$.
>
> - For example, if A *Contains* B and B *Contains* C, then A *Contains* C.
>
> The **transitive** relations are:
>
> - Equals, Contains, Shelters, Surrounds.

|Relationship|Symmetric|Transitive|Complement|
|------------|---------|----------|----------|
|Disjoint    |Yes      |No        |Disjoint  |
|Shelters    |No       |Yes       |Sheltered |
|Surrounds   |No       |Yes       |Enclosed  |
|Borders     |Yes      |No        |Borders   |
|Confines    |No       |No        |Confined  |
|Contains    |No       |Yes       |Within    |
|Equals      |Yes      |Yes       |Equals    |
|Partitioned |No       |No        |Partitions|
|Overlaps    |Yes      |No        |Overlaps  |

## Tests to verify correct identification of **Logical** relationships.

### Is Logical

1. A *Contains* B and B *Contains* C.
    > Verify that the relationship A *Contains* C is **Logical**.<br>
    > - Solid sphere within a middle sphere within an outside sphere.<br>
    > ![Sphere in Sphere in Sphere](<../src/Images/FreeCAD Images/Sphere in Spheres in Shell.png>)


2. A *Contains* B and B *Contains* C and C *Contains* D.
    > Verify that the relationships A *Contains* D and B *Contains* D are both **Logical**.
    > - 4 nested spheres

3. A *Confines* B and B *Confines* C.
    > Verify that the relationships A *Surrounds* C is **Logical**.<br>
    > - A hollow sphere confines a middle hollow sphere,which confines an inner solid sphere.<br>
    > ![Sphere in Sphere in Sphere](<../src/Images/Logical/confined spheres.png>)

4. A *Is Partitioned by* B and B *Contains* C.
    > Verify that the relationship A *Contains* C is **Logical**.<br>
    > ![Not Logical Partitions](<../src/Images/Logical/Partitions,Contains.png>)

5. A *Contains* B and B *Is Partitioned by* C.
    > Verify that the relationship A *Contains* C is **Logical**.<br>
    > ![Logical Partitions](<../src/Images/Logical/Contains,Partitions.png>)

6. A *Equals* B and B *Borders* C.
    > Verify that A *Borders* C is **Logical**.<br>
    > ![Embedded Boxes](<../src/Images/Boundaries\ExteriorBorders2D_SUP.png>)

7. A *Equals* B and B *Is Disjoint With* C.
    > Verify that A *Is Disjoint With* C is **Logical**.<br>
    > ![Nested Cylinders](<../src/Images/Logical/logical disjoint.png>)

8. A *Shelters* B (cylinder in cylinder), B *Surrounds* C, C *Contains* D.
    > Verify that A *Shelters* C is **Logical**,
    > A *Shelters* D is **Logical**, and
    > B *Surrounds* D is **Logical**.<br>
    > ![Nested Cylinders](<../src/Images/Logical/shelters_surrounds_contains.png>)

### Is Not Logical

1. A *Contains* B and A *Contains* C, but B is Disjoint from C.
    > Verify that A *Contains* C is **Not Logical**.<br>
    > ![Embedded Boxes](<../src/Images/Logical/Not Logical Disjoint Cylinders.png>)

2. A *Borders* B and B *Borders* C.
    > Verify that the relationship A *Borders* C is **Not Logical**.<br>
    > ![Bordered Boxes](<../src/Images/Logical/borders 3 way.png>)

3. A *Is Partitioned by* B and B *Is Partitioned by* C.
    > Verify that the relationship A *Is Partitioned by* C is **Not Logical**.<br>
    > ![Partitioned Boxes](<../src/Images/Logical/Nested Partitioned cubes.png>)

4. A *Is Partitioned by* B and B *Is Partitioned by* C.
    > Verify that A *Contains* C is **Not Logical**.
    > ![Partitioned Boxes](<../src/Images/Logical/Nested partial Partitioned cubes.png>)
