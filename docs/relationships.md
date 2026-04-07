The *DE-27IM* relationship is derived from three *DE-9IM* relationships. It provides a more comprehensive relationship between two polygons by including the relationships between the polygons, their exteriors, and their convex hulls.  For example, a polygon with a hole can be compared to another polygon residing within the hole. A *DE-9IM* relationship would classify the two polygons as *disjoint*, but a *DE-27IM* relationship would capture the fact that one polygon is contained within the other.

The *DE-27IM* relationship string is composed of three *DE-9IM* relationship strings concatenated. The left-most 9-bit string represents the *DE-9IM* relationship between the two polygons.  The middle 9-bit string represents the *DE-9IM* relationship between the second polygon and the *exterior* of the first polygon. The right-most 9-bit string represents the *DE-9IM* relationship between the second polygon and the *convex hull* of the first polygon. From this 27-bit string, a number of physical relationship can be derived.

Named relationships are identified by logical patterns such as: `T*T*F*FF*`
- The '*T*' indicates the bit must be True.
- The '*F*' indicates the bit must be False.
- The '__\*__' indicates the bit can be either True or False.
Ane example of a complete relationship logic is:
Surrounds (One structure resides completely within a hole in another
           structure):
>- Region Test = `FF*FF****`
> - The contours of the two structures have no regions in common.

> - Exterior Test = `T***F*F**`
> - With holes filled, one structure is within the other.

> - Hull Test = `*********`
> - Together, the Region and Exterior Tests sufficiently identifies the
> relationship, so the Hull Test is not necessary.

The mask binary is a sequence of *0*s and *1*s with every '__\*__' as a '*0*' and every '*T*' or '*F*' bit as a '*1*'.  The operation: *relationship_integer* **&** *mask* will set all of the bit that are allowed to be either True or False to *0*. The value binary is a sequence of *0*s and *1*s with every '*T*' as a '*1*' and every '__\*__' or '*F*' bit as a '*0*'. The relationship is identified when value binary is equal to the result of the *relationship_integer* **&** *mask*
operation.

The relationships are defined as follows:

|Relationship  |Region Test  |Exterior Test  |Hull Test   |Description|
|--------------|-------------|---------------|------------|-----------|
|Disjoint      |`FF*FF****`  |`****F*T**`    |`****F*T**` |There is no overlap between ***A*** and ***B***.|
|Shelters      |`FF*FF****`  |`****F*T**`    |`T*T*F*F**` |The Convex Hull of ***A*** contains ***B***.|
|Surrounds     |`FF*FF****`  |`T*T*F*F**`    |`*********` |The Exterior of ***A*** contains ***B***.|
|Borders       |`F***T****`  |`******T**`    |`*********` |Part of the *exterior* boundary of ***A*** touches the exterior boundary of ***B***.|
|Confines      |`F***T****`  |`T*T***F**`    |`*********` |Part of the *interior* boundary of ***A*** touches the exterior boundary of ***B***.|
|Partitions    |`T*T*T*F**`  |`*********`    |`*********` |***A*** contains ***B*** and part of the *boundary* of ***B*** touches part of the *boundary* of ***A***.|
|Contains      |`TT*FF*F**`  |`T********`    |`T********` |***B*** is fully within ***A***.|
|Overlaps      |`T*****T**`  |`T********`    |`T********` |***A*** and ***B*** intersect, but neither contains the other.|
|Equals        |`T*F*T*F**`  |`T***T****`    |`T********` |***A*** and ***B*** enclose the identical area.|

A number of these relationships also have a complementary relationship, e.g.*Contains* has a complementary relationship with *B is fully within A*.However, by requiring that the primary polygon (**A**) is larger than the secondary polygon (**B**), we can ensure that the relationship is properly defined without the need to include tests for the complementary relationships.

This size requirement is not explicitly checked in the DE-27IM class because DE-27IM relationships are usually obtained in the context of a 3D volume and it is possible for individual contours from the two volumes to have the opposite size difference to their respective volumes.

One of the values of the binary DE-27IM relationship is that the individual relationships between contours that define two 3D volumes can be merged using a logical *OR* to obtain the relationship between the 3D volumes.
