# Shelters Definition Issues

<link rel="stylesheet" href="relations.css">

## Shelters Definition
<table width="350px">
<tr class="l"><th>Shelters</th><th>Transitive</t></tr>
<td class="d" colspan="2">
<span class="a">a</span> and <span class="b">b</span> have no points in common, but the Convex Hull of <span class="a">a</span> contains <span class="b">b</span>.
</td></tr><tr><td colspan="2">
<img src="Images/Relationships/shelters.png" alt="shelters">
</td></tr></table>

- This relation is intended to identify structure relations base on an expansion margin followed by a crop.
- For example, the case of an avoidance ring expansion around a PTV, followed by a cropping of the avoidance ring and PTV to the skin. The cropping may cause the avoidance ring to no longer surround the PTV.
- Without the "Shelters" relation, the avoidance ring will be classified as "Disjoint" from the PTV.
- The Convex Hull geometrical operation is used to identify the "Shelters" relation.



## Challenges with the Shelters Definition
<table>
<tr>
<td width="350">
The simplest example of <b>"Shelters"</b> is a "C" shape with a semi-circle in the
middle.
<ul><li>The semi-circle is not <b>"Surrounded"</b> by the "C" shape because the shape is not closed.  </li>
<li>However, the Convex Hull of
the "C" shape contains the circle.</li>
<li> When the cut-plane is approximately perpendicular to the slice plane, this method of identifying shelters works.</li></ul></td>
<td width="250px">
<img src="Images/FreeCAD Images/Sheltered Horizontal cylinder End On View.png" alt="shelters">
</td></tr>
<tr><td width="350">
As soon as the cut-plane is no longer perpendicular to the slice plane, the difficulties begin.<br><br>

<ul><li>In this example the a hollow cylinder <b>"Shelters"</b> an inner cylinder. cut-plane is essentially parallel to the slice plane.</li></ul></td><td width="250px">
<img src="Images/FreeCAD Images/Sheltered cylinder angled view.png" alt="shelters">
</td></tr>
<tr><td width="350">
<ul><li>If the information is limited to the slice plane, then using the basic relation definitions, the hollow cylinder should be classified as <b>"Surrounding"</b> the inner cylinder.</li>
<li> The hollow cylinder appears as a ring around the inner circle on the slice, so in the 2D plane the ring encloses the circle.</li></ul></td><td width="250px"><img src="Images/FreeCAD Images/Sheltered cylinder Top View.png" alt="shelters"></td></tr>
<tr><td width="350"><ul><li> However, the hollow cylinder does not <b>"Surround"</b> the inner cylinder in 3D space.</li>
</ul></td>
<td width="250px">
<img src="Images/FreeCAD Images/Sheltered cylinder angled view.png" alt="shelters"
></td></tr>
<tr><td width="350">
To add another complication, if the same hollow cylinder with another cylinder inside it is rotated so that the cylinder axis is in the x-axis (horizontal on the slice plane), then the same geometry will be classified as <b>"Disjoint"</b>.</li>
</ul></td>
<td width="250px">
<img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Angled View.png" alt="shelters">
</td></tr>
<tr><td width="350">
<ul><li>In the slice plane, the horizontal hollow cylinder and inner cylinder appear as two thin rectangles above and below a fatter rectangle.</li>
<li>On the slice plane there is no indication that the two larger rectangles are part of the same 3D volume.
</td>
<td width="250px">
<br><br>
<img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Top View.png"></td>
</tr>
<tr><td width="350">
My current definition of <i>Convex Hull</i> is actually the combined convex hulls of each distinct region in the contour. <br>
<ul><li>i.e. The <i>Convex Hull</i> consists of multiple elastic bands stretched around each
external contour, rather that one elastic band stretched around all external
contours.  </li>
<ul><li>Using this definition, The horizontal hollow cylinder with another
cylinder inside it will be classified as <b>Disjoint</b></td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Almost Top View.png" alt="shelters"></td>
</tr>
<tr><td width="350">
If the convex hull is defined as "The smallest convex polygon that contains
all the regions on the plane", then The horizontal hollow cylinder with another
cylinder inside it will be correctly classified as <b>Shelters</b>.</td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder With Hull.png" alt="shelters"></td></tr>
<tr><td width="350">
However this definition may result in spurious relationships being identified.
<ul><li>For example, the convex hull of a set of distinct regions will contain
the area between the regions.</ul></td>
<td width="250px">
<img src="Images/FreeCAD Images/Three Parallel Cylinders.png" alt="shelters"></td></tr>
</table>

## Hull Definition

**Current Definition:**

A bounding contour generated from the entire contour MultiPolygon.

A convex hull can be pictures as an elastic band stretched around the
external contour.

If contour contains more than one distinct region the hull will be the
combination of the convex_hulls for each distinct region.  It will not
contain the area between the regions.  in other words, **the convex hull
will consist of multiple elastic bands stretched around each external
contour rather that one elastic band stretched around all external
contours.**


- This definition is not consistent with the definition of a convex hull in
  computational geometry.  The convex hull of a set of points is the smallest
  convex polygon that contains all the points.  The convex hull of a set of
  polygons is the smallest convex polygon that contains all the polygons.

- The current definition also prevents Shelters relations from being identified.

- Going with the broader definition of a convex hull may result in spurious
  relationships being identified.  For example, the convex hull of a set of
  distinct regions will contain the area between the regions.


## Relation Testing Changes Required
- Use the Regions Graph to identify the regions that are part of the same
  3D volume.
  [Identify Volumes](https://stackoverflow.com/questions/33088008/fetch-connected-nodes-in-a-networkx-graph)

- Do all relation testing by volume rather than by structure or region.
- Use the standard definition of "Convex Hull" of the volume to
  identify the Shelters relation.
- For Hole volumes, test whether they are closed or open in 3D space.
[Netwokx Algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html)

- Use this information to distinguish between the "Shelters" and
  "Surrounds" relations.

## Required Data Structure Changes
- Need to move away from the slice table to the Regions Graph.
- Regions need to have a volume attribute, with unique volume IDs.
- Hole volumes need to be identified as open or closed.
- Most of the StructureSlice methods will need to be moved to the
  Regions class.
- The StructureSlice class will still be needed to generate the Regions Graph,
   but the slice table will become redundant.


## Related Disjoint Issue

- Disjoint Relations are always overridden by other relation.
- A hollow cylindrical shell with an interior cylinder ending inside the shell should be reported as **Shelters**
- In the above example, if the interior cylinder extends beyond the outer cylinder's hole, then the relation should be **Disjoint**.

![Extended Inner Cylinder](<Images/FreeCAD Images/Extended Inner cylinder.png>)

- Similar to the above issue with the Convex Hull, the **Disjoint** relation will only be identified correctly if the hole on the plane is recognized as *exterior* to the structure.

- The same issue occurs for a concentric hollow cylinder surrounding one of two smaller cylinders, where the Second cylinder is **Disjoint**.
- The relationship is **Disjoint** because the Second cylinder is outside of the First Structure.
- However, the Disjoint relation is being overridden by the **Surrounds** relation.

<table><tr><td>
<img src="Images/FreeCAD Images/Disjoint Parallel Cylinders.png">
</td><td>
<img src="Images/FreeCAD Images/Disjoint Axial Cylinders.png" alt="shelters">
</td></tr></table>
