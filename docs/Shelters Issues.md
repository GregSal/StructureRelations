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


- Surrounds geometry with hole opened to exterior


## Challenges with the Shelters Definition
<table>
<tr>
<td width="350">
The simplest example of <b>Shelters</b> is a "C" shape with a circle in the
middle. The circle is not surrounded by the "C" shape, but the Convex Hull of
the "C" shape contains the circle.</td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder End On View.png" alt="shelters"></td>
</tr>
<tr><td width="350">
A hollow cylinder with another cylinder inside it, should also be considered
<b>Shelters</b></td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Angled View.png" alt="shelters"></td>
</tr>
<tr><td width="350">
But in the slice plane it appears as a rectangle between two larger rectangles.
On the slice plane there is no indication that the two larger rectangles are
part of the same 3D volume.

My current definition of <i>Convex Hull</i> is:
The <i>Convex Hull</i> consists of multiple elastic bands stretched around each
external contour, rather that one elastic band stretched around all external
contours.  Using this definition, The horizontal hollow cylinder with another
cylinder inside it will be classified as <b>Disjoint</b></td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Almost Top View.png" alt="shelters"></td>
</tr>
<tr><td width="350">
If the convex hull is defined as "The smallest convex polygon that contains
all the regions on the plane, then The horizontal hollow cylinder with another
cylinder inside it will be correctly classified as <b>Shelters</b>.</td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered Horizontal cylinder Almost Top View.png" alt="shelters"></td>
</tr>
<tr><td width="350">
To add another complication, if the same hollow cylinder with another cylinder
inside it is rotated so that the cylinder axis is perpendicular to the slice
plane, then the same geometry will be classified as <b>Surrounds</b>.</td>
<td width="250px"><img src="Images/FreeCAD Images/Sheltered cylinder Top View.png" alt="shelters"></td>
</tr>
</table>

#### Hull Definition

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
