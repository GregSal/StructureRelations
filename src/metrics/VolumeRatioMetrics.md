# Volume Metrics and Their Ratios
- Only Meaningful for "Shared" Relationships (Equal, Contains, Partitioned, Overlaps)
- Volume Metrics can be applied to situations with > 2 structures, but are then not tied to a relationship.

<link rel="stylesheet" href="../relations.css">

## Color Coding
<table style="border: 2px solid black; width=50px;"><tr><td>
<ul style="font-weight: 900; font-size: 20px;">
<li style="color: blue;">Structure A</li>
<li style="color: green;">Structure B</li>
<li style="color: orange;">intersection of a & b</li></ul>
</tr></td></table>


## Volume Metrics
<table width="450px">
<tr><th>Structures</th><th>Union</th><th>Intersection</th><th>Difference</th></tr>
<tr>
<td><img src="../Images/2D Relations/2 volumes.png" alt="Equals"  class="d100"></td>
<td><img src="../Images/2D Relations/Volume Union.png" alt="Equals" class="d100"></td>
<td><img src="../Images/2D Relations/Volume intersection.png" alt="Equals"  class="d100"></td>
<td><img src="../Images/2D Relations/Volume Difference.png" alt="Equals" class="d100"></td>
</tr></table>

### EQUAL
<table width="450px">
<tr class="l"><th>EQUAL</th><th>Shared</th><th>Symmetric, Transitive</th></tr>
<td class="d" colspan="3">
The interiors of <span class="a">A</span> and <span class="b">B</span>
intersect and no part of the interior of one geometry intersects the exterior of the other.
</td></tr>
<tr><td>
<img src="../Images/Relationships/equals.png" alt="Equals" >
</td>
<td colspan="2"> Union = Intersection = V<sub>A</sub> = V<sub>B</sub><br> Difference = 0</td>
</tr></table>

### CONTAINS
<table width="450px">
<tr class="l"><th>CONTAINS / WITHIN</th><th>Shared</th><th>Transitive</th></tr>
<td class="d" colspan="3">
All points of <span class="b">B</span> lie in the interior of <span class="a">A</span>, no points of <span class="b">B</span> lie in the exterior of <span class="a">A</span>, some points in <span class="a">A</span> are exterior to <span class="b">B</span>, and the boundaries of <span class="a">A</span> and <span class="b">B</span> do not intersect.
</td></tr>
<tr><td>
<img src="../Images/Relationships/contains simple.png" alt="Contains">
<img src="../Images/Relationships/contains with hole.png" alt="Contains With Hole">
</td>
<td colspan="2">
Union = V<sub>A</sub><br>
Intersection = V<sub>B</sub><br>
Difference =V<sub>A</sub> - V<sub>B</sub></td>
</tr>
</table>

### PARTITIONED
<table width="450px">
<tr class="l"><th>PARTITIONED / PARTITIONS</th><th>Shared</th><th></th></tr>
<td class="d" colspan="3">
All points of <span class="b">B</span> lie in the interior of <span class="a">A</span>,
no points of <span class="b">B</span> lie in the exterior of <span class="a">A</span>,
some points in <span class="a">A</span> are exterior to <span class="b">B</span>,
and the boundaries of <span class="a">A</span> and <span class="b">B</span> have more than one point in common.
</td></tr>
<tr><td>
<img src="../Images/Relationships/partitions simple.png" alt="Partition">
<img src="../Images/Relationships/partitions archipelago with island.png" alt="Partition With Island">
</td>
<td colspan="2">
Union = V<sub>A</sub><br>
Intersection = V<sub>B</sub><br>
Difference =V<sub>A</sub> - V<sub>B</sub></td>
</td>
</tr></table>

### OVERLAPS
<table width="450px">
<tr class="l"><th>OVERLAPS</th><th>Shared</th><th>Symmetric</th></tr>
<td class="d" colspan="3">
<span class="a">A</span> and <span class="b">B</span>
have some but not all points in common.
</td></tr>
<tr><td>
<img src="../Images/Relationships/overlaps simple.png" alt="Overlaps">
<img src="../Images/Relationships/overlaps ring surrounds simple.png" alt="Ring Overlaps Simple">
</td>
<td colspan="2">
V<sub>A</sub> + V<sub>B</sub> > Union<br>
V<sub>A</sub> > Intersection<br>
V<sub>B</sub> > Intersection<br>
Difference > 0</td>
</tr></table>

### BORDERS
<table width="450px">
<tr class="l"><th>BORDERS</th><th>Adjoining</th><th>Symmetric</th></tr>
<td class="d" colspan="3">
The exterior boundaries of <span class="a">A</span> and <span class="b">B</span>
have more than one point in common, but their interiors do not intersect.
</td></tr>
<tr><td>
<img src="../Images/Relationships/borders.png" alt="Overlaps">
<img src="../Images/Relationships/Concave Borders.png" alt="Ring Overlaps Simple">
</td>
<td colspan="2">
Union = V<sub>A</sub> + V<sub>B</sub><br>
Intersection = 0<br>
Difference = 0</td>
</tr></table>

### CONFINES (Interior Borders)
<table width="450px">
<tr class="l"><th>CONFINES / CONFINED</th><th>Adjoining</th><th></th></tr>
<td class="d" colspan="3">
Part of the interior boundary of <span class="a">A</span> and
the exterior boundary of <span class="b">B</span> have more than one point in common
but their interiors do not intersect.
</td></tr>
<tr><td>
<img src="../Images/Relationships/confines.png" alt="Confines">
<img src="../Images/Relationships/confines with island.png" alt="Confines With Island">
</td>
<td colspan="2">
Union = V<sub>A</sub> + V<sub>B</sub><br>
Intersection = 0<br>
Difference = 0</td>
</tr></table>

### SURROUNDS
<table width="450px">
<tr class="l"><th>SURROUNDS / ENCLOSED</th><th>Separate</th><th>Transitive</t></tr>
<td class="d" colspan="3">
<span class="a">A</span> and <span class="b">B</span> have no interior points in common, but the Exterior of <span class="a">A</span> (<span class="a">A</span> with its holes of filled) contains <span class="b">B</span>.
</td></tr><tr><td>
<img src="../Images/Relationships/surrounds simple.png" alt="shelters">
<img src="../Images/Relationships/surrounds with ring.png" alt="shelters">
</td>
<td colspan="2">
Union = V<sub>A</sub> + V<sub>B</sub><br>
Intersection = 0<br>
Difference = 0</td>
</tr></table>

### SHELTERS
<table width="450px">
<tr class="l"><th>SHELTERS / SHELTERED</th><th>Separate</th><th>Transitive</t></tr>
<td class="d" colspan="3">
<span class="a">A</span> and <span class="b">B</span> have no points in common,
but the Convex Hull around <span class="a">A</span> contains <span class="b">B</span>.
</td></tr><tr><td>
<img src="../Images/Relationships/shelters.png" alt="shelters">
</td>
<td colspan="2">
Union = V<sub>A</sub> + V<sub>B</sub><br>
Intersection = 0<br>
Difference = 0</td>
</tr></table>

### DISJOINT
<link rel="stylesheet" href="relations.css">
<table width="450px">
<tr class="l"><th>DISJOINT</th><th>Separate</th><th>Symmetric</th></tr>
<td class="d" colspan="3">The Convex Hull around <span class="a">A</span>
has no points in common with <span class="b">B</span>.
</td></tr>
<tr><td>
<img src="../Images/Relationships/disjoint.png" alt="Disjoint">
</td>
<td colspan="2">
Union = V<sub>A</sub> + V<sub>B</sub><br>
Intersection = 0<br>
Difference = 0</td>
</tr></table>

## Volume Ratios
### Possible Ratios
- Intersection / Union (Range: 0 to 1)
- Difference / Union (Range: 0 to 1)
- Intersection / Difference (Range: 0 to infinity)

- *OVERLAPS* only
  - V<sub>A</sub> / Union (Range: 0 to 1)
  - Intersection / V<sub>A</sub> (Range: 0 to 1)
  - V<sub>B</sub> / V<sub>A</sub> (Range: 0 to 1)
  - Difference / V<sub>A</sub> (Range: 0 to 1)

- Only implement these two to begin with.
  Others can be implemented later by editing Json file with Ratio definitions.
  - Intersection / Union
  - Difference / Union (= 1 - Intersection / Union for *CONTAINS* and *PARTITIONED*)

**Note:** For *CONTAINS* & *PARTITIONED*
  - Union = V<sub>A</sub>
  - Intersection = V<sub>B</sub>
  - Difference =V<sub>A</sub> - V<sub>B</sub>
