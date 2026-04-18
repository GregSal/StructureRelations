# Structure Metrics

## Metrics Categories

There are 4 kinds of Metrics, based on the type of units.

- **Length** (one dimensional)
- **Surface Area** (two dimensional)
- **Volume** (three dimensional)
- **Ratio** (ratio of two other metrics, no units)

### Length
    - Units of cm or mm

    - For a single structure there can be different types of length metrics.
      - The maximum distance between any two points on a contour.
      - The orthogonal bounds of the structure.bounds(a) − bounds(b)
      - The equivalent diameter of the structure (diameter of a circle with the
          same area as the structure).

    - For two structures it is some form of measurement of the distance between
        the contours of two structures.
      - The minimum distance between the contours of the two structures.
          ($distance(a,b)$)
      - The maximum distance between the contours of the two structures
          (Hausdorff distance).
      - The average distance between the contours of two structures.
      - The distance between the contours of two structures in the
          orthogonal directions (L, R, Ant, Post, Sup, Inf).

### Surface Area
    - Units of $cm^2$, $mm^2$

    - For a single structure this is the surface area of that structure.

    - For two structures there can be different types of surface area metrics.
      - A measure of the area of the surfaces where the two structures touch.
          ($A_{boundary} \cap B_{boundary}$)
      - A measure of the area of the surface of one structure that extends
          within (or without) the other structure. ($[A - B]_{boundary}$)


### Volume
    - Units of $cm^3$, cc, ml
    - Volume of overlap ($A \cap B$), or difference ($A - B$) between two structures
    - For a single structure this is the volume of that structure


  - **Ratio**
    - Ratio of two other metrics
    - The two metrics must be of the same type and units
    - The two metrics can be the same of from different structures
    - Ratios have no units (None) or \%


## Metric Definitions

### Distance
The Distance metric is the distance between two structures, which is the
minimum distance between any point on the contour of one structure to any point
on the contour of the other structure. On a given slice, this can be calculated
using the Shapely `distance` function.
> $Distance = distance(a,b)$

In the Z direction it is the distance between the closest boundary slices of
$a$ and $b$
> $\Delta Z$

#### Calculation
For the entire structure, the distance is the minimum of the 2D distance and the Z distance:
> $Distance = min( distance(a,b), \Delta Z )$

#### Usage
The Distance metric is well defined for two structures that do not overlap:
- DISJOINT
- SURROUNDS
- SHELTERS

For structures that touch each other (BORDERS), the distance is zero.

For structures that overlap in any way (OVERLAPS, CONTAINS etc.), the distance
is undefined (`NaN`).

## Ratio of Volumes

Ratio of the overlapping volume to the average or larger volume.
$R_V =\frac{V( a \cap b )}{\overline{V_{a,b}}}$

Since the structures are compared slice by slice, the Volume function can be written as:<br>
$V(x) = t \times \mathsf{area} ( x )$,<br>
where $t$ is the slice thickness.

When calculating the ratio, slice thickness is a constant, so the Volume ratio is equal to the ratio sums of the areas over all slices:

$R_V =\frac{\sum_s \mathsf{area}( a \cap b )}{\sum_s \overline{\mathsf{area}_{a,b}}}$

*Used By:*

- **Overlaps**
> - Ratio of the volume of overlap to the average volume of $a$ and $b$:
> - $R_v =\frac{\sum_s \left[ 2 \times \mathsf{area}( a \cap b ) \right] }{\sum_s \left[ \mathsf{area}( a ) + \mathsf{area}( b ) \right] }$


- **Partition**
> - Ratio of the volume of the overlap to the volume of the larger structure ($a$):
> - $R_V =\frac{\sum_s \mathsf{area}( a \cap b )}{\sum_s \mathsf{area}( a )}$


## Ratio of Surface Area

Ratio of the touching surface area of the two structures, to the surface area of the average or larger structure.
$R_A =\frac{A( a \cap b )}{\overline{A_{a,b}}}$,<br>
where $A$ is the Surface Area.

Since the structures are compared slice by slice, the Surface Area function can be written as:<br>
$A(x) = t \times \ell_s( x )$,<br>
where $t$ is the slice thickness and $\ell_s$ is the relevant perimeter length on a given slice.

When calculating the ratio, slice thickness is a constant, so the surface area ratio is equal to the ratio sums of the relevant perimeter lengths over all slices:

$R_A =\frac{\sum_{S} \ell_s( a \cap b )}{\sum_{S} \overline{\ell_s}}$

*Used By:*

- Exterior Borders
> - Ratio of the length of the touching exterior perimeters to the average length of the exterior perimeters of $a$ and $b$.
> - $R_A = \frac{ 2 \sum_{S} \ell_s⁡(a_p \cap b_p) }{ \sum_{S} \left[ \ell_s(a_{px}) + \ell_s(b_{px}) \right] }$

- Interior Borders
> - Ratio of the length of touching perimeters to the length of the perimeter of the hole in $a$ containing $b$.
> - $R_l = \frac{ \sum_{S} \ell_s⁡(a_p \cap b_p)}{ \sum_{S} \left[ \ell_s(a_{ph}) + \ell_s(b_{px}) \right] }$

Where:
- $a_p$ is the perimeter of polygon $a$
- $b_p$ is the perimeter of polygon $b$
- $a_{px}$ is the exterior perimeter of polygon $a$
- $b_{px}$ is the exterior perimeter of polygon $b$
- $a_{ph}$ is the perimeter of the relevant hole within polygon $a$
- $\ell_s⁡(p_i)$ is the length of perimeter $i$ on slice $s$


## Margins
- $Margin_\perp = bounds(a) − bounds(b)$

- $Margin_{min} = distance(a,b)$

- $Margin_{max} = distance_{housdorff}(a,b)$

**Z direction metrics**
- Orthogonal:
  - The distance between the last slice containing $a$ and the last slice
      of $b$ where the last contour of $a$ overlaps with the contour of $b$
- Max:
  - The larger of $\Delta Z$ or $d_{2D}$
- Min:
  - The Smaller of $d_{min}^{2D}$ and $\Delta Z$,<br>


**Used By:**
- Contains
- Surrounds
- Shelters (Does not use $Margin_{max}$)


**Z direction metrics**
  - % Volume and % Surface Area:
    - No additional calculations.
  - Distance:
    - The larger of $\Delta Z$ or $d_{2D}$


```
Relationship: Partition
Expected Margins: 2 cm in all directions.
        2.0   0.0
        ANT  SUP
         | /
         |/
2.0 RT--- ---LT 2.0
        /|
       / |
   INF  POST
  0.0   2.0

Min: 0.0   Max: 2.0
```
