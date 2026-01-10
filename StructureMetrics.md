# Structure Metrics

## Ratio of Volumes

Ratio of the overlapping volume to the average or larger volume.
$R_V =\frac{V( a \cap b )}{\overline{V_{a,b}}}$

Since the structures are compared slice by slice, the Volume function can be written as:<br>
$V(x) = t \times \operatorname{area} ( x )$,<br>
where $t$ is the slice thickness.

When calculating the ratio, slice thickness is a constant, so the Volume ratio is equal to the ratio sums of the areas over all slices:

$R_V =\frac{\sum_s \operatorname{area}( a \cap b )}{\sum_s \overline{\operatorname{area}_{a,b}}}$

*Used By:*

- **Overlaps**
> - Ratio of the volume of overlap to the average volume of $a$ and $b$:
> - $R_v =\frac{\sum_s \left[ 2 \times \operatorname{area}( a \cap b ) \right] }{\sum_s \left[ \operatorname{area}( a ) + \operatorname{area}( b ) \right] }$


- **Partition**
> - Ratio of the volume of the overlap to the volume of the larger structure ($a$):
> - $R_V =\frac{\sum_s \operatorname{area}( a \cap b )}{\sum_s \operatorname{area}( a )}$


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


## Distance
- $Distance = distance(a,b)$

**Z direction metrics**

- The larger of $\Delta Z$ or $d_{2D}$

**Used By:**
- Disjoint


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
