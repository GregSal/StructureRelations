'''Calculate metrics for relationships between structures.
'''
# %% Imports
# Type imports
from typing import Dict, List

# Standard Libraries
from itertools import product

# Shared Packages
import numpy as np
import shapely

# Local packages
from types_and_classes import PRECISION

from region_slice import RegionSlice

from utilities import find_intersection, generate_orthogonal_lines


# %% Distance Functions
def calculate_orthogonal_distances(primary: RegionSlice,
                                   secondary: RegionSlice,
                                   coverage: str,
                                   use_centre: str,
                                   precision: int = PRECISION
                                   ) -> List[Dict[str, float]]:
    '''Calculate the orthogonal distances between two structures.

    The orthogonal distances are the minimum distances between the two
    structures in the positive an negative x, y, and z directions.

    The distances are determined as follows:
        1. Each polygon in the first StructureSlice is paired with each polygon
             in the second StructureSlice.
        2. The maximum extent of the combined polygon pair is obtained.
        3. The center of the one of the two polygons is obtained.
        4. Four orthogonal lines are generated; each extending from the center
            to the appropriate extent.
        4. The intersections of each orthogonal line with the two polygons are
            determined.
        5. The orthogonal distance in a given direction is calculated from the
            minimum distance between the orthogonal line intersections of the
            primary polygons with those of the other polygon.

    Args:
        primary (StructureSlice): The primary structure to compare.

        secondary (StructureSlice): The secondary structure to compare.

        coverage (str): The coverage type to use when selecting the contours.

        use_centre (str): The method to use to calculate the centre of the
            secondary structure.

        precision (int, optional): The number of decimal points to round to.
            Defaults to global PRECISION constant.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing the orthogonal
            distances between the two structures. The keys of the dictionaries
            are: ['x_neg', 'y_neg', 'x_pos', 'y_pos', 'z_neg', 'z_pos'].
    '''

    def calculate_distance(poly_a: shapely.Polygon, poly_b: shapely.Polygon,
                        use_centre: str, precision: int = PRECISION) -> Dict[str, float]:
        # Calculate the orthogonal distances between the two polygons.
        orthogonal_lines = generate_orthogonal_lines(poly_a, poly_b, use_centre)

        ortho_dist = {}
        for label, line in orthogonal_lines.items():
            points_a = find_intersection(line, poly_a)
            points_b = find_intersection(line, poly_b)

            # The orthogonal distance is the minimum distance between the
            # intersections of the two structures with orthogonal line in that
            # direction.
            if points_a.is_empty or points_b.is_empty:
                dist = np.nan
            else:
                distances = []
                for point_a, point_b in product(points_a.geoms, points_b.geoms):
                    distances.append(shapely.distance(point_a, point_b))
                dist = round(min(distances), precision)
            ortho_dist[label] = dist
        return ortho_dist

    # Calculate the orthogonal distances between the each pair of polygons in
    # the two structures.
    primary_contours = primary.select(coverage).geoms
    secondary_contours = secondary.contour.geoms
    distances = []
    for poly_a, poly_b in product(primary_contours, secondary_contours):
        ortho_dist = calculate_distance(poly_a, poly_b, use_centre, precision)
        distances.append(ortho_dist)
        #if distances is None:
        #    distances = Extent(**ortho_dist)
        #else:
        #    distances = distances.get_min(Extent(**ortho_dist))
    #return asdict(distances)
    return distances
