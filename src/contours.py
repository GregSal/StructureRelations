'''Contour Classes and related Functions
'''
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import Polygon

from types_and_classes import ROI_Type, SliceIndexType, ContourPointsType
from types_and_classes import ContourIndex
from types_and_classes import InvalidContour
from types_and_classes import SliceSequence
from types_and_classes import PRECISION, SliceIndexSequenceType


# %% Polygon Functions
def points_to_polygon(points: List[Tuple[float, float]]) -> Polygon:
    '''Convert a list of points to a Shapely polygon and validate it.

    Args:
        points (List[Tuple[float, float]]): A list of tuples containing 2D or
            3D points.

    Raises:
        InvalidContour: If the points cannot form a valid polygon.

    Returns:
        Polygon: A valid Shapely polygon.
    '''
    if not points:
        return shapely.Polygon()
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise InvalidContour("Invalid polygon created from points.")
    return polygon



#%% Interpolation Functions
def calculate_new_slice_index(slices: SliceIndexSequenceType,
                              precision=PRECISION) -> float:
    '''Calculate the new z value based on the given slices.

    Args:
        slices (Union[List[SliceIndexType], SliceIndexType]): The slices to
            calculate the new z value from.

    Returns:
        float: The calculated new z value.
    '''
    if isinstance(slices, (list, tuple)):
        new_slice = round(np.mean(slices), precision)
        min_slice = round(min(slices), precision)
        max_slice = round(max(slices), precision)
        if not (min_slice <= new_slice <= max_slice):
            raise ValueError("Calculated slice is out of bounds after rounding.")
        return new_slice
    else:
        return slices


def interpolate_polygon(slices: SliceIndexSequenceType, p1: shapely.Polygon,
                        p2: shapely.Polygon = None) -> shapely.Polygon:
    '''Interpolate a polygon between two polygons based on the given slices.

    This function takes two polygons and interpolates a new polygon based on
    the given slices. The new polygon is created by interpolating the
    coordinates of the two polygons,

    **any holes of the first polygon are also interpolated.**

    The new polygon is then assigned a z value based on the given slices.
    The function also handles the case where one of the polygons
    is empty. The function raises a ValueError if either of the polygons are
    multi-polygons.
    The function also raises a ValueError if the first polygon is empty and no
    second polygon is given.
    Args:
        slices (SliceIndexSequenceType): _description_
        p1 (shapely.Polygon): _description_
        p2 (shapely.Polygon, optional): _description_. Defaults to None.

    Raises:
        ValueError: When either of the polygons are multi-polygons
        ValueError: When the first polygon is empty and no second polygon is
            given.
    Returns:
        shapely.Polygon: _description_
    '''
    # TODO Use shapely.affinity.scale to interpolate polygons
    def align_polygons(p1, p2):
        # Get the point between the centroid of the first and second polygons.
        cm_shift = ((p2.centroid.x - p1.centroid.x) / 2,
                    (p2.centroid.y - p1.centroid.y) / 2)
        # Shift the two polygons to the same mid point.
        ctr_poly1 = shapely.affinity.translate(p1,
                                               xoff=cm_shift[0],
                                               yoff=cm_shift[1])
        ctr_poly2 = shapely.affinity.translate(p2,
                                               xoff=-cm_shift[0],
                                               yoff=-cm_shift[1])
        # get the scaling factors for the two polygons.
        x_size1 = p1.bounds[2] - p1.bounds[0]
        y_size1 = p1.bounds[3] - p1.bounds[1]
        x_size2 = p2.bounds[2] - p2.bounds[0]
        y_size2 = p2.bounds[3] - p2.bounds[1]
        scale_x1 = (x_size1 + x_size2) /( 2 * x_size1)
        scale_y1 = (y_size1 + y_size2) /( 2 * y_size1)
        scale_x2 = (x_size1 + x_size2) /( 2 * x_size2)
        scale_y2 = (y_size1 + y_size2) /( 2 * y_size2)
        scale_poly1 = shapely.affinity.scale(ctr_poly1,
                                             xfact=scale_x1,
                                             yfact=scale_y1)
        scale_poly2 = shapely.affinity.scale(ctr_poly2,
                                             xfact=scale_x2,
                                             yfact=scale_y2)
        return scale_poly1, scale_poly2

    def match_holes(p1, p2):
        if p1.is_empty:
            holes1 = []
        else:
            holes1 = list(p1.interiors)
        # If no second polygon given, use the centroid of each first hole as the
        # matching second hole boundary.
        if p2 is None:
            if holes1:
                matched_holes = [(hole, hole.centroid) for hole in holes1]
            else:
                matched_holes = []
            return matched_holes
        # If the first polygon does not have any holes, and the second polygon
        # does, use the centroid of the second hole as the matching first hole
        # boundary.
        holes2 =  list(p2.interiors)
        if not holes1:
            matched_holes = [(hole, hole.centroid) for hole in holes2]
            return matched_holes
        # If both polygons have holes, match the holes match the holes of the
        # second polygon to the first polygon.
        matched_holes = []
        # set each second hole as not matched.
        hole2_matched = {i: False for i in range(len(holes2))}
        for hole1 in holes1:
            matched1 = False  # set the first hole as not matched.
            for idx, hole2 in enumerate(holes2):
                if hole1.overlaps(hole2):
                    matched_holes.append((hole1, hole2))
                    matched1 = True # set the first hole as matched.
                    hole2_matched[idx] = True  # set the second hole as matched.
            # If the first hole is not matched, use the centroid of the first
            # hole as the matching second hole boundary.
            if not matched1:
                matched_holes.append((hole1, hole1.centroid))
        # Add any unmatched holes from the second polygon, using the centroid of
        # the second hole as the matching first hole boundary.
        for idx, hole2 in enumerate(holes2):
            if not hole2_matched[idx]:
                matched_holes.append((hole2, hole2.centroid))
        return matched_holes

    def interpolate_boundaries(boundary1, boundary2):
        new_cords = []
        for crd in boundary1.coords:
            ln = shapely.shortest_line(shapely.Point(crd), boundary2)
            ptn = ln.interpolate(0.5, normalized=True)
            new_cords.append(ptn)
        return new_cords

    # Get the z value for the new polygon.
    new_z = calculate_new_slice_index(slices)
    # Error Checking
    # If either of the polygons are multi-polygons, raise an error.
    if isinstance(p1, shapely.MultiPolygon):
        raise ValueError('Only single polygons are supported.')
    if isinstance(p2, shapely.MultiPolygon):
        raise ValueError('Only single polygons are supported.')
    # Cannot interpolate two empty polygons.
    if p1.is_empty & (p2 is None):
        raise ValueError('No second polygon given and first polygon is empty.')

    # If only one polygon is given, scale the polygon to half its size.
    if p2 is None:
        itp_poly = shapely.affinity.scale(p1, xfact=0.5, yfact=0.5)
        itp_poly = shapely.force_3d(itp_poly, new_z)
        return itp_poly
    elif p1.is_empty:
        itp_poly = shapely.affinity.scale(p2, xfact=0.5, yfact=0.5)
        itp_poly = shapely.force_3d(itp_poly, new_z)
        return itp_poly

    # If two polygons given, align the polygons to the same center and size and
    # then adjust the shape using a linear interpolation to the half way point
    # between the two polygons.
    # Align and scale the polygons to the same centre and size.
    aligned_poly1, aligned_poly2 = align_polygons(p1, p2)
    # Interpolate the new polygon coordinates as half way between the p1
    # boundary and boundary 2.
    boundary1 = aligned_poly1.exterior
    boundary2 = aligned_poly2.exterior
    new_cords = interpolate_boundaries(boundary1, boundary2)
    # Add the holes to the new polygon.
    new_holes = []
    matched_holes = match_holes(p1, p2)
    for hole1, hole2 in matched_holes:
        new_hole = interpolate_boundaries(hole1, hole2)
        new_holes.append(new_hole)
    # Build the new polygon from the interpolated coordinates.
    itp_poly = shapely.Polygon(new_cords, holes=new_holes)
    # Add the z value to the polygon.
    itp_poly = shapely.force_3d(itp_poly, new_z)
    return itp_poly


# %% Contour Classes
class ContourPoints(dict):
    '''A dictionary of contour points.

    The dictionary has three required keys:
        'ROI': (ROI_Type) The value will contain the ROI number for the contour.
        'Slice': (SliceIndexType) The value will contain the slice index for
            the contour.
        'Points': A list of length 2 or three tuples for float containing the
            coordinates of the points that define the contour.

    The dictionary will accept additional keys related to the contour.

    Args:
        points (ContourPointsType: A list of tuples containing 2 or 3 float
            values representing 2D or 3D contour points.
        roi (ROI_Type): An integer referencing a structure.
        slice_index (SliceIndexType, optional): The slice index for the contour.
            If not provided, the slice index is extracted from the z coordinate
            of the first point or set to 0.0. Defaults to None.

    Raises:
        InvalidContour: Raised if the input data is invalid.
    '''

    def __init__(self, points: ContourPointsType, roi: ROI_Type,
                 slice_index: SliceIndexType = None, **dict_items):
        # Validate the ROI
        if not isinstance(roi, int):
            raise InvalidContour("ROI must be an integer.")
        slice_index = self.validate_slice_index(slice_index, points)
        points = self.validate_points(points, slice_index)
        dict_items['ROI'] = roi
        dict_items['Slice'] = slice_index
        dict_items['Points'] = points
        super().__init__(**dict_items)

    @staticmethod
    def validate_slice_index(slice_index: SliceIndexType,
                             points: ContourPointsType)-> SliceIndexType:
        '''Validate or set the slice_index for the contour.

    - slice_index if provided, must be float or integer.  If not provided and
        the points tuples are 3D, then slice_index is extracted from the z
        coordinate for the first point. Otherwise slice_index is set to 0.0.

    Args:
        slice_index (SliceIndexType): The slice index for the contour.  If None,
            the slice index is extracted from the z coordinate of the first
            point.
        points (ContourPointsType: A list of tuples containing 2 or 3 float
            values representing 2D or 3D contour points.

    Raises:
        InvalidContour: Raised if the input data is invalid.

    Returns:
        SliceIndexType: The validated or set slice_index value.
        '''
        if slice_index is None:
            if len(points[0]) == 3:
                slice_index = points[0][2]
            else:
                raise InvalidContour("Slice index not provided and points are 2D.")
        else:
            # Verify that the slice is a float or integer
            try:
                slice_index = float(slice_index)
            except ValueError as err:
                raise InvalidContour("Slice must be an integer or float.") from err
        return slice_index

    @staticmethod
    def validate_points(points: ContourPointsType,
                        slice_index: SliceIndexType = None
                       )-> tuple[ContourPointsType, ROI_Type, SliceIndexType]:
        '''Validate the contour points and add a z coordinate if necessary.

        - points must be a list of 3 or more tuples containing 2 or 3 float
            values representing 2D or 3D contour points.
        - If the points are 2D, slice_index is added to the points as the
            z coordinate.

        Args:
            points (ContourPointsType: A list of tuples containing 2 or 3 float
                values representing 2D or 3D contour points.
            slice_index (SliceIndexType): The slice index for the contour.

        Raises:
            InvalidContour: Raised if the input data is invalid.

        Returns:
            ContourPointsType: The validated 3D points.
        '''
        # Validate the points
        if len(points) < 3:
            raise InvalidContour("Contour must have at least three points.")
        # Verify that the points are tuples of floats of length 2 or 3
        clean_points = []
        point_dim = None
        for point in points:
            # Verify that each point is a tuple of float.
            if not isinstance(point, tuple):
                raise InvalidContour("Points must be tuples of floats.")
            if not all(isinstance(coord, (float, int)) for coord in point):
                raise InvalidContour("Points must be tuples of floats.")
            # Verify that each point is a tuple of length 2 or 3.
            this_point_dim = len(point)
            if this_point_dim not in (2, 3):
                raise InvalidContour("Points must be tuples of length 2 or 3.")
            # Verify that all points have the same length.
            if not point_dim:
                point_dim = this_point_dim
            else:
                if point_dim != this_point_dim:
                    raise InvalidContour("All points must have the same length.")
                # If points are 3D, verify that all z coordinates match the
                # slice index
                if this_point_dim == 3 and point[2] != slice_index:
                    raise InvalidContour('All 3D points must have the same z '
                                         'coordinate (slice index).')
            # If points are 2D, add slice as the z-coordinate.
            if this_point_dim == 2:
                point = point + (slice_index,)
            clean_points.append(point)
        return clean_points


class Contour:
    '''Class representing a contour with associated metadata.

    Attributes:
       Index attributes:
        roi (ROI_Type): The ROI number of the contour.
        slice_index (SliceIndexType): The slice index of the contour.
        contour_index (int): Auto-incremented contour index.
        region_index (int): The region index of the contour.
            Defaults to None until regions are assigned.
        index (ContourIndex): A tuple of (ROI, SliceIndex, ContourIndex).
            This is a read-only property.
      shape:
        polygon (shapely.Polygon): The polygon generated from the contour.
        exterior (shapely.Polygon): The solid exterior polygon of the contour.
        hull (shapely.Polygon): The convex hull of the contour.
        thickness (float): The thickness of the contour.  Defaults to
            Contour.default_thickness (0.0).

      hole information:
        is_hole (bool): Whether the contour is a hole.
        hole_reference (ContourIndex): If is_hole is True, contains the index
            of the smallest non-hole contour that contains this hole.
            If is_hole is False, hole_reference is None.
        hole_type (str): If is_hole is True, the type of the hole. One of:
                Open,
                Closed, or
                Unknown (Default).
            If is_hole is False, hole_type is 'None'.

      boundary information:
        is_boundary (bool): Whether the contour is a boundary.
            Defaults to False.
        is_interpolated (bool): Whether the contour is interpolated.
            Defaults to False.
    '''
    # Class Variables
    # Incremented counter for contour index
    counter = 0
    # Default thickness for contours
    default_thickness = 0.0

    def __init__(self, roi: ROI_Type, slice_index: SliceIndexType,
                 polygon: shapely.Polygon,
                 contours: List['Contour']) -> None:
        self.roi = roi
        self.slice_index = slice_index
        self.polygon = polygon
        self.thickness = self.default_thickness
        self.is_hole = False
        self.related_contours = []
        self.hole_type = 'None'
        self.is_boundary = False
        self.is_interpolated = False
        self.region_index = ''  # Default to an empty string
        self.related_regions = []
        self.contour_index = Contour.counter
        Contour.counter += 1
        self.validate_polygon()
        self.compare_with_existing_contours(contours)

    @property
    def index(self) -> ContourIndex:
        '''Return a tuple representing (ROI, SliceIndex, ContourIndex).'''
        return (self.roi, self.slice_index, self.contour_index)

    @property
    def exterior(self)-> shapely.Polygon:
        '''The solid exterior Polygon.


        Returns:
            shapely.Polygon: The contour Polygon with all holes
                filled in.
        '''
        ext_poly = shapely.Polygon(shapely.get_exterior_ring(self.polygon))
        return ext_poly

    @property
    def hull(self)-> shapely.Polygon:
        '''A bounding contour generated from the entire contour Polygon.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        Returns:
            shapely.Polygon: The bounding contour for the entire contour.
        '''
        hull = shapely.convex_hull(self.polygon)
        return hull

    def compare_with_existing_contours(self, contours: List['Contour']) -> None:
        '''Compare the polygon to each existing Contour in the list.

        If the polygon is within an existing contour, the new contour is a hole.
        If the polygon overlaps an existing contour, raise an error.

        Args:
            contours (List['Contour']): An ordered list of existing contours
                from largest area to smallest area.

        Raises:
            InvalidContour: Raised if the new contour overlaps an existing
                contour in the list.
        '''
        for existing_contour in reversed(contours):
            # starting from the largest contour and working down to the smallest
            if self.polygon.within(existing_contour.polygon):
                if existing_contour.is_hole:
                    # If the existing contour is a hole, the new contour cannot
                    # be its hole, but it could be an island and should be
                    # recorded as an embedded contour.
                    self.related_contours.append(existing_contour.contour_index)
                    existing_contour.related_contours.append(self.contour_index)
                else:
                    # New contour is completely within the existing contour
                    self.is_hole = True
                     # Set the hole reference to the existing contour index
                    self.related_contours.append(existing_contour.contour_index)
                    existing_contour.related_contours.append(self.contour_index)
                    self.hole_type = 'Unknown'

            elif self.polygon.overlaps(existing_contour.polygon):
                # New contour overlaps an existing contour, raise an error
                raise InvalidContour('New contour overlaps an existing contour.')

    def validate_polygon(self) -> None:
        '''Validate the polygon to ensure it is valid.'''
        if not self.polygon.is_valid:
            raise InvalidContour('Invalid polygon provided for the contour.')

    def area(self) -> float:
        '''Calculate the area of the contour polygon.'''
        return self.polygon.area

    def centroid(self) -> Tuple[float, float]:
        '''Calculate the centroid of the contour polygon.'''
        return self.polygon.centroid.coords[0]


class ContourMatch:
    '''Class representing a match between two contours.

    Attributes:
        contour1 (Contour): The first contour.
        contour2 (Contour): The second contour.
        thickness (float): Half the difference between the two slice indices.
        combined_area (float): The sum of the areas of the two contours.
        direction (int): 1 if the difference between the slice_index of the
            first and second contours is positive, -1 otherwise.
    '''

    def __init__(self, contour1: Contour, contour2: Contour) -> None:
        self.contour1 = contour1
        self.contour2 = contour2
        self.thickness = abs(contour1.slice_index - contour2.slice_index) / 2

    def direction(self, node: Contour) -> int:
        '''Get the direction of the contour match.

        Args:
            node (Contour): The contour node.

        Returns:
            int: The direction of the match.
        '''
        if node.index == self.contour1.index:
            offset = self.contour2.slice_index - node.slice_index
            direction = 1 if offset > 0 else -1
            return direction
        elif node.index == self.contour2.index:
            offset = self.contour1.slice_index - node.slice_index
            direction = 1 if offset > 0 else -1
            return direction
        else:
            raise ValueError('Node is not part of the match.')


# %% Contour Table Construction
def build_contour_table(slice_data: List[ContourPoints]) -> Tuple[pd.DataFrame,
                                                                  SliceSequence]:
    '''Build a contour table from a list of Contour objects.

    The table contains the following columns:
        ROI, Slice, Points, Polygon, Area
    The Polygon column contains the polygons generated from the points. The Area
    column contains the area of the polygons. The table is sorted by ROI, Slice
    and by descending area. The slice sequence is generated from the Slice
    column.

    Args:
        slice_data (List[ContourPoints]): A list of Contour objects.

    Returns:
        tuple: A tuple containing the contour table and the slice sequence.
            contour_table (pd.DataFrame): The contour table.
        slice_sequence (SliceSequence): An ordered list of all slice indexes in
            use and their neighbours.
    '''
    contour_table = pd.DataFrame(slice_data)
    # Convert the contours points to polygons and calculate their areas
    contour_table['Polygon'] = contour_table['Points'].apply(points_to_polygon)
    contour_table['Area'] = contour_table['Polygon'].apply(lambda poly: poly.area)
    # Sort the contours by ROI, Slice and decreasing Area
    # Decreasing area is important because that an earlier contour cannot be
    # inside a later one.
    contour_table.sort_values(by=['ROI', 'Slice', 'Area'],
                        ascending=[True, True, False],
                        inplace=True)
    # Generate the slice sequence for the contours
    slice_sequence = SliceSequence(contour_table.Slice)
    return contour_table, slice_sequence


# %% Contour Graph Construction Functions
def build_contours(contour_table, roi):
    '''Build contours for a given ROI from the contour table.

    This function filters the contour table for the specified ROI and creates
    Contour objects for each slice. It organizes the contours by slice
    in a dictionary.

    Args:
        contour_table (pd.DataFrame): The contour table containing contour data.
        roi (int): The ROI number to filter contours.

    Returns:
        dict: A dictionary where keys are slice indices and values are lists of
              Contour objects for that slice. Each list of contours is sorted
              by area in order of descending area.
    '''
    # Filter the contour table for the specified ROI
    contour_set = contour_table[contour_table.ROI == roi]
    # Create a dictionary to hold contours by slice
    contour_by_slice = defaultdict(list)
    # Set the index to 'Slice' for easier access
    contour_set.set_index('Slice', inplace=True)

    # Iterate over each slice and create Contour objects
    for slice_index, contour in contour_set.Polygon.items():
        contours_on_slice = contour_by_slice[slice_index]
        new_contour = Contour(roi, slice_index, contour, contours_on_slice)
        contour_by_slice[slice_index].append(new_contour)
    # Sort the contours on each slice by area in order of descending area
    for slice_index in contour_by_slice:
        contour_by_slice[slice_index].sort(key=lambda c: c.area(), reverse=True)
    return contour_by_slice


def build_contour_lookup(contour_graph: nx.Graph) -> pd.DataFrame:
    '''Build a lookup table for contours.

    This function creates a DataFrame that serves as a lookup table for contours.
    The hole type is categorized into 'Open', 'Closed', 'Unknown', and 'None'.
    The DataFrame is sorted by slice index and contour index.

    Args:
        contour_graph (nx.Graph): A Contour Graph object containing contour
            information. Each node in the graph should represent a contour and
            should have a 'contour' attribute that is an instance of the
            Contour class.

    Returns:
        pd.DataFrame: A DataFrame containing the contour lookup table.
            The DataFrame includes the following columns:
                - ROI,
                - SliceIndex,
                - HoleType,
                - Interpolated,
                - Boundary,
                - ContourIndex,
                - RegionIndex, and
                - Label.
    '''
    lookup_list = []
    for _, data in contour_graph.nodes(data=True):
        contour = data['contour']
        lookup_list.append({
            'ROI': contour.roi,
            'SliceIndex': contour.slice_index,
            'HoleType': contour.hole_type,
            'Interpolated': contour.is_interpolated,
            'Boundary': contour.is_boundary,
            'ContourIndex': contour.contour_index,
            'RegionIndex': contour.region_index,
            'Label': contour.index
        })
    contour_lookup = pd.DataFrame(lookup_list)
    contour_lookup.HoleType = contour_lookup.HoleType.astype('category')
    contour_lookup.HoleType.cat.set_categories(['Open', 'Closed',
                                                'Unknown', 'None'])
    contour_lookup.sort_values(by=['SliceIndex', 'ContourIndex'], inplace=True)
    return contour_lookup


def add_graph_edges(contour_graph: nx.Graph, contour_lookup: pd.DataFrame,
                    slice_sequence: SliceSequence) -> nx.Graph:
    '''Add edges to link Contour Neighbours.

    Edges are added between contours that are neighbours in the slice sequence
    and have the same hole type.  Matching contours are identified by the
    intersection of the contours' convex hulls.
    convex hulls.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table
            for contours, including slice index, contour index, and hole type.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.

    Returns:
        nx.Graph: The updated contour graph with edges added.
    '''
    # Iterate through each contour reference in the lookup table
    for contour_ref in contour_lookup.itertuples(index=False):
        # Get the contour reference
        this_slice = contour_ref.SliceIndex
        this_label = contour_ref.Label
        hole_type = contour_ref.HoleType
        this_contour = contour_graph.nodes(data=True)[this_label]['contour']
        # Identify the next slice in the sequence
        next_slice = slice_sequence.get_neighbors(this_slice).next_slice
        neighbour_idx = ((contour_lookup.SliceIndex == next_slice) &
                        (contour_lookup.HoleType == hole_type))
        # match this contour to the contours on the next slice
        for neighbour in contour_lookup.loc[neighbour_idx, 'Label']:
            neighbour_contour = contour_graph.nodes(data=True)[neighbour]['contour']
            if this_contour.hull.intersects(neighbour_contour.hull):
                contour_match = ContourMatch(this_contour, neighbour_contour)
                contour_graph.add_edge(this_label, neighbour, match=contour_match)
    return contour_graph


def add_boundary_contours(contour_graph: nx.Graph,
                          slice_sequence: SliceSequence) -> None:
    '''Add interpolated boundary contours to the graph.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.

    Returns:
        tuple: A tuple containing the updated contour graph and slice sequence.
            contour_graph (nx.Graph): The updated graph representation of the
                contours with interpolated boundary contours added.
            slice_sequence (SliceSequence): The updated slice sequence object
                with the interpolated slice indices added.
    '''
    # Select all nodes with only one edge (degree=1)
    boundary_nodes = [node for node, degree in contour_graph.degree()
                      if degree == 1]
    for original_boundary in boundary_nodes:
        # Get the contour and its slice index
        contour = contour_graph.nodes[original_boundary]['contour']
        this_slice = contour.slice_index
        # Determine the neighbor slice not linked with an edge
        neighbors = slice_sequence.get_neighbors(this_slice)
        # Get the slice index of the neighbouring contours.
        # This is the starting point for the neighbours of the interpolated slice
        slice_ref = {'PreviousSlice': neighbors.previous_slice,
                     'NextSlice': neighbors.next_slice}
        # Determine the slice index that is a neighbour of the original
        # boundary contour, so that the other slice index can be used to
        # interpolate the slice index of the interpolated contour.
        neighbouring_nodes = contour_graph.adj[original_boundary].keys()
        # Because degree=1, there should only be one neighbouring node.
        neighbour_slice = [nbr[1] for nbr in neighbouring_nodes][0]
        # Get slice index to use for interpolating (slice_beyond) and the
        # neighbouring slice references for the interpolated slice.
        if neighbors.previous_slice == neighbour_slice:
            # The next slice is the neighbour for interpolation
            slice_beyond = neighbors.next_slice
            # If the next slice is not set, use the gap to estimate the next
            # slice index.
            if pd.isna(slice_beyond):
                slice_beyond = this_slice + neighbors.gap(absolute=False)
            # The current boundary slice is the previous slice for the
            # interpolated slice.
            slice_ref['PreviousSlice'] = this_slice
        else:
            # The previous slice is the neighbour for interpolation
            slice_beyond = neighbors.previous_slice
            # If the next slice is not set, use the gap to estimate the previous
            # slice index.
            if pd.isna(slice_beyond):
                slice_beyond = this_slice - neighbors.gap(absolute=False)
            # The current boundary slice is the next slice for the
            # interpolated slice.
            slice_ref['NextSlice'] = this_slice
        # Calculate the interpolated slice index
        interpolated_slice = (this_slice + slice_beyond) / 2
        slice_ref['ThisSlice'] = interpolated_slice
        slice_ref['Original'] = False
        # Generate the interpolated boundary contour
        interpolated_polygon = interpolate_polygon([this_slice, slice_beyond],
                                                   contour.polygon)
        interpolated_contour = Contour(
            roi=contour.roi,
            slice_index=interpolated_slice,
            polygon=interpolated_polygon,
            contours=[]
            )
        interpolated_contour.is_interpolated = True
        interpolated_contour.is_boundary = True
        interpolated_contour.is_hole = contour.is_hole
        interpolated_contour.hole_type = contour.hole_type
        # Add the interpolated slice index to the slice sequence
        slice_sequence.add_slice(**slice_ref)
        # Add the interpolated contour to the graph
        interpolated_label = interpolated_contour.index
        contour_graph.add_node(interpolated_label, contour=interpolated_contour)
        # Add a ContourMatch edge between the original and interpolated contours
        contour_match = ContourMatch(contour, interpolated_contour)
        contour_graph.add_edge(original_boundary, interpolated_label,
                               match=contour_match)
    return contour_graph, slice_sequence


def set_enclosed_regions(contour_graph: nx.Graph) -> List[nx.Graph]:
    '''Create EnclosedRegion SubGraphs and assign RegionIndexes to the contours.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.

    Returns:
        List[nx.Graph]: A list of SubGraphs, each representing an enclosed region.
    '''
    region_counter = 0
    region_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Find connected contours in the contour graph.
    for region_nodes in nx.connected_components(contour_graph):
        # Create a SubGraph for the connected component.
        enclosed_region = contour_graph.subgraph(region_nodes)
        # Assign a unique RegionIndex to each contour in the SubGraph.
        # This index is composed of the ROI number, an uppercase letter and a
        # suffix number (if needed).)
        # Extract the ROI from the first node in the SubGraph.
        # The ROI is the same for all nodes in the Graph.
        roi = enclosed_region.nodes[list(region_nodes)[0]]['contour'].roi
        # Check if the region_counter exceeds the length of region_labels
        # If so, add a suffix to the label.
        if region_counter > len(region_labels):
            suffix = str(region_counter // len(region_labels) - 1)
        else:
            suffix = ''
        # Create a label for the enclosed region.
        region_label = f'{roi}{region_labels[region_counter]}{suffix}'
        # Assign the label to each contour in the SubGraph.
        for node in enclosed_region.nodes:
            contour = enclosed_region.nodes[node]['contour']
            contour.region_index = region_label
            # Add the region label to the related_regions
            # get the related_contours for the contour
            related_contours_list = contour.related_contours
            # Find all contours where the contour_index is in the
            # related_contours list.
            node_data = dict(contour_graph.nodes.data('contour'))
            for related_contour in node_data.values():
                if related_contour.contour_index in related_contours_list:
                    related_contour.related_regions.append(region_label)
        region_counter += 1
    return contour_graph


def set_hole_type(contour_graph: nx.Graph, contour_lookup: pd.DataFrame,
                  slice_sequence: SliceSequence) -> None:
    '''Determine whether the regions that are holes are 'Open' or 'Closed'.

    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table
            for contours, including slice index, contour index, and hole type.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.
    '''
    # Select boundary contours that are holes using contour_lookup
    hole_boundaries = ((contour_lookup['Boundary']) &
                       (contour_lookup['HoleType'] == 'Unknown'))
    # Initialize the region hole type to 'Unknown'
    # Get a list of the RegionIndexes that reference holes
    hole_regions = contour_lookup.loc[hole_boundaries, 'RegionIndex']
    hole_regions = list(hole_regions.drop_duplicates())
    region_hole_type = {region: 'Unknown' for region in hole_regions}
    # Iterate through each region and determine the hole type
    boundary_contours = contour_lookup.loc[hole_boundaries, 'Label']
    for boundary_label in boundary_contours:
        # Get the contour from the graph
        boundary_contour = contour_graph.nodes[boundary_label]['contour']
        # Get the RegionIndex for the boundary contour
        region_index = boundary_contour.region_index
        # Get the slice index of the boundary contour
        this_slice = boundary_contour.slice_index
        # Determine the slice index just beyond the boundary contour.
        neighbouring_nodes = contour_graph.adj[boundary_label].keys()
        # Because it is a boundary contour, there should only be one
        # neighbouring node.
        neighbour_slice = [nbr[1] for nbr in neighbouring_nodes][0]
        # Get the slice index of the boundary contour
        this_slice = boundary_contour.slice_index
        # Get the slice indexes of neighbouring contours
        neighbors = slice_sequence.get_neighbors(this_slice).neighbour_list()
        # The slice that is not a neighbour of the boundary contour is the
        # slice beyond the boundary contour.
        slice_beyond = [nbr for nbr in neighbors if nbr != neighbour_slice][0]
        beyond = contour_lookup.SliceIndex==slice_beyond
        contour_labels = list(contour_lookup.loc[beyond].Label)
        # The boundary is open if no contour contains the boundary contour
        boundary_closed = False
        for label in contour_labels:
            # Get the contour from the graph
            contour = contour_graph.nodes[label]['contour']
            if contour.polygon.contains(boundary_contour.polygon):
                # The hole is closed by the contour
                boundary_closed = True
                break
        if boundary_closed:
            # The hole is closed by the contour
            # Region is closed if all boundaries are closed
            if region_hole_type[region_index] == 'Unknown':
                region_hole_type[region_index] = 'Closed'
        else:
            # The hole is open by the contour
            # Region is open if any boundary is open
            if region_hole_type[region_index] == 'Unknown':
                region_hole_type[region_index] = 'Open'
    # Set the hole type for the each region
    for region, hole_type in region_hole_type.items():
        # Get the contours in the region
        region_contours = contour_lookup.RegionIndex == region
        region_labels = list(contour_lookup.loc[region_contours, 'Label'])
        # Set the hole type for each contour in the region
        for label in region_labels:
            # Get the contour from the graph
            contour = contour_graph.nodes[label]['contour']
            # Set the hole type for the contour
            contour.hole_type = hole_type
    return contour_graph


def build_contour_graph(contour_table: pd.DataFrame,
                        slice_sequence: SliceSequence,
                        roi: ROI_Type) -> Tuple[nx.Graph, pd.DataFrame]:
    '''Build a graph of contours for the specified ROI.

    This function creates a graph representation of the contours for a given
    ROI. Each contour is represented as a node in the graph, and edges are
    created based on the relationships between the contours.

    Args:
        contour_table (pd.DataFrame): The contour table containing contour data.
            The table must be sorted by descending area, or holes will not be identified properly.
        slice_sequence (SliceSequence): The slice sequence object containing
            the slice indices and their neighbors.
        roi (int): The ROI number to filter contours.

    Returns:
        tuple: A tuple containing the contour graph and a lookup table.
            contour_graph (nx.Graph): The graph representation of the contours.
            contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table
                for contours, including slice index, contour index, and hole type.
    '''
    contour_by_slice = build_contours(contour_table, roi)
    # Create an empty graph
    contour_graph = nx.Graph()
    # Add nodes to the graph
    for contour_data in contour_by_slice.values():
        for contour in contour_data:
            contour_label = contour.index
            contour_graph.add_node(contour_label, contour=contour)
    # Create the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Add the edges to the graph
    contour_graph = add_graph_edges(contour_graph, contour_lookup,
                                    slice_sequence)
    # Add the boundary contours to the graph
    contour_graph, slice_sequence = add_boundary_contours(contour_graph,
                                                          slice_sequence)
    # Re-build the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Identify the distinct regions in the graph
    # and assign the region index to each contour
    contour_graph = set_enclosed_regions(contour_graph)
    # Re-build the Graph indexer
    contour_lookup = build_contour_lookup(contour_graph)
    # Set the hole type for the regions
    contour_graph = set_hole_type(contour_graph, contour_lookup, slice_sequence)
    # Re-build the Graph indexer
    #contour_lookup = build_contour_lookup(contour_graph)
    return contour_graph, slice_sequence
