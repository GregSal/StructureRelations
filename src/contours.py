'''Contour Classes and related Functions
'''
from collections import defaultdict
from typing import List, Tuple

from networkx import Graph
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import Polygon

from types_and_classes import ROI_Type, SliceIndexType, ContourPointsType
from types_and_classes import ContourIndex
from types_and_classes import InvalidContour
from types_and_classes import SliceSequence


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
                slice_index = 0.0
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
            If is_hole is False, hole_type is None.

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
        self.hole_reference = None
        self.hole_type = None
        self.is_boundary = False
        self.is_interpolated = False
        self.region_index = None
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
            contours (List['Contour']): A list of existing contours.

        Raises:
            InvalidContour: Raised if the new contour overlaps an existing
                contour in the list.
        '''
        for existing_contour in reversed(contours):
            if existing_contour.is_hole:
                # If the existing contour is a hole, the new contour cannot be
                # its hole
                continue
            if self.polygon.within(existing_contour.polygon):
                # New contour is completely within the existing contour
                self.is_hole = True
                self.hole_reference = existing_contour.contour_index
                self.hole_type = "Unknown"
                break  # Stop checking once a containing contour is found
            if self.polygon.overlaps(existing_contour.polygon):
                # New contour overlaps an existing contour, raise an error
                raise InvalidContour("New contour overlaps an existing contour.")

    def validate_polygon(self) -> None:
        '''Validate the polygon to ensure it is valid.'''
        if not self.polygon.is_valid:
            raise InvalidContour("Invalid polygon provided for the contour.")

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
    '''

    def __init__(self, contour1: Contour, contour2: Contour) -> None:
        self.contour1 = contour1
        self.contour2 = contour2
        self.thickness = abs(contour1.slice_index - contour2.slice_index) / 2
        self.combined_area = contour1.area() + contour2.area()


# %% Contour Functions
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


def build_contour_table(slice_data: List[ContourPoints]) -> Tuple[pd.DataFrame,
                                                                  SliceSequence]:
    '''Build a contour table from a list of Contour objects.
    The table contains the following columns:
        ROI, Slice, Points, Polygon, Area
        The Polygon column contains the polygons generated from the points.
        The Area column contains the area of the polygons.
        The table is sorted by ROI, Slice and Area.
        The slice sequence is generated from the Slice column.
        The slice sequence is a list of slices in the order they are encountered.

    Args:
        slice_data (List[ContourPoints]): A list of Contour objects.

    Returns:
        tuple: A tuple containing the contour table and the slice sequence.
            contour_table (pd.DataFrame): The contour table.
            slice_sequence (SliceSequence): The slice sequence.
    '''
    contour_table = pd.DataFrame(slice_data)
    # Convert the contours points to polygons and calculate their areas
    contour_table['Polygon'] = contour_table['Points'].apply(points_to_polygon)
    contour_table['Area'] = contour_table['Polygon'].apply(lambda poly: poly.area)
    # Sort the contours by ROI, Slice and decreasing Area
    # Decreasing area is important because that an earlier contour cannot be inside
    # a later one.
    contour_table.sort_values(by=['ROI', 'Slice', 'Area'],
                        ascending=[True, True, False],
                        inplace=True)
    # Generate the slice sequence for the contours
    slice_sequence = SliceSequence(contour_table.Slice)
    return contour_table, slice_sequence


# %% Contour Graph Building Functions
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
              Contour objects for that slice.
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

    return contour_by_slice


def build_contour_lookup(label_list) -> pd.DataFrame:
    '''Build a lookup table for contours.

    This function creates a DataFrame that serves as a lookup table for contours.
    It includes information about the slice index, contour index, and hole type.
    The hole type is categorized into 'Open', 'Closed', 'Unknown', and 'None'.
    The DataFrame is sorted by slice index and contour index.

    Args:
        label_list (list): A list of dictionaries containing contour
            information. Each dictionary should have keys
                - 'SliceIndex',
                - 'ContourIndex', and
                - 'HoleType'.

    Returns:
        pd.DataFrame: A DataFrame containing the contour lookup table.
    '''
    contour_lookup = pd.DataFrame(label_list)
    contour_lookup.HoleType = contour_lookup.HoleType.fillna('None')
    contour_lookup.HoleType = contour_lookup.HoleType.astype('category')
    contour_lookup.HoleType.cat.set_categories(['Open', 'Closed',
                                                'Unknown', 'None'])
    contour_lookup.sort_values(by=['SliceIndex', 'ContourIndex'],  inplace=True)
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


def build_contour_graph(contour_table, slice_sequence: SliceSequence,
                        roi: ROI_Type) -> Tuple[nx.Graph, pd.DataFrame]:
    '''Build a graph of contours for the specified ROI.

    This function creates a graph representation of the contours for a given
    ROI. Each contour is represented as a node in the graph, and edges are
    created based on the relationships between the contours.

    Args:
        contour_table (pd.DataFrame): The contour table containing contour data.
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
    label_list = []

    # Add nodes to the graph
    for contour_data in contour_by_slice.values():
        for contour in contour_data:
            contour_label = contour.index
            contour_graph.add_node(contour_label, contour=contour)
            roi, slice_index, idx = contour_label
            label_list.append({
                'ROI': roi,
                'SliceIndex': slice_index,
                'HoleType': contour.hole_type,
                'ContourIndex': idx,
                'Label': contour_label
                })
    # Create the Graph indexer
    contour_lookup = build_contour_lookup(label_list)
    # Add the edges to the graph
    contour_graph = add_graph_edges(contour_graph, contour_lookup,
                                    slice_sequence)
    return contour_graph, contour_lookup
