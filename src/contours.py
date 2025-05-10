'''Contour Classes and related Functions
'''
from typing import List, Tuple, Union, Dict, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

from types_and_classes import ROI_Type, SliceIndexType, ContourPointsType
from types_and_classes import ContourIndex, RegionIndex
from types_and_classes import InvalidContour
from types_and_classes import PRECISION, SliceIndexSequenceType


# %% Classes for Slice Indexing and Neighbours
@dataclass
class SliceNeighbours:
    '''A class to represent the neighbours of a slice.

    Attributes:
        this_slice (SliceIndexType): The current slice index.
        previous_slice (SliceIndexType): The previous slice index.
        next_slice (SliceIndexType): The next slice index.

        force_types: Method to ensure the slice indices are of the correct type.
        is_neighbour: Method to check if another slice index is a neighbour.
        gaps: Method to calculate the gaps between slices.
    '''

    this_slice: SliceIndexType
    previous_slice: SliceIndexType
    next_slice: SliceIndexType

    def __post_init__(self) -> None:
        self.force_types()

    def force_types(self):
        '''Ensure the slice indices are of the correct type.'''
        # Convert to float and then to SliceIndexType
        # to ensure they are of the correct type.
        if self.this_slice is None:
            self.this_slice = np.nan
        if self.previous_slice is None:
            self.previous_slice = np.nan
        if self.next_slice is None:
            self.next_slice = np.nan
        self.this_slice = SliceIndexType(float(self.this_slice))
        self.previous_slice = SliceIndexType(float(self.previous_slice))
        self.next_slice = SliceIndexType(float(self.next_slice))

    def gap(self, absolute=True) -> Union[int, float]:
        '''Calculate the gaps between slices.

        If one of the two neighbours is None, then calculate the gap based on
        the distance between the current slice and the other slice. if absolute
        is True, return the absolute value of the gap. If both neighbours are
        None, return nan.

        Returns:
            Union[int, float]: The gap between slices.
        '''
        # If both neighbours are None, return nan
        if pd.isna(self.previous_slice) and pd.isna(self.next_slice):
            return np.nan
        # If one of the neighbours is None, calculate the gap based on the
        # distance between the current slice and the other slice.
        if pd.isna(self.previous_slice):
            gap = self.next_slice - self.this_slice
        elif pd.isna(self.next_slice):
            gap = self.this_slice - self.previous_slice
        else:
            # Calculate the gap between the previous and next slice
            # and divide by 2 to get the average gap.
            gap = (self.next_slice - self.previous_slice) / 2
        if absolute:
            return abs(gap)
        return gap

    def is_neighbour(self, other_slice: SliceIndexType) -> bool:
        '''Check if another slice index is a neighbour.'''
        return self.previous_slice <= other_slice <= self.next_slice

    def neighbour_list(self) -> List[SliceIndexType]:
        '''Return a list of neighbours, excluding the current slice.'''
        return [self.previous_slice, self.next_slice]


class SliceSequence:
    '''An ordered list of all slice indexes in use.

    Attributes:
        sequence (pd.DataFrame): A DataFrame containing slice information.
        slices (List[SliceIndexType]): A list of all slice indices in the
            sequence.
    Methods:
        add_slice: Method to add a slice index to the sequence.
        remove_slice: Method to remove a slice index from the sequence.
        get_nearest_slice: Method to find the nearest slice index to a given
            value.
        get_neighbors: Method to get the neighbors of a given slice index.
    '''
    sequence: pd.DataFrame

    def __init__(self, slice_indices: Union[List[SliceIndexType], pd.Series]) -> None:
        '''Initialize the SliceSequence.

        Args:
            slice_indices (Union[List[SliceIndexType], pd.Series]): A list or
                Series of slice indices.
        '''
        # Convert to Series if not already
        slice_series = pd.Series(slice_indices).drop_duplicates().sort_values()

        # Create the DataFrame
        self.sequence = pd.DataFrame({
            'ThisSlice': slice_series,
            'NextSlice': slice_series.shift(-1),
            'PreviousSlice': slice_series.shift(1),
            'Original': True
        }).set_index('ThisSlice', drop=False)

    @property
    def slices(self) -> List[SliceIndexType]:
        '''Return a list of all slice indices in the sequence.'''
        return self.sequence['ThisSlice'].tolist()

    def add_slice(self, slice_index: SliceIndexType = None,
                  **slice_ref) -> None:
        '''Add a slice index to the sequence.

        Args:
            slice_index (SliceIndexType, optional): The slice index to add.
                If None, the slice index will be taken from 'ThisSlice' in the
                slice_ref dictionary. Defaults to None.
            slice_ref (dict): A dictionary containing the slice reference
                information. It should contain some or all of the following
                keys:
                    - 'ThisSlice'
                    - 'PreviousSlice'
                    - 'NextSlice'
                    - 'Original'
                If slice_index is None, the slice index will be taken from
                'ThisSlice' in the slice_ref dictionary. If 'PreviousSlice' or
                'NextSlice' are not provided, they will be set to None. If
                dictionary. If 'Original' is not provided, it will be set to
                False.

        Raises:
            - ValueError: If slice_index is None and 'ThisSlice' is not in
                slice_ref.
        '''
        if slice_index is None:
            slice_index = slice_ref.get('ThisSlice')
            if slice_index is None:
                raise ValueError('slice_index must be provided or "ThisSlice" '
                                  'must be in slice_ref.')
        slice_ref['ThisSlice'] = slice_index
        if 'Original' not in slice_ref:
            slice_ref['Original'] = False
        # Check if the slice index is already in the sequence
        if slice_index not in self.sequence.index:
            new_row = pd.DataFrame([{**slice_ref}])
            new_row.set_index('ThisSlice', drop=False, inplace=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.sequence = pd.concat([self.sequence, new_row]).sort_index()

    def remove_slice(self, slice_index: SliceIndexType) -> None:
        '''Remove a slice index from the sequence.'''
        if slice_index in self.sequence.index:
            self.sequence = self.sequence.drop(slice_index)
            self.sequence['NextSlice'] = self.sequence['ThisSlice'].shift(-1)
            self.sequence['PreviousSlice'] = self.sequence['ThisSlice'].shift(1)

    def get_nearest_slice(self, value: float) -> SliceIndexType:
        '''Find the nearest slice index to a given value.'''
        distance = abs(self.sequence.index - value)
        return self.sequence.index[distance.argmin()]

    def get_neighbors(self, slice_index: SliceIndexType) -> Union[SliceNeighbours, None]:
        '''Return a SliceNeighbours object for a given SliceIndex.

        Args:
            slice_index (SliceIndexType): The slice index to find neighbors for.

        Returns:
            SliceNeighbours: An object containing this slice, next slice, and previous slice.
            None: If the given SliceIndex is not in the sequence.
        '''
        if slice_index not in self.sequence.index:
            return None
        row = self.sequence.loc[slice_index]
        return SliceNeighbours(
            this_slice=slice_index,
            next_slice=row['NextSlice'],
            previous_slice=row['PreviousSlice']
        )

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index: SliceIndexType) -> Dict[str, Any]:
        return self.sequence.loc[index, :].to_dict()

    def __iter__(self):
        # TODO evaluate this function.  Is it returning the appropriate value?
        return iter(self.sequence['ThisSlice'])

    def __contains__(self, slice_index: SliceIndexType) -> bool:
        return slice_index in self.sequence.index


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
        min_slice = min(slices)
        max_slice = max(slices)
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
        region_index (str): The region index of the contour.
            Defaults to None until regions are assigned.
        index (ContourIndex): A tuple of (ROI, SliceIndex, ContourIndex).
            This is a read-only property.
      shape:
        polygon (shapely.Polygon): The polygon generated from the contour.
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
        self.related_contours: List[ContourIndex] = []
        self.hole_type = 'None'
        self.is_boundary = False
        self.is_interpolated = False
        self.region_index: RegionIndex = ''  # Default to an empty string
        self.contour_index = Contour.counter
        Contour.counter += 1
        self.validate_polygon()
        self.compare_with_existing_contours(contours)

    @property
    def index(self) -> ContourIndex:
        '''Return a tuple representing (ROI, SliceIndex, ContourIndex).'''
        return (self.roi, self.slice_index, self.contour_index)

    @property
    def area(self) -> float:
        '''Calculate the area of the contour polygon.'''
        return self.polygon.area

    @property
    def centroid(self) -> Tuple[float, float]:
        '''Calculate the centroid of the contour polygon.'''
        return self.polygon.centroid.coords

    @property
    def hull(self) -> Polygon:
        '''Calculate the convex hull of the contour polygon.'''
        return self.polygon.convex_hull

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


class ContourMatch:
    '''Class representing a match between two contours.

    Attributes:
        contour1 (Contour): The first contour.
        contour2 (Contour): The second contour.
        gap (float): Half the difference between the two slice indices.
        combined_area (float): The sum of the areas of the two contours.
        direction (int): 1 if the difference between the slice_index of the
            first and second contours is positive, -1 otherwise.
    '''

    def __init__(self, contour1: Contour, contour2: Contour) -> None:
        self.contour1 = contour1
        self.contour2 = contour2
        self.gap = abs(contour1.slice_index - contour2.slice_index)

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
