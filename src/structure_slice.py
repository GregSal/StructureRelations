'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from itertools import chain
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict
# Shared Packages
import numpy as np
import pandas as pd
#import xlwings as xw
import shapely

# Local packages
from types_and_classes import PRECISION, SliceIndexType, StructurePairType
from types_and_classes import ROI_Type
from types_and_classes import InvalidContour, InvalidContourRelation
from utilities import interpolate_polygon, point_round, poly_round



# %% Type definitions and Globals
# An enclosed region representing either a structure area, or a hole within
# that structure.  The float type is present to allow for np.nan values.
ContourType = Union["Region", "StructureSlice", shapely.Polygon, float]


#%% Region Class
class Region:
    def __init__(self, roi: ROI_Type, slice_position: SliceIndexType,
                 polygon: ContourType, is_hole: bool = False,
                 is_boundary: bool = False,
                 is_interpolated: bool = False):
        self.roi = roi
        self.slice = slice_position
        self.is_hole = is_hole
        self.is_boundary = is_boundary
        self.is_interpolated = is_interpolated
        self.region_labels = []
        if isinstance(polygon, shapely.Polygon):
            self.polygon = polygon
        else:
            self.polygon = None

    def part_of(self, other: 'Region') -> bool:
        # Check if the region is part of another region
        # This is done to ensure that If the region is a hole, it is not part
        # of the parent region.
        # regions can only be part of other regions with the same roi
        if not self.roi == other.roi:
            return False
        # TODO add check for slice position. Regions can only be part of other
        # regions on the neighbouring slice.
        # Holes can only be part of other holes
        if not self.is_hole == other.is_hole:
            return False
        # The interior of both polygons must overlap.
        pattern = '2********'
        return self.polygon.relate_pattern(other.polygon, pattern)

    @property
    def is_empty(self)-> bool:
        '''Check if the slice is empty.

        Returns:
            bool: True if the slice is empty, False otherwise.
        '''
        if self.polygon is None:
            return True
        if self.polygon.is_empty:
            return True
        if self.polygon.area == 0:
            return True
        return False

    @property
    def area(self)-> float:
        '''The area encompassed by the polygon.

        Returns:
            float: The area encompassed by the polygon.
        '''
        return self.polygon.area

    def __repr__(self):
        return ''.join([f'Region(roi={self.roi}, ',
                        f'slice={self.slice}, ',
                        f'is_hole={self.is_hole}, ',
                        f'is_boundary={self.is_boundary}, ',
                        f'region_labels={self.region_labels}, ',
                        #f'polygon={self.polygon})'
            ])


# %% StructureSlice Class
class StructureSlice():
    '''Assemble a shapely.MultiPolygon.

    Iteratively create a shapely MultiPolygon from a list of shapely Polygons.
    polygons that are contained within the already formed MultiPolygon are
    treated as holes and subtracted from the MultiPolygon.  Polygons
    overlapping with the already formed MultiPolygon are rejected. Polygons that
    are disjoint with the already formed MultiPolygon are combined with a union.

    Two custom properties exterior and hull are defined. Exterior returns the
    equivalent with all holes filled in.  Hull returns a MultiPolygon that is
    the convex hull surrounding the entire MultiPolygon.

    Args:
        contours (List[shapely.Polygon]): A list of polygons to be merged
        into a single MultiPolygon.

    Attributes:
        contour (shapely.MultiPolygon): The MultiPolygon created by combining
            the supplied list of polygons.
        exterior (shapely.MultiPolygon): The contour MultiPolygon with all
            holes filled in.
        hull (shapely.MultiPolygon): The MultiPolygon that is the convex hull
            surrounding the contour MultiPolygon.
    '''

    def __init__(self, contours: List[shapely.Polygon], **kwargs) -> None:
        '''Iteratively create a shapely MultiPolygon from a list of shapely
        Polygons.

        Polygons that are contained within the already formed MultiPolygon are
        treated as holes and subtracted from the MultiPolygon.  Polygons
        overlapping with the already formed MultiPolygon are rejected. Polygons
        that are disjoint with the already formed MultiPolygon are combined.

        Args:
            contours (List[shapely.Polygon]): A list of polygons to be merged
            into a single MultiPolygon.
        '''
        if 'roi' in kwargs:
            self.roi = kwargs['roi']
        else:
            self.roi = None
        if 'precision' in kwargs:
            self.precision = kwargs['precision']
        else:
            self.precision = PRECISION
        if 'ignore_errors' in kwargs:
            ignore_errors = kwargs['ignore_errors']
        else:
            ignore_errors = False
        if 'slice_position' in kwargs:
            self.slice_position = kwargs['slice_position']
        else:
            self.slice_position = None
        self.contour = shapely.MultiPolygon()
        for contour in contours:
            self.add_contour(contour, ignore_errors=ignore_errors)

    def add_contour(self, contour: shapely.Polygon, ignore_errors=False) -> None:
        '''Add a shapely Polygon to the current MultiPolygon from a list of
        shapely Polygons.

        Polygons that are contained within the already formed MultiPolygon are
        treated as holes and subtracted from the MultiPolygon.  Polygons
        overlapping with the already formed MultiPolygon are rejected. Polygons
        that are disjoint with the already formed MultiPolygon are combined.

        Args:
            contour (shapely.Polygon): The shapely Polygon to be added.
                The shapely Polygon must either be contained in or be disjoint
                with the existing MultiPolygon.
            ignore_errors (bool, optional): If True, the function will not raise
                an error when the supplied shapely Polygon overlaps with the
                existing MultiPolygon. Defaults to False.

        Raises:
            ValueError: When the supplied shapely Polygon overlaps with the
                existing MultiPolygon.
        '''
        # check for slice position
        dim = shapely.get_coordinate_dimension(contour)
        if dim == 3:
            slice_position = shapely.get_coordinates(contour, include_z=True)[0][2]
            if self.slice_position is None:
                self.slice_position = slice_position
            elif slice_position != self.slice_position:
                raise ValueError('Slice position mismatch.')
        # Apply requisite rounding to polygon
        contour_round = poly_round(contour, self.precision)
        # Check for valid contour
        if not shapely.is_valid(contour_round):
            if ignore_errors:
                # TODO Add optional text stream / function to receive warning
                # messages when contours are skipped.
                return
            error_str = shapely.is_valid_reason(contour_round)
            raise InvalidContour(error_str)
        # Check for non-overlapping structures
        if self.contour.disjoint(contour_round):
            # Combine non-overlapping structures
            new_contours = self.contour.union(contour_round)
        # Check for hole contour
        elif self.contour.contains(contour_round):
            # Subtract hole contour
            new_contours = self.contour.difference(contour_round)
        else:
            if ignore_errors:
                return
            raise InvalidContourRelation('Cannot merge overlapping contours.')
        # Enforce the MultiPolygon type for self.contour
        if isinstance(new_contours, shapely.MultiPolygon):
            self.contour = new_contours
        else:
            self.contour = shapely.MultiPolygon([new_contours])

    def parameters(self)->Dict[str, Any]:
        '''Return a dictionary of the structure parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the structure parameters.
                The dictionary contains the following keys:
                    'precision': The contour coordinate precision.
                    'slice_position': The slice position of the structure.
        '''
        return {'precision': self.precision,
                'slice_position': self.slice_position}

    @property
    def exterior(self)-> shapely.MultiPolygon:
        '''The solid exterior contour MultiPolygon.

        Returns:
            shapely.MultiPolygon: The contour MultiPolygon with all holes
                filled in.
        '''
        solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                  for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        if isinstance(solid, shapely.MultiPolygon):
            ext_poly = shapely.MultiPolygon(solid)
        else:
            ext_poly = shapely.MultiPolygon([solid])
        return ext_poly

    @property
    def hull(self)-> shapely.MultiPolygon:
        '''A bounding contour generated from the entire contour MultiPolygon.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        If contour contains more than one distinct region the hull will be the
        combination of the convex_hulls for each distinct region.  It will not
        contain the area between the regions.  in other words, the convex hull
        will consist of multiple elastic bands stretched around each external
        contour rather that one elastic band stretched around all external
        contours.

        Returns:
            shapely.MultiPolygon: The bounding contour for the entire contour
                MultiPolygon.
        '''
        solids = [shapely.convex_hull(poly) for poly in self.contour.geoms]
        combined = shapely.unary_union(solids)
        if isinstance(combined, shapely.MultiPolygon):
            hull = combined
        else:
            hull = shapely.MultiPolygon([combined])
        return hull

    @property
    def interiors(self)-> List[shapely.Polygon]:
        '''A list of the holes in the contour as shapely.Polygon objects.
        '''
        holes = []
        for poly in self.contour.geoms:
            holes.extend(poly.interiors)
        hole_polygons = [shapely.Polygon(hole) for hole in holes]
        return hole_polygons

    def select(self, coverage: str) -> shapely.MultiPolygon:
        # select the polygon type
        if coverage == 'contour':
            polygon = self.contour
        elif coverage == 'exterior':
            polygon = self.exterior
        elif coverage == 'hull':
            polygon = self.hull
        else:
            raise ValueError('Invalid coverage type')
        return polygon


    @property
    def area(self)-> float:
        '''The area encompassed by solid exterior contour MultiPolygon.

        Returns:
            float: The area encompassed by each polygon on the slice.
        '''
        area_p = sum(poly.area for poly in self.contour.geoms)
        return area_p

    @property
    def external_area(self)-> float:
        '''The area encompassed by the convex hulls of each polygon on the
        slice.

        Returns:
            float: The area encompassed by the convex hulls of each polygon on
                the slice.
        '''
        solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                  for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        return solid.area

    @property
    def hull_area(self)-> float:
        '''The area encompassed by the convex hulls of each polygon on the
        slice.

        Returns:
            float: The area encompassed by the convex hulls of each polygon on
                the slice.
        '''
        solids = [shapely.convex_hull(poly) for poly in self.contour.geoms]
        solid = shapely.unary_union(solids)
        return solid.area

    @property
    def is_empty(self)-> bool:
        '''Check if the slice is empty.

        Returns:
            bool: True if the slice is empty, False otherwise.
        '''
        count = len(self.contour.geoms)
        if count == 0:
            return True
        if float(self.area) == 0.0:
            return True
        return False

    def extract_regions(self, extract_holes=False)->List[Region]:
        '''Extract the individual regions from the contour MultiPolygon.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted region data.
        '''
        regions_list = []
        roi = self.roi
        slice_index = self.slice_position
        for poly in self.contour.geoms:
            # Create Region instances for each polygon in the slice
            # Note: the polygon includes its holes.
            region = Region(roi, slice_index, poly,
                            is_hole=False, is_boundary=False)
            regions_list.append(region)
            if extract_holes:
                # Create Region instances for each hole in the polygon
                for interior in poly.interiors:
                    hole = shapely.Polygon(interior)
                    region_hole = Region(self.roi, self.slice_position,
                                         hole,
                                         is_hole=True, is_boundary=False)
                    regions_list.append(region_hole)
        return regions_list

    def centers(self, coverage: str = 'contour')->List[shapely.Point]:
        '''A list of the geometric centers of each polygon in the ContourSlice.

        Args:
            coverage (str): The type of coverage to use for the centroid
                calculations.  Must be one of 'external', 'hull' or 'contour'.

        Returns:
            List[shapely.Point]: A list of the geometric centers of each polygon
            in the ContourSlice.
        '''
        centre_list = []
        polygons = self.select(coverage)
        # calculate the centroid for each polygon
        for poly in polygons.geoms:
            centre_point = point_round(poly.centroid, self.precision)
            centre_list.append(centre_point)
        return centre_list





# %% Slice related functions
def empty_structure(structure: ContourType, invert=False) -> bool:
    '''Check if the structure is empty.

    Tests whether structure is NaN or an empty StructureSlice.
    If the structure is a StructureSlice, it is considered empty if it has
    no contours.

    Args:
        structure (Union[StructureSlice, float]): A StructureSlice or NaN object.

    Returns:
        bool: False if the structure is type StructureSlice and is not empty.
            Otherwise True.
    '''
    if not isinstance(structure, (StructureSlice, Region, shapely.Polygon)):
        is_empty = True
    elif structure.is_empty:
        is_empty =  True
    elif structure.area == 0:
        is_empty = True
    else:
        is_empty = False
    if invert:
        return not is_empty
    return is_empty


def has_area(poly: ContourType)->bool:
    '''Check if the structure has area.

    Tests whether the structure has an area greater than zero or is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.

    Returns:
        bool: True if the structure has an area greater than zero, False
            otherwise.
    '''
    if empty_structure(poly):
        return False
    area = poly.area
    return area > 0


def contains_point(poly: Union[StructureSlice, float],
                   point: shapely.Point)->bool:
    '''Check if the structure contains the given point.

    Tests whether the structure contains the given point or is empty.
    This is a convenience function that wraps the shapely Polygon.contains
    method, allowing it to be applied to a series of StructureSlice objects.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.
        point (shapely.Point): A shapely Point object.

    Returns:
        bool: True if the structure contains the point, False otherwise.
    '''
    if empty_structure(poly):
        return False
    if isinstance(poly, StructureSlice):
        return poly.contour.contains(point)
    if isinstance(poly, Region):
        return poly.polygon.contains(point)
    if isinstance(poly, shapely.Polygon):
        return poly.contains(point)
    return False


def get_centroid(poly: Union[StructureSlice, float])->shapely.Point:
    '''Get the centroid of the structure.

    Returns the centroid of the structure or NaN if the structure is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.

    Returns:
        shapely.Point: The centroid of the structure.
    '''
    if empty_structure(poly):
        return shapely.Point()
    if isinstance(poly, StructureSlice):
        return poly.contour.centroid
    if isinstance(poly, Region):
        return poly.polygon.centroid
    if isinstance(poly, shapely.Polygon):
        return poly.centroid
    return shapely.Point()


def merge_contours(slice_contours: pd.Series,
                   ignore_errors=False) -> StructureSlice:
    '''Merge contours for a single slice into a single StructureSlice.

    the supplied slice_contours are sorted by area in descending order and
    merged into a single StructureSlice instance. StructureSlice treats every
    contour as a hole or a solid region.  The largest contour is treated as a
    solid region and for each of the remaining contours, if the contour is
    contained within the solid region, it is treated as a hole.  If the contour
    is disjoint with the solid region, it is combined with the solid region.

    Args:
        slice_data (pd.Series): A series of individual structure contours.
        ignore_errors (bool, optional): If True, overlapping contours are
            allowed and combined to generate a larger solid region.
            Defaults to False.
    '''
    ranked_contours = slice_contours.sort_values('Area', ascending=False)
    slice_position = slice_contours.index.get_level_values('Slice Index')[0]
    roi_num = ranked_contours.index[0][0]
    try:
        structure_slice = StructureSlice(list(ranked_contours.Contour),
                                         slice_position=slice_position,
                                         roi=roi_num,
                                         ignore_errors=ignore_errors)
    except InvalidContour as err:
        msg = str(err)
        roi_num = ranked_contours.index[0][0]
        slice_idx = ranked_contours.index[0][1]
        print(f'{msg}\t for ROI: {roi_num} on slice: {slice_idx}')
        structure_slice = StructureSlice([])
    return structure_slice


def make_slice_table(slice_data: pd.Series, ignore_errors=False)->pd.DataFrame:
    '''Merge contour data to build a table of StructureSlice data

    The table index is SliceIndex, sorted from smallest to largest. The table
    columns are ROI_Num.
    Individual structure contours with the same ROI_Num and SliceIndex are
    merged into a single StructureSlice instance

    Args:
        slice_data (pd.Series): A series of individual structure contours.
        ignore_errors (bool, optional): If True, overlapping contours are
            allowed and combined to generate a larger solid region.
            Defaults to False.

    Returns:
        pd.DataFrame: A table of StructureSlice data with an SliceIndex and
            ROI_Num as the index and columns respectively.
    '''
    sorted_data = slice_data.sort_index(level=['Slice Index', 'ROI Num']).copy()
    sorted_data['Area'] = sorted_data.map(lambda x: x.area)
    structure_group = sorted_data.groupby(level=['Slice Index', 'ROI Num'])
    structure_data = structure_group.apply(merge_contours,
                                           ignore_errors=ignore_errors)
    slice_table = structure_data.unstack('ROI Num')
    return slice_table


#%% Select Regions
RegionDict = Dict[ROI_Type, Dict[SliceIndexType, List[Region]]]


def slice_regions(structure_slice):
    # Create Region instances for each polygon and hole in the slice
    local_regions = []
    roi = structure_slice.roi
    slice_index = structure_slice.slice_position
    for polygon in structure_slice.contour.geoms:
        # Create Region instances for each polygon in the slice
        # Note: the polygon includes its holes.
        region = Region(roi, slice_index, polygon, is_hole=False,
                        is_boundary=False)
        local_regions.append(region)
        # Create Region instances for each hole in the polygon
        for interior in polygon.interiors:
            hole = shapely.Polygon(interior)
            region_hole = Region(roi, slice_index, hole, is_hole=True,
                                 is_boundary=False)
            local_regions.append(region_hole)
    return local_regions


def add_new_label(idx: int, region: Region) -> int:
    region.region_labels.append(chr(97 + idx))  # 'a', 'b', 'c', ...
    idx += 1
    return idx


def match_regions(current_region: Region, previous_regions: List[Region])->List[str]:
    region_labels = current_region.region_labels
    for prev_region in previous_regions:
        if current_region.part_of(prev_region):
            region_labels.extend(prev_region.region_labels)
    return region_labels


def boundary_region(previous_slice: SliceIndexType, current_region: Region):
    if previous_slice is None:
        # If there is no previous slice, (i.e. this is the first slice in
        # slice_table) then don't create an interpolated region.
        return None
    roi = current_region.roi
    current_slice = current_region.slice
    slice_pair = (previous_slice, current_slice)
    # Interpolated boundary slices are placed between the first slice with the
    # region and the last slice before the region.
    intp_poly = interpolate_polygon(slice_pair, current_region.polygon)
    intp_slice = intp_poly.centroid.z
    bdry_region = Region(roi, intp_slice, intp_poly,
                            is_hole=current_region.is_hole,
                            is_boundary=True,
                            is_interpolated=True)
    bdry_region.region_labels = current_region.region_labels.copy()
    return bdry_region


def expand_regions(region_collection: RegionDict) -> pd.DataFrame:
    expanded_data = []
    for roi, slices in region_collection.items():
        for slice_index, regions in slices.items():
            for region in regions:
                for label in region.region_labels:
                    expanded_data.append({
                        'ROI': roi,
                        'Slice': slice_index,
                        'Label': label,
                        'Region': region
                    })
    return pd.DataFrame(expanded_data)


# Function to create Region instances from slice-table DataFrame
def make_region_table(slice_table: pd.DataFrame) -> RegionDict:
    region_collection = {}
    idx = 0
    for roi in slice_table.columns:
        region_collection[roi] = defaultdict(list)
        previous_regions = []
        previous_slice = None
        # Iterate over slices in the slice_table for a given ROI
        for slice_index, structure_slice in slice_table[roi].items():
            if empty_structure(structure_slice):
                # Ignore empty slices
                continue
            # Identify whether this is the first slice in for a given ROI:
            first_slice = (slice_index == slice_table[roi].first_valid_index())
            # Create a list of Region instances for each slice
            current_regions = slice_regions(structure_slice)
            for region in current_regions:
                if first_slice:
                    # If this is the first slice in for a given ROI, then
                    # set unique labels for each region on the first slice and
                    # create an interpolated boundary region.
                    idx = add_new_label(idx, region)
                    bdry_region = boundary_region(previous_slice, region)
                    if bdry_region:
                        # If the boundary region is created, add it to the
                        # region_collection in the interpolated slice.
                        intp_slice = bdry_region.slice
                        region_collection[roi][intp_slice].append(bdry_region)
                    else:
                        # If the boundary region is note created, set the
                        # current region as a boundary region.
                        region.is_boundary = True
                else:
                    # Find overlapping polygons and give them the same region
                    # labels.
                    region_labels = match_regions(region, previous_regions)
                    if region_labels:
                        region.region_labels = region_labels
                    else:
                        # If the region does not match any previous region, then
                        # give it a unique label and create an interpolated
                        # region half way to the previous slice.
                        idx = add_new_label(idx, region)
                        bdry_region = boundary_region(previous_slice, region)
                        intp_slice = bdry_region.slice
                        region_collection[roi][intp_slice].append(bdry_region)
            # Add the current regions to the region_collection.
            region_collection[roi][slice_index] = current_regions
            # Identify any previous regions that did not match with a current
            # region and create interpolated boundaries.
            all_labels = [region.region_labels for region in current_regions]
            matched_labels = set(label for label in chain(all_labels))
            for prev_region in previous_regions:
                if set(prev_region.region_labels).isdisjoint(matched_labels):
                    bdry_region = boundary_region(slice_index, prev_region)
                    if bdry_region:
                        intp_slice = bdry_region.slice
                        region_collection[roi][intp_slice].append(bdry_region)
                    else:
                        prev_region.is_boundary = True
            # Update the previous_slice and previous_regions for the next slice.
            previous_slice = slice_index
            previous_regions = current_regions
        # Mark regions in the last slice as boundaries.
        if previous_slice is not None:
            for region in previous_regions:
                region.is_boundary = True
    # Expand the regions_dict into a DataFrame with one column per region
    region_table = expand_regions(region_collection)
    region_table.set_index(['ROI', 'Label', 'Slice'], inplace=True)
    region_table = region_table.unstack(['ROI', 'Label'])
    region_table.columns = region_table.columns.droplevel(0)
    return region_table





#%% Select Slices and neighbours
def select_slices(slice_table: pd.DataFrame,
                  selected_roi: StructurePairType) -> pd.DataFrame:
    '''Select the slices that have either of the structures.

    Select all slices that have either of the structures.

    Args:
        slice_table (pd.DataFrame): A table of StructureSlice data with
            SliceIndex as the index, ROI_Num for columns and StructureSlice or
            NaN as the values.
        selected_roi (StructurePairType): A tuple of two ROI_Num to select.

    Returns:
        pd.DataFrame:  A subset of slice_table with the two selected_roi as the
            columns and the range of slices hat have either of the structures as
            the index.
    '''
    start = SliceIndexType(slice_table[selected_roi].first_valid_index())
    end = SliceIndexType(slice_table[selected_roi].last_valid_index())
    structure_slices = slice_table.loc[start:end, selected_roi]
    return structure_slices


def structure_neighbours(slice_structures: pd.DataFrame,
                         shift_direction=-1) -> pd.DataFrame:
    '''Generate a table with the second of the two structure columns shifted by
    shift_direction.

    Take a DataFrame with two columns of StructureSlice.  Shift the second
    column by the amount specified with shift_direction. Calculate the gap
    between the the their Slice_index's (The DataFrame index).  Return a
    DataFrame with columns labeled ['a', 'b', 'height'], where 'a' is the
    original first column, 'b' is the shifted second column and 'height' is the
    calculated SliceIndex gap.

    Args:
        slice_structures (pd.DataFrame): a DataFrame containing two columns of
            StructureSlice with SliceIndex values as the index.
        shift_direction (int, optional): The number of rows to shift the second
            structure's slices. Defaults to -1.

    Returns:
        pd.DataFrame: a table with columns labeled ['a', 'b', 'height'], where
            'a' is the original first column, 'b' is the shifted second column
            and 'height' is the calculated SliceIndex gap.
    '''
    used_slices = slice_structures.copy()
    used_slices.dropna(how='all', inplace=True)
    used_slices.columns = ['a', 'b']
    slices_index = used_slices.index.to_series()
    slices_gaps = slices_index.shift(shift_direction) - slices_index
    neighbour = used_slices['b'].shift(shift_direction)
    slice_shift = pd.concat([used_slices['a'], neighbour, slices_gaps],
                            axis='columns')
    slice_shift.dropna(inplace=True)
    slice_shift.columns = ['a', 'b', 'height']
    return slice_shift


def find_neighbouring_slice(structure_slices) -> pd.DataFrame:
    '''Find the neighbouring slices for each missing slice.

    For each slice that is missing a structure, find the neighbouring slices
    that contain the structure.  The neighbouring slices are found by shifting
    the slice index in the positive and negative direction.

    Args:
        structure_slices (pd.Series): A series with SliceIndexType as the index
            and StructureSlice or NaN as the values.

    Returns:
        pd.DataFrame: A table with columns labeled ['z_pos', 'z_neg'], where
            'z_pos' is the SliceIndex of the neighbouring slice in the positive
            direction and 'z_neg' is the SliceIndex of the neighbouring slice
            in the negative direction.
    '''
    def neighbouring_slice(slice_index: pd.Series, missing_slices: pd.Series,
                           shift_direction=1, shift_start=0)->pd.Series:
        '''Find the neighbouring slices for each missing slice.

        For each slice that is missing a structure, find the neighbouring slices
        that contain the structure.  The neighbouring slices are found by
        shifting the slice index until a non-empty structure is found.

        Args:
            slice_index (pd.Series): A series with SliceIndexType as the index
                and the values.
            missing_slices (pd.Series): A subset of slice_index containing index
                values that ar missing a structure.
            shift_direction (int, optional): The direction to shift the slice
                index. Defaults to 1.
            shift_start (int, optional): The starting shift size. Defaults to 0.
        '''
        # TODO this function (neighbouring_slice) seems very awkward.  It should
        # be possible to simplify this function.
        ref = slice_index[missing_slices]
        ref_missing = list(missing_slices)
        shift_size = shift_start
        while ref_missing:
            shift_size += shift_direction
            # Shift the slice index by the shift size and select the indexes
            # that need a neighbour.
            shift_slice = slice_index.shift(shift_size)[missing_slices]
            ref_idx = ref.isin(ref_missing)
            ref[ref_idx] = shift_slice[ref_idx]
            ref_missing = list(set(ref) & set(missing_slices))
            ref_missing.sort()
        return ref

    slice_index = structure_slices.index.to_series()
    # Find the slices that are missing a structure
    missing_slices = slice_index[structure_slices.apply(empty_structure)]
    z_neg = neighbouring_slice(slice_index, missing_slices, shift_direction=-1)
    z_pos = neighbouring_slice(slice_index, missing_slices, shift_direction=1)
    ref = pd.concat([z_pos, z_neg], axis='columns')
    ref.columns = ['z_pos', 'z_neg']
    return ref


def find_boundary_slices(structure_slices: pd.Series) -> List[SliceIndexType]:
    '''Identify the first and last slices of a structure region.

    Slices without the structure are identified by `isna()`.
    Any slice that contains the structure, but has a neighbouring slices that
    does not contain the structure is considered a boundary slice.

    Args:
        structure_slices (pd.Series): A series with SliceIndex as the index and
            StructureSlice or na as the values.

    Returns:
        List[SliceIndexType]: A list of all slice indexes where the structure is
            not present on a neighbouring slice.
    '''
    # create a mask for the slices that contain the structure
    used_slices = ~structure_slices.apply(empty_structure)
    # Identify the slices that contain the structure but have a neighbouring
    # slice that does not contain the structure.
    start = used_slices & (used_slices ^ used_slices.shift(1))
    end = used_slices & (used_slices ^ used_slices.shift(-1))
    # Combine the start and end slices to create a list of boundary slices.
    start_slices = list(structure_slices[start].index)
    end_slices = list(structure_slices[end].index)
    boundaries = start_slices + end_slices
    return boundaries


def identify_boundary_slices(structure_slices: Union[pd.Series, pd.DataFrame],
                             selected_roi: StructurePairType = None) -> pd.Series:
    '''Identify boundary slices for the given structure or structures.

    Identifies the first and last slice that has a structure contour for the
    given structure or structures. If a DataFrame is provided, the boundary
    slices are identified for the structures specified in selected_roi.
    If a Series is provided, the boundary slices are identified for the single
    structure and the selected_roi is ignored.

    Args:
        structure_slices (Union[pd.Series, pd.DataFrame]): A Series or DataFrame
            containing StructureSlice data.
        selected_roi (StructurePairType, optional): A list of two ROI_Num to
            select when structure_slices is a DataFrame. Defaults to None.

    Returns:
        pd.Series: A Series indicating whether each slice is a boundary slice.
    '''
    if isinstance(structure_slices, pd.Series):
        boundary_slice_index = set(find_boundary_slices(structure_slices))
    elif isinstance(structure_slices, pd.DataFrame):
        try:
            roi_a, roi_b = selected_roi
        except ValueError as err:
            raise ValueError('selected_roi must be a tuple of two integers when'
                             ' structure_slices is a DataFrame.') from err
        # Identify the slices that are boundary slices for either of the
        # structures.
        boundary_slice_index = set(find_boundary_slices(
            structure_slices[roi_a]))
        boundary_slice_index.update(find_boundary_slices(
            structure_slices[roi_b]))
    else:
        raise ValueError('structure_slices must be either a Series or a '
                         'DataFrame.')
    is_boundary_slice = structure_slices.index.isin(boundary_slice_index)
    return pd.Series(is_boundary_slice, index=structure_slices.index)


def get_region_centres(structure_slice)->pd.Series:
    '''Get the centroid of each region in the structure slice.

    Args:
        structure_slice (StructureSlice): A StructureSlice object.

    Returns:
        pd.Series: A Series containing the centroid of each region in the
            structure slice.
    '''
    if structure_slice.is_empty:
        return pd.Series()
    region_data = structure_slice.extract_regions()
    region_centres = region_data['centroid']
    return region_centres
