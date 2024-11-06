'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from math import pi, sqrt
from typing import List, Union

# Shared Packages
import pandas as pd
#import xlwings as xw
import shapely

# Local packages
from types_and_classes import PRECISION, SliceIndexType, StructurePairType
from types_and_classes import InvalidContour, InvalidContourRelation
from utilities import poly_round

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
        if 'precision' in kwargs:
            self.precision = kwargs['precision']
        else:
            self.precision = PRECISION
        if 'ignore_errors' in kwargs:
            ignore_errors = kwargs['ignore_errors']
        else:
            ignore_errors = False
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
        return count == 0

    def extract_regions(self)->pd.DataFrame:
        '''Extract the individual regions from the contour MultiPolygon.

        Calculate the area, center, perimeter and circular radius for each
        region. The resulting DataFrame has the following columns:
            'polygon', 'area', 'centroid', 'perimeter' and 'radius'.
        The 'polygon' column contains the shapely Polygon for each region.
        The circular radius is the radius of a circle with the same area
        as the region.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted region data.
        '''
        region_list = []
        for poly in self.contour.geoms:
            poly_dict = {'polygon': poly,
                         'area': poly.area,
                         'centroid': poly.centroid,
                         'perimeter': poly.length,
                         'radius': round(sqrt(self.contour.area / pi),
                                         self.precision)}

            region_list.append(poly_dict)
        return pd.DataFrame(region_list)


# %% Slice related functions
def empty_structure(structure:  Union[StructureSlice, float]) -> bool:
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
    if not isinstance(structure, StructureSlice):
        return True
    return structure.is_empty


def has_area(poly: Union[StructureSlice, float])->bool:
    '''Check if the structure has area.

    Tests whether the structure has an area greater than zero or is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.

    Returns:
        bool: True if the structure has an area greater than zero, False
            otherwise.
    '''
    if poly.is_empty:
        return False
    area = poly.contour.area
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
    if poly.is_empty:
        return False
    return poly.contour.contains(point)


def get_centroid(poly: Union[StructureSlice, float])->shapely.Point:
    '''Get the centroid of the structure.

    Returns the centroid of the structure or NaN if the structure is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice or NaN object.

    Returns:
        shapely.Point: The centroid of the structure.
    '''
    if poly.is_empty:
        return shapely.Point()
    return poly.contour.centroid


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
    try:
        structure_slice = StructureSlice(list(ranked_contours.Contour),
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
    used_slices = ~structure_slices.apply(empty_structure)
    start = used_slices & (used_slices ^ used_slices.shift(1))
    end = used_slices & (used_slices ^ used_slices.shift(-1))
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
