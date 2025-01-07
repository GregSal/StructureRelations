'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import Any, Dict, List, Union

# Standard Libraries

# Shared Packages
import pandas as pd
import shapely

# Local packages
from types_and_classes import PRECISION
from types_and_classes import ROI_Type, SliceIndexType, SliceNeighbours
from types_and_classes import RegionNode, RegionNodeType, RegionGraph
from types_and_classes import InvalidContour, InvalidContourRelation
from utilities import point_round, poly_round


# %% Type definitions and Globals
# An enclosed region representing either a structure area, or a hole within
# that structure. The type also include a dictionary that contains the string
# 'polygon' as a key and a shapely.Polygon as the matching value.
# The float type is present to allow for np.nan values.
ContourType = Union["StructureSlice", RegionNodeType, shapely.Polygon, float]


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
            self.roi: ROI_Type = kwargs['roi']
        else:
            self.roi: ROI_Type = None
        if 'precision' in kwargs:
            self.precision = kwargs['precision']
        else:
            self.precision = PRECISION
        if 'ignore_errors' in kwargs:
            ignore_errors = kwargs['ignore_errors']
        else:
            ignore_errors = False
        if 'slice_position' in kwargs:
            self.slice_position: SliceIndexType = kwargs['slice_position']
        else:
            self.slice_position: SliceIndexType = None
        self.slice_neighbours: SliceNeighbours = None
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
        if (coverage == 'contour'):
            polygon = self.contour
        elif (coverage == 'exterior'):
            polygon = self.exterior
        elif (coverage == 'hull'):
            polygon = self.hull
        else:
            raise ValueError('Invalid coverage type')
        return polygon

    def extract_regions(self, graph: RegionGraph, extract_holes=True) -> None:
        '''Extract the individual regions from the contour MultiPolygon and add
        them as nodes to the graph.

        Args:
            graph (nx.Graph): The graph to add the regions to.
            extract_holes (bool, optional): Whether to extract holes as
                separate regions. Defaults to True.
        '''
        node_data = RegionNode(roi=self.roi, slice_index=self.slice_position,
                               slice_neighbours=self.slice_neighbours)
        for poly in self.contour.geoms:
            # Ensure that node settings specific to individual regions on a
            # slice are not propagated to other regions on the slice.
            node_data.reset()
            if poly.area == 0:
                node_data.is_empty = True
            else:
                node_data.polygon = poly
            # Create nodes for each polygon in the slice
            node_data.add_node(graph)
            if extract_holes:
                # Create nodes for each hole in the polygon
                for interior in poly.interiors:
                    # Ensure that node settings specific to individual holes
                    # on a slice are not propagated to other holes on the slice.
                    node_data.reset()
                    node_data.is_hole = True
                    hole = shapely.Polygon(interior)
                    if hole.area == 0:
                        node_data.is_empty = True
                    else:
                        node_data.polygon = hole
                    node_data.add_node(graph)

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

    def set_slice_neighbours(self, previous_slice: SliceIndexType,
                             next_slice: SliceIndexType) -> None:
        '''Set the SliceNeighbours attribute for the StructureSlice.

        Args:
            previous_slice (SliceIndexType): The previous slice index.
            next_slice (SliceIndexType): The next slice index.
        '''
        self.slice_neighbours = SliceNeighbours(this_slice=self.slice_position,
                                                previous_slice=previous_slice,
                                                next_slice=next_slice)

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

    Tests whether structure has an 'is_empty' attribute, an 'is_empty' key.
    If so, use the value obtained from this attribute or dictionary value.
    Otherwise, if it has a zero area or does not have an area attribute
    `is_empty` is False.

    Args:
        structure (Union[StructureSlice, float]): A StructureSlice or NaN object.
        invert (bool, optional): If True, return the opposite of the result.

    Returns:
        bool: False if the structure is type StructureSlice and is not empty.
            Otherwise True.  If invert True, the result is opposite.
    '''
    # Check for None
    if structure is None:
        is_empty = True
    else:
        # check for an is_empty attribute.
        # This is used for StructureSlice objects.
        try:
            is_empty = structure.is_empty
        except AttributeError:
            # if there is no is_empty attribute, check for an 'is_empty' key.
            # This is used for RegionNode objects.
            try:
                is_empty = structure['is_empty']
            except (TypeError, KeyError):
                # if there is no 'is_empty' key, check for an area attribute.
                # This is used for shapely.Polygon objects.
                try:
                    is_empty = structure.area == 0
                except AttributeError:
                    # if there is no area attribute, the structure is considered empty.
                    is_empty = True
    if invert:
        return not is_empty
    return is_empty


def has_area(poly: ContourType)->bool:
    '''Check if the structure has area.

    Tests whether the structure has an area greater than zero or is empty.

    Args:
        poly (ContourType): A StructureSlice, RegionNode, Polygon, or NaN object.

    Returns:
        bool: True if the structure has an area greater than zero, False
            otherwise.
    '''
    if empty_structure(poly):
        area = False
    else:
        try:
             # check for a StructureSlice or shapely.Polygon object.
            area = poly.area
        except AttributeError:
            # check for a RegionNodeType object.
            try:
                area = poly['polygon'].area
            except (TypeError, KeyError):
                # Anything else is considered to have no area.
                area = False
    if area:
        return area > 0
    return area


def contains_point(poly: ContourType, point: shapely.Point)->bool:
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
    contains = False
    # check for an empty poly.
    if empty_structure(poly):
        contains = False
    else:
        try:
            # check for a StructureSlice object.
            contains = poly.contour.contains(point)
        except AttributeError:
            try:
                # check for a shapely.Polygon object.
                contains = poly.contains(point)
            except AttributeError:
                try:
                    # check for a RegionNodeType object.
                    contains = poly['polygon'].contains(point)
                except (TypeError, KeyError):
                    contains = False
    return contains


def get_centroid(poly: Union[StructureSlice, float])->shapely.Point:
    '''Get the centroid of the structure.

    Returns the centroid of the structure or an empty Point object the
    structure is empty.

    Args:
        poly (Union[StructureSlice, float]): A StructureSlice, shapely.Polygon
            or object containing a 'polygon' attribute.

    Returns:
        shapely.Point: The centroid of the structure.
    '''
    if empty_structure(poly):
        centroid = shapely.Point()
    else:
        try:
            # check for a StructureSlice object.
            centroid = poly.contour.centroid
        except AttributeError:
            try:
                # check for a shapely.Polygon object.
                centroid = poly.centroid
            except AttributeError:
                try:
                    # check for a RegionNodeType object.
                    centroid = poly['polygon'].centroid
                except (TypeError, KeyError):
                    # Anything else is considered to have no centroid.
                    centroid = shapely.Point()
    return centroid


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
