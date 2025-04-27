'''Structures from DICOM files

Types, Classes and utility function definitions.

'''
# %% Imports
# Type imports
from typing import List, Union

# Standard Libraries

# Shared Packages
import pandas as pd
import networkx as nx
import shapely

from contours import Contour
from types_and_classes import SliceIndexType

# Local packages



# %% StructureSlice Class
class RegionSlice():
    '''Class representing a slice of an enclosed region.

    *** This text need fixing***
    Iteratively create a shapely MultiPolygon from a list of shapely Polygons.
    polygons that are contained within the already formed MultiPolygon are
    treated as holes and subtracted from the MultiPolygon.  Polygons
    overlapping with the already formed MultiPolygon are rejected. Polygons that
    are disjoint with the already formed MultiPolygon are combined with a union.

    Two custom properties exterior and hull are defined. Exterior returns the
    equivalent with all holes filled in.  Hull returns a MultiPolygon that is
    the convex hull surrounding the entire MultiPolygon.
    		○ For each slice:
			§ For each region on the slice
				□ Select all contours with the specified region on the slice
				□ Select all contours from related regions that are on the same slice
				□ Sort the selected contours in order of descending size.
			§ Subtract any holes on the slice
			§ Holes and contours must be subtracted in size order or islands will be lost.
			§ Combine all exterior holes on the slice and assign to the holes attribute
			§ If contour is a boundary assign it to the boundary attribute. ,


    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table
            for contours.
        slice_index (SliceIndexType): The slice index of the region.

    Attributes:
        slice_index (SliceIndexType): The slice index of the regions.
        contour_indexes (List[List[str]]): The indexes of the contours combined
            within each region, in the same order as the list of regions.
        region_indexes (List[List[str]]): The indexes of the merged regions in
            the same order as the list of regions.
        regions (shapely.MultiPolygon): The MultiPolygons created by combining
            the contours on a given region and slice, along with the contours
            for all related regions (holes and islands).
        boundaries (shapely.MultiPolygon): The MultiPolygons that are the
            combination of all related boundary polygons on the slice.
        open_holes (List[shapely.Polygon]): A list of the holes in the contour
            that connect to the exterior of the contour.
        exterior (shapely.MultiPolygon): The contour MultiPolygon with all
            holes filled in.
        hull (shapely.MultiPolygon): The MultiPolygon that is the convex hull
            surrounding the contour MultiPolygon.
    '''

    def __init__(self, contour_graph: nx.Graph, contour_lookup: pd.DataFrame,
                 slice_index: SliceIndexType) -> None:
        '''Iteratively create a list of regions for the specified slice, from a
        contour graph.

        Args:
            contours (List[shapely.Polygon]): A list of polygons to be merged
            into a single MultiPolygon.
        '''
        def get_region_contours(contour_reference: pd.DataFrame,
                                related_regions: List[str])->List[Contour]:
            '''Return the contours for the specified regions.'''
            region_selection = contour_reference['RegionIndex'].isin(related_regions)
            region_reference = contour_reference.loc[region_selection]
            contour_labels = region_reference['Label'].tolist()
            region_contours = contour_graph.nodes.data('contour')[contour_labels]
            return region_contours

        self.slice_index = slice_index
        # Filter contours by SliceIndex and contour type
        # Only contours that are not interpolated, holes or boundaries are
        # used to select the regions.
        selected_contours = contour_lookup['SliceIndex'] == slice_index
        reference_columns = ['Label', 'RegionIndex',
                             'HoleType', 'Interpolated', 'Boundary']
        contour_reference = contour_lookup.loc[selected_contours,
                                               reference_columns]
        regular_contours = (~contour_reference['Boundary'] &
                            ~contour_reference['Interpolated'] &
                            (contour_reference['HoleType'] == 'None'))
        region_set = set(contour_reference.loc[regular_contours, 'RegionIndex'])
        # Initialize the region lists
        self.region_indexes = []
        self.contour_indexes = []
        self.regions = []
        self.boundaries = []
        self.open_holes = []
        # Iterate through the regions on the slice
        while len(region_set) > 0:
            # Get the contours for the selected region.
            region_index = region_set.pop()
            region_contours = get_region_contours(contour_reference,
                                                  [region_index])
            # Get expand the initial selection to include all related regions
            # on the slice.
            related_regions = {region_index}
            for contour in region_contours:
                related_regions.update(set(contour['related_regions']))
            # Modify region_selection to include all related regions
            region_selection = contour_reference['RegionIndex'].isin(related_regions)
            region_reference = contour_reference.loc[region_selection]
            contour_labels = region_reference['Label'].tolist()
            region_contours = contour_graph.nodes.data('contour')[contour_labels]
            # Sort the contours by area in descending order
            region_contours = sorted(region_contours, key=lambda x: x[1].area,
                                     reverse=True)
            region = shapely.MultiPolygon()
            boundary = shapely.MultiPolygon()
            open_hole = shapely.MultiPolygon()
            contour_labels = []
            # Combine the contours into the appropriate MultiPolygons
            for contour in region_contours:
                contour_labels.append(contour.index)
                # Check whether the contour is a boundary
                if contour.is_boundary:
                    if contour.is_hole:
                        # Subtract the hole from the region
                        boundary = boundary - contour.polygon
                        # Holes on boundaries must be open holes
                        open_hole = open_hole.union(contour.polygon)
                    else:
                        # Add the boundary to the region
                        boundary = boundary.union(contour.polygon)
                elif contour.is_hole:
                    # Subtract the hole from the region
                    region = region - contour.polygon
                    if contour.hole_type == 'Open':
                        # if the hole is open, add it to the open_hole list
                        open_hole = open_hole.union(contour.polygon)
                else:
                    region = region.union(contour.polygon)
            # Assign attributes
            self.regions.append(region)
            self.boundaries.append(boundary)
            self.open_holes.append(open_hole)
            self.contour_indexes.append(contour_labels)
            self.region_indexes.append(related_regions)

    @property
    def exterior(self)-> List[shapely.MultiPolygon]:
        '''The solid exterior contour MultiPolygon.

        Boundaries are not included in the exterior contour MultiPolygon.
        Open holes are subtracted from the exterior contour MultiPolygon,
        because these are not considered to be part of the solid exterior.

        Returns:
            List[shapely.MultiPolygon]: The region MultiPolygons with all holes
                filled in.
        '''
        exterior_regions = []
        for region, hole in zip(self.regions, self.open_holes):
            if region.is_empty:
                continue
            solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                      for poly in region.geoms]
            solid = shapely.unary_union(solids)
            if isinstance(solid, shapely.MultiPolygon):
                ext_poly = shapely.MultiPolygon(solid)
            else:
                ext_poly = shapely.MultiPolygon([solid])
            # Subtract open holes
            if not hole.is_empty:
                ext_poly = ext_poly = ext_poly - hole
            exterior_regions.append(ext_poly)
        return exterior_regions

    @property
    def hull(self)-> List[shapely.MultiPolygon]:
        '''A list of bounding polygons generated from each region.

        A convex hull can be pictures as an elastic band stretched around the
        external contour.

        If a region contains more than one polygon the hull will contain the
        area between the regions.  In other words, each convex hull
        will consist of one elastic bands stretched around each external
        polygon that is part of the region.
        Returns:
            List[shapely.MultiPolygon]: The convex hulls for each region.
        '''
        hull_regions = []
        for region in self.regions:
            if region.is_empty:
                continue
            solids = [shapely.Polygon(shapely.get_exterior_ring(poly))
                      for poly in region.geoms]
            solid = shapely.unary_union(solids)
            if isinstance(solid, shapely.MultiPolygon):
                hull_poly = shapely.MultiPolygon(solid)
            else:
                hull_poly = shapely.MultiPolygon([solid])
            hull_regions.append(hull_poly)
        return hull_regions

    @property
    def is_empty(self)-> bool:
        '''Check if the entire slice is empty.

        Returns:
            bool: True if the slice is empty, False otherwise.
        '''
        if self.regions:
            no_regions = all(region.is_empty for region in self.regions)
        else:
            no_regions = True
        if self.boundaries:
            no_boundaries = all(boundary.is_empty
                                for boundary in self.boundaries)
        else:
            no_boundaries = True
        return no_regions | no_boundaries

    def __bool__(self) -> bool:
        '''Check if the slice is empty.

        This is the opposite of the is_empty property, and is used to check
        if the slice is empty in a boolean context.

        Returns:
            bool: False if the slice is empty, True otherwise.
        '''
        return not self.is_empty


# %% Slice related functions
def empty_structure(structure: Union[RegionSlice, float], invert=False) -> bool:
    '''Check if the structure is empty.

    If the structure is a RegionSlice, it is considered empty if it has no
    contours. If the structure is a NaN object, it is considered empty. If the
    structure is None, it is considered empty. If the structure has an is_empty
    attribute, it is used to determine if the structure is empty.  If the
    structure has an area attribute, it is used to determine if the structure
    is empty.

    This function is primarily used to test shape objects, such as RegionSlice,
    in a DataFrame context

    Args:
        structure (Union[RegionSlice, float]): A RegionSlice or NaN object.
        invert (bool, optional): If True, return the opposite of the result.

    Returns:
        bool: True if the structure is None, NaN, or meets one of the other
            conditions for being "empty". Otherwise False.  If invert True,
            the result is opposite.
    '''
    # Check for None
    if structure is None:
        is_empty = True
    elif isinstance(structure, float):
        # check for NaN
        is_empty = pd.isna(structure)
    else:
        # check for an is_empty attribute.
        # This is used for RegionSlice objects.
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



def build_region_table(contour_graph: nx.Graph,
                       contour_lookup: pd.DataFrame) -> pd.DataFrame:
    '''Build a DataFrame of RegionSlices for each RegionIndex and SliceIndex.

    ***This Text needs Correcting***
    Args:
        contour_graph (nx.Graph): The graph representation of the contours.
        contour_lookup (pd.DataFrame): A DataFrame serving as a lookup table
            for contours.

    Returns:
        pd.DataFrame: A DataFrame containing RegionSlices with the following
        columns:
            - RegionIndex
            - SliceIndex
            - RegionSlice
    '''
    enclosed_region_data = []

    # Iterate through each unique combination of RegionIndex and SliceIndex
    for slice_index in contour_lookup['SliceIndex'].unique():
        # Create a RegionSlice for the given SliceIndex
        region_slice = RegionSlice(contour_graph, contour_lookup, slice_index)
        # Add the RegionSlice to the DataFrame
        enclosed_region_data.append({
            'SliceIndex': slice_index,
            'RegionSlice': region_slice
        })

    # Create the enclosed_region DataFrame
    enclosed_region_table = pd.DataFrame(enclosed_region_data)
    return enclosed_region_table
