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

# Local packages
from contours import Contour, build_contour_lookup
from types_and_classes import SliceIndexType


# %% StructureSlice Class
class RegionSlice():
    '''Class representing all regions of a structure on a specific slice.

    The class contains multiple lists of Shapley MultiPolygons. The lists
    are:
        - regions: Each MultiPolygon defines the polygons on the slice that
            are part of a unique contiguous 3D region (A contiguous 3D region
            may have multiple polygons on a given slice that converge on a
            different slice). The Polygons also include any holes (voids) in the
            region.
        - boundaries: Each MultiPolygon includes the polygons on the slice that
            are part of the related 3D region but are local boundaries of that
            region.  The boundaries are kept distinct from the regions to allow
            for corrections to relationship tests for the structure.
        - open_holes: Each MultiPolygon includes the polygons on the slice that
            define holes in the related  region which are open to the exterior
            of the structure. In other words, holes that describe concave
            portions of the 3D region, which appear as interior holes on the
            particular slice. The holes are not part of the region.  The holes
            are kept distinct from the regions to allow for accurate
            identification of the exterior surface of the 3D region.

    The RegionSlice is constructed from the structure's contour_graph. The
    contour_graph is a NetworkX graph that contains all the contours of the
    structure. The nodes of the graph contain Contour objects and the edges link
    to matching contours on the previous and next slices. Each contour in the
    graph contains an index identifying which unique region it belongs to. The
    contour also contains information regarding the type of contour/regions it
    defines: a boundary, an open or closed hole, or a regular part of the region.

    The custom properties exterior and hull are defined:
        - Exterior: Returns the equivalent list of MultiPolygon regions with all
            closed (internal) holes filled in.
        - Hull: Returns an equivalent list of MultiPolygons, representing the
            convex hulls of each region.

    The is_empty property is defined to check if the slice is empty. The slice is
        considered empty if all the regions and boundaries are empty.
    The __bool__ method uses the is_empty function in a boolean context.


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
        is_empty (bool): True if the slice is empty, False otherwise.
        is_interpolated (bool): True if all contours in the slice are
            interpolated, False otherwise.
    '''

    def __init__(self, contour_graph: nx.Graph, slice_index: SliceIndexType) -> None:
        '''Iteratively create a list of regions for the specified slice, from a
        contour graph.

        Args:
            contour_graph (nx.Graph): A NetworkX graph that contains all the
                contours of the structure. The nodes of the graph contain
                a 'contour' data item which is a Contour object.  The edges of
                the graph link to matching contours on the previous and next
                slices. Each Contour object in the graph contains an index
                identifying which unique region it belongs to. The Contour
                object also contains information regarding the type of
                contour/regions it defines: a boundary, an open or closed hole,
                or a regular part of the region.
        '''
        def get_region_contours(contour_reference: pd.DataFrame,
                                related_regions: List[str])->List[Contour]:
            '''Return the contours for the specified regions.'''
            region_selection = contour_reference['RegionIndex'].isin(related_regions)
            region_reference = contour_reference.loc[region_selection]
            contour_labels = region_reference['Label'].tolist()
            contour_data = dict(contour_graph.nodes.data('contour'))
            region_contours = [contour_data[label] for label in contour_labels]
            return region_contours

        self.slice_index = slice_index
        contour_lookup = build_contour_lookup(contour_graph)
        # Filter contours by SliceIndex and contour type
        # Only contours that are not interpolated, holes or boundaries are
        # used to select the regions.
        selected_contours = contour_lookup['SliceIndex'] == slice_index
        reference_columns = ['Label', 'RegionIndex',
                             'HoleType', 'Interpolated', 'Boundary']
        contour_reference = contour_lookup.loc[selected_contours,
                                               reference_columns]
        #regular_contours = (~contour_reference['Boundary'] &
        #                    ~contour_reference['Interpolated'] &
        #                    (contour_reference['HoleType'] == 'None'))
        #region_set = set(contour_reference.loc[regular_contours, 'RegionIndex'])
        region_set = set(contour_reference.RegionIndex)
        # Initialize the region lists
        self.region_indexes = []
        self.contour_indexes = []
        self.regions = []
        self.boundaries = []
        self.open_holes = []
        interpolated_contours = []
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
                related_regions.update(set(contour.related_regions))
            # Remove the regions that have already been processed.
            region_set = region_set - related_regions
            # Modify region_contours to include all related regions
            region_contours = get_region_contours(contour_reference,
                                                  list(related_regions))
            # Sort the contours by area in descending order
            region_contours = sorted(region_contours, key=lambda x: x.area(),
                                     reverse=True)
            # Initialize region, boundary and open holes with empty MultiPolygons.
            region = shapely.MultiPolygon()
            boundary = shapely.MultiPolygon()
            open_hole = shapely.MultiPolygon()
            contour_labels = []
            # Add the contours into the appropriate MultiPolygons
            for contour in region_contours:
                contour_labels.append(contour.index)
                # record whether the contour is interpolated
                interpolated_contours.append(contour.is_interpolated)
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
            # Add the MultiPolygons into the appropriate lists.  An empty
            # MultiPolygon is added to the list if there are no appropriate
            # contours in the regions.  As a result the lengths of the lists
            # will be the same and the region, boundary and open holes
            # MultiPolygons will match one-to-one. The contour_indexes and
            # region_indexes reference lists will also match.
            self.regions.append(region)
            self.boundaries.append(boundary)
            self.open_holes.append(open_hole)
            self.contour_indexes.append(contour_labels)
            self.region_indexes.append(related_regions)
        # Label the RegionSlice as Interpolated if all contours on the slice are
        # interpolated.
        self.is_interpolated = all(interpolated_contours)

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
                hull_regions.append(region)
            else:
                hull = region.convex_hull
                if isinstance(hull, shapely.MultiPolygon):
                    hull_poly = shapely.MultiPolygon(hull)
                else:
                    hull_poly = shapely.MultiPolygon([hull])
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
        return no_regions & no_boundaries

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
                       slice_sequence: pd.DataFrame) -> pd.DataFrame:
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
    for slice_index in slice_sequence.slices:
        # Create a RegionSlice for the given SliceIndex
        region_slice = RegionSlice(contour_graph, slice_index)
        # Add the RegionSlice to the DataFrame
        enclosed_region_data.append({
            'SliceIndex': slice_index,
            'RegionSlice': region_slice,
            'Empty': region_slice.is_empty,
            'Interpolated': region_slice.is_interpolated
        })

    # Create the enclosed_region DataFrame
    enclosed_region_table = pd.DataFrame(enclosed_region_data)
    return enclosed_region_table
