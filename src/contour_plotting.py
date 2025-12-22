'''Plotting Functions

Functions in here are used to plot polygons and structure contours for
debugging and visualization purposes.'''
# %% Imports
# Type imports
from typing import List

import logging

# Shared Packages
import shapely
from shapely.plotting import plot_polygon, plot_line
import matplotlib.pyplot as plt

# Local Packages
from types_and_classes import SliceIndexType

from structure_set import StructureSet
from region_slice import RegionSlice

# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% functions
def plot_ab(poly_a, poly_b, add_axis=True, axes=None):
    '''Plot the difference between two polygons.

    This function plots the difference between two polygons, showing the
    parts that are only in poly_a, only in poly_b, and the intersection of
    both.   The parts that are only in poly_a are plotted in blue, the parts
    that are only in poly_b are plotted in green, and the intersection is
    plotted in orange. The axes are set to equal aspect ratio and the
    horizontal and vertical lines at 0 are shown as dashed gray lines.

        If the polygons are MultiPolygons, they are plotted as a single polygon.
    If the polygons are LineStrings, they are plotted as a single line.

    Args:
        poly_a (shapely.Polygon | RegionSlice): The first polygon.
        poly_b (shapely.Polygon | RegionSlice): The second polygon.

    Returns:
        ax: The matplotlib axis with the plotted polygons.
    '''
    def plot_geom(ax, geom, color='black'):
        if isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            plot_polygon(geom, ax=ax, add_points=False, color=color,
                         facecolor=color)
        elif isinstance(geom, (shapely.LineString, shapely.MultiLineString,
                               shapely.LinearRing, shapely.LinearRing)):
            plot_line(geom, ax=ax, add_points=False, color=color)
        elif isinstance(geom, shapely.GeometryCollection):
            # plot each of the geometry objects in the collection
            for g in geom.geoms:
                plot_geom(ax, g, color)

    if isinstance(poly_a, (RegionSlice)):
        poly_a = poly_a.merge_regions()
    if isinstance(poly_b, (RegionSlice)):
        poly_b = poly_b.merge_regions()
    if axes is not None:
        ax = axes
    else:
        fig = plt.figure(1, figsize=(4,2))
        ax = fig.add_subplot(121)
    ax.set_axis_off()
    ax.axis('equal')

    only_a = shapely.difference(poly_a, poly_b)
    plot_geom(ax, only_a, color='blue')
    only_b = shapely.difference(poly_b, poly_a)
    plot_geom(ax, only_b, color='green')
    both_ab = shapely.intersection(poly_a, poly_b)
    plot_geom(ax, both_ab, color='orange')

    if add_axis:
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
    if not axes:
        plt.show()
    return ax


def plot_roi_slice(structure_set: StructureSet,
                   slice_index: SliceIndexType,
                   roi_list: List[int] = None,
                   structure_names: List[str] = None,
                   axes=None,
                   add_axis=False,
                   tolerance=0.0):
    '''Plot the contours of one or two structures on a specific slice.

    This function is the extension of plot_ab that accepts a structure set and
    structure identifiers. It visualizes structure contours on a specified slice,
    plotting one or two structures to compare their spatial relationships.
    When plotting two structures:
    - Blue: regions only in first structure
    - Green: regions only in second structure
    - Orange: overlapping regions

    Args:
        structure_set (StructureSet): The structure set containing the structures.
        slice_index (SliceIndexType): The slice index to plot contours from.
        roi_list (List[int], optional): A list of 1 or 2 ROI numbers to plot.
            Either roi_list or structure_names must be provided.
        structure_names (List[str], optional): A list of 1 or 2 structure names
            to plot. Either roi_list or structure_names must be provided.
        axes (matplotlib.axes, optional): Existing axes to plot on. If None,
            creates a new figure. Defaults to None.
        add_axis (bool, optional): Whether to add axis lines at 0. Defaults to False.
        tolerance (float, optional): If greater than 0.0, plots a dotted boundary
            line at the specified distance around the contours. Defaults to 0.0.

    Returns:
        matplotlib.axes: The axes with the plotted contours.

    Raises:
        ValueError: If neither roi_list nor structure_names is provided, if more
            than 2 structures are specified, or if a structure name is not found.
    '''
    # Validate inputs
    if roi_list is None and structure_names is None:
        raise ValueError('Either roi_list or structure_names must be provided')

    # Convert structure names to ROI numbers if needed
    if structure_names is not None:
        if len(structure_names) > 2:
            raise ValueError('Can only plot 1 or 2 structures. '
                           f'Got {len(structure_names)} structure names.')

        roi_list = []
        if structure_set.dicom_structure_file is None:
            raise ValueError('structure_set must have dicom_structure_file to '
                           'use structure_names')

        name_dict = structure_set.dicom_structure_file.structure_names
        for name in structure_names:
            # Case-insensitive match - structure_names is a dict {ROI: name}
            matching_roi = None
            for roi, structure_name in name_dict.items():
                if structure_name.lower() == name.lower():
                    matching_roi = roi
                    break

            if matching_roi is None:
                raise ValueError(f"Structure name '{name}' not found in "
                               f"structure set")
            roi_list.append(matching_roi)

    # Validate roi_list length
    if len(roi_list) > 2:
        raise ValueError(f'Can only plot 1 or 2 structures. Got {len(roi_list)} ROIs.')

    # Validate ROIs exist
    for roi in roi_list:
        if roi not in structure_set.structures:
            raise ValueError(f'ROI {roi} not found in structure set')

    # Get structures
    structure_a = structure_set.structures[roi_list[0]]
    slice_a = structure_a.get_slice(slice_index)

    # Track if we need to show the plot
    show_plot = (axes is None)

    def plot_tolerance_bands(ax, poly_a, poly_b):
        '''Plot tolerance bands with proper color handling for overlaps.

        Each buffer is colored based on whether it overlaps with the other
        structure's full buffered region (polygon + buffer):
        - Cornflowerblue: A's buffer outside B's buffered region
        - Lightgreen: B's buffer outside A's buffered region
        - Lightsalmon: Either buffer overlapping with the other structure's buffered region
        '''
        if tolerance > 0.0:
            # Create full buffered regions (polygon + buffer ring)
            buffered_a = shapely.GeometryCollection()
            buffered_b = shapely.GeometryCollection()

            if not poly_a.is_empty:
                buffered_a = poly_a.buffer(tolerance / 2)

            if not poly_b.is_empty:
                buffered_b = poly_b.buffer(tolerance / 2)

            # Extract just the buffer rings (excluding the original polygons)
            band_a = shapely.difference(buffered_a, poly_a)
            band_b = shapely.difference(buffered_b, poly_b)

            # Determine which parts of each buffer overlap with the other's buffered region
            # band_a_salmon: A's buffer that overlaps with B's buffered region (poly_b + buffer_b)
            # band_b_salmon: B's buffer that overlaps with A's buffered region (poly_a + buffer_a)
            band_a_salmon = shapely.intersection(band_a, buffered_b)
            band_b_salmon = shapely.intersection(band_b, buffered_a)

            # The rest stays their original color
            band_a_blue = shapely.difference(band_a, buffered_b)
            band_b_green = shapely.difference(band_b, buffered_a)

            # Plot non-overlapping tolerance bands
            if not band_a_blue.is_empty:
                if isinstance(band_a_blue, (shapely.Polygon, shapely.MultiPolygon)):
                    plot_polygon(band_a_blue, ax=ax, add_points=False,
                               color='cornflowerblue', facecolor='cornflowerblue')

            if not band_b_green.is_empty:
                if isinstance(band_b_green, (shapely.Polygon, shapely.MultiPolygon)):
                    plot_polygon(band_b_green, ax=ax, add_points=False,
                               color='lightgreen', facecolor='lightgreen')

            # Plot overlapping tolerance bands in coral (union both regions)
            band_overlap = shapely.union(band_a_salmon, band_b_salmon)
            if not band_overlap.is_empty:
                if isinstance(band_overlap, (shapely.Polygon, shapely.MultiPolygon)):
                    plot_polygon(band_overlap, ax=ax, add_points=False,
                               color='coral', facecolor='coral')

    if len(roi_list) == 1:
        # Single structure - extract polygon and plot
        poly_a = shapely.GeometryCollection()
        if slice_a and not slice_a.is_empty:
            poly_a = slice_a.merge_regions()

        # Call plot_ab with only first polygon
        ax = plot_ab(poly_a, shapely.GeometryCollection(),
                    add_axis=add_axis, axes=(axes if axes else plt.gca()))

        # Plot tolerance bands
        plot_tolerance_bands(ax, poly_a, shapely.GeometryCollection())

        ax.set_title(f'{structure_a.name} at slice {slice_index}', color='gray')

    else:
        # Two structures - extract both polygons and plot
        structure_b = structure_set.structures[roi_list[1]]
        slice_b = structure_b.get_slice(slice_index)

        poly_a = shapely.GeometryCollection()
        poly_b = shapely.GeometryCollection()

        if slice_a and not slice_a.is_empty:
            poly_a = slice_a.merge_regions()

        if slice_b and not slice_b.is_empty:
            poly_b = slice_b.merge_regions()

        # Call plot_ab to handle the comparison plotting
        ax = plot_ab(poly_a, poly_b, add_axis=add_axis, axes=(axes if axes else plt.gca()))

        # Plot tolerance bands
        plot_tolerance_bands(ax, poly_a, poly_b)

        ax.set_title(f'{structure_a.name} vs {structure_b.name} at slice {slice_index}',
                    color='gray')

    if show_plot:
        plt.show()

    return ax
