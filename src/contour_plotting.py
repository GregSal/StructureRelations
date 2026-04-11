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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
                   tolerance=0.0,
                   plot_mode: str = 'contour',
                   relationship_overlay: str = 'none',
                   show_legend: bool = True):
    '''Plot the contours of one or two structures on a specific slice.

    This function is the extension of plot_ab that accepts a structure set and
    structure identifiers. It visualizes structure contours on a specified slice,
    plotting one or two structures to compare their spatial relationships.
    In contour mode, structures are rendered as outlines. In relationship
    mode, regions are rendered as filled areas:
    - Blue: regions only in first structure
    - Green: regions only in second structure
    - Orange: overlapping regions

    Args:
        structure_set (StructureSet): The structure set containing the structures.
        slice_index (SliceIndexType): The slice index to plot contours from.
        roi_list (List[int], optional): ROI numbers to plot. Contour mode accepts
            one or more structures. Relationship mode uses the first two and can
            optionally use a third ROI when relationship_overlay is
            'third_structure'.
            Either roi_list or structure_names must be provided.
        structure_names (List[str], optional): A list of 1 or 2 structure names
            to plot. Either roi_list or structure_names must be provided.
        axes (matplotlib.axes, optional): Existing axes to plot on. If None,
            creates a new figure. Defaults to None.
        add_axis (bool, optional): Whether to add axis lines at 0. Defaults to False.
        tolerance (float, optional): If greater than 0.0, plots a dotted boundary
            line at the specified distance around the contours. Defaults to 0.0.
        plot_mode (str, optional): One of 'contour' or 'relationship'. Defaults to
            'contour'.
        relationship_overlay (str, optional): Optional third relationship layer.
            One of 'none', 'third_structure', 'structure_1', 'structure_2',
            'intersection_ab', 'a_minus_b', or 'a_xor_b'. Defaults to 'none'.
        show_legend (bool, optional): Whether to draw a legend. Defaults to True.

    Returns:
        matplotlib.axes: The axes with the plotted contours.

    Raises:
        ValueError: If neither roi_list nor structure_names is provided, if a
            structure name is not found, or if plot configuration is invalid.
    '''
    # Validate inputs
    if roi_list is None and structure_names is None:
        raise ValueError('Either roi_list or structure_names must be provided')

    # Convert structure names to ROI numbers if needed
    if structure_names is not None:
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

    if plot_mode not in {'contour', 'relationship'}:
        raise ValueError(f'plot_mode must be contour or relationship. Got {plot_mode}.')

    allowed_overlays = {
        'none',
        'third_structure',
        'structure_1',
        'structure_2',
        'intersection_ab',
        'a_minus_b',
        'a_xor_b',
    }
    if relationship_overlay not in allowed_overlays:
        raise ValueError(
            'relationship_overlay must be one of '
            f"{sorted(allowed_overlays)}. Got {relationship_overlay}."
        )

    if len(roi_list) == 0:
        raise ValueError('At least one ROI must be provided.')

    if plot_mode == 'relationship' and len(roi_list) < 2:
        raise ValueError('Relationship mode requires at least 2 ROIs.')

    if plot_mode == 'relationship' and len(roi_list) > 3:
        raise ValueError('Relationship mode supports at most 3 ROIs.')

    if relationship_overlay == 'third_structure' and len(roi_list) < 3:
        raise ValueError('third_structure overlay requires a third ROI.')

    # Validate ROIs exist
    for roi in roi_list:
        if roi not in structure_set.structures:
            raise ValueError(f'ROI {roi} not found in structure set')

    # Track if we need to show the plot
    show_plot = (axes is None)

    color_cycle = [
        'blue',
        'green',
        'red',
        'darkorange',
        'teal',
        'brown',
        'magenta',
    ]

    def plot_filled_geometry(ax, geom, color, alpha=1.0):
        if geom.is_empty:
            return
        if isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            plot_polygon(
                geom,
                ax=ax,
                add_points=False,
                color=color,
                facecolor=color,
                alpha=alpha,
            )
            return
        if isinstance(geom, shapely.GeometryCollection):
            for item in geom.geoms:
                plot_filled_geometry(ax, item, color, alpha=alpha)

    def plot_outline_geometry(ax, geom, color):
        if geom.is_empty:
            return
        if isinstance(geom, shapely.Polygon):
            plot_line(geom.exterior, ax=ax, add_points=False, color=color)
            for interior in geom.interiors:
                plot_line(interior, ax=ax, add_points=False, color=color)
            return
        if isinstance(geom, shapely.MultiPolygon):
            for polygon in geom.geoms:
                plot_outline_geometry(ax, polygon, color)
            return
        if isinstance(geom, (shapely.LineString, shapely.MultiLineString, shapely.LinearRing)):
            plot_line(geom, ax=ax, add_points=False, color=color)
            return
        if isinstance(geom, shapely.GeometryCollection):
            for item in geom.geoms:
                plot_outline_geometry(ax, item, color)

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

    def plot_tolerance_outline(ax, geom, color):
        if tolerance <= 0.0 or geom.is_empty:
            return
        buffered = geom.buffer(tolerance / 2)
        if isinstance(buffered, shapely.Polygon):
            plot_line(buffered.exterior, ax=ax, add_points=False, color=color, linestyle=':')
            return
        if isinstance(buffered, shapely.MultiPolygon):
            for polygon in buffered.geoms:
                plot_line(polygon.exterior, ax=ax, add_points=False, color=color, linestyle=':')

    def get_roi_geometry(roi):
        structure = structure_set.structures[roi]
        structure_slice = structure.get_slice(slice_index)
        geometry = shapely.GeometryCollection()
        if structure_slice and not structure_slice.is_empty:
            geometry = structure_slice.merge_regions()
        return structure, geometry

    structures = []
    geometries = []
    for roi in roi_list:
        structure, geometry = get_roi_geometry(roi)
        structures.append(structure)
        geometries.append(geometry)

    ax = axes if axes else plt.gca()
    ax.set_axis_off()
    ax.axis('equal')

    if plot_mode == 'contour':
        legend_handles = []
        for idx, (structure, geometry) in enumerate(zip(structures, geometries)):
            color = color_cycle[idx % len(color_cycle)]
            plot_outline_geometry(ax, geometry, color)
            plot_tolerance_outline(ax, geometry, color)
            if show_legend:
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2, label=structure.name)
                )

        title = ', '.join(structure.name for structure in structures[:3])
        if len(structures) > 3:
            title = f'{title} (+{len(structures) - 3} more)'
        ax.set_title(f'{title} at slice {slice_index}', color='gray')
        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, loc='upper right')
    else:
        structure_a = structures[0]
        structure_b = structures[1]
        poly_a = geometries[0]
        poly_b = geometries[1]

        ax = plot_ab(poly_a, poly_b, add_axis=add_axis, axes=ax)
        plot_tolerance_bands(ax, poly_a, poly_b)

        overlay_geom = None
        overlay_label = None
        if relationship_overlay == 'third_structure':
            overlay_geom = geometries[2]
            overlay_label = structures[2].name
        elif relationship_overlay == 'structure_1':
            overlay_geom = poly_a
            overlay_label = structure_a.name
        elif relationship_overlay == 'structure_2':
            overlay_geom = poly_b
            overlay_label = structure_b.name
        elif relationship_overlay == 'intersection_ab':
            overlay_geom = shapely.intersection(poly_a, poly_b)
            overlay_label = 'A ∩ B'
        elif relationship_overlay == 'a_minus_b':
            overlay_geom = shapely.difference(poly_a, poly_b)
            overlay_label = 'A - B'
        elif relationship_overlay == 'a_xor_b':
            overlay_geom = shapely.symmetric_difference(poly_a, poly_b)
            overlay_label = 'A XOR B'

        if overlay_geom is not None and not overlay_geom.is_empty:
            plot_filled_geometry(ax, overlay_geom, 'red', alpha=0.45)
            plot_outline_geometry(ax, overlay_geom, 'red')
            plot_tolerance_outline(ax, overlay_geom, 'red')

        ax.set_title(
            f'{structure_a.name} vs {structure_b.name} at slice {slice_index}',
            color='gray',
        )

        if show_legend:
            handles = [
                Patch(facecolor='blue', edgecolor='blue', label=f'{structure_a.name} only'),
                Patch(facecolor='green', edgecolor='green', label=f'{structure_b.name} only'),
                Patch(facecolor='orange', edgecolor='orange', label='Intersection'),
            ]
            if overlay_geom is not None and not overlay_geom.is_empty:
                handles.append(
                    Patch(facecolor='red', edgecolor='red', alpha=0.45, label=overlay_label)
                )
            ax.legend(handles=handles, loc='upper right')

    if add_axis and plot_mode == 'contour':
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')

    if show_plot:
        plt.show()

    return ax
