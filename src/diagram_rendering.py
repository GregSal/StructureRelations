'''Utilities for rendering relationship diagrams from a StructureSet.

This module extracts notebook-oriented plotting logic into reusable functions.
The implementation is intentionally iteration-friendly rather than finalized,
so consumers can keep refining geometry and readability heuristics.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd


VIS_TO_MPL_MARKER = {
    'ellipse': 'o',
    'dot': '.',
    'box': 's',
    'square': 's',
    'diamond': 'D',
    'triangle': '^',
    'triangledown': 'v',
    'star': '*',
    'hexagon': 'h',
}


DEFAULT_VERTICAL_ORDER = [
    'GTV',
    'CTV',
    'PTV',
    'TREATED VOLUME',
    'SHELL',
]


@dataclass
class DiagramRenderResult:
    '''Container for diagram rendering outputs.

    Attributes:
        fig (plt.Figure): Matplotlib figure used for rendering.
        axis (plt.Axes): Matplotlib axis containing nodes, edges, and labels.
        relationship_graph (nx.DiGraph): Filtered graph used for visualization.
        edge_render_items (list[dict[str, Any]]): Per-edge render metadata.
        positions (dict[int, tuple[float, float]]): Node positions in data space.
        plot_nodes (pd.DataFrame): Node table with grouping/sort metadata.
        relationship_counts (pd.Series): Count of each relation type displayed.
        figure_background (str): Background color currently applied.
    '''

    fig: plt.Figure
    axis: plt.Axes
    relationship_graph: nx.DiGraph
    edge_render_items: list[dict[str, Any]]
    positions: dict[int, tuple[float, float]]
    plot_nodes: pd.DataFrame
    relationship_counts: pd.Series
    figure_background: str


def vis_shape_to_marker(vis_shape: str) -> str:
    '''Map vis-network shape names to matplotlib markers.

    Args:
        vis_shape (str): Node shape from diagram settings.

    Returns:
        str: Matplotlib marker.
    '''
    return VIS_TO_MPL_MARKER.get(str(vis_shape).lower(), 'o')


def vis_shape_to_boxstyle(vis_shape: str) -> str:
    '''Map vis-network shape names to text-box style strings.

    Args:
        vis_shape (str): Node shape from diagram settings.

    Returns:
        str: Matplotlib ``boxstyle`` value.
    '''
    normalized = str(vis_shape).lower()
    if normalized == 'ellipse':
        return 'round,pad=0.35,rounding_size=0.9'
    if normalized in {'box', 'square'}:
        return 'round,pad=0.28,rounding_size=0.1'
    return 'round,pad=0.28,rounding_size=0.2'


def darken_color(hex_color: str) -> str:
    '''Return a darkened version of an RGB hex color.

    Args:
        hex_color (str): Color string like ``#AABBCC``.

    Returns:
        str: Darkened color; fallback is ``#1f2937`` for invalid inputs.
    '''
    color_value = str(hex_color).replace('#', '')
    if len(color_value) != 6:
        return '#1f2937'

    red = int(color_value[0:2], 16)
    green = int(color_value[2:4], 16)
    blue = int(color_value[4:6], 16)

    darkened_red = int(red * 0.7)
    darkened_green = int(green * 0.7)
    darkened_blue = int(blue * 0.7)

    return (
        f'#{darkened_red:02x}'
        f'{darkened_green:02x}'
        f'{darkened_blue:02x}'
    )


def get_text_color(background_color: str,
                   dark_color: str = '#000000',
                   light_color: str = '#FFFFFF') -> str:
    '''Pick a readable foreground color for a background.

    Args:
        background_color (str): RGB hex color string.
        dark_color (str): Preferred text color on bright backgrounds.
        light_color (str): Preferred text color on dark backgrounds.

    Returns:
        str: ``dark_color`` or ``light_color`` based on luminance.
    '''
    color_value = str(background_color).replace('#', '')
    if len(color_value) != 6:
        return dark_color

    red = int(color_value[0:2], 16)
    green = int(color_value[2:4], 16)
    blue = int(color_value[4:6], 16)

    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    return dark_color if brightness > 128 else light_color


def get_node_color_map(current_structure_set,
                       default_color: str = '#b7bec8') -> dict[int, str]:
    '''Build ROI-to-color mapping from DICOM display colors.

    Args:
        current_structure_set: StructureSet-like object with ``summary()`` and
            optional ``dicom_structure_file.dataset.ROIContourSequence``.
        default_color (str): Color used when no DICOM display color exists.

    Returns:
        dict[int, str]: ROI number to color hex string.
    '''
    node_color_map = {
        int(row['ROI']): default_color
        for _, row in current_structure_set.summary().iterrows()
    }

    dicom_structure_file = getattr(current_structure_set, 'dicom_structure_file', None)
    dataset = getattr(dicom_structure_file, 'dataset', None)
    if dataset is None:
        return node_color_map

    for roi_contour in getattr(dataset, 'ROIContourSequence', []):
        roi_num = int(getattr(roi_contour, 'ReferencedROINumber', 0))
        rgb = getattr(roi_contour, 'ROIDisplayColor', None)
        if rgb and len(rgb) >= 3:
            node_color_map[roi_num] = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]), int(rgb[1]), int(rgb[2])
            )

    return node_color_map


def normalize_edge_style(style: dict[str, Any],
                         fallback: dict[str, Any]) -> dict[str, Any]:
    '''Normalize edge style dictionary with fallback defaults.

    Args:
        style (dict[str, Any]): Style entry for one relationship type.
        fallback (dict[str, Any]): Default style entry.

    Returns:
        dict[str, Any]: Normalized style values.
    '''
    return {
        'color': style.get('color', fallback.get('color', '#999999')),
        'width': float(style.get('width', fallback.get('width', 1))),
        'dashes': bool(style.get('dashes', fallback.get('dashes', False))),
        'arrows': style.get('arrows', fallback.get('arrows', None)),
    }


def _coerce_horizontal_group(raw_value: Any) -> str:
    '''Normalize horizontal group values for predictable ordering.'''
    if pd.isna(raw_value):
        return 'missing'
    text_value = str(raw_value).strip()
    if text_value == '':
        return 'blank'

    lowered = text_value.lower()
    if lowered == 'none':
        return 'None'
    if lowered in {'(ungrouped)', 'ungrouped', 'missing'}:
        return 'missing'

    return text_value


def _is_numeric_group(group_value: Any) -> bool:
    '''Return whether group text can be interpreted as a number.'''
    try:
        float(str(group_value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _horizontal_sort_key(group_value: str) -> tuple[int, float, str]:
    '''Build deterministic left-to-right ordering key.

    Ordering policy:
    1) numeric groups, 2) text groups, 3) ``None``, 4) blank, 5) missing.
    '''
    if group_value == 'None':
        return (2, 0.0, '')
    if group_value == 'blank':
        return (3, 0.0, '')
    if group_value == 'missing':
        return (4, 0.0, '')
    if _is_numeric_group(group_value):
        return (0, float(str(group_value).strip()), '')
    return (1, 0.0, str(group_value).lower())


def _vertical_order_name(group_value: Any) -> str:
    '''Map vertical group text to normalized category names.'''
    upper_value = str(group_value).strip().upper().replace('_', ' ')
    if upper_value.startswith('GTV'):
        return 'GTV'
    if upper_value.startswith('CTV'):
        return 'CTV'
    if upper_value.startswith('PTV'):
        return 'PTV'
    if upper_value in {'TREATED VOLUME', 'TREATED_VOLUME'}:
        return 'TREATED VOLUME'
    if upper_value.startswith('SHELL'):
        return 'SHELL'
    return upper_value


def _vertical_sort_key(group_value: str,
                       vertical_order_index: dict[str, int]) -> tuple[int, str]:
    '''Build deterministic top-to-bottom ordering key.'''
    normalized = _vertical_order_name(group_value)
    return (vertical_order_index.get(normalized, len(vertical_order_index)), normalized)


def _extract_shaft_vertices(edge_item: dict[str, Any],
                            ax: plt.Axes,
                            positions: dict[int, tuple[float, float]]) -> list[tuple[float, float]]:
    '''Return a display-space polyline for the drawn edge shaft.

    Args:
        edge_item (dict[str, Any]): Metadata produced while drawing one edge.
        ax (plt.Axes): Axis used for rendering.
        positions (dict[int, tuple[float, float]]): Node positions.

    Returns:
        list[tuple[float, float]]: Display-space path points for one edge shaft.
    '''
    source, target = edge_item['edge_to_draw']
    source_point = positions[source]
    target_point = positions[target]
    arc_rad = float(edge_item.get('edge_arc_rad', 0.0))

    source_disp = ax.transData.transform(source_point)
    target_disp = ax.transData.transform(target_point)

    if edge_item.get('draw_arrows') and edge_item.get('edge_artists'):
        arrow_patch = edge_item['edge_artists'][0]
        rendered_path = arrow_patch.get_path().transformed(arrow_patch.get_transform())
        codes = rendered_path.codes
        verts = rendered_path.vertices

        if codes is not None and len(codes) >= 2 and len(verts) >= 2:
            shaft_vertices = [verts[0]]
            for idx in range(1, len(codes)):
                code_val = int(codes[idx])
                if code_val in (1, 79):
                    break
                shaft_vertices.append(verts[idx])

            if len(shaft_vertices) >= 2:
                dense_path = [
                    (float(point[0]), float(point[1])) for point in shaft_vertices
                ]
                if (
                    len(dense_path) == 3
                    and int(codes[1]) == 3
                    and int(codes[2]) == 3
                ):
                    p0 = dense_path[0]
                    p1 = dense_path[1]
                    p2 = dense_path[2]
                    dense_path = []
                    for sample_idx in range(201):
                        t_val = sample_idx / 200.0
                        omt = 1.0 - t_val
                        bx = (
                            (omt * omt * p0[0])
                            + (2.0 * omt * t_val * p1[0])
                            + (t_val * t_val * p2[0])
                        )
                        by = (
                            (omt * omt * p0[1])
                            + (2.0 * omt * t_val * p1[1])
                            + (t_val * t_val * p2[1])
                        )
                        dense_path.append((float(bx), float(by)))
                return dense_path

    if abs(arc_rad) < 1e-12:
        return [
            (float(source_disp[0]), float(source_disp[1])),
            (float(target_disp[0]), float(target_disp[1])),
        ]

    from matplotlib.patches import ConnectionStyle

    connector = ConnectionStyle.Arc3(rad=arc_rad)
    curve_path = connector(
        (float(source_disp[0]), float(source_disp[1])),
        (float(target_disp[0]), float(target_disp[1])),
        patchA=None,
        patchB=None,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    return [
        (float(vertex[0]), float(vertex[1]))
        for vertex in curve_path.interpolated(120).vertices
    ]


def _normalize_angle(angle_deg: float) -> float:
    '''Keep text rotation within an upright range.'''
    if angle_deg > 90.0:
        return angle_deg - 180.0
    if angle_deg < -90.0:
        return angle_deg + 180.0
    return angle_deg


def _path_point_display(vertices: list[tuple[float, float]],
                        fraction: float = 0.5) -> tuple[float, float, float]:
    '''Return point and tangent angle at a fractional arc-length position.

    Args:
        vertices (list[tuple[float, float]]): Polyline in display coordinates.
        fraction (float): Arc-length position in [0, 1].

    Returns:
        tuple[float, float, float]: x, y, angle (degrees).
    '''
    if len(vertices) == 0:
        return 0.0, 0.0, 0.0
    if len(vertices) == 1:
        return float(vertices[0][0]), float(vertices[0][1]), 0.0

    frac = min(max(float(fraction), 0.0), 1.0)
    segment_lengths = []
    total_length = 0.0
    for idx in range(1, len(vertices)):
        dx = float(vertices[idx][0] - vertices[idx - 1][0])
        dy = float(vertices[idx][1] - vertices[idx - 1][1])
        seg_len = (dx * dx + dy * dy) ** 0.5
        segment_lengths.append(seg_len)
        total_length += seg_len

    if total_length < 1e-9:
        return float(vertices[0][0]), float(vertices[0][1]), 0.0

    target_len = frac * total_length
    traversed = 0.0
    for idx, seg_len in enumerate(segment_lengths, start=1):
        next_traversed = traversed + seg_len
        if next_traversed >= target_len and seg_len > 0.0:
            seg_frac = (target_len - traversed) / seg_len
            x0, y0 = vertices[idx - 1]
            x1, y1 = vertices[idx]
            angle_deg = _normalize_angle(math.degrees(math.atan2(y1 - y0, x1 - x0)))
            return (
                float(x0 + seg_frac * (x1 - x0)),
                float(y0 + seg_frac * (y1 - y0)),
                float(angle_deg),
            )
        traversed = next_traversed

    x0, y0 = vertices[-2]
    x1, y1 = vertices[-1]
    angle_deg = _normalize_angle(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    return float(vertices[-1][0]), float(vertices[-1][1]), float(angle_deg)


def _point_to_segment_distance(px: float, py: float,
                               ax0: float, ay0: float,
                               bx0: float, by0: float) -> float:
    '''Return point-to-segment distance in display space.'''
    vx = bx0 - ax0
    vy = by0 - ay0
    wx = px - ax0
    wy = py - ay0
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        dx = px - ax0
        dy = py - ay0
        return (dx * dx + dy * dy) ** 0.5
    t_val = (wx * vx + wy * vy) / vv
    t_val = min(1.0, max(0.0, t_val))
    proj_x = ax0 + t_val * vx
    proj_y = ay0 + t_val * vy
    dx = px - proj_x
    dy = py - proj_y
    return (dx * dx + dy * dy) ** 0.5


def _point_to_polyline_distance(px: float,
                                py: float,
                                vertices: list[tuple[float, float]]) -> float:
    '''Return minimum point-to-polyline distance in display space.'''
    if len(vertices) == 0:
        return float('inf')
    if len(vertices) == 1:
        dx = px - vertices[0][0]
        dy = py - vertices[0][1]
        return (dx * dx + dy * dy) ** 0.5

    best = float('inf')
    for idx in range(1, len(vertices)):
        ax0, ay0 = vertices[idx - 1]
        bx0, by0 = vertices[idx]
        dist = _point_to_segment_distance(px, py, ax0, ay0, bx0, by0)
        if dist < best:
            best = dist
    return best


def apply_crossing_heavy_label_offsets(result: DiagramRenderResult,
                                       near_threshold_px: float = 18.0,
                                       node_near_threshold_px: float = 24.0,
                                       heavy_threshold: int = 2,
                                       node_zorder: int = 20,
                                       label_zorder: int = 7) -> None:
    '''Reposition crossing-heavy labels while keeping them edge-locked.

    The function removes existing relation labels, then redraws each label at a
    deterministic point on its own edge. Crossing-heavy means the midpoint is
    near multiple other edge paths and/or near any node center.

    Args:
        result (DiagramRenderResult): Output from ``render_template_diagram``.
        near_threshold_px (float): Edge-path proximity threshold in pixels.
        node_near_threshold_px (float): Node-center proximity threshold.
        heavy_threshold (int): Min nearby edges to count as crossing-heavy.
        node_zorder (int): z-order for node labels/boxes after relabeling.
        label_zorder (int): z-order for relation labels.
    '''
    fig = result.fig
    axis = result.axis
    relationship_graph = result.relationship_graph

    if relationship_graph.number_of_edges() == 0:
        return

    fig.canvas.draw()

    relation_label_texts = {
        str(edge_item['style_data'].get('label', ''))
        for edge_item in result.edge_render_items
    }

    for text_artist in list(axis.texts):
        if (
            int(text_artist.get_fontsize()) == 10
            and text_artist.get_text() in relation_label_texts
        ):
            text_artist.remove()

    edge_paths = []
    for edge_item in result.edge_render_items:
        display_vertices = _extract_shaft_vertices(edge_item, axis, result.positions)
        mid_x, mid_y, _ = _path_point_display(display_vertices, 0.5)
        edge_paths.append({
            'edge_item': edge_item,
            'vertices': display_vertices,
            'mid_x': mid_x,
            'mid_y': mid_y,
        })

    node_display_points = [
        axis.transData.transform(result.positions[roi])
        for roi in relationship_graph.nodes()
    ]
    center_x = sum(float(point[0]) for point in node_display_points)
    center_x = center_x / max(len(node_display_points), 1)
    center_y = sum(float(point[1]) for point in node_display_points)
    center_y = center_y / max(len(node_display_points), 1)

    for idx, path_item in enumerate(edge_paths):
        near_count = 0
        for other_idx, other_item in enumerate(edge_paths):
            if other_idx == idx:
                continue
            dist = _point_to_polyline_distance(
                path_item['mid_x'],
                path_item['mid_y'],
                other_item['vertices'],
            )
            if dist <= near_threshold_px:
                near_count += 1

        near_node_count = 0
        for node_point in node_display_points:
            node_dx = float(path_item['mid_x'] - node_point[0])
            node_dy = float(path_item['mid_y'] - node_point[1])
            node_dist = (node_dx * node_dx + node_dy * node_dy) ** 0.5
            if node_dist <= node_near_threshold_px:
                near_node_count += 1

        label_fraction = 0.5
        if near_count >= heavy_threshold or near_node_count > 0:
            start_x, start_y = path_item['vertices'][0]
            end_x, end_y = path_item['vertices'][-1]
            start_dist = ((start_x - center_x) ** 2 + (start_y - center_y) ** 2) ** 0.5
            end_dist = ((end_x - center_x) ** 2 + (end_y - center_y) ** 2) ** 0.5
            label_fraction = 0.32 if start_dist >= end_dist else 0.68

        label_dx, label_dy, label_angle = _path_point_display(
            path_item['vertices'],
            label_fraction,
        )
        label_data_x, label_data_y = axis.transData.inverted().transform((label_dx, label_dy))

        style_data = path_item['edge_item']['style_data']
        axis.text(
            float(label_data_x),
            float(label_data_y),
            style_data['label'],
            color=style_data['color'],
            fontsize=10,
            ha='center',
            va='center',
            rotation=label_angle,
            rotation_mode='anchor',
            bbox={
                'boxstyle': 'round,pad=0.18',
                'facecolor': result.figure_background,
                'edgecolor': result.figure_background,
                'linewidth': 1,
            },
            zorder=label_zorder,
        )

    node_label_texts = {
        str(node_data.get('label', ''))
        for _, node_data in relationship_graph.nodes(data=True)
    }
    for text_artist in axis.texts:
        if text_artist.get_text() in node_label_texts:
            text_artist.set_zorder(node_zorder)
            bbox_patch = text_artist.get_bbox_patch()
            if bbox_patch is not None:
                bbox_patch.set_zorder(node_zorder)


def render_template_diagram(structure_set,
                            structures_df: pd.DataFrame,
                            diagram_settings_path: str | Path,
                            hide_logical_edges: bool = True,
                            horizontal_spacing: float = 3.3,
                            vertical_spacing: float = 1.2,
                            duplicate_vertical_spread: float = 0.35,
                            node_fill_color: str = '#b7bec8',
                            vertical_order: list[str] | None = None,
                            show_plot: bool = True) -> DiagramRenderResult:
    '''Render a relationship diagram using template-based grouping.

    Args:
        structure_set: StructureSet instance with ``summary()`` and
            ``relationship_graph``.
        structures_df (pd.DataFrame): Parsed structure metadata indexed by name,
            containing ``h_grouping`` and ``v_grouping`` columns.
        diagram_settings_path (str | Path): Path to diagram settings JSON.
        hide_logical_edges (bool): Whether to hide logical/inferred edges.
        horizontal_spacing (float): Distance between horizontal groups.
        vertical_spacing (float): Distance between vertical groups.
        duplicate_vertical_spread (float): Offset for duplicate entries in the
            same vertical group.
        node_fill_color (str): Fallback node color.
        vertical_order (list[str] | None): Optional override for canonical
            top-to-bottom category order.
        show_plot (bool): Whether to call ``plt.show()``.

    Returns:
        DiagramRenderResult: Rendering artifacts and metadata.
    '''
    settings_path = Path(diagram_settings_path)
    with open(settings_path, 'r', encoding='utf-8') as settings_file:
        diagram_settings = json.load(settings_file)

    node_shapes_cfg = diagram_settings.get('node_shapes', {})
    relationship_styles_cfg = diagram_settings.get('relationship_styles', {})
    if not node_shapes_cfg:
        raise ValueError('Missing node_shapes section in diagram_settings.json')
    if not relationship_styles_cfg:
        raise ValueError('Missing relationship_styles section in diagram_settings.json')

    shape_map_raw = node_shapes_cfg.get('shape_map', {})
    default_vis_shape = node_shapes_cfg.get('default_shape')
    if not isinstance(shape_map_raw, dict):
        raise ValueError('node_shapes.shape_map must be a dictionary')
    if not default_vis_shape:
        raise ValueError('node_shapes.default_shape must be defined')

    shape_map = {str(key).upper(): str(value) for key, value in shape_map_raw.items()}

    diagram_font_cfg = diagram_settings.get('diagram_options', {}).get('font', {})
    font_dark_color = diagram_font_cfg.get('dark_color', '#000000')
    font_light_color = diagram_font_cfg.get('light_color', '#FFFFFF')

    diagram_defaults = diagram_settings.get('relationship_display_defaults', {})
    show_disjoint = bool(diagram_defaults.get('show_disjoint', False))
    show_unknown = bool(diagram_defaults.get('show_unknown', False))
    show_edge_labels = bool(diagram_defaults.get('show_edge_labels', True))

    figure_background = (
        diagram_settings
        .get('diagram_options', {})
        .get('background', {})
        .get('color', '#FFFFFF')
    )

    summary_df = structure_set.summary().copy()
    node_color_map = get_node_color_map(structure_set, default_color=node_fill_color)

    dicom_type_summary_col = None
    for candidate in ['DICOM Type', 'DICOM_Type']:
        if candidate in summary_df.columns:
            dicom_type_summary_col = candidate
            break

    plot_nodes = summary_df[['ROI', 'Name']].copy()
    plot_nodes['Name'] = plot_nodes['Name'].astype(str)
    plot_nodes = plot_nodes.merge(
        structures_df[['h_grouping', 'v_grouping']],
        left_on='Name',
        right_index=True,
        how='left',
    )

    if dicom_type_summary_col is not None:
        plot_nodes['DICOM Type'] = summary_df[dicom_type_summary_col].astype(str)
    elif 'DICOM Type' in structures_df.columns:
        plot_nodes = plot_nodes.merge(
            structures_df[['DICOM Type']],
            left_on='Name',
            right_index=True,
            how='left',
        )
        plot_nodes['DICOM Type'] = plot_nodes['DICOM Type'].fillna('Unknown').astype(str)
    else:
        plot_nodes['DICOM Type'] = 'Unknown'

    vertical_order_value = vertical_order or DEFAULT_VERTICAL_ORDER
    vertical_order_index = {
        name: idx for idx, name in enumerate(vertical_order_value)
    }

    plot_nodes['h_grouping'] = plot_nodes['h_grouping'].apply(_coerce_horizontal_group)
    plot_nodes['v_grouping'] = plot_nodes['v_grouping'].fillna('(ungrouped)').astype(str)

    h_order = sorted(plot_nodes['h_grouping'].unique(), key=_horizontal_sort_key)
    h_index_map = {group_name: idx for idx, group_name in enumerate(h_order)}
    plot_nodes['h_index'] = plot_nodes['h_grouping'].map(h_index_map)
    plot_nodes['v_sort_key'] = plot_nodes['v_grouping'].map(
        lambda val: _vertical_sort_key(val, vertical_order_index)
    )

    plot_nodes = plot_nodes.sort_values(
        by=['h_index', 'v_sort_key', 'Name', 'ROI']
    ).reset_index(drop=True)

    vertical_index_map = {}
    for h_group, group_rows in plot_nodes.groupby('h_grouping', sort=False):
        v_order = (
            group_rows[['v_grouping', 'v_sort_key']]
            .drop_duplicates()
            .sort_values(by='v_sort_key')
        )
        vertical_index_map[h_group] = {
            group_name: idx
            for idx, group_name in enumerate(v_order['v_grouping'].tolist())
        }

    plot_nodes['v_index'] = plot_nodes.apply(
        lambda row: vertical_index_map[row['h_grouping']][row['v_grouping']],
        axis=1,
    ).astype(int)
    plot_nodes['v_dup_index'] = plot_nodes.groupby(['h_grouping', 'v_grouping']).cumcount()
    plot_nodes = plot_nodes.drop(columns=['v_sort_key'])

    plot_nodes['vis_shape'] = plot_nodes['DICOM Type'].str.upper().map(shape_map)
    plot_nodes['vis_shape'] = plot_nodes['vis_shape'].fillna(default_vis_shape)
    plot_nodes['mpl_marker'] = plot_nodes['vis_shape'].map(vis_shape_to_marker)

    roi_to_node_meta = {int(row['ROI']): row for _, row in plot_nodes.iterrows()}
    selected_rois = set(roi_to_node_meta.keys())

    relationship_graph = nx.DiGraph()
    for roi, row in roi_to_node_meta.items():
        node_color = node_color_map.get(int(roi), node_fill_color)
        relationship_graph.add_node(
            roi,
            label=row['Name'],
            dicom_type=row['DICOM Type'],
            vis_shape=row['vis_shape'],
            mpl_marker=row['mpl_marker'],
            node_color=node_color,
            node_border_color=darken_color(node_color),
            node_text_color=get_text_color(
                node_color,
                dark_color=font_dark_color,
                light_color=font_light_color,
            ),
            subset=int(row['h_index']),
            h_grouping=row['h_grouping'],
            v_grouping=row['v_grouping'],
            v_index=int(row['v_index']),
            v_dup_index=int(row['v_dup_index']),
        )

    fallback_style = relationship_styles_cfg.get('UNKNOWN', {})
    for source, target, edge_data in structure_set.relationship_graph.edges(data=True):
        if source not in selected_rois or target not in selected_rois:
            continue

        relationship = edge_data.get('relationship')
        if relationship is None:
            continue

        relation_type_obj = getattr(relationship, 'relationship_type', None)
        relation_type = getattr(relation_type_obj, 'relation_type', None) or 'UNKNOWN'

        if relation_type == 'DISJOINT' and not show_disjoint:
            continue
        if relation_type == 'UNKNOWN' and not show_unknown:
            continue
        if hide_logical_edges and bool(getattr(relationship, 'is_logical', False)):
            continue

        style = normalize_edge_style(
            relationship_styles_cfg.get(relation_type, {}),
            fallback_style,
        )
        relation_label = getattr(relation_type_obj, 'label', relation_type)

        relationship_graph.add_edge(
            int(source),
            int(target),
            relation_type=relation_type,
            label=relation_label,
            color=style['color'],
            width=style['width'],
            dashes=style['dashes'],
            arrows=style['arrows'],
        )

    if relationship_graph.number_of_nodes() == 0:
        raise ValueError('No nodes available for plotting')

    positions = {}
    for roi, node_data in relationship_graph.nodes(data=True):
        x_coord = node_data['subset'] * horizontal_spacing
        y_coord = -(
            node_data['v_index'] * vertical_spacing
            + node_data['v_dup_index'] * duplicate_vertical_spread
        )
        positions[roi] = (x_coord, y_coord)

    fig, axis = plt.subplots(figsize=(16, 10))
    axis.set_facecolor(figure_background)

    edge_render_items: list[dict[str, Any]] = []

    for source, target, style_data in relationship_graph.edges(data=True):
        line_style = 'dashed' if style_data['dashes'] else 'solid'
        arrow_setting = style_data.get('arrows')

        edge_to_draw = (source, target)
        arrow_style = '-|>'
        draw_arrows = True

        if arrow_setting is None:
            draw_arrows = False
        elif arrow_setting == 'to;from':
            arrow_style = '<|-|>'
        elif arrow_setting == 'from':
            edge_to_draw = (target, source)

        edge_draw_kwargs = {
            'G': relationship_graph,
            'pos': positions,
            'edgelist': [edge_to_draw],
            'edge_color': style_data['color'],
            'width': style_data['width'],
            'style': line_style,
            'ax': axis,
        }

        if draw_arrows:
            edge_draw_kwargs.update({
                'arrows': True,
                'arrowstyle': arrow_style,
                'arrowsize': 18,
                'connectionstyle': 'arc3,rad=0.08',
            })
        else:
            edge_draw_kwargs.update({'arrows': False})

        edge_artists = nx.draw_networkx_edges(**edge_draw_kwargs)
        edge_render_items.append({
            'source': source,
            'target': target,
            'style_data': style_data,
            'draw_arrows': draw_arrows,
            'edge_to_draw': edge_to_draw,
            'edge_arc_rad': 0.08 if draw_arrows else 0.0,
            'edge_artists': edge_artists,
        })

    # Draw nodes as labeled annotation boxes so shape/style can track config.
    for roi, node_data in relationship_graph.nodes(data=True):
        x_coord, y_coord = positions[roi]
        axis.text(
            x_coord,
            y_coord,
            node_data['label'],
            ha='center',
            va='center',
            fontsize=12,
            color=node_data['node_text_color'],
            bbox={
                'boxstyle': vis_shape_to_boxstyle(node_data['vis_shape']),
                'facecolor': node_data['node_color'],
                'edgecolor': node_data['node_border_color'],
                'linewidth': 2,
                'pad': 0.35,
            },
            zorder=20,
        )

    if show_edge_labels and relationship_graph.number_of_edges() > 0:
        fig.canvas.draw()
        for edge_item in edge_render_items:
            display_vertices = _extract_shaft_vertices(edge_item, axis, positions)
            label_dx, label_dy, label_angle = _path_point_display(display_vertices, 0.5)
            label_x, label_y = axis.transData.inverted().transform((label_dx, label_dy))
            style_data = edge_item['style_data']
            axis.text(
                float(label_x),
                float(label_y),
                style_data['label'],
                color=style_data['color'],
                fontsize=10,
                ha='center',
                va='center',
                rotation=label_angle,
                rotation_mode='anchor',
                bbox={
                    'boxstyle': 'round,pad=0.18',
                    'facecolor': figure_background,
                    'edgecolor': figure_background,
                    'linewidth': 1,
                },
                zorder=7,
            )

    node_legend_handles = []
    for dicom_type, type_rows in plot_nodes.groupby('DICOM Type'):
        vis_shape = type_rows['vis_shape'].iloc[0]
        sample_roi = int(type_rows['ROI'].iloc[0])
        sample_color = node_color_map.get(sample_roi, node_fill_color)
        node_legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=vis_shape_to_marker(vis_shape),
                color='w',
                markerfacecolor=sample_color,
                markeredgecolor=darken_color(sample_color),
                markersize=9,
                label=f'{dicom_type} ({vis_shape})',
            )
        )

    edge_legend_handles = []
    present_relations = sorted(
        {
            style_data['relation_type']
            for _, _, style_data in relationship_graph.edges(data=True)
        }
    )
    for relation_name in present_relations:
        relation_style = normalize_edge_style(
            relationship_styles_cfg.get(relation_name, {}),
            fallback_style,
        )
        edge_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=relation_style['color'],
                linestyle='--' if relation_style['dashes'] else '-',
                linewidth=relation_style['width'],
                label=relation_name,
            )
        )

    if node_legend_handles:
        node_legend = axis.legend(
            handles=node_legend_handles,
            title='Node shapes (diagram settings)',
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )
        axis.add_artist(node_legend)

    if edge_legend_handles:
        axis.legend(
            handles=edge_legend_handles,
            title='Edge styles (diagram settings)',
            loc='upper left',
            bbox_to_anchor=(1.02, 0.42),
            frameon=True,
        )

    axis.set_title(
        'Target Structure Relationships\n'
        'Horizontal: h_grouping | Vertical: v_grouping'
    )
    axis.axis('off')
    plt.tight_layout()

    if show_plot:
        plt.show()

    relationship_counts = Counter(
        style_data['relation_type']
        for _, _, style_data in relationship_graph.edges(data=True)
    )
    count_series = pd.Series(relationship_counts).sort_index()

    return DiagramRenderResult(
        fig=fig,
        axis=axis,
        relationship_graph=relationship_graph,
        edge_render_items=edge_render_items,
        positions=positions,
        plot_nodes=plot_nodes,
        relationship_counts=count_series,
        figure_background=figure_background,
    )
