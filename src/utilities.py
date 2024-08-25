'''Utility Functions'''
# %% Imports
# Type imports

from itertools import chain
from typing import List, Tuple, Dict

# Standard Libraries
from math import ceil, sin, cos, radians, sqrt


# Shared Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line

from types_and_classes import ROI_Num, SliceIndex, Contour, StructurePair, poly_round
from types_and_classes import InvalidContour

from types_and_classes import StructureSlice, relate

# Global Default Settings
PRECISION = 3


# %% Tuples to Strings
def colour_text(roi_colour):
    colour_fmt = ''.join([
        f'({roi_colour[0]:0d}, ',
        f'{roi_colour[1]:0d}, ',
        f'{roi_colour[2]:0d})'
        ])
    return colour_fmt


def com_text(com):
    com_fmt = ''.join([
        f'({com[0]:-5.2f}, ',
        f'{com[1]:-5.2f}, ',
        f'{com[2]:-5.2f})'
        ])
    return com_fmt


# %% Debugging display functions
def bin_format(bin_val: int):
    bin_str = bin(bin_val)
    if len(bin_str) < 29:
        zero_pad = 29 - len(bin_str)
        bin_str = bin_str[0:2] + '0' * zero_pad + bin_str[2:]
    bin_fmt = '{bin1:^11s} | {bin2:^11s} | {bin3:^11s}'
    bin_dict = {
        'bin1': bin_str[2:11],
        'bin2': bin_str[11:20],
        'bin3': bin_str[20:29]
        }
    return bin_fmt.format(**bin_dict)


def plot_ab(poly_a, poly_b):
    def plot_geom(ax, geom, color='black'):
        if isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            plot_polygon(geom, ax=ax, add_points=False, color=color, facecolor=color)
        elif isinstance(geom, (shapely.LineString, shapely.MultiLineString,
                               shapely.LinearRing, shapely.LinearRing)):
            plot_line(geom, ax=ax, add_points=False, color=color)
        elif isinstance(geom, shapely.GeometryCollection):
            # plot each of the geometry objects in the collection
            for g in geom.geoms:
                plot_geom(ax, g, color)

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
    plt.show()


def plot_roi(slice_table, roi_list: List[int]):
    def make_array(slice_table, roi_num: List[int]):
        all_points = []
        for slice, structure_slice in slice_table[roi_num].dropna().items():
            poly = structure_slice.contour
            points = [tuple(p) for p in chain(shapely.get_coordinates(poly))]
            xy = np.array(points)
            num_points = np.size(xy, 0)
            z = np.ones((num_points, 1)) * slice
            xyz = np.concatenate([xy, z], axis=1)
            all_points.append(xyz)
        point_array = np.concatenate(all_points, axis=0)
        return point_array

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    color_list = ['red', 'blue', 'green', 'yellow']
    for i, roi in enumerate(roi_list):
        data = make_array(slice_table, roi)
        ax.plot(data[:,0], data[:,1], data[:,2], linestyle='none', marker='.',
                color=color_list[i])
    plt.show()


def c_type(obj):
    if isinstance(obj, StructureSlice):
        n = str(type(obj.contour))
        s = n.replace('shapely.geometry.', '')
    else:
        s = str(type(obj))
    s = s.replace('<class ', '')
    s = s.replace('>', '')
    return s


def type_table(sr):
    def f(ss):
        if isinstance(ss, StructureSlice):
            obj_str = c_type(ss.contour)
            type_str = '\n'.join(c_type(poly) for poly in ss.contour.geoms)
            return type_str
        return c_type(ss)

    type_str = sr.map(f)
    lbl_dict = {idx: type_lbl for idx, type_lbl in type_str.items()}
    return lbl_dict


# %% Slice related functions
def make_slice_list(height: float = None, number_slices: int = None,
                    start: float = 0.0, spacing: float = 0.1,
                    precision=PRECISION):
    if height:
        if number_slices:
            spacing = height / number_slices
        else:
            number_slices = ceil(height / spacing)
    elif not number_slices:
        msg = 'At least one of height or number_slices must be specified.'
        raise ValueError(msg)
    slices = [round(SliceIndex(num*spacing + start), precision)
              for num in range(number_slices)]
    return slices


def slice_spacing(contour):
    # Index is the slice position of all slices in the image set
    # Columns are structure IDs
    # Values are the distance (INF) to the next contour
    inf = contour.dropna().index.min()
    sup = contour.dropna().index.max()
    contour_range = (contour.index <= sup) & (contour.index >= inf)
    slices = contour.loc[contour_range].dropna().index.to_series()
    gaps = slices.shift(-1) - slices
    return gaps


def make_slice_table(slice_data: pd.DataFrame, ignore_errors=False)->pd.DataFrame:
    def merge_contours(slice_contours: pd.DataFrame, ignore_errors=False):
        ranked_contours = slice_contours.sort_values('Area', ascending=False)
        try:
            structure_slice = StructureSlice(list(ranked_contours.Contour),
                                             ignore_errors=ignore_errors)
        except InvalidContour as err:
            msg = err.__str__()
            roi_num = ranked_contours.index[0][0]
            slice_idx = ranked_contours.index[0][1]
            print(f'{msg}\t for ROI: {roi_num} on slice: {slice_idx}')
            structure_slice = None
        return structure_slice

    sorted_data = slice_data.sort_index(level=['Slice Index', 'ROI Num']).copy()
    sorted_data['Area'] = sorted_data.map(lambda x: x.area)
    structure_group = sorted_data.groupby(level=['Slice Index', 'ROI Num'])
    structure_data = structure_group.apply(merge_contours,
                                           ignore_errors=ignore_errors)
    slice_table = structure_data.unstack('ROI Num')
    return slice_table


def build_slice_spacing_table(slice_table)->pd.DataFrame:
    # Find distance between slices with contours
    def get_slices(structure: pd.Series):
        used_slices = structure.dropna().index.to_series()
        return used_slices

    contour_slices = slice_table.apply(get_slices)
    slice_spacing_data = contour_slices.apply(slice_spacing)
    return slice_spacing_data


def neighbouring_slice(slice_index, missing_slices, shift_direction=1,
                       shift_start=0):
    ref = slice_index[missing_slices]
    ref_missing = list(missing_slices)
    shift_size = shift_start
    while ref_missing:
        shift_size += shift_direction
        shift_slice = slice_index.shift(shift_size)[missing_slices]
        ref_idx = ref.isin(ref_missing)
        ref[ref_idx] = shift_slice[ref_idx]
        ref_missing = list(set(ref) & set(missing_slices))
        ref_missing.sort()
    return ref

def find_neighbouring_slice(structure_slices):
    slice_index = structure_slices.index.to_series()
    missing_slices = slice_index[structure_slices.isna()]
    z_neg = neighbouring_slice(slice_index, missing_slices, shift_direction=-1)
    z_pos = neighbouring_slice(slice_index, missing_slices, shift_direction=1)
    ref = pd.concat([z_pos, z_neg], axis='columns')
    ref.columns = ['z_pos', 'z_neg']
    return ref


# %% Contour Creation Functions
def circle_points(radius: float, offset_x: float = 0, offset_y: float = 0,
                  num_points: int = 16, precision=3)->list[tuple[float, float]]:
    deg_step = radians(360/num_points)
    degree_points = np.arange(stop=radians(360), step=deg_step)
    if radius == 0:
        radius = 10**(-precision)
    x_coord = np.array([round(radius*sin(d), precision) for d in degree_points])
    y_coord = np.array([round(radius*cos(d), precision) for d in degree_points])
    x_coord = x_coord + offset_x
    y_coord = y_coord + offset_y
    coords = [(x,y) for x,y in zip(x_coord,y_coord)]
    return coords


def circle_x_points(radius: float, x_list: List[float],
                    offset_x: float = 0, offset_y: float = 0,
                    precision=3)->list[tuple[float, float]]:
    coords_pos = []
    coords_neg = []
    for x in x_list:
        x0 = x - offset_x
        try:
            y = sqrt(radius**2 - x0**2)
        except ValueError:
            y = 10**(-precision)
        else:
            y_pos = round(offset_y + y, precision)
            y_neg = round(offset_y - y, precision)
        coords_pos.append((x, y_pos))
        coords_neg.append((x, y_neg))
    coords_neg.reverse()
    coords = coords_pos + coords_neg[1:]
    return coords


def box_points(width: float, height: float = None, offset_x: float = 0,
               offset_y: float = 0, precision=3) -> list[tuple[float, float]]:
    x1_unit = width / 2
    if x1_unit == 0:
        x1_unit = 10**(-precision)
    if not height:
        y1_unit = x1_unit
    else:
        y1_unit = height / 2
    coords = [
        ( x1_unit + offset_x,  y1_unit + offset_y),
        ( x1_unit + offset_x, -y1_unit + offset_y),
        (-x1_unit + offset_x, -y1_unit + offset_y),
        (-x1_unit + offset_x,  y1_unit + offset_y)
        ]
    return coords


def sphere_points(radius: float, spacing: float = 0.1, num_points: int = 16,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
                precision=3)->Dict[SliceIndex, tuple[float, float]]:
    number_slices = ceil(radius * 2 / spacing) + 1
    start_slice = offset_z - radius
    z_coord = make_slice_list(number_slices=number_slices, spacing=spacing,
                              start=start_slice, precision=precision)
    r_coord = circle_x_points(radius, z_coord, offset_z, precision=precision+1)
    # Generate circle for each slices
    slice_data = {}
    for slice_idx, radius in r_coord:
        slice_points = circle_points(radius, offset_x, offset_y, num_points,
                                      precision)
        slice_data[SliceIndex(slice_idx)] = slice_points
    return slice_data


def cylinder_points(radius: float, length: float, spacing: float = 0.1,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
                precision=3)->Dict[SliceIndex, tuple[float, float]]:
    number_slices = ceil(radius * 2 / spacing) + 1
    start_slice = offset_z - radius
    z_coord = make_slice_list(number_slices=number_slices, spacing=spacing,
                              start=start_slice, precision=precision)
    r_coord = circle_x_points(radius, z_coord, offset_z, precision=precision)
    # Generate circle for each slices
    slice_data = {}
    for slice_idx, r in r_coord:
        slice_points = box_points(r, length, offset_x, offset_y, precision)
        slice_data[SliceIndex(slice_idx)] = slice_points
    return slice_data


def make_sphere(radius: float, spacing: float = 0.1, num_points: int = 16,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
                precision=3, roi_num=0)->pd.DataFrame:
    slice_list = []
    points_dict = sphere_points(radius, spacing, num_points,
                                offset_x, offset_y, offset_z, precision)
    for slice_idx, xy_points in points_dict.items():
        slice_contour = shapely.Polygon(xy_points)
        roi_slice = {'ROI Num': roi_num,
                     'Slice Index': SliceIndex(slice_idx),
                     'Contour': slice_contour}
        slice_list.append(roi_slice)
    slice_contours = pd.DataFrame(slice_list)
    slice_contours.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_contours


def make_vertical_cylinder(radius: float, length: float, spacing: float = 0.1,
                           num_points: int = 16, offset_x: float = 0,
                           offset_y: float = 0, offset_z: float = 0,
                           precision=PRECISION, roi_num=0)->pd.DataFrame:
    z_coord = make_slice_list(height=length, spacing=spacing, start=offset_z,
                              precision=precision)
    xy_points = circle_points(radius, offset_x, offset_y, num_points, precision)
    contour = shapely.Polygon(xy_points)
    slice_list = []
    for slice_idx in z_coord:
        roi_slice = {'ROI Num': roi_num,
                     'Slice Index': SliceIndex(slice_idx),
                     'Contour': contour}
        slice_list.append(roi_slice)
    slice_contours = pd.DataFrame(slice_list)
    slice_contours.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_contours


def make_horizontal_cylinder(radius: float, length: float, spacing: float = 0.1,
                             offset_x: float = 0, offset_y: float = 0,
                             offset_z: float = 0, precision=PRECISION,
                             roi_num=0)->pd.DataFrame:
    slice_list = []
    points_dict = cylinder_points(radius, length, spacing,
                                   offset_x, offset_y, offset_z, precision)
    for slice_idx, xy_points in points_dict.items():
        slice_contour = shapely.Polygon(xy_points)
        roi_slice = {'ROI Num': roi_num,
                     'Slice Index': SliceIndex(slice_idx),
                     'Contour': slice_contour}
        slice_list.append(roi_slice)
    slice_contours = pd.DataFrame(slice_list)
    slice_contours.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_contours


def make_box(width: float, length: float, height: float, spacing: float = 0.1,
             offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
             precision=PRECISION, roi_num=0)->pd.DataFrame:
    z_coord = make_slice_list(height=height, spacing=spacing, start=offset_z,
                              precision=precision)
    xy_points = box_points(width, length, offset_x, offset_y, precision)
    contour = shapely.Polygon(xy_points)
    slice_list = []
    for slice_idx in z_coord:
        roi_slice = {'ROI Num': roi_num,
                     'Slice Index': SliceIndex(slice_idx),
                     'Contour': contour}
        slice_list.append(roi_slice)
    slice_contours = pd.DataFrame(slice_list)
    slice_contours.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_contours


def make_contour_slices(shape: shapely.Polygon, spacing: float = 0.1,
                        height: float = None, number_slices: int = None,
                        offset_z: float = 0, precision=PRECISION,
                        roi_num=0)->pd.DataFrame:
    z_coord = make_slice_list(height=height, number_slices=number_slices,
                              spacing=spacing, start=offset_z,
                              precision=precision)
    contour = poly_round(shape, precision)
    slice_list = []
    for slice_idx in z_coord:
        roi_slice = {'ROI Num': roi_num,
                     'Slice Index': SliceIndex(slice_idx),
                     'Contour': contour}
        slice_list.append(roi_slice)
    slice_contours = pd.DataFrame(slice_list)
    slice_contours.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_contours
