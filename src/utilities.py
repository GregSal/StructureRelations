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

from types_and_classes import StructureSlice, relate



# %%| Type definitions and Globals
ROI_Num = int  # Index to structures defined in Structure RT DICOM file
SliceIndex = float
Contour = shapely.Polygon
StructurePair =  Tuple[ROI_Num, ROI_Num]


# Global Settings
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


# %% Contour Creation Functions
def make_slice_list(number: int, start: float = 0.0, spacing: float = 0.1):
    slices = [round(SliceIndex(num*spacing + start), PRECISION)
              for num in range(number)]
    return slices


def make_slice_table(slice_data: pd.DataFrame)->pd.DataFrame:
    slice_table = slice_data.unstack('ROI Num')
    slice_table.columns = slice_table.columns.droplevel()
    return slice_table


def circle_points(radius: float, offset_x: float = 0, offset_y: float = 0,
                  num_points: int = 16, precision=3)->list[tuple[float, float]]:
    deg_step = radians(360/num_points)
    degree_points = np.arange(stop=radians(360), step=deg_step)
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
            y=np.nan
        else:
            y_pos = round(offset_y + y, precision)
            y_neg = round(offset_y - y, precision)
        coords_pos.append((x, y_pos))
        coords_neg.append((x, y_neg))
    coords_neg.reverse()
    coords = coords_pos + coords_neg[1:]
    return coords


def box_points(width: float, height: float = None, offset_x: float = 0,
               offset_y: float = 0) -> list[tuple[float, float]]:
    x1_unit = width / 2
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
    start_slice = - radius
    z_coord = make_slice_list(number_slices, start_slice, spacing)
    r_coord = circle_x_points(radius, z_coord, offset_z, precision=precision+1)
    # Generate circle for each slices
    slice_data = {}
    for slice_idx, radius in r_coord:
        slice_points = circle_points(radius, offset_x, offset_y, num_points,
                                      precision)
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
    number_slices = ceil(length / spacing)
    z_coord = make_slice_list(number_slices, offset_z, spacing)
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


# %%|
def make_horizontal_cylinder():
    pass

def make_box():
    pass

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

# %% Test plot function not working

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import numpy as np
#from scipy.interpolate import griddata
#
#def plot_3d_surface(x, y, z):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#
#    # Convert lists to numpy arrays
#    x = np.array(x)
#    y = np.array(y)
#    z = np.array(z)
#
#    # Create grid data for the surface plot
#    xi = np.linspace(x.min(), x.max(), 100)
#    yi = np.linspace(y.min(), y.max(), 100)
#    xi, yi = np.meshgrid(xi, yi)
#    zi = griddata((x, y), z, (xi, yi), method='cubic')
#
#    # Plot the surface
#    surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
#
#    ax.set_xlabel('X axis')
#    ax.set_ylabel('Y axis')
#    ax.set_zlabel('Z axis')
#
#    plt.show()







# %% Retired functions
# slice_table = slice_data.index.to_frame()
# slice_table = slice_table['Slice Index'].unstack('ROI Num')
# contour_slices = slice_table.apply(slice_spacing)


# def build_slice_table(contour_sets)->pd.DataFrame:
#    def form_table(slice_index):
#        slice_index.reset_index(inplace=True)
#        slice_index.sort_values('Slice', inplace=True)
#        slice_index.set_index(['Slice','StructureID'], inplace=True)
#        slice_table = slice_index.unstack()
#        slice_table.columns = slice_table.columns.droplevel()
#        return slice_table
#
#    slice_index = build_contour_index(contour_sets)
#    slice_table = form_table(slice_index)
#    contour_slices = slice_table.apply(slice_spacing)
#    return contour_slices
#
#
# def build_contour_index(contour_sets: Dict[int, ContourSet])->pd.DataFrame:
#    '''Build an index of structures in a contour set.
#
#    The table columns contain the structure names, the ROI number, and the
#    slice positions where the contours for that structure are located.  There
#    is one row for each slice and structure on that slice.  Multiple contours
#    for a single structure on a given slice, have only one row in teh contour
#    index
#
#    Args:
#        contour_sets (Dict[int, RS_DICOM_Utilities.ContourSet]): A dictionary
#            of structure data.
#
#    Returns:
#        pd.DataFrame: An index of structures in a contour set indication which
#            slices contains contours for each structure.
#    '''
#    slice_ref = {}
#    name_ref = {}
#    for structure in contour_sets.values():
#        slice_ref[structure.roi_num] = list(structure.contours.keys())
#        name_ref[structure.roi_num] = structure.structure_id
#    slice_seq = pd.Series(slice_ref).explode()
#    slice_seq.name = 'Slice'
#    name_lookup = pd.Series(name_ref)
#    name_lookup.name = 'StructureID'
#    slice_lookup = pd.DataFrame(name_lookup).join(slice_seq, how='outer')
#    return slice_lookup
#
