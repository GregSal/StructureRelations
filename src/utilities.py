'''Utility Functions'''
# %% Imports
# Type imports

from typing import List, Tuple, Dict

# Standard Libraries
from math import sin, cos, radians


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


# %% Contour Creation Functions
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


def make_slice_list(number: int, start: float = 0.0, spacing: float = 0.1):
    slices = [round(SliceIndex(num*spacing + start), PRECISION)
              for num in range(number)]
    return slices


def make_sphere():
    pass


def make_vertical_cylinder():
    pass

def make_horizontal_cylinder():
    pass

def make_box():
    pass



def make_contour_slices(roi_num: ROI_Num, slices: List[SliceIndex],
                        contours: List[Contour]):
    data_list = []
    for slice_idx in slices:
        data_item = {
            'ROI Num': roi_num,
            'Slice Index': SliceIndex(slice_idx),
            'Structure Slice': StructureSlice(contours)
            }
        data_list.append(data_item)
    slice_data = pd.DataFrame(data_list)
    slice_data.set_index(['ROI Num', 'Slice Index'], inplace=True)
    return slice_data


def make_slice_table(slice_data: pd.DataFrame)->pd.DataFrame:
    slice_table = slice_data.unstack('ROI Num')
    slice_table.columns = slice_table.columns.droplevel()
    return slice_table


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
