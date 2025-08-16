'''Utility Functions'''
# %% Imports
# Type imports

from itertools import chain
from typing import List, Dict

# Standard Libraries
from math import ceil, sin, cos, radians, sqrt


# Shared Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon, plot_line

from types_and_classes import PRECISION, SliceIndexType
from contours import ContourPoints
from region_slice import RegionSlice




# %% Tuples to Strings
def colour_text(roi_colour):
    '''Format the ROI colour as a string.
    '''
    colour_fmt = ''.join([
        f'({roi_colour[0]:0d}, ',
        f'{roi_colour[1]:0d}, ',
        f'{roi_colour[2]:0d})'
        ])
    return colour_fmt


def com_text(com):
    '''Format the centre of mass as a string.
    '''
    com_fmt = ''.join([
        f'({com[0]:-5.2f}, ',
        f'{com[1]:-5.2f}, ',
        f'{com[2]:-5.2f})'
        ])
    return com_fmt


def to_str(relation_int: int, size=27)->str:
    '''Convert an integer to a binary string of the specified size.
    The integer must be less than 2**size. If the integer is larger than
    2**size, a ValueError is raised.
    Args:
        relation_int (int): The integer to convert to a binary string.
        size (int, optional): The number of bits in the binary string.
            Defaults to 27.
    Returns:
        str: The binary string representation of the integer.
    Raises:
        ValueError: If the integer is larger than 2**size.
    '''
    str_size = size + 2  # Accounts for '0b' prefix.
    bin_str = bin(relation_int)
    if len(bin_str) < str_size:
        zero_pad = str_size - len(bin_str)
        bin_str = '0' * zero_pad + bin_str[2:]
    elif len(bin_str) > str_size:
        raise ValueError(''.join([
            'The input integer must be {size} bits long. The input integer ',
            'was: ', f'{len(bin_str) - 2}'
            ]))
    else:
        bin_str = bin_str[2:]
    return bin_str


# %% Debugging display functions
def bin_format(bin_val: int, ignore_errors=False):
    '''Format a binary value as a string with three 9-bit sections.

    Args:
        bin_val (int): The binary value to format.
        ignore_errors (bool, optional): If True, ignore errors and return 'Error'
            if the input is not an integer. Defaults to False.

    Returns:
        str: The formatted binary string.
    '''
    if np.isnan(bin_val):
        return ''
    try:
        bin_val = int(bin_val)
    except ValueError as err:
        if ignore_errors:
            return 'Error'
        raise ValueError('bin_val must be an integer.') from err
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

def bin2matrix(bin_val: int):
    '''Convert a binary value to a 3x3 matrix string.

    Args:
        bin_val (int): The binary value to convert.

    Returns:
        str: A string representation of the binary value as a 3x3 matrix.
    '''
    bin_str = to_str(bin_val)
    if len(bin_str) < 9:
        zero_pad = 9 - len(bin_str)
        bin_str = '0' * zero_pad + bin_str[2:]
    bin_fmt = '|{bin1}|\n|{bin2}|\n|{bin3}|'
    bin_dict = {'bin1': bin_str[0:3],
                'bin2': bin_str[3:6],
                'bin3': bin_str[6:9]}
    return bin_fmt.format(**bin_dict)


def show_bounds(polygon):
    '''Display the bounds of a polygon as a DataFrame.
    '''
    a = polygon.bounds
    b = pd.DataFrame([[a[0], a[1]],[a[2], a[3]]],
             columns=['x', 'y'],
             index=['min', 'max'])
    return b


def plot_ab(poly_a, poly_b, add_axis=True):
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
    plt.show()
    return ax


def plot_roi(slice_table, roi_list: List[int]):
    '''Plot the contours of the specified ROIs in 3D.

    This function plots the contours of the specified ROIs in 3D. Each ROI is
    plotted in a different color. The contours are plotted as points in 3D space.

    Args:
        slice_table (pd.DataFrame): The table of slices containing the contours.
        roi_list (List[int]): A list of ROI numbers to plot.
    '''
    def make_array(slice_table, roi_num: List[int]):
        all_points = []
        for slice_idx, structure_slice in slice_table[roi_num].dropna().items():
            poly = structure_slice.contour
            points = [tuple(p) for p in chain(shapely.get_coordinates(poly))]
            xy = np.array(points)
            num_points = np.size(xy, 0)
            z = np.ones((num_points, 1)) * slice_idx
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
def make_slice_list(height: float = None, number_slices: int = None,
                    start: float = 0.0, spacing: float = 0.1,
                    precision=PRECISION) -> List[SliceIndexType]:
    '''Generate a list of SliceIndexType with the desired range and increment.

    Height or number_slices must be provided.
    If height is supplied then then calculate the spacing from
            height / number_slices
        or calculate the number of slices from
            height / spacing
    Args:
        height (float, optional): The distance in cm from start to end.
            If None, number_slices is required. Defaults to None.
        number_slices (int, optional): The number of slices to include in the
            list. If None, height is required. Defaults to None.
        start (float, optional): The lowest SliceIndexType value. Defaults to 0.0.
        spacing (float, optional): The gap between slices. Not used if both
            height and number_slices are provided. Defaults to 0.1.
        precision (int, optional): SliceIndexType values are rounded to this number
            of decimal places.precision. Defaults to PRECISION (3).

    Raises:
        ValueError: if neither height nor number_slices is provided.

    Returns:
        List[SliceIndexType]: A list of SliceIndexType for testing purposes.
    '''
    if height:
        # number of slices = number of spaces + 1
        if number_slices:
            spacing = height / (number_slices - 1) # Subtract one to include the end
        else:
            number_slices = ceil(height / spacing) + 1  # Add one to include the end
    elif not number_slices:
        msg = 'At least one of height or number_slices must be specified.'
        raise ValueError(msg)
    slices = [round(SliceIndexType(num*spacing + start), precision)
              for num in range(number_slices)]
    return slices


def circle_points(radius: float, offset_x=0.0, offset_y=0.0, num_points=16,
                  precision=3, z: float = None)->list[tuple[float, float]]:
    '''Generate points for a circle with the specified radius.

    The circle is centered at (offset_x, offset_y) and the coordinates are
    rounded to the specified precision. If z is not None, the points will be
    3D points with the specified z coordinate. The points are at equal
    intervals around the circle.

    Args:
        radius (float): The radius of the circle.
        offset_x (float, optional): The x position of the center of the circle.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the circle.
            Defaults to 0.
        num_points (int, optional): The number of points to generate for the
            circle. Defaults to 16.
        precision (int, optional): The number of decimal points to use when
            rounding the coordinates. Defaults to 3.
        z (float, optional): The z coordinate of the points. If None, the points
            will be 2D points. Defaults to None.

    Returns:
        list[tuple[float, float]]: A list of tuples containing the x and y
            coordinates of the points on the circle. If z is not None, the
            tuples will contain the z coordinate as well.
    '''
    deg_step = radians(360/num_points)
    degree_points = np.arange(stop=radians(360), step=deg_step)
    if radius == 0:
        radius = 10**(-precision)
    x_coord = np.array([round(radius*sin(d), precision) for d in degree_points])
    y_coord = np.array([round(radius*cos(d), precision) for d in degree_points])
    x_coord = x_coord + offset_x
    y_coord = y_coord + offset_y
    if z is not None:
        z = float(z)
        coords = [(x,y,z) for x,y in zip(x_coord,y_coord)]
    else:
        coords = list(zip(x_coord,y_coord))
    return coords


def circle_x_points(radius: float, x_list: List[float], offset_x=0, offset_y=0,
                    precision=3)->list[tuple[float, float]]:
    '''Generate points for a circle with the specified radius and x coordinates.

    Args:
        radius (float): The radius of the circle.
        x_list (List[float]): A list of x coordinates for the circle.
        offset_x (float, optional): The x position of the center of the circle.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the circle.
            Defaults to 0.
        precision (int, optional): The number of decimal points to use when
            rounding the coordinates. Defaults to 3.

    Returns:
        list[tuple[float, float]]: A list of tuples containing the x and y
            coordinates of the points on the circle.
    '''
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


def box_points(width: float, height: float = None, offset_x=0.0, offset_y=0.0,
               precision=3) -> list[tuple[float, float]]:
    '''Generate points for a rectangle with the specified width and height.

    The rectangle is centered at (offset_x, offset_y) and the coordinates are
    rounded to the specified precision. If height is not supplied, it is set to
    the same value as width. The points are in the order: top left, bottom left,
    bottom right, top right.

    Args:
        width (float): The width of the rectangle.
        height (float, optional): The height of the rectangle. If None, height
            is set to width. Defaults to None.
        offset_x (float, optional): The x position of the center of the rectangle.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the rectangle.
            Defaults to 0.
        precision (int, optional): The number of decimal points to use when
            rounding the coordinates. Defaults to 3.

    Returns:
        list[tuple[float, float]]: A list of tuples containing the x and y
            coordinates of the points on the rectangle.
    '''
    x1_unit = width / 2
    if x1_unit == 0:
        x1_unit = 10**(-precision)
    if height is None:
        y1_unit = x1_unit
    else:
        if height == 0:
            y1_unit = 10**(-precision)
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
                precision=3)->Dict[SliceIndexType, tuple[float, float]]:
    '''Generate points for a sphere with the specified radius.

    The center of the sphere is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified. The sphere is sliced
    into horizontal slices at the specified spacing. The points are at equal
    intervals around the circle. The z coordinate of the points is the slice
    index. The points are returned as a dictionary with the slice index as the
    key and a tuple of (x, y) coordinates as the value.

    Args:
        radius (float): The radius of the sphere.
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        num_points (int, optional): The number of points to use for each contour
            (polygon). Defaults to 16.
        offset_x (float, optional): The x position of the center of the sphere.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the sphere.
            Defaults to 0.
        offset_z (float, optional): The z position of the center of the sphere.
            Defaults to 0.
        precision (int, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.
    Returns:
        Dict[SliceIndexType, tuple[float, float]]: A dictionary containing the
            slice index as the key and a tuple of (x, y) coordinates as the value.
    '''
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
        slice_data[SliceIndexType(slice_idx)] = slice_points
    return slice_data


def cylinder_points(radius: float, length: float, spacing: float = 0.1,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
                precision=3)->Dict[SliceIndexType, tuple[float, float]]:
    '''Generate points for a cylinder with the specified radius and length.

    The center of the cylinder is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified. The cylinder is sliced
    into horizontal slices at the specified spacing. The points are at equal
    intervals around the circle. The z coordinate of the points is the slice
    index. The points are returned as a dictionary with the slice index as the
    key and a tuple of (x, y) coordinates as the value.

    Args:
        radius (float): The radius of the cylinder.
        length (float): The length of the cylinder in the z direction.
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        offset_x (float, optional): The x position of the center of the cylinder.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the cylinder.
            Defaults to 0.
        offset_z (float, optional): The z position of the center of the cylinder.
            Defaults to 0.
        precision (int, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.

    Returns:
        Dict[SliceIndexType, tuple[float, float]]: A dictionary containing the
            slice index as the key and a tuple of (x, y) coordinates as the value.
    '''
    number_slices = ceil(radius * 2 / spacing) + 1
    start_slice = offset_z - radius
    z_coord = make_slice_list(number_slices=number_slices, spacing=spacing,
                              start=start_slice, precision=precision)
    r_coord = circle_x_points(radius, z_coord, offset_z, precision=precision)
    # Generate circle for each slices
    slice_data = {}
    for slice_idx, r in r_coord:
        slice_points = box_points(length, r, offset_x, offset_y, precision)
        slice_data[SliceIndexType(slice_idx)] = slice_points
    return slice_data


def make_sphere(radius: float, spacing: float = 0.1, num_points: int = 16,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
                precision=3, roi_num=0)->List[ContourPoints]:
    '''Generate contour slices for a sphere.

    The center of the sphere is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified.

    Args:
        radius (float): The radius of the sphere.
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        num_points (int, optional): The number of points to use for each contour
            (polygon). Defaults to 16.
        offset_x (float, optional): The x position of the center of the sphere.
            Defaults to 0.
        offset_y (float, optional): The y position of the center of the sphere.
            Defaults to 0.
        offset_z (float, optional): The y position of the center of the sphere.
            Defaults to 0.
        precision (int, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.
        roi_num (int, optional): Thr structure index number. Defaults to 0.

    Returns:
        List[ContourPoints]: A list of dictionaries containing the roi,
            slice index and list of points delimiting the sphere on that slice.
    '''
    slice_list = []
    points_dict = sphere_points(radius, spacing, num_points,
                                offset_x, offset_y, offset_z, precision)
    for slice_idx, xy_points in points_dict.items():
        roi_slice = ContourPoints(xy_points, roi_num, slice_idx)
        slice_list.append(roi_slice)
    return slice_list


def make_vertical_cylinder(radius: float, length: float, spacing: float = 0.1,
                           num_points: int = 16, offset_x: float = 0,
                           offset_y: float = 0, offset_z: float = 0,
                           precision=PRECISION, roi_num=0)->List[ContourPoints]:
    '''Generate contour slices for a vertical cylinder.

    The center of the cylinder is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified.

    Args:
        radius (float): The radius of the cylinder.
        length (float): The length of the cylinder in the z direction
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        num_points (int, optional): The number of points to use for each contour
            (polygon). Defaults to 16.
        offset_x (float, optional): The x position of the center of the
            cylinder. Defaults to 0.
        offset_y (float, optional): The y position of the center of the
            cylinder. Defaults to 0.
        offset_z (float, optional): The z position of the center of the
            cylinder. Defaults to 0.
        precision (_type_, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.
        roi_num (int, optional): Thr structure index number. Defaults to 0.

    Returns:
        List[ContourPoints]: A list of dictionaries containing the roi,
            slice index and list of points delimiting the cylinder on that slice.
    '''
    starting_z = offset_z - length / 2
    z_coord = make_slice_list(height=length, spacing=spacing, start=starting_z,
                              precision=precision)
    xy_points = circle_points(radius, offset_x, offset_y, num_points, precision)
    slice_list = []
    for slice_idx in z_coord:
        roi_slice = ContourPoints(xy_points, roi_num, slice_idx)
        slice_list.append(roi_slice)
    return slice_list


def make_horizontal_cylinder(radius: float, length: float, spacing: float = 0.1,
                             offset_x: float = 0, offset_y: float = 0,
                             offset_z: float = 0, precision=PRECISION,
                             roi_num=0)->List[ContourPoints]:
    '''Generate contour slices for a horizontal cylinder.

    The center of the cylinder is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified.

    Args:
        radius (float): The radius of the cylinder.
        length (float): The length of the cylinder in the z direction
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        num_points (int, optional): The number of points to use for each contour
            (polygon). Defaults to 16.
        offset_x (float, optional): The x position of the center of the
            cylinder. Defaults to 0.
        offset_y (float, optional): The y position of the center of the
            cylinder. Defaults to 0.
        offset_z (float, optional): The z position of the center of the
            cylinder. Defaults to 0.
        precision (_type_, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.
        roi_num (int, optional): Thr structure index number. Defaults to 0.

    Returns:
        List[ContourPoints]: A list of dictionaries containing the roi,
            slice index and list of points delimiting the cylinder on that slice.
    '''
    slice_list = []
    points_dict = cylinder_points(radius, length, spacing,
                                   offset_x, offset_y, offset_z, precision)
    for slice_idx, xy_points in points_dict.items():
        roi_slice = ContourPoints(xy_points, roi_num, slice_idx)
        slice_list.append(roi_slice)
    return slice_list


def make_box(width: float, length: float = None, height: float = None,
             offset_x: float = 0, offset_y: float = 0, offset_z: float = 0,
             spacing: float = 0.1, precision=PRECISION,
             roi_num=0)->List[ContourPoints]:
    '''Generate contour slices for a rectangular prism cylinder.

    The center of the box is at (offset_x, offset_y, offset_z). The
    coordinates are rounded to the precision specified.

    Args:
        width (float): The x dimension of the box.
        length (float): The y dimension of the box. If length is not supplied,
            length=width.
        height (float): The z dimension of the box.  If height is not supplied,
            height=width.
        spacing (float, optional): The spacing of the slices. Defaults to 0.1.
        num_points (int, optional): The number of points to use for each contour
            (polygon). Defaults to 16.
        offset_x (float, optional): The x position of the center of the
            box. Defaults to 0.
        offset_y (float, optional): The y position of the center of the
            box. Defaults to 0.
        offset_z (float, optional): The z position of the center of the
            box. Defaults to 0.
        precision (_type_, optional): The number of decimal points to use when
            rounding the polygon coordinates. Defaults to 3.
        roi_num (int, optional): Thr structure index number. Defaults to 0.

    Returns:
        List[ContourPoints]: A list of dictionaries containing the roi,
            slice index and list of points delimiting the box on that slice.
    '''
    if not height:
        if height == 0:
            height = 10**(-precision)
        else:
            height = width
    starting_z = offset_z - height / 2
    z_coord = make_slice_list(height=height, spacing=spacing, start=starting_z,
                              precision=precision)
    xy_points = box_points(width, length, offset_x, offset_y, precision)
    slice_list = []
    for slice_idx in z_coord:
        roi_slice = ContourPoints(xy_points, roi_num, slice_idx)
        slice_list.append(roi_slice)
    return slice_list
