#from math import sqrt
from typing import Tuple, List, Union

from matplotlib.pylab import f
vector = Tuple[float, float, float]

import FreeCAD as App
import FreeCADGui as Gui
import Part
from BOPTools import BOPFeatures

SCRIPT_PATH = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD_Scripts"
IMAGE_PATH = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\Images\FreeCAD Images"


# %% Part Functions
def make_sphere(radius: float,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    placement = App.Vector(
        (offset_x) * 10,
        (offset_y) * 10,
        (offset_z) * 10
        )
    sphere = Part.makeSphere(radius * 10, placement)
    return sphere


def make_vertical_cylinder(radius: float, length: float,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    starting_z = offset_z - length / 2
    placement = App.Vector(
        (offset_x) * 10,
        (offset_y) * 10,
        (starting_z) * 10
        )
    cylinder = Part.makeCylinder(radius * 10, length * 10, placement)
    return cylinder


def make_horizontal_cylinder(radius: float, length: float,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    starting_x = offset_x - length / 2
    placement = App.Vector(
        (starting_x) * 10,
        (offset_y) * 10,
        (offset_z) * 10
        )
    direction = App.Vector(1, 0, 0)
    cylinder = Part.makeCylinder(radius * 10, length * 10, placement, direction)
    return cylinder


def make_box(width: float, length: float = None, height: float = None,
             offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    if height:
        if height <= 0.0:
            raise ValueError('Height must be greater than 0')
    else:
        height = width
    if length:
        if length <= 0.0:
            raise ValueError('Length must be greater than 0')
    else:
        length = width
    starting_x = offset_x - length / 2
    starting_y = offset_y - width / 2
    starting_z = offset_z - height / 2
    placement = App.Vector(
        (starting_x) * 10,
        (starting_y) * 10,
        (starting_z) * 10
        )
    cylinder = Part.makeBox(length * 10, width * 10, height * 10, placement)
    return cylinder


def merge_parts(parts_list: List[Part.Shape])->Part.Shape:
    shape = parts_list[0]
    for part in parts_list[1:]:
        overlap = shape.common(part)
        if overlap.Volume == 0:
            shape = shape.fuse(part)
        else:
            shape = shape.cut(part)
    return shape


# %% Interaction Functions
def get_struct_1_only(struct_1, struct_2)->Union[Part.Shape, None]:
    struct_1_only = struct_1.cut(struct_2)
    if struct_1_only.Volume == 0:
        return None
    return struct_1_only


def get_both_structs(struct_1, struct_2)->Union[Part.Shape, None]:
    both_structs = struct_1.common(struct_2)
    if both_structs.Volume == 0:
        return None
    return both_structs


def interactions(struct_a, struct_b)->Tuple[Part.Shape]:
    struct_a_only = get_struct_1_only(struct_a, struct_b)
    struct_b_only = get_struct_1_only(struct_b, struct_a)
    both_ab = get_both_structs(struct_b, struct_a)
    return struct_a_only, struct_b_only, both_ab


def show_structure(structure: Part.Shape, label:str,
                   color:Tuple[int, int, int],
                   transparency=75,
                   display_as='Flat Lines',
                   line_style='Solid',
                   line_color=(0,0,0))->Part.Feature:
    if structure is None:
        return None
    struct = Part.show(structure)
    struct.Label = label
    struct.ViewObject.ShapeColor = color
    struct.ViewObject.Transparency = transparency
    struct.ViewObject.DisplayMode = display_as
    # Options are 'Flat Lines' 'Shaded', 'Wireframe', 'Points'
    struct.ViewObject.DrawStyle = line_style
    # Options are 'solid' 'Dashed', 'Dotted', 'Dashdot'
    struct.ViewObject.LineColor = line_color
    return struct


def display_interactions(struct_a, struct_b):
    # Define Structure Colors
    struct_a_color = (0,0,255)  #  Blue
    struct_b_color = (0,255,0)  #  Green
    intersect_color = (255,170,0)  # Orange
    # Get Interactions
    struct_a_only, struct_b_only, struct_ab = interactions(struct_a, struct_b)
    # Display Interactions
    struct_a = show_structure(struct_a_only, 'a', struct_a_color)
    struct_b = show_structure(struct_b_only, 'b', struct_b_color)
    both_struct = show_structure(struct_ab, 'both', intersect_color)
    return struct_a, struct_b, both_struct





# %% Other Display related functions
def add_slice_plane(structures: List[Part.Shape], slice_position: float,
                    slice_color = (200, 200, 200),
                    display_style='Wireframe',
                    line_style='Dashdot'):
    combined = merge_parts(structures)
    region_size = combined.BoundBox
    placement = App.Vector(region_size.XMin,
                           region_size.YMin,
                           slice_position * 10)
    slice_plane = Part.makePlane(region_size.XLength, region_size.YLength,
                                 placement)
    label = f'Slice: {slice_position:2.1f}'
    slice_display = show_structure(slice_plane, label, color=slice_color,
                                   display_as=display_style, line_style='Dashdot')
    return slice_display

def swap_shape(orig: Part.Feature, new_shape: Part.Shape, suffix: str)->Part.Feature:
    new_struct = Part.show(new_shape)
    new_struct.Label = orig.Label + suffix
    new_struct.ViewObject.ShapeColor = orig.ViewObject.ShapeColor
    new_struct.ViewObject.Transparency = orig.ViewObject.Transparency
    new_struct.ViewObject.DisplayMode = orig.ViewObject.DisplayMode
    new_struct.ViewObject.DrawStyle = orig.ViewObject.DrawStyle
    new_struct.ViewObject.LineColor = orig.ViewObject.LineColor
    new_struct.ViewObject.LineWidth = orig.ViewObject.LineWidth
    new_struct.ViewObject.LineWidth = orig.ViewObject.LineWidth
    new_struct.ViewObject.PointSize = orig.ViewObject.PointSize
    new_struct.ViewObject.PointColor = orig.ViewObject.PointColor
    orig.ViewObject.Visibility = False
    return new_struct

def crop_quarter(feature_list: List[Part.Feature],
                 quarter = (1,-1,1))->Part.Shape:
    quarter_rotations = {
        (1,1,1): (App.Vector(0, 0, 1), 0),
        (1,-1,1): (App.Vector(1, 0, 0), 90),
        (1,-1,-1): (App.Vector(1, 0, 0), 180),
        # (-1,-1,1): (0,0,180),  # To be added as required
        # (1,1,-1): (0,90,0),
        # (1,-1,-1): (0,-90,0),
        # (-1,1,-1): (0,-90,180),
        # (-1,-1,-1): (0,90,180)
        }
    rotation = quarter_rotations[quarter]
    combined = None
    for feature in feature_list:
        if feature is not None:
            if combined is None:
                combined = feature.Shape
            else:
                combined = combined.fuse(feature.Shape)
    region_size = combined.BoundBox
    placement = region_size.Center
    quarter_box = Part.makeBox(region_size.XMax, -region_size.YMin, region_size.ZMax,
                               placement)
    quarter_box = quarter_box.rotate(App.Vector(0, 0, 0), *rotation)
    quarter_crop = combined.common(quarter_box)
    cropped_list = []
    for feature in feature_list:
        if feature is not None:
            feature_cropped = feature.Shape.cut(quarter_box)
            feature_crop = swap_shape(feature, feature_cropped, '_cropped')
            feature.ViewObject.Visibility = False
        else:
            feature_crop = None
        cropped_list.append(feature_crop)
    crop_box = show_structure(quarter_crop, 'Crop Lines', color=(255,255,255),
                              display_as='Wireframe',
                              line_style='Dotted', line_color= (255,0,0))
    quarter_box = show_structure(quarter_box, 'Crop Lines', color=(255,255,255),
                              display_as='Wireframe',
                              line_style='Dotted', line_color= (255,0,0))
    return cropped_list, crop_box


# %% OLD Functions



def add_measurement(doc, start, end, offset=1.0, label='m1',
                    text_color=(0,0,0), line_color=(0,0,0)):
    m1 = doc.addObject("App::MeasureDistance", label)
    m1.P1 = App.Vector(start[0] * 10, start[1] * 10, start[2] * 10)
    m1.P2 = App.Vector(end[0] * 10, end[1] * 10, end[2] * 10)
    m1.ViewObject.LineColor = line_color
    m1.ViewObject.TextColor = text_color
    m1.ViewObject.FontSize = 30
    m1.ViewObject.DistFactor = offset
    doc.recompute()
    return m1




# %% For future reference
# _text_ = Draft.make_text(["Slice = 1.0"], placement=pl)
# Gui.Selection.addSelection('D__OneDrive___Queen_s_University_Python_Projects_StructureRelations_src_Images_FreeCAD_Images__StackedSpheres_FCStd','Text')
# Draft.autogroup(_text_)


#add_measurement((0, 0, 1), (0, 0, -1), offset=2.0, label='Distance Between Upper Spheres')
#file_name = 'StackedSpheres1'
#file_path = save_path + "//" + file_name + ".png"
#Gui.ActiveDocument.ActiveView.saveImage(file_path)#

#file_path = save_path + "//" + file_name + ".FCStd"
#App.activeDocument().saveAs(file_path)


#crop_box.DrawStyle = u"Dotted"
#crop_box.DisplayMode = u"Wireframe"
#struct_1.ViewObject.Transparency = transparency
#struct_1_shp = struct_1.Shape
# Gui.Selection.addSelection('TestSpheres','SectionCutBoxX')
# Gui.Selection.addSelection('TestSpheres','SectionCutCompound')
# crop_box.Visibility = True
# doc.removeObject('SectionCutX')
#Gui.Selection.clearSelection()
#Gui.ActiveDocument.ActiveView.setAxisCross(False)

#file_path = save_path + "//" + file_name + ".png"
#Gui.ActiveDocument.ActiveView.saveImage(file_path)

#file_name = 'TestSpheres'
#file_path = save_path + "//" + file_name + ".FCStd"
##doc = App.getDocument(file_name)
#App.activeDocument().saveAs(file_path)



#Gui.runCommand('Std_ToggleVisibility',0)
#Gui.runCommand('Std_ToggleClipPlane',0)
#
