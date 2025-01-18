#from math import sqrt
from typing import Tuple, List, Union
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


def make_vertical_cylinder(radius: float, height: float,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    starting_z = offset_z - height / 2
    placement = App.Vector(
        (offset_x) * 10,
        (offset_y) * 10,
        (starting_z) * 10
        )
    cylinder = Part.makeCylinder(radius * 10, height * 10, placement)
    return cylinder


def make_horizontal_cylinder(radius: float, height: float,
                offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    starting_x = offset_x - height / 2
    placement = App.Vector(
        (starting_x) * 10,
        (offset_y) * 10,
        (offset_z) * 10
        )
    direction = App.Vector(1, 0, 0)
    cylinder = Part.makeCylinder(radius * 10, height * 10, placement, direction)
    return cylinder


def make_box(length: float, width:float, height: float,
             offset_x: float = 0, offset_y: float = 0, offset_z: float = 0
                )->Part.Shape:
    starting_x = offset_x - height / 2
    placement = App.Vector(
        (starting_x) * 10,
        (offset_y) * 10,
        (offset_z) * 10
        )
    cylinder = Part.makeBox(length * 10, width * 10, height * 10, placement)
    return cylinder


def merge_parts(parts_list: List[Part.Shape])->Part.Shape:
    shape = parts_list[0]
    for part in parts_list[1:]:
        shape = shape.fuse(part)
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
                   display_as='Shaded',
                   line_style='Solid')->Part.Feature:
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
                    slice_color = (200, 200, 200)):
    combined = merge_parts(structures)
    region_size = combined.BoundBox
    placement = App.Vector(region_size.XMin,
                           region_size.YMin,
                           slice_position * 10)
    slice_plane = Part.makePlane(region_size.XLength, region_size.YLength,
                                 placement)
    label = f'Slice: {slice_position:2.1f}'
    show_structure(slice_plane, label, color=slice_color,
                   display_as='Wireframe',
                   line_style='Dashdot')
    return slice_plane


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


def make_structure(doc: App.Document, shape: Part.Shape,
                   part_name: str)->Part.Feature:
    struct = doc.addObject("Part::Feature", part_name)
    struct.Label = part_name
    struct.Shape = shape
    doc.recompute()
    return struct


# %% Cropped views
def crop_dir(cropping_size, direction):
    if 'X' in direction:
        cropping_center = cropping_size.Center
        cropping_center[1] = cropping_size.YMin - 1
        cropping_center[2] = cropping_size.ZMin - 1
        box_size = (-cropping_size.XMin + 2,
                    cropping_size.YLength + 2,
                    cropping_size.ZLength + 2)
    if 'Y' in direction:
        cropping_center = cropping_size.Center
        cropping_center[0] = cropping_size.XMin - 1
        cropping_center[2] = cropping_size.ZMin -1
        box_size = (cropping_size.XLength + 2,
                    -cropping_size.YMin + 2,
                    cropping_size.ZLength + 2)
    if 'Z' in direction:
        cropping_center = cropping_size.Center
        cropping_center[0] = cropping_size.XMin - 1
        cropping_center[1] = cropping_size.YMin - 1
        box_size = (cropping_size.XLength + 2,
                    cropping_size.YLength + 2,
                    -cropping_size.ZMin + 2)
    return cropping_center, box_size


def struct_crop(struct, name, direction):
    cropping_size = struct.Shape.BoundBox
    cropping_center, box_size = crop_dir(cropping_size, direction)
    cut_view = struct.Shape.cut(Part.makeBox(*box_size, cropping_center))
    cropped_struct = doc.addObject("Part::Feature", name)
    cropped_struct.Shape = cut_view
    cropped_struct.ViewObject.ShapeColor = struct.ViewObject.ShapeColor
    struct.Visibility = False


def crop_orth(a_only, b_only, both, direction):
    if a_only:
        struct_crop(a_only, f'Crop_A_{direction}', direction)
    if b_only:
        struct_crop(b_only, f'Crop_B_{direction}', direction)
    if both:
        struct_crop(both, f'Crop_Both_{direction}', direction)
    doc.recompute()
    if 'X' in direction:
        Gui.activeDocument().activeView().viewRight()
        Gui.SendMsgToActiveView("ViewFit")
    if 'Y' in direction:
        Gui.activeDocument().activeView().viewRear()
        Gui.SendMsgToActiveView("ViewFit")
    if 'Z' in direction:
        Gui.activeDocument().activeView().viewTop()
        Gui.SendMsgToActiveView("ViewFit")
    #Gui.ActiveDocument.ActiveView.setAxisCross(False)



def make_crop(doc, struct_1_only, overlapping):

    # Define Structure Colors
    struct_1_color = (0,0,255)  #  Blue
    struct_2_color = (0,255,0)  #  Green
    intersect_color = (255,170,0)  # Orange


    crop_box = doc.addObject("Part::Box", "CropX")
    cropping_size = struct_1_only.Shape.BoundBox
    cropping_center = cropping_size.Center
    cropping_center[1] = cropping_size.YMin - 1
    cropping_center[2] = cropping_size.ZMin -1
    crop_box.Placement=App.Placement(cropping_center, App.Rotation(0,0,0), App.Vector(0,0,0))
    crop_box.Length = f'{-cropping_size.XMin + 2} mm'
    crop_box.Width = f'{cropping_size.YLength + 2} mm'
    crop_box.Height = f'{cropping_size.ZLength + 2} mm'
    doc.recompute()

    # Crop & color structures
    bp = BOPFeatures.BOPFeatures(doc)
    cropped_struct_1 = bp.make_cut([struct_1_only.Name, "CropX"])
    doc.recompute()
    cropped_struct_1.ViewObject.ShapeColor = struct_1_color
    cropped_struct_1.Label = "Struct_1_crop_X"

    cropped_overlap = bp.make_cut([overlapping.Name, "CropX"])
    doc.recompute()
    cropped_overlap.ViewObject.ShapeColor = intersect_color
    cropped_overlap.Label = "Structure Overlap Crop_X"


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
