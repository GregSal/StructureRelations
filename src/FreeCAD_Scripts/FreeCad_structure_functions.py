#from math import sqrt
from typing import Tuple
vector = Tuple[float, float, float]

import FreeCAD as App
import FreeCADGui as Gui
import Part
from BOPTools import BOPFeatures

script_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD Scripts"
image_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\Images\FreeCAD Images"


# %% Functions
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


def make_sphere(doc: App.Document, part_name: str, radius: float,
                offset: vector)->Part.Feature:
    placement = App.Vector(offset[0] * 10, offset[1] * 10, offset[2] * 10)
    sphere = Part.makeSphere(radius * 10, placement)
    struct = make_structure(doc, sphere, part_name)
    return struct


def make_cylinder(doc: App.Document, part_name: str,
                  radius: float, height: float,
                  offset: vector, direction: vector)->Part.Feature:
    orientation = App.Vector(*direction)
    placement = App.Vector(offset[0] * 10, offset[1] * 10, offset[2] * 10)
    cylinder = Part.makeCylinder(radius * 10, height * 10, placement, orientation)
    struct = make_structure(doc, cylinder, part_name)
    return struct

# Structure Interactions Coloring
def interactions(doc, struct_1, struct_2)->Tuple[Part.Feature]:
    def cut_a(doc, struct_a, struct_b, color, label)->Part.Feature:
        bp = BOPFeatures.BOPFeatures(doc)
        struct_a_only = bp.make_cut([struct_a.Name, struct_b.Name])
        struct_a_only.Label = label
        if struct_a_only.Shape.isNull():
            doc.removeObject(struct_a_only.Name)
            struct_a_only = None
        else:
            struct_a_only.ViewObject.ShapeColor = color
        doc.recompute()
        return struct_a_only

    def find_overlapping(doc, struct_a, struct_b, color, label)->Part.Feature:
        bp = BOPFeatures.BOPFeatures(doc)
        overlapping = bp.make_multi_common([struct_a.Name, struct_b.Name])
        doc.recompute()
        overlapping.Label = label
        if overlapping.Shape.isNull():
            doc.removeObject(overlapping.Name)
            overlapping = None
        else:
            overlapping.ViewObject.ShapeColor = color
        #doc.recompute()
        return overlapping

    # Define Structure Colors
    struct_1_color = (0,0,255)  #  Blue
    struct_2_color = (0,255,0)  #  Green
    intersect_color = (255,170,0)  # Orange
    # Structure Intersections
    struct_1_only = cut_a(doc, struct_1, struct_2, struct_1_color,
                          "Structure 1 Only")
    struct_2_only = cut_a(doc, struct_2, struct_1, struct_2_color,
                          "Structure 2 Only")
    overlapping = find_overlapping(doc, struct_1, struct_2, intersect_color,
                                   "Overlapping Structures")
    doc.recompute()
    return struct_1_only, struct_2_only, overlapping



# Cropped views
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
