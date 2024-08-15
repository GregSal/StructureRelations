# This is a test script for embedded spheres

#from math import sqrt
#rom BOPTools import BOPFeatures
from typing import Tuple
vector = Tuple[float, float, float]
import Part

# %% Functions
def make_sphere(doc: App.Document, part_name: str, radius: float, offset: vector)->Part.Feature:
    placement = App.Vector(offset[0] * 10, offset[1] * 10, offset[2] * 10)
    sphere = Part.makeSphere(radius * 10, placement)
    struct = doc.addObject("Part::Feature", part_name)
    struct.Label = part_name
    struct.Shape = sphere
    doc.recompute()
    return struct


# Structure Interactions Coloring
def interactions(doc, struct_a, struct_b):
    # Structure Intersections
    def find_overlapping(doc, struct_a, struct_b, color, label):
        overlapping = struct_a.Shape.common(struct_b.Shape)
        if overlapping.Volume == 0:
            return None
        struct = doc.addObject("Part::Feature", label)
        struct.Label = label
        struct.Shape = overlapping
        struct.ViewObject.ShapeColor = color
        doc.recompute()
        return struct

    # Structure Differences
    def find_exclusion(doc, struct_a, struct_b, color, label):
        exclusion = struct_a.Shape.cut(struct_b.Shape)
        print(label, exclusion.Volume)
        if exclusion.Volume == 0:
            return None
        struct = doc.addObject("Part::Feature", label)
        struct.Label = label
        struct.Shape = exclusion
        struct.ViewObject.ShapeColor = color
        doc.recompute()
        return struct

    # Define Structure Colors
    a_color = (0,0,255)  #  Blue
    b_color = (0,255,0)  #  Green
    both_color = (255,170,0)  # Orange
    # Define Structure Labels
    a_label = "Only A"
    b_label = "Only B"
    both_label = "Both"
    a_only = find_exclusion(doc, struct_a, struct_b, a_color, a_label)
    b_only = find_exclusion(doc, struct_b, struct_a, b_color, b_label)
    both = find_overlapping(doc, struct_a, struct_b, both_color, both_label)
    return a_only, b_only, both

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

#save_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD Scripts"
#file_name = 'TestSpheres'
#file_path = save_path + "//" + file_name + ".FCStd"
##doc = App.getDocument(file_name)
#App.activeDocument().saveAs(file_path)
# %% Main
Gui.activateWorkbench("PartWorkbench")
doc = App.newDocument()

# Make Structures
struct_a = make_sphere(doc, 'A', 3, (0, 0, 0))
struct_b = make_sphere(doc, 'B', 2, (0, 0, 0))

# Color interatctions
a_only, b_only, both = interactions(doc, struct_a, struct_b)
struct_a.Visibility = False
struct_b.Visibility = False
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

# X Axis Crop
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')

# Y Axis Crop
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'Y')

# Z Axis Crop
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'Z')
