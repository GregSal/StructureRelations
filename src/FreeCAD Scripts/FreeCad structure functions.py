# This is a test script for embedded spheres

#from math import sqrt
#rom BOPTools import BOPFeatures
from typing import Tuple
vector = Tuple[float, float, float]
import Part

save_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\Images\FreeCAD Images"
# %% Functions
def add_measurement(start, end, offset):
    m1 = doc.addObject("App::MeasureDistance", 'm1')
    m1.P1 = App.Vector(start[0] * 10, start[1] * 10, start[2] * 10)
    m1.P2 = App.Vector(end[0] * 10, end[1] * 10, end[2] * 10)
    m1.ViewObject.LineColor = (0,0,0)
    m1.ViewObject.TextColor = (0,0,0)
    m1.ViewObject.FontSize = 30
    m1.ViewObject.DistFactor = offset
    m1.Distance
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
    placement = App.Vector(offset[0] * 10, offset[1] * 10, offset[2] * 10)
    cylinder = Part.makeCylinder(radius * 10, height * 10, placement, direction)
    struct = make_structure(doc, cylinder, part_name)
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

#file_name = 'TestSpheres'
#file_path = save_path + "//" + file_name + ".FCStd"
##doc = App.getDocument(file_name)
#App.activeDocument().saveAs(file_path)


# %% Stacked Spheres1
Gui.activateWorkbench("PartWorkbench")
doc = App.newDocument()

sup_placement = App.Vector(0, 0, 4.0 * 10)
inf_placement = App.Vector(0, 0, -4.0 * 10)
inf2_placement = App.Vector(0, 0, -12.0 * 10)
sphere3_sup = Part.makeSphere(3.0 * 10, sup_placement)
sphere3_inf = Part.makeSphere(3.0 * 10, inf_placement)
sphere3_inf2 = Part.makeSphere(3.0 * 10, inf2_placement)
stacked_spheres_part = sphere3_sup.fuse(sphere3_inf)
stacked_spheres_part = stacked_spheres_part.fuse(sphere3_inf2)
stacked_spheres = make_structure(doc, stacked_spheres_part, '0')

sphere2_inf = make_sphere(doc, '1', 2, (0, 0, -4.0))
#cylinder2 = make_cylinder(doc, '1', 2.0, 10.0, App.Vector(0, 0, 0), App.Vector(0, 0, 1))

a_only, b_only, both = interactions(doc, stacked_spheres, sphere2_inf)
doc.recompute()
#Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
#Gui.ActiveDocument.ActiveView.setAxisCross(False)
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

file_name = 'StackedSpheres1'
file_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)
#
file_path = save_path + "//" + file_name + ".FCStd"
App.activeDocument().saveAs(file_path)
# %% Stacked Spheres2
Gui.activateWorkbench("PartWorkbench")
doc = App.newDocument()

sup_placement = App.Vector(0, 0, 4.0 * 10)
inf_placement = App.Vector(0, 0, -4.0 * 10)
inf2_placement = App.Vector(0, 0, -12.0 * 10)
sphere3_sup = Part.makeSphere(3.0 * 10, sup_placement)
sphere3_inf = Part.makeSphere(3.0 * 10, inf_placement)
sphere3_inf2 = Part.makeSphere(3.0 * 10, inf2_placement)
stacked_spheres_part = sphere3_sup.fuse(sphere3_inf)
stacked_spheres_part = stacked_spheres_part.fuse(sphere3_inf2)
stacked_spheres = make_structure(doc, stacked_spheres_part, '0')

sphere2_inf = make_sphere(doc, '1', 2, (0, 0, -3.5))
#cylinder2 = make_cylinder(doc, '1', 2.0, 10.0, App.Vector(0, 0, 0), App.Vector(0, 0, 1))

a_only, b_only, both = interactions(doc, stacked_spheres, sphere2_inf)
doc.recompute()
#Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
#Gui.ActiveDocument.ActiveView.setAxisCross(False)
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

file_path = save_path + "//" + 'StackedSpheres2' + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)





# %% Main
ppGui.activateWorkbench("PartWorkbench")
doc = App.newDocument()

# Make Test Structures
cylinder6 = make_cylinder(doc, '1', 6.0, 10.0,
                          App.Vector(0, 0, 0), App.Vector(0, 0, 1))
cylinder4 = make_cylinder(doc, '2', 4.0, 10.0,
                          App.Vector(0, 0, 0), App.Vector(0, 0, 1))
sphere5 = make_sphere(doc, '3', 5, (0, 0, 0))
sphere2 = make_sphere(doc, '4', 2, (0, 0, 0))

outer_sphere = Part.makeSphere(6 * 10)
hole = Part.makeSphere(4 * 10)
inner_sphere = Part.makeSphere(3 * 10)
embeded_spheres_part = outer_sphere.cut(hole).fuse(inner_sphere)
embeded_spheres = make_structure(doc, embeded_spheres_part, '5')

cylinderH2 = make_cylinder(doc, '6', 6.0, 10.0,
                          App.Vector(0, 0, 0), App.Vector(1, 0, 0))
cylinderH1 = make_cylinder(doc, '7', 4.0, 10.0,
                          App.Vector(0, 0, 0), App.Vector(1, 0, 0))

# %% Structures 1 & 2
a_only, b_only, both = interactions(doc, cylinder6, cylinder4)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(False)
file_path = save_path + "//" + 'ConcentricCylinders' + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)


# %% Structures 5 & 4
a_only, b_only, both = interactions(doc, embeded_spheres, sphere2)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(False)

file_path = save_path + "//" + 'EmbeddedSpheres' + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)


# %% Structures 6 & 7
a_only, b_only, both = interactions(doc, cylinderH6, cylinderH4)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'Z')
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(False)

file_path = save_path + "//" + 'HorizontalCylinders' + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)


# %% Crop Views
# Color interactions
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


# %% Save
file_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)

#save_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD Scripts"
file_name = 'ContainsTests'
file_path = save_path + "//" + file_name + ".FCStd"

App.activeDocument().saveAs(file_path)
##doc = App.getDocument(file_name)
