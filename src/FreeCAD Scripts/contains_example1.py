# This is a test script for embedded spheres

#from math import sqrt
from BOPTools import BOPFeatures

# Structure Interactions Coloring
def interactions(bp, struct_a, struct_b):
    # Structure Intersections
    def find_overlapping(bp, struct_1, struct_2, color, label):
        overlapping = bp.make_multi_common([struct_1.Name, struct_2.Name, ])
        if
        doc.recompute()
        overlapping.Label = label
        overlapping.ViewObject.ShapeColor = color
        doc.recompute()
        return overlapping

    # Structure Differences
    def find_exclusion(bp, struct_a, struct_b, color, label):
        struct_a_only = bp.make_cut([struct_a.Name, struct_b.Name, ])
        doc.recompute()
        struct_a_only.Label = label
        struct_a_only.ViewObject.ShapeColor = color
        return struct_a_only

    # Define Structure Colors
    struct_1_color = (0,0,255)  #  Blue
    struct_2_color = (0,255,0)  #  Green
    intersect_color = (255,170,0)  # Orange
    # Define Structure Labels
    struct_1_label = "Structure 1 Only"
    struct_2_label = "Structure 2 Only"
    intersect_label = "Overlapping Structures"
    struct_a_only = find_exclusion(bp, struct_a, struct_b,
                                   struct_1_color, struct_1_label)
    struct_b_only = find_exclusion(bp, struct_b, struct_a,
                                   struct_2_color, struct_2_label)
    overlapping = find_overlapping(bp, struct_1, struct_2,
                                   intersect_color, intersect_label)
    return struct_a_only, struct_b_only, overlapping

struct_a_only = bp.make_cut([struct_2.Name, struct_1.Name, ])
try:
    doc.recompute()
except Exception as err:
    pass

# Object Functions
def make_sphere(doc, part_name, radius, offset):
    placement = App.Vector(offset[0] * 10, offset[1] * 10, offset[2] * 10)
    no_rotation = App.Rotation(0,0,0)
    zero_axis = App.Vector(0,0,0)
    struct = doc.addObject("Part::Sphere", part_name)
    struct.Label = part_name
    struct.Radius = radius
    struct.Placement = App.Placement(placement, no_rotation, zero_axis)
    doc.recompute()
    return struct


save_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD Scripts"
file_name = 'TestSpheres'
file_path = save_path + "//" + file_name + ".FCStd"
Gui.activateWorkbench("PartWorkbench")
doc = App.newDocument(file_name)
bp = BOPFeatures.BOPFeatures(doc)
#doc.saveAs(file_path)

#doc = FreeCAD.newDocument()

# First Sphere
struct_1_name = "Sphere6"
struct_1_radius = '3 cm'
struct_1_offset = (3, 0, 1)
struct_1 = make_sphere(doc, struct_1_name, struct_1_radius, struct_1_offset)


myPart = FreeCAD.ActiveDocument.addObject("Part::Feature", "myPartName")

# Second Sphere
struct_2_name = "Sphere4"
struct_2_radius = '2 cm'
struct_2_offset = (3, 0, 1)
struct_2 = make_sphere(doc, struct_2_name, struct_2_radius, struct_2_offset)


only_1 = struct_1.Shape.cut(struct_2.Shape)

only_1, only_2, both = interactions(bp, struct_1, struct_1)

# Structure Intersections
struct_1_only = bp.make_cut([struct_1_name, struct_2_name, ])
doc.recompute()
struct_1_only.Label = "Structure 1 Only"
struct_1_only.ViewObject.ShapeColor = struct_1_color

struct_2_only = bp.make_cut([struct_2_name, struct_1_name, ])
doc.recompute()
struct_2_only.Label = "Structure 2 Only"
struct_2_only.ViewObject.ShapeColor = struct_2_color

overlapping = bp.make_multi_common([struct_1_name, struct_2_name, ])
doc.recompute()
overlapping.Label = "Overlapping Structures"
overlapping.ViewObject.ShapeColor = intersect_color

doc.recompute()

# Orthogonal X crop Box
crop_name = "CropX"
crop_box = doc.addObject("Part::Box", crop_name)
cropping_size = struct_1.Shape.BoundBox
cropping_center = cropping_size.Center
cropping_center[1] = cropping_size.YMin - 1
cropping_center[2] = cropping_size.ZMin -1
crop_box.Placement=App.Placement(cropping_center, App.Rotation(0,0,0), App.Vector(0,0,0))
crop_box.Length = f'{-cropping_size.XMin + 2} mm'
crop_box.Width = f'{cropping_size.YLength + 2} mm'
crop_box.Height = f'{cropping_size.ZLength + 2} mm'
doc.recompute()

# Crop & color structures
cropped_struct_1 = bp.make_cut([struct_1_only.Name, crop_name, ])
doc.recompute()
cropped_struct_1.ViewObject.ShapeColor = struct_1_color
cropped_struct_1.Label = "Struct_1_crop_X"

#cropped_struct_2 = bp.make_cut([struct_2_only.Name, crop_name, ])
#doc.recompute()
#cropped_struct_2.ViewObject.ShapeColor = struct_2_color
#cropped_struct_2.Label = "Struct_2_crop_X"

cropped_overlap = bp.make_cut([overlapping.Name, crop_name, ])
doc.recompute()
cropped_overlap.ViewObject.ShapeColor = intersect_color
cropped_overlap.Label = "Structure Overlap Crop_X"

doc.recompute()
#doc.activeView().viewIsometric()
Gui.activeDocument().activeView().viewIsometric()
Gui.ActiveDocument.ActiveView.setAxisCross(False)
Gui.SendMsgToActiveView("ViewFit")
Gui.Selection.clearSelection()

#crop_box.DrawStyle = u"Dotted"
#crop_box.DisplayMode = u"Wireframe"
#struct_1.ViewObject.Transparency = transparency
#struct_1_shp = struct_1.Shape
# Gui.Selection.addSelection('TestSpheres','SectionCutBoxX')
# Gui.Selection.addSelection('TestSpheres','SectionCutCompound')
# crop_box.Visibility = True
# doc.removeObject('SectionCutX')

file_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)
