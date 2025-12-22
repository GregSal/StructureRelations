# This is a test script for dual embedded cylinders

#from math import sqrt
from BOPTools import BOPFeatures


save_path = r"D:\OneDrive - Queen's University\Python\Projects\StructureRelations\src\FreeCAD Scripts"
file_name = 'DualEmbededCylinders'
file_path = save_path + "//" + file_name + ".FCStd"
Gui.activateWorkbench("PartWorkbench")
doc = App.newDocument(file_name)
bp = BOPFeatures.BOPFeatures(doc)
#doc.saveAs(file_path)

#doc = FreeCAD.newDocument()

# Body, radius 6, length 11
body = make_cylinder(doc, 'body', 12, 11, (0, 0, -5), (0,0,1))

# Second Sphere
primary = make_cylinder(doc, 'primary', 5, 7, (0, 0, -3), (0,0,1))

left_hole = make_cylinder(doc, 'left_hole', 2, 5, (-2.5, 0, -2), (0,0,1))
right_hole = make_cylinder(doc, 'right_hole', 2, 5, (2.5, 0, -2), (0,0,1))
primary = bp.make_cut(['primary', 'left_hole', ])

primary = bp.make_cut(["primary", "left_hole", ])
primary = bp.make_cut(["Cut", "right_hole", ])
primary.Label = "1"


confines_cylinder = make_cylinder(doc, '2', 1, 5, (2.5, 0, -2), (0,0,1))


Gui.runCommand('Std_ToggleVisibility',0)
Gui.runCommand('Std_ToggleVisibility',0)




primary = primary.Shape.cut(left_hole.Shape)

only_1, only_2, both = interactions(doc, struct_1, struct_1)



bordering_cylinder2 = make_cylinder(doc, '4', 3, 2, (0, 0, -4), (0,0,1))
primary = primary.Shape.cut(left_hole.Shape)

only_1, only_2, both = interactions(doc, struct_1, struct_1)

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

file_path = save_path + "\\" + 'confines_cylinder' + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)
save_path = r"C:\Users\gsalomon\Python Scripts\StructureRelations\src\FreeCAD_Scripts"
