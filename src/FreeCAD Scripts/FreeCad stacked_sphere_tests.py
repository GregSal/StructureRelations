
# %% Stacked Spheres1
file_name = 'StackedSpheres1'
file_path = save_path + "//" + file_name + ".FCStd"
doc = App.newDocument(file_path)

# 3 Spheres centres on Z=4, Z=-4, Z=-12
sup_placement = App.Vector(0, 0, 4.0 * 10)
inf_placement = App.Vector(0, 0, -4.0 * 10)
inf2_placement = App.Vector(0, 0, -12.0 * 10)
sphere3_sup = Part.makeSphere(3.0 * 10, sup_placement)
sphere3_inf = Part.makeSphere(3.0 * 10, inf_placement)
sphere3_inf2 = Part.makeSphere(3.0 * 10, inf2_placement)
stacked_spheres_part = sphere3_sup.fuse(sphere3_inf)
stacked_spheres_part = stacked_spheres_part.fuse(sphere3_inf2)
stacked_spheres = make_structure(doc, stacked_spheres_part, '0')

# Single 2 cm sphere within centre sphere of other structure
sphere2_inf = make_sphere(doc, '1', 2, (0, 0, -4.0))

a_only, b_only, both = interactions(doc, stacked_spheres, sphere2_inf)
doc.recompute()
stacked_spheres.Visibility = False
sphere2_inf.Visibility = False
#Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
#Gui.ActiveDocument.ActiveView.setAxisCross(False)

add_measurement((0, 0, 1), (0, 0, -1), offset=2.0, label='Distance Between Upper Spheres')
add_measurement((0, 0, -7), (0, 0, -9), offset=1.5, label='Distance Between Lower Spheres')
add_measurement((0, 0, -15), (0, 0, -9), offset=-1.0, label='Sphere Diameter')
add_measurement((0, 0, 4), (0, 0, 7), offset=-1.0, label='Sphere Radius')
add_measurement((0, 0, -6), (0, 0, -7), offset=3.0,
                text_color=(0, 255, 0), line_color=(0, 255, 0),
                label='Inner to Outer Distance')

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

file_name = 'StackedSpheres1'
file_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)

file_path = save_path + "//" + file_name + ".FCStd"
App.activeDocument().saveAs(file_path)
# %% Stacked Spheres Vertical Offset
file_name = 'StackedSpheresOffset'
file_path = save_path + "//" + file_name + ".FCStd"
doc = App.newDocument(file_path)

# 3 Spheres centres on Z=4, Z=-4, Z=-12
sup_placement = App.Vector(0, 0, 4.0 * 10)
inf_placement = App.Vector(0, 0, -4.0 * 10)
inf2_placement = App.Vector(0, 0, -12.0 * 10)
sphere3_sup = Part.makeSphere(3.0 * 10, sup_placement)
sphere3_inf = Part.makeSphere(3.0 * 10, inf_placement)
sphere3_inf2 = Part.makeSphere(3.0 * 10, inf2_placement)
stacked_spheres_part = sphere3_sup.fuse(sphere3_inf)
stacked_spheres_part = stacked_spheres_part.fuse(sphere3_inf2)
stacked_spheres = make_structure(doc, stacked_spheres_part, '0')

# Single 2 cm sphere within centre sphere of other structure
sphere2_inf = make_sphere(doc, '1', 2, (0, 0, -3.5))

a_only, b_only, both = interactions(doc, stacked_spheres, sphere2_inf)
doc.recompute()
stacked_spheres.Visibility = False
sphere2_inf.Visibility = False
#Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
#Gui.ActiveDocument.ActiveView.setAxisCross(False)

add_measurement((0, 0, 1), (0, 0, -1), offset=2.0, label='Distance Between Upper Spheres')
add_measurement((0, 0, -7), (0, 0, -9), offset=1.5, label='Distance Between Lower Spheres')
add_measurement((0, 0, -15), (0, 0, -9), offset=-1.0, label='Sphere Diameter')
add_measurement((0, 0, 4), (0, 0, 7), offset=-1.0, label='Sphere Radius')
add_measurement((0, 0, -1), (0, 0, -1.5), offset=3.0,
                text_color=(0, 255, 0), line_color=(0, 255, 0),
                label='Positive Z Margin')
add_measurement((0, 0, -5.5), (0, 0, -7), offset=3.0,
                text_color=(0, 255, 0), line_color=(0, 255, 0),
                label='Negative Z Margin')

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")


image_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(image_path)

file_path = save_path + "//" + file_name + ".FCStd"
App.activeDocument().saveAs(file_path)


# %% Stacked Spheres Vertical and Horizontal Offset
file_name = 'StackedSpheresShiftedOffset'
file_path = save_path + "//" + file_name + ".FCStd"
doc = App.newDocument(file_path)

# 3 Spheres centres on Z=4, Z=-4, Z=-12
sup_placement = App.Vector(0, 0, 4.0 * 10)
inf_placement = App.Vector(0, 0, -4.0 * 10)
inf2_placement = App.Vector(0, 0, -12.0 * 10)
sphere3_sup = Part.makeSphere(3.0 * 10, sup_placement)
sphere3_inf = Part.makeSphere(3.0 * 10, inf_placement)
sphere3_inf2 = Part.makeSphere(3.0 * 10, inf2_placement)
stacked_spheres_part = sphere3_sup.fuse(sphere3_inf)
stacked_spheres_part = stacked_spheres_part.fuse(sphere3_inf2)
stacked_spheres = make_structure(doc, stacked_spheres_part, '0')

# Single 2 cm sphere within centre sphere of other structure
sphere2_inf = make_sphere(doc, '1', 2, (0.5, 0, -3.5))

a_only, b_only, both = interactions(doc, stacked_spheres, sphere2_inf)
doc.recompute()
stacked_spheres.Visibility = False
sphere2_inf.Visibility = False
#Gui.runCommand('Std_ViewCreate',0)
crop_orth(a_only, b_only, both, 'X')
#Gui.ActiveDocument.ActiveView.setAxisCross(False)

add_measurement((0, 0, 1), (0, 0, -1), offset=2.0, label='Distance Between Upper Spheres')
add_measurement((0, 0, -7), (0, 0, -9), offset=1.5, label='Distance Between Lower Spheres')
add_measurement((0, 0, -15), (0, 0, -9), offset=-1.0, label='Sphere Diameter')
add_measurement((0, 0, 4), (0, 0, 7), offset=-1.0, label='Sphere Radius')
add_measurement((0, 0, -1), (0, 0, -1.5), offset=3.0,
                text_color=(0, 255, 0), line_color=(0, 255, 0),
                label='Positive Z Margin')
add_measurement((0, 0, -5.5), (0, 0, -7), offset=3.0,
                text_color=(0, 255, 0), line_color=(0, 255, 0),
                label='Negative Z Margin')

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")


image_path = save_path + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(image_path)

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
