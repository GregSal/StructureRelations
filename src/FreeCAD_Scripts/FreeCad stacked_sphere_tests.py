
# %%  Create a new document
file_name = 'Embedded Spheres'
file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(file_path)
# %% Embedded Spheres
sphere6 = make_sphere(radius=3, offset_x=0, offset_y=0, offset_z=0)
sphere3 = make_sphere(radius=1.5, offset_x=0, offset_y=0, offset_z=0)
a, b, both = display_interactions(sphere6, sphere3)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

file_name = 'Embedded Spheres'
file_path = IMAGE_PATH + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)

file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
App.activeDocument().saveAs(file_path)

# %% Make Concentric cylinders
file_name = 'Concentric cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

primary_cylinder = make_vertical_cylinder(radius=5, height=0.7,
                                          offset_x=0, offset_y=0, offset_z=0)
contained_cylinder = make_vertical_cylinder(radius=3, height=0.5,
                                            offset_x=0, offset_y=0, offset_z=0)

a, b, both = display_interactions(primary_cylinder, contained_cylinder)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Embedded Boxes
cube6 = make_box(width=6, offset_x=0, offset_y=0, offset_z=0)
cube3 = make_box(width=3, offset_x=0, offset_y=0, offset_z=0)
a, b, both = display_interactions(cube6, cube3)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

file_name = 'Embedded Boxes'
file_path = IMAGE_PATH + "//" + file_name + ".png"
Gui.ActiveDocument.ActiveView.saveImage(file_path)

file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
App.activeDocument().saveAs(file_path)

# %% Stacked Spheres1
file_name = 'EmbededSpheres'
file_path = image_path + "//" + file_name + ".FCStd"
doc = App.newDocument(file_path)

center_sphere = make_sphere(radius=1, offset_x=0, offset_y=0, offset_z=0)
sup_sphere = make_sphere(radius=1, offset_x=0, offset_y=0, offset_z=3)
inf_sphere = make_sphere(radius=1, offset_x=0, offset_y=0, offset_z=-3)
stacked_spheres = merge_parts([center_sphere, sup_sphere, inf_sphere])

# %% Stacked Spheres2
center_sphere = make_sphere(radius=0.5, offset_x=0, offset_y=0, offset_z=0)
sup_sphere = make_sphere(radius=0.5, offset_x=0, offset_y=0, offset_z=3)
inf_sphere = make_sphere(radius=0.5, offset_x=0, offset_y=0, offset_z=-3)
inner_stacked_spheres = merge_parts([center_sphere, sup_sphere, inf_sphere])


a, b, both = display_interactions(stacked_spheres, inner_stacked_spheres)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)
a.Visibility = True








# %%
sphere5 = make_sphere(doc, '3', 5, (0, 0, 0))
sphere2 = make_sphere(doc, '4', 2, (0, 0, 0))

outer_sphere = Part.makeSphere(6 * 10)
hole = Part.makeSphere(4 * 10)
inner_sphere = Part.makeSphere(3 * 10)
embeded_spheres_part = outer_sphere.cut(hole).fuse(inner_sphere)
embeded_spheres = make_structure(doc, embeded_spheres_part, '5')
