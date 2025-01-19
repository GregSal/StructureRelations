
# %%  Create a new document
from test_3D_relations import make_box


file_name = 'Embedded Spheres'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)
# %% Embedded Spheres
sphere6 = make_sphere(radius=3, offset_x=0, offset_y=0, offset_z=0)
sphere3 = make_sphere(radius=1.5, offset_x=0, offset_y=0, offset_z=0)
a, b, both = display_interactions(sphere6, sphere3)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

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
file_name = 'Embedded Boxes'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

cube6 = make_box(width=6, offset_x=0, offset_y=0, offset_z=0)
cube3 = make_box(width=3, offset_x=0, offset_y=0, offset_z=0)
a, b, both = display_interactions(cube6, cube3)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Parallel Cylinders
file_name = 'Parallel Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

left_cylinder = make_vertical_cylinder(radius=2, length=8,
                                       offset_x=-2.5, offset_y=0, offset_z=0)
right_cylinder = make_vertical_cylinder(radius=2, length=8,
                                        offset_x=2.5, offset_y=0, offset_z=0)
parallel_cylinders = merge_parts([left_cylinder, right_cylinder])

right_middle_cylinder = make_vertical_cylinder(radius=1, length=6,
                                               offset_x=2.5,
                                               offset_y=0, offset_z=0)

a, b, both = display_interactions(parallel_cylinders, right_middle_cylinder)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

plane_4 = add_slice_plane([parallel_cylinders], slice_position=4.0)
plane_3 = add_slice_plane([parallel_cylinders], slice_position=3.0)
plane_3.ViewObject.DisplayMode = 'Flat Lines'

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)



# %% Sphere in Sphere in Sphere
file_name = 'Sphere in Spheres in Shell'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

sphere12 = make_sphere(radius=6, offset_x=0, offset_y=0, offset_z=0)
hole10 = make_sphere(radius=5, offset_x=0, offset_y=0, offset_z=0)
sphere8 = make_sphere(radius=4, offset_x=0, offset_y=0, offset_z=0)
spheres_a = merge_parts([sphere12, hole10, sphere8])

sphere6 = make_sphere(radius=3, offset_x=0, offset_y=0, offset_z=0)

a, b, both = display_interactions(spheres_a, sphere6)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)
cropped_list, crop_box = crop_quarter([a, b, both])
a_crop, b_crop, both_crop = cropped_list

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)
