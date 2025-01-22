
# %%  Create a new document
from shapely import length
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

primary_cylinder = make_vertical_cylinder(radius=5, length=0.7,
                                          offset_x=0, offset_y=0, offset_z=0)
contained_cylinder = make_vertical_cylinder(radius=3, length=0.5,
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

# %% Make Concentric Surrounding cylinders
file_name = 'Surrounded cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_vertical_cylinder(radius=6, length=10)
cylinder_hole = make_vertical_cylinder(radius=5, length=8)
primary_cylinder = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_vertical_cylinder(radius=3, length=6)

a, b, both = display_interactions(primary_cylinder, surrounded_cylinder)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
plane_3 = add_slice_plane([outer_cylinder], slice_position=-3.0, display_style='Flat Lines')

crop_box = crop_quarter([a, b, both], quarter = (1,-1,-1))

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Concentric Surrounding cylinders
file_name = 'Horizontal Surrounded cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_horizontal_cylinder(radius=6, length=10)
cylinder_hole = make_horizontal_cylinder(radius=5, length=8)
primary_cylinder = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_horizontal_cylinder(radius=3, length=6)

a, b, both = display_interactions(primary_cylinder, surrounded_cylinder)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

crop_box = crop_quarter([a, b, both], quarter = (1,-1,1))

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Sphere in Sphere in Sphere
file_name = 'Sphere in Shell'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

sphere12 = make_sphere(radius=6, offset_x=0, offset_y=0, offset_z=0)
hole10 = make_sphere(radius=5, offset_x=0, offset_y=0, offset_z=0)
spheres_a = merge_parts([sphere12, hole10])

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

# %% Sphere in Sphere in Box
file_name = 'Sphere in Cylinder in Box'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

cube6 = make_box(width=10)
left_cylinder = make_vertical_cylinder(radius=2, length=8, offset_x=-2.5)
right_cylinder = make_vertical_cylinder(radius=2, length=8, offset_x=2.5)
holes_in_box = merge_parts([cube6, left_cylinder, right_cylinder])

right_sphere = make_sphere(radius=1, offset_x=2.5)

a, b, both = display_interactions(holes_in_box, right_sphere)
both.ViewObject.Transparency = 0
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Concentric Sheltered cylinder
file_name = 'Sheltered cylinder'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_vertical_cylinder(radius=6, length=10)
cylinder_hole = make_vertical_cylinder(radius=5, length=10)
primary_cylinder = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_vertical_cylinder(radius=3, length=6)

a, b, both = display_interactions(primary_cylinder, surrounded_cylinder)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

#plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
plane_3 = add_slice_plane([outer_cylinder], slice_position=-3.0, display_style='Flat Lines')

#crop_box = crop_quarter([a, b, both], quarter = (1,-1,-1))

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Concentric Sheltered horizontal cylinder
file_name = 'Sheltered Horizontal cylinder'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_horizontal_cylinder(radius=6, length=10)
cylinder_hole = make_horizontal_cylinder(radius=5, length=10)
primary_cylinder = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_horizontal_cylinder(radius=3, length=6)

a, b, both = display_interactions(primary_cylinder, surrounded_cylinder)
doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

#plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
#plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
plane_0 = add_slice_plane([outer_cylinder], slice_position=0.0,
                          display_style='Flat Lines', scale_factor=1.1)

crop_box = crop_half([a, b, both], quadrant='+Z')

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Three Parallel Cylinders
file_name = 'Three Parallel Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

first_cylinder = make_vertical_cylinder(radius=1, length=10,
                                        offset_x=-2, offset_y=-2)
second_cylinder = make_vertical_cylinder(radius=1, length=10,
                                        offset_x=2, offset_y=-2)
third_cylinder = make_vertical_cylinder(radius=1, length=10,
                                        offset_x=0, offset_y=2)
cylinder_set = merge_parts([first_cylinder, second_cylinder, third_cylinder])

central_cylinder = make_vertical_cylinder(radius=1, length=10, offset_y=-0.5)

a, b, both = display_interactions(cylinder_set, central_cylinder)

hull_box = make_box(length=4, width=2, height=10,
                                        offset_x=0, offset_y=-2)
hull = show_structure(hull_box, 'Hull of a1', color=(255,255,0), transparency=75,
                   display_as='Flat Lines',
                   line_style='Dotted', line_color=(0,0,0))
hull_box_2 = make_box(length=4.4, width=2, height=10,
                                        offset_x=0, offset_y=0)
rotation = (App.Vector(0, 0, 1), 65.4)
placement = App.Placement(App.Vector(0, -2, 0), App.Rotation(*rotation))
#hull_box_2 = hull_box_2.rotate(App.Vector(0, -2, 0), *rotation)
hull_box_2.Placement = placement
#hull_box_2 = hull_box_2.applyTranslation(App.Vector(0, 0, 1))
hull2 = show_structure(hull_box_2, 'Hull of a2', color=(255,255,0), transparency=75,
                   display_as='Flat Lines',
                   line_style='Dotted', line_color=(0,0,0))


doc.recompute()
hull_box_3 = make_box(length=4.4, width=2, height=10,
                                        offset_x=0, offset_y=0)
rotation = (App.Vector(0, 0, 1), -65.4)
#placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(*rotation))
hull_box_3 = hull_box_3.rotate(App.Vector(0, -2, 0), *rotation)
hull3 = show_structure(hull_box_3, 'Hull of a2', color=(255,255,0), transparency=75,
                   display_as='Flat Lines',
                   line_style='Dotted', line_color=(0,0,0))
#hull_box_3.Placement = placement

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

#plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
#plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
plane_0 = add_slice_plane([outer_cylinder], slice_position=0.0,
                          display_style='Flat Lines', scale_factor=1.1)

crop_box = crop_half([a, b, both], quadrant='+Z')

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)
