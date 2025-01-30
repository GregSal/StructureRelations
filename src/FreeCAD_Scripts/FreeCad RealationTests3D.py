
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

plane_0 = add_slice_plane([outer_cylinder], slice_position=0.0,
                          display_style='Flat Lines', scale_factor=1.1)

crop_box = crop_half([a, b, both], quadrant='+Z')

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

plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
plane_3 = add_slice_plane([outer_cylinder], slice_position=-3.0, display_style='Flat Lines')

crop_box = crop_quarter([a, b, both], quarter = (1,-1,-1))

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Three Parallel Cylinders
from math import sqrt

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
a.ViewObject.Transparency = 0
b.ViewObject.Transparency = 0
intersect_color = (255,170,0)  # Orange
b.ViewObject.ShapeColor = intersect_color

start_plane = -4 * 10
length = 8 * 10
r = 1 * 10
x_offset = 2 * 10
y_offset = 2 * 10
d = 100 / sqrt(r**2 +  r**2)

V1 = App.Vector(x_offset, -y_offset-r, start_plane)
V2 = App.Vector(-x_offset, -y_offset-r, start_plane)
VC1 = App.Vector(V2.x-d, V2.y+10-d, start_plane)
VC2 = App.Vector(V2.x-d, V2.y+10+d, start_plane)
V3 = App.Vector(0, y_offset+r, start_plane)
VC3 = App.Vector(V3.x-d, V3.y-10+d, start_plane)
VC4 = App.Vector(V3.x+d, V3.y-10+d, start_plane)
V4 = App.Vector(x_offset+r, -y_offset, start_plane)
VC5 = App.Vector(V4.x-r+d, V4.y+d, start_plane)

L1 = Part.LineSegment(V1, V2)
C1 = Part.Arc(V2, VC1, VC2)
L2 = Part.LineSegment(VC2, VC3)
C2 = Part.Arc(VC3, V3, VC4)
L3 = Part.LineSegment(VC4, VC5)
C3 = Part.Arc(VC5, V4, V1)


S1 = Part.Shape([L1, C1, L2, C2, L3, C3])
W = Part.Wire(S1.Edges)
F = Part.Face(W)
hull_frame = F.extrude(App.Vector(0, 0, length))
hull = show_structure(hull_frame, 'Hull of a', color=(255,255,128), transparency=75,
                   display_as='Flat Lines',
                   line_style='Dotted', line_color=(0,0,0))
doc.recompute()
Gui.activeDocument().activeView().viewTop()
Gui.ActiveDocument.ActiveView.setAxisCross(False)
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Disjoint Boxes
file_name = 'Disjoint Boxes'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

left_cube = make_box(width=2, offset_x=-2, offset_y=0, offset_z=0)
right_cube = make_box(width=2, offset_x=2, offset_y=0, offset_z=0)
a, b, both = display_interactions(left_cube, right_cube)
a.ViewObject.Transparency = 0
b.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Extended Inner cylinder
file_name = 'Extended Inner cylinder'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_vertical_cylinder(radius=6, length=10)
cylinder_hole = make_vertical_cylinder(radius=5, length=10)
primary_cylinder = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_vertical_cylinder(radius=3, length=12)

a, b, both = display_interactions(primary_cylinder, surrounded_cylinder)
a.ViewObject.Transparency = 20
b.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Make Disjoint Parallel Cylinders
file_name = 'Disjoint Parallel Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_vertical_cylinder(radius=6, length=10)
cylinder_hole = make_vertical_cylinder(radius=5, length=8)
primary_structure = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_vertical_cylinder(radius=3, length=6)
disjoint_cylinder = make_vertical_cylinder(radius=3, length=6, offset_x=10)
secondary_structure = merge_parts([surrounded_cylinder, disjoint_cylinder])

a, b, both = display_interactions(primary_structure, secondary_structure)
b.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

crop_box = crop_quarter([a], quarter = (1,-1,-1))
doc.recompute()

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Disjoint Axial Cylinders
file_name = 'Disjoint Axial Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outer_cylinder = make_vertical_cylinder(radius=6, length=10)
cylinder_hole = make_vertical_cylinder(radius=5, length=8)
primary_structure = merge_parts([outer_cylinder, cylinder_hole])

surrounded_cylinder = make_vertical_cylinder(radius=3, length=6)
disjoint_cylinder = make_vertical_cylinder(radius=3, length=6, offset_z=12)
secondary_structure = merge_parts([surrounded_cylinder, disjoint_cylinder])

a, b, both = display_interactions(primary_structure, secondary_structure)
b.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

crop_box = crop_quarter([a], quarter = (1,-1,-1))
doc.recompute()

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Make Disjoint Concentric Cylinders
file_name = 'Disjoint Concentric Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

primary_cylinder = make_vertical_cylinder(radius=3, length=8)

bordering_cylinder1 = make_vertical_cylinder(radius=2, length=2, offset_z=7)
bordering_cylinder2 = make_vertical_cylinder(radius=2, length=2, offset_z=-7)
secondary_structure = merge_parts([bordering_cylinder1, bordering_cylinder2])

a, b, both = display_interactions(primary_cylinder, secondary_structure)
b.ViewObject.Transparency = 0
plane_6 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=6.0, display_style='Flat Lines')
plane_5 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=5.0, display_style='Flat Lines')
plane_4 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=4.0, display_style='Flat Lines')
plane_6i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-6.0, display_style='Flat Lines')
plane_5i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-5.0, display_style='Flat Lines')
plane_4i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-4.0, display_style='Flat Lines')

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Bordering Concentric Cylinders
file_name = 'Bordering Concentric Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

primary_cylinder = make_vertical_cylinder(radius=3, length=8)

bordering_cylinder1 = make_vertical_cylinder(radius=2, length=2, offset_z=6)
bordering_cylinder2 = make_vertical_cylinder(radius=2, length=2, offset_z=-6)
secondary_structure = merge_parts([bordering_cylinder1, bordering_cylinder2])

a, b, both = display_interactions(primary_cylinder, secondary_structure)
b.ViewObject.Transparency = 0
plane_5 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=5.0, display_style='Flat Lines')
plane_4 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=4.0, display_style='Flat Lines')
plane_5i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-5.0, display_style='Flat Lines')
plane_4i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-4.0, display_style='Flat Lines')

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Bordering Boxes
file_name = 'Bordering Boxes'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

left_cube = make_box(width=2, offset_x=-1, offset_y=0, offset_z=0)
right_cube = make_box(width=2, offset_x=1, offset_y=0, offset_z=0)
a, b, both = display_interactions(left_cube, right_cube)
a.ViewObject.Transparency = 10
b.ViewObject.Transparency = 10

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Bordering Boxes With Disjoint Box
file_name = 'Bordering Boxes With Disjoint Box'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

left_cube = make_box(width=2, offset_x=-1, offset_y=0, offset_z=0)
right_cube = make_box(width=2, offset_x=1, offset_y=0, offset_z=0)
disjoint_cube = make_box(width=2, offset_x=-4, offset_y=0, offset_z=0)
secondary_structure = merge_parts([right_cube, disjoint_cube])

a, b, both = display_interactions(left_cube, secondary_structure)
a.ViewObject.Transparency = 0
b.ViewObject.Transparency = 10

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Bordering Concentric Cylinder SUP Offset
file_name = 'Bordering Concentric Cylinder SUP Offset'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

left_cube = make_box(width=2, offset_x=-1, offset_y=0, offset_z=0)
right_cube = make_box(width=2, offset_x=1, offset_y=0, offset_z=0)
a, b, both = display_interactions(left_cube, right_cube)
a.ViewObject.Transparency = 10
b.ViewObject.Transparency = 10


primary_cylinder = make_vertical_cylinder(radius=0.2, length=0.4, offset_z=-0.5)
secondary_structure = make_vertical_cylinder(radius=0.2, length=0.4, offset_z=0)

a, b, both = display_interactions(primary_cylinder, secondary_structure)
b.ViewObject.Transparency = 20
b.ViewObject.Transparency = 20
plane_2 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-0.2, display_style='Flat Lines')
plane_3 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-0.3, display_style='Flat Lines')

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Overlapping Concentric Cylinders
file_name = 'Overlapping Concentric Cylinders'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

primary_cylinder = make_vertical_cylinder(radius=3, length=8)

overlapping_cylinder1 = make_vertical_cylinder(radius=2, length=2, offset_z=5)
overlapping_cylinder2 = make_vertical_cylinder(radius=2, length=2, offset_z=-5)
secondary_structure = merge_parts([overlapping_cylinder1, overlapping_cylinder2])

a, b, both = display_interactions(primary_cylinder, secondary_structure)
b.ViewObject.Transparency = 0
plane_4 = add_slice_plane([primary_cylinder, secondary_structure], slice_position=4.0, display_style='Flat Lines')
plane_4i = add_slice_plane([primary_cylinder, secondary_structure], slice_position=-4.0, display_style='Flat Lines')

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Confines Bordering Boxes
file_name = 'Confines Bordering Boxes'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6, offset_x=0, offset_y=0, offset_z=0)
box4 = make_box(width=4, offset_x=0, offset_y=0, offset_z=0)
primary_structure = merge_parts([box6, box4])

a, b, both = display_interactions(primary_structure, box4)

doc.recompute()
cropped_list, crop_box = crop_quarter([a], quarter = (1,-1,1))
b.ViewObject.Transparency = 0
cropped_list[0].ViewObject.Transparency = 25
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Confines Embedded Cylinder
file_name = 'Confines Embedded Cylinder'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

outside_cylinder = make_vertical_cylinder(radius=4, length=8)
center_hole = make_vertical_cylinder(radius=2, length=6)
primary_structure = merge_parts([outside_cylinder, center_hole])

middle_cylinder = make_vertical_cylinder(radius=1, length=6)

a, b, both = display_interactions(primary_structure, middle_cylinder)

doc.recompute()
cropped_list, crop_box = crop_quarter([a], quarter = (1,-1,1))
b.ViewObject.Transparency = 0
cropped_list[0].ViewObject.Transparency = 25
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Make Confines Embedded Spheres
file_name = 'Confines Embedded Spheres'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

sphere12 = make_sphere(radius=6, offset_x=0, offset_y=0, offset_z=0)
hole10 = make_sphere(radius=5, offset_x=0, offset_y=0, offset_z=0)
sphere8 = make_sphere(radius=4, offset_x=0, offset_y=0, offset_z=0)
primary_structure = merge_parts([sphere12, hole10, sphere8])

sphere10 = make_sphere(radius=5, offset_x=0, offset_y=0, offset_z=0)
hole8 = make_sphere(radius=4, offset_x=0, offset_y=0, offset_z=0)
secondary_structure = merge_parts([sphere10, hole8])

a, b, both = display_interactions(primary_structure, secondary_structure)

doc.recompute()
cropped_list, crop_box = crop_quarter([a, b], quarter = (1,-1,1))
cropped_list[0].ViewObject.Transparency = 0
cropped_list[1].ViewObject.Transparency = 0
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Confines Box in Box on z surface
file_name = 'Confines Box in Box on z surface'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6)
hole4 = make_box(width=4)

primary_structure = merge_parts([box6, hole4])

secondary_structure = make_box(width=2, offset_z=1)

a, b, both = display_interactions(primary_structure, secondary_structure)

doc.recompute()
cropped_list, crop_box = crop_quarter([a], quarter = (1,-1,1))
cropped_list[0].ViewObject.Transparency = 0
b.ViewObject.Transparency = 0
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Confines Box in Box on y surface
file_name = 'Confines Box in Box on y surface'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6)
hole4 = make_box(width=4)

primary_structure = merge_parts([box6, hole4])

secondary_structure = make_box(width=2, offset_y=1)

a, b, both = display_interactions(primary_structure, secondary_structure)

doc.recompute()
cropped_list, crop_box = crop_quarter([a], quarter = (1,-1,1))
cropped_list[0].ViewObject.Transparency = 0
b.ViewObject.Transparency = 0
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)


# %% Partition Box in Box on y surface
file_name = 'Partition Box in Box on y surface'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6)
box6_3 = make_box(width=3, length=6, height=6, offset_y=1.5)
a, b, both = display_interactions(box6, box6_3)
a.ViewObject.Transparency = 20
both.ViewObject.Transparency = 20

doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Partition Box in Box on z surface
file_name = 'Partition Box in Box on z surface'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6)
box6_3 = make_box(width=6, length=6, height=3, offset_z=1.5)
a, b, both = display_interactions(box6, box6_3)
a.ViewObject.Transparency = 20
both.ViewObject.Transparency = 20

doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Partition Sphere Island
file_name = 'Partition Sphere Island'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

sphere12 = make_sphere(radius=6, offset_x=0, offset_y=0, offset_z=0)
hole8 = make_sphere(radius=4, offset_x=0, offset_y=0, offset_z=0)
sphere4 = make_sphere(radius=2, offset_x=0, offset_y=0, offset_z=0)
primary_structure = merge_parts([sphere12, hole8, sphere4])

a, b, both = display_interactions(primary_structure, sphere4)

doc.recompute()
cropped_list, crop_box = crop_quarter([a, b], quarter = (1,-1,1))
cropped_list[0].ViewObject.Transparency = 0
cropped_list[1].ViewObject.Transparency = 0
both.ViewObject.Transparency = 20
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Overlapping Spheres
file_name = 'Overlapping Spheres'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

right_sphere6 = make_sphere(radius=6, offset_x=-2.0, offset_y=0, offset_z=0)
left_sphere6 = make_sphere(radius=6, offset_x=2.0, offset_y=0, offset_z=0)

a, b, both = display_interactions(right_sphere6, left_sphere6)
a.ViewObject.DisplayMode = "Shaded"
b.ViewObject.DisplayMode = "Shaded"
both.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewFront()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Overlapping Boxes in Y direction
file_name = 'Overlapping Boxes in Y direction'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=0.6)
box6_y = make_box(width=0.6, offset_y=0.2)
a, b, both = display_interactions(box6, box6_y)
both.ViewObject.Transparency = 20
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Overlapping Boxes in Z direction
file_name = 'Overlapping Boxes in Z direction'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=0.6)
box6_y = make_box(width=0.6, offset_z=0.3)
a, b, both = display_interactions(box6, box6_y)
both.ViewObject.Transparency = 20
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Stacked Boxes
file_name = 'Stacked Boxes'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=0.6)
box6_y = make_box(width=0.6, offset_z=0.6)
a, b, both = display_interactions(box6, box6_y)
both.ViewObject.Transparency = 20
doc.recompute()

box8 = make_box(width=0.8)
plane_3 = add_slice_plane([box8], slice_position=0.3, display_style='Flat Lines')

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Overlapping Cubes INF RT
file_name = 'Overlapping Cubes INF RT'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

box6 = make_box(width=6)
box6_y = make_box(width=6, offset_z=-3, offset_x=3)
a, b, both = display_interactions(box6, box6_y)
both.ViewObject.Transparency = 20
doc.recompute()

Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")

Gui.ActiveDocument.ActiveView.setAxisCross(True)

Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)

# %% Make Overlapping Extended Cylinder
file_name = 'Overlapping Extended Cylinder'
image_file_path = IMAGE_PATH + "//" + file_name + ".png"
fcad_file_path = SCRIPT_PATH + "//" + file_name + ".FCStd"
doc = App.newDocument(fcad_file_path)

primary_cylinder = make_vertical_cylinder(radius=3, length=8)

primary_cylinder = make_vertical_cylinder(radius=5, length=7)
overlapping_cylinder = make_vertical_cylinder(radius=3, length=9)

a, b, both = display_interactions(primary_cylinder, overlapping_cylinder)
b.ViewObject.Transparency = 0

doc.recompute()
Gui.activeDocument().activeView().viewIsometric()
Gui.SendMsgToActiveView("ViewFit")
Gui.ActiveDocument.ActiveView.setAxisCross(True)


Gui.ActiveDocument.ActiveView.saveImage(image_file_path)
App.activeDocument().saveAs(fcad_file_path)
