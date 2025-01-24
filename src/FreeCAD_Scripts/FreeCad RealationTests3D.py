
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

#plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
#plane_4 = add_slice_plane([outer_cylinder], slice_position=-4.0, display_style='Flat Lines')
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

#plane_5 = add_slice_plane([outer_cylinder], slice_position=-5.0)
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

left_cube = make_box(width=2, offset_x=-3, offset_y=0, offset_z=0)
right_cube = make_box(width=2, offset_x=3, offset_y=0, offset_z=0)
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
