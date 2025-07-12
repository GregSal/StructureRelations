'''Test the DE27IM class.'''
import networkx as nx
import shapely
from matplotlib import contour

from contours import Contour, ContourMatch, SliceNeighbours
from debug_tools import circle_points
from region_slice import RegionSlice
from relations import DE9IM, DE27IM


def build_region_slice(roi=1, radius=2):
    '''Builds a RegionSlice with three contours and their matches.

    Returns:
        tuple: (contour_graph, contour0, contour1, contour2)
    '''
    # Create a polygon with a circle of the specified radius
    poly = shapely.Polygon(circle_points(radius))
    # Make contours with the specified roi at different slice indices
    contour0 = Contour(roi, slice_index=0, polygon=poly, existing_contours=[])
    contour1 = Contour(roi, slice_index=1, polygon=poly, existing_contours=[])
    contour2 = Contour(roi, slice_index=2, polygon=poly, existing_contours=[])
    # Create an ContourGraph with the contours
    contour_graph = nx.Graph()
    contour_graph.add_node(contour0.index, contour=contour0)
    contour_graph.add_node(contour1.index, contour=contour1)
    contour_graph.add_node(contour2.index, contour=contour2)
    # Add edges to the graph
    contour_match01 = ContourMatch(contour0, contour1)
    contour_match12 = ContourMatch(contour1, contour2)
    contour_graph.add_edge(contour0.index, contour1.index, match=contour_match01)
    contour_graph.add_edge(contour0.index, contour1.index, match=contour_match01)
    contour_graph.add_edge(contour1.index, contour2.index, match=contour_match12)
    # Create a RegionSlice from the contour graph
    region_slice = RegionSlice(contour_graph, slice_index=1)
    return region_slice


def test_de27im_initialization_with_contours():
    poly_a = shapely.Polygon(circle_points(2))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation is not None
    assert de27im.int is not None

def test_de27im_padding_with_contours():
    poly_a = shapely.Polygon(circle_points(2))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation[9:]  == '0' * 18
    assert de27im.relation[:9]  == '100010001'

def test_de27im_initialization_with_contour():
    roi = 1
    slice_index = 0.5
    poly_a = Contour(roi, slice_index, shapely.Polygon(circle_points(2)), [])
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
    assert de27im.relation is not None
    assert de27im.int is not None

def test_de27im_initialization_with_region_slice():
    region_slice_a = build_region_slice(roi=0, radius=2)
    region_slice_b = build_region_slice(roi=1, radius=1)
    de27im = DE27IM(region_slice_a, region_slice_b)
    assert de27im.relation is not None
    assert de27im.int is not None

def test_de27im_initialization_with_relation_str():
    relation_str = '111000000111000000111000000'
    de27im = DE27IM(relation_str=relation_str)
    assert de27im.relation == relation_str
    assert de27im.int == int(relation_str, 2)

def test_de27im_initialization_with_relation_int():
    relation_str = '111000000111000000111000000'
    relation_int = int(relation_str, 2)
    de27im = DE27IM(relation_int=relation_int)
    assert de27im.relation == relation_str
    assert de27im.int == relation_int

def test_de27im_to_str():
    relation_str = '111000000111000000111000000'
    relation_int = int(relation_str, 2)
    relation_str = DE27IM.to_str(relation_int)
    assert relation_str == relation_str

def test_de27im_to_int():
    relation_str = '111000000111000000111000000'
    relation_int = DE27IM.to_int(relation_str)
    assert relation_int == int(relation_str, 2)

def test_de27im_identify_relation():
    relation_int = DE27IM.test_binaries[7].value
    de27im = DE27IM(relation_int=relation_int)
    assert de27im.identify_relation() == DE27IM.test_binaries[7].relation_type

def test_de27im_merge():
    relation_str1 = '111000000111000000111000000'
    relation_str2 = '000111000000111000000111000'
    de27im1 = DE27IM(relation_str=relation_str1)
    de27im2 = DE27IM(relation_str=relation_str2)
    de27im1.merge(de27im2)
    expected_relation = '111111000111111000111111000'
    assert de27im1.relation == expected_relation
    assert de27im1.int == int(expected_relation, 2)

def test_de27im_apply_adjustments_boundary_a():
    initial_relation_group = (DE9IM(relation_str='FF2FF1212'),
                              DE9IM(relation_str='212FF1FF2'),
                              DE9IM(relation_str='212FF1FF2'))
    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                               ['boundary_a'])
    expected_relation = (DE9IM(relation_str='FFFFF2212'),
                         DE9IM(relation_str='FFF212FF2'),
                         DE9IM(relation_str='FFF212FF2'))
    assert adjusted_de27im == expected_relation

def test_de27im_apply_adjustments_boundary_b():
    initial_relation_group = (DE9IM(relation_str='FF2FF1212'),
                              DE9IM(relation_str='212FF1FF2'),
                              DE9IM(relation_str='212FF1FF2'))
    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                               ['boundary_b'])
    expected_relation = (DE9IM(relation_str='FF2FF1F22'),
                         DE9IM(relation_str='F22FF1FF2'),
                         DE9IM(relation_str='F22FF1FF2'))
    assert adjusted_de27im == expected_relation

def test_de27im_apply_adjustments_both_boundaries():
    initial_relation_group = (DE9IM(relation_str='FF2FF1212'),
                              DE9IM(relation_str='212FF1FF2'),
                              DE9IM(relation_str='212FF1FF2'))
    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                               ['boundary_a', 'boundary_b'])
    expected_relation = (DE9IM(relation_str='FFFFF2F22'),
                         DE9IM(relation_str='FFFF22FF2'),
                         DE9IM(relation_str='FFFF22FF2'))
    assert adjusted_de27im == expected_relation

def test_de27im_apply_adjustments_hole_a():
    initial_relation_group = (DE9IM(relation_str='FF2FF1212'),
                              DE9IM(relation_str='FFFFFFFFF'),
                              DE9IM(relation_str='FFFFFFFFF'))

    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                            ['hole_a'])
    hole_a_adjustment = (DE9IM(relation_str='FFFFF1FF2'),
                        DE9IM(relation_str='FFFFFFFFF'),
                        DE9IM(relation_str='FFFFFFFFF'))
    assert adjusted_de27im == hole_a_adjustment

def test_de9im_apply_adjustments_hole_b():
    initial_relation_group = (DE9IM(relation_str='FF2FF1212'),
                            DE9IM(relation_str='FFFFFFFFF'),
                            DE9IM(relation_str='FFFFFFFFF'))

    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                            ['hole_b'])
    hole_b_adjustment = (DE9IM(relation_str='FFFFFFF12'),
                        DE9IM(relation_str='FFFFFFFFF'),
                        DE9IM(relation_str='FFFFFFFFF'))
    assert adjusted_de27im == hole_b_adjustment

def test_de9im_apply_adjustments_transpose():
    initial_relation_group = (DE9IM(relation_str='212FF1FF2'),
                              DE9IM(relation_str='212FF1FF2'),
                              DE9IM(relation_str='212FF1FF2'))

    de27im = DE27IM(relation_int=0)
    adjusted_de27im = de27im.apply_adjustments(initial_relation_group,
                                            ['transpose'])
    transpose_adjustment = (DE9IM(relation_str='2FF1FF212'),
                            DE9IM(relation_str='2FF1FF212'),
                            DE9IM(relation_str='2FF1FF212'))
    assert adjusted_de27im == transpose_adjustment
