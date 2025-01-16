import shapely

from types_and_classes import RegionNode, SliceNeighbours

from relations import DE27IM, DE9IM
from debug_tools import circle_points

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

def test_de27im_initialization_with_regions():
    roi = 1
    slice_index = 0.5
    slice_neighbours = SliceNeighbours(slice_index, 0, 1)
    poly_a = RegionNode(roi, slice_index, slice_neighbours,
                        shapely.Polygon(circle_points(2)))
    poly_b = shapely.Polygon(circle_points(2))
    de27im = DE27IM(poly_a, poly_b)
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
    merged_de27im = de27im1.merge(de27im2)
    expected_relation = '111111000111111000111111000'
    assert merged_de27im.relation == expected_relation
    assert merged_de27im.int == int(expected_relation, 2)

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
