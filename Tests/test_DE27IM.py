'''Test the DE27IM class.'''
import json
from pathlib import Path

import networkx as nx
import pytest
import shapely

from contours import Contour, ContourMatch
from debug_tools import circle_points
from region_slice import RegionSlice
from relations import DE9IM, DE27IM, RELATIONSHIP_TYPES


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


def _transpose_27bit(value: int) -> int:
    bits = list(f'{value:027b}')
    for offset in (0, 9, 18):
        for idx_a, idx_b in ((1, 3), (2, 6), (5, 7)):
            a = offset + idx_a
            b = offset + idx_b
            bits[a], bits[b] = bits[b], bits[a]
    return int(''.join(bits), 2)


def _find_primary_test(relation_name: str):
    relation_type = RELATIONSHIP_TYPES[relation_name]
    for test_binary in DE27IM.test_binaries:
        if test_binary.relation_type.relation_type != relation_name:
            continue
        if test_binary.relation_type.reversed_arrow:
            continue
        return relation_type, test_binary
    raise AssertionError(f'Primary test binary missing for {relation_name}')


class TestRelationshipDefinitionTranspose:
    @pytest.mark.parametrize(
        'relation_name',
        [
            relation.relation_type
            for relation in RELATIONSHIP_TYPES.values()
            if relation.symmetric and relation.relation_type != 'UNKNOWN'
        ],
    )
    def test_runtime_transpose_identity_for_symmetric(self, relation_name):
        relation_type, primary_test = _find_primary_test(relation_name)
        transposed = primary_test.transpose(relation_type=relation_type)
        assert transposed.mask == primary_test.mask
        assert transposed.value == primary_test.value

    @pytest.mark.parametrize(
        'relation_name',
        ['DISJOINT', 'BORDERS', 'EQUAL', 'OVERLAPS'],
    )
    def test_symmetric_core_set_identity(self, relation_name):
        relation_type, primary_test = _find_primary_test(relation_name)
        transposed = primary_test.transpose(relation_type=relation_type)
        assert transposed.mask == primary_test.mask
        assert transposed.value == primary_test.value

    def test_json_symmetric_definitions_are_transpose_invariant(self):
        json_path = Path(__file__).resolve().parents[1] / 'src' / 'relationship_definitions.json'
        data = json.loads(json_path.read_text(encoding='utf-8'))
        for definition in data['Relationships']:
            if definition.get('relation_type') == 'UNKNOWN':
                continue
            if not definition.get('symmetric', False):
                continue
            mask_str = definition.get('mask', '')
            value_str = definition.get('value', '')
            if not mask_str or not value_str:
                continue
            mask = int(mask_str, 2)
            value = int(value_str, 2)
            assert _transpose_27bit(mask) == mask
            assert _transpose_27bit(value) == value

    def test_asymmetric_control_contains_to_within(self):
        contains_type, contains_primary = _find_primary_test('CONTAINS')
        within_type = RELATIONSHIP_TYPES['WITHIN']
        transposed = contains_primary.transpose(relation_type=within_type)
        assert ((transposed.mask != contains_primary.mask) or
                (transposed.value != contains_primary.value))
        assert transposed.relation_type != contains_type
