'''Tests for multi-region per-region relationship computation.

Covers:
- StructureShape.relate_regions() returns a dict keyed by (region_idx, region_idx)
- StructureRelationship.per_region_relations is populated after calculate_relationships()
- StructureRelationship.has_multiple_regions() returns correct booleans
- StructureRelationship.region_relationship_types maps each pair to a RelationshipType
- StructureRelationship.display_label() bracketing logic
- StructureSet.describe_relationship() structure
- count_relations_by_rank() ordering and filtering
'''
# %% Imports
import pytest
from unittest.mock import PropertyMock, patch

from structure_set import StructureSet
from relations import (
    DE27IM, RELATIONSHIP_TYPES,
    RELATION_RANK, count_relations_by_rank,
    compute_region_pair_de27im,
)
from relationships import StructureRelationship
from debug_tools import make_vertical_cylinder, make_box
from types_and_classes import DEFAULT_TRANSVERSE_TOLERANCE


# %% Helper factories

def _two_disjoint_cylinders_vs_container(
    spacing: float = 0.5,
) -> StructureSet:
    '''Build a StructureSet where ROI 1 has two disjoint cylinders and ROI 2
    is a large box that contains both of them.

    ROI 1 — cylinder A: radius=1, centred at (-3, 0, 0)
    ROI 1 — cylinder B: radius=1, centred at (+3, 0, 0)
    ROI 2 — box:         10 x 4 x 2 cm centred at (0, 0, 0)

    Volume order: ROI 2 (box) > ROI 1 (two cylinders), so the directed
    edge stored in the graph is roi_a=2 → roi_b=1.
    '''
    cyl_a = make_vertical_cylinder(
        radius=1.0, length=2.0, spacing=spacing,
        offset_x=-3.0, offset_y=0.0, offset_z=0.0,
        roi_num=1,
    )
    cyl_b = make_vertical_cylinder(
        radius=1.0, length=2.0, spacing=spacing,
        offset_x=3.0, offset_y=0.0, offset_z=0.0,
        roi_num=1,
    )
    # make_box: width=x-dim, length=y-dim, height=z-dim
    container = make_box(
        width=10.0, length=4.0, height=2.0,
        offset_x=0.0, offset_y=0.0, offset_z=0.0,
        roi_num=2,
    )
    slices = cyl_a + cyl_b + container
    return StructureSet(slices)


def _two_cylinders_same_container(
    spacing: float = 0.5,
) -> StructureSet:
    '''ROI 1: single large cylinder; ROI 2: two smaller disjoint cylinders
    both inside ROI 1.  Each ROI-2 sub-cylinder is WITHIN ROI 1.

    Volume order: ROI 1 (large) > ROI 2 (small×2), so the directed
    edge is roi_a=1 → roi_b=2.
    '''
    outer = make_vertical_cylinder(
        radius=6.0, length=2.0, spacing=spacing,
        offset_x=0.0, offset_y=0.0, offset_z=0.0,
        roi_num=1,
    )
    inner_a = make_vertical_cylinder(
        radius=1.0, length=2.0, spacing=spacing,
        offset_x=-3.0, offset_y=0.0, offset_z=0.0,
        roi_num=2,
    )
    inner_b = make_vertical_cylinder(
        radius=1.0, length=2.0, spacing=spacing,
        offset_x=3.0, offset_y=0.0, offset_z=0.0,
        roi_num=2,
    )
    slices = outer + inner_a + inner_b
    return StructureSet(slices)


# %% Unit tests for count_relations_by_rank

class TestCountRelationsByRank:
    '''Unit tests for the count_relations_by_rank() helper.'''

    def test_single_type_returns_one_entry(self):
        rel = RELATIONSHIP_TYPES.get('CONTAINS') or RELATIONSHIP_TYPES.get('contains')
        # Build a mapping with 3 pairs all sharing the same type
        dummy = {(f'r{i}', 'r0'): rel for i in range(3)}
        result = count_relations_by_rank(dummy)
        assert len(result) == 1
        assert result[0][1] == 3

    def test_sorted_by_rank(self):
        contains = RELATIONSHIP_TYPES.get('CONTAINS') or RELATIONSHIP_TYPES.get('contains')
        disjoint = RELATIONSHIP_TYPES.get('DISJOINT') or RELATIONSHIP_TYPES.get('disjoint')
        dummy = {('r0', 'r0'): contains, ('r1', 'r1'): disjoint}
        result = count_relations_by_rank(dummy)
        assert result[0][0].relation_type == contains.relation_type
        assert result[1][0].relation_type == disjoint.relation_type

    def test_drop_disjoint(self):
        contains = RELATIONSHIP_TYPES.get('CONTAINS') or RELATIONSHIP_TYPES.get('contains')
        disjoint = RELATIONSHIP_TYPES.get('DISJOINT') or RELATIONSHIP_TYPES.get('disjoint')
        dummy = {('r0', 'r0'): contains, ('r1', 'r1'): disjoint}
        result = count_relations_by_rank(dummy, drop_disjoint=True)
        assert len(result) == 1
        assert result[0][0].relation_type == contains.relation_type

    def test_drop_overlaps(self):
        overlaps = RELATIONSHIP_TYPES.get('OVERLAPS') or RELATIONSHIP_TYPES.get('overlaps')
        disjoint = RELATIONSHIP_TYPES.get('DISJOINT') or RELATIONSHIP_TYPES.get('disjoint')
        dummy = {('r0', 'r0'): overlaps, ('r1', 'r1'): disjoint}
        result = count_relations_by_rank(dummy, drop_overlaps=True)
        assert len(result) == 1
        assert result[0][0].relation_type == disjoint.relation_type

    def test_empty_returns_empty(self):
        assert count_relations_by_rank({}) == []


# %% Unit tests for display_label

class TestDisplayLabel:
    '''Unit tests for StructureRelationship.display_label() bracketing rules.

    Uses unittest.mock to inject pre-computed RelationshipType values so that
    tests are decoupled from DE27IM bit-pattern internals.
    '''

    def test_no_multi_show_false_returns_string(self):
        rel = StructureRelationship()
        label = rel.display_label(show_multi=False)
        assert isinstance(label, str)

    def test_single_region_pair_no_brackets(self):
        '''Single region pair → all types same → no curly brackets.'''
        contains = RELATIONSHIP_TYPES.get('CONTAINS')
        assert contains is not None
        rel = StructureRelationship()
        with patch.object(
            type(rel), 'region_relationship_types',
            new_callable=PropertyMock,
            return_value={('r0', 'r0'): contains},
        ):
            with patch.object(
                type(rel), 'relationship_type',
                new_callable=PropertyMock,
                return_value=contains,
            ):
                label = rel.display_label(show_multi=True)
        assert '{' not in label
        assert contains.label in label

    def test_non_unknown_consolidated_uses_curly(self):
        '''Multiple types, consolidated NOT UNKNOWN → {consolidated_label}.'''
        contains = RELATIONSHIP_TYPES.get('CONTAINS')
        disjoint = RELATIONSHIP_TYPES.get('DISJOINT')
        assert contains is not None
        assert disjoint is not None
        # The consolidated type is CONTAINS (not UNKNOWN)
        rel = StructureRelationship()
        # Set two dummy region-pair entries so has_multiple_regions() returns True
        rel.per_region_relations = {'dummy1': None, 'dummy2': None}
        with patch.object(
            type(rel), 'region_relationship_types',
            new_callable=PropertyMock,
            return_value={('r0', 'r0'): contains, ('r1', 'r1'): disjoint},
        ):
            with patch.object(
                type(rel), 'relationship_type',
                new_callable=PropertyMock,
                return_value=contains,
            ):
                label = rel.display_label(show_multi=True)
        assert label == '{' + contains.label + '}'

    def test_unknown_consolidated_uses_ampersand(self):
        '''Multiple types, consolidated UNKNOWN → {label_1 & label_2 & ...}.'''
        contains = RELATIONSHIP_TYPES.get('CONTAINS')
        disjoint = RELATIONSHIP_TYPES.get('DISJOINT')
        unknown = RELATIONSHIP_TYPES.get('UNKNOWN')
        assert contains is not None
        assert disjoint is not None
        assert unknown is not None
        rel = StructureRelationship()
        # Set two dummy region-pair entries so has_multiple_regions() returns True
        rel.per_region_relations = {'dummy1': None, 'dummy2': None}
        with patch.object(
            type(rel), 'region_relationship_types',
            new_callable=PropertyMock,
            return_value={('r0', 'r0'): contains, ('r1', 'r1'): disjoint},
        ):
            with patch.object(
                type(rel), 'relationship_type',
                new_callable=PropertyMock,
                return_value=unknown,
            ):
                label = rel.display_label(show_multi=True)
        assert label.startswith('{')
        assert label.endswith('}')
        assert '&' in label


# %% Integration tests using synthetic geometries

class TestRelateRegionsMethod:
    '''Tests for StructureShape.relate_regions().'''

    @pytest.fixture(scope='class')
    def two_disjoint_in_box(self):
        return _two_disjoint_cylinders_vs_container()

    def test_relate_regions_returns_dict(self, two_disjoint_in_box):
        ss = two_disjoint_in_box
        # Volume order: ROI 2 (box) > ROI 1 (cylinders) → edge is 2→1
        structure_a = ss.structures[2]
        structure_b = ss.structures[1]
        result = structure_a.relate_regions(structure_b)
        assert isinstance(result, dict)

    def test_relate_regions_has_two_keys(self, two_disjoint_in_box):
        '''ROI 1 has two disjoint regions; ROI 2 has one.
        Expected: at least 2 distinct second-element region keys (one per cylinder)
        and at most one first-element key (single box region).
        RegionIndex values are per connected component and can change per slice
        pair, so we check uniqueness of second elements and minimum count.'''
        ss = two_disjoint_in_box
        structure_a = ss.structures[2]  # box (larger, single region)
        structure_b = ss.structures[1]  # two cylinders
        result = structure_a.relate_regions(structure_b)
        # Box should contribute only '2A' as the first element
        first_elements = {k[0] for k in result.keys()}
        assert first_elements == {'2A'}
        # Two cylinders contribute at least 2 distinct region labels on some slice
        second_elements = {k[1] for k in result.keys()}
        assert len(second_elements) >= 2

    def test_relate_regions_values_are_de27im(self, two_disjoint_in_box):
        ss = two_disjoint_in_box
        result = ss.structures[2].relate_regions(ss.structures[1])
        for v in result.values():
            assert isinstance(v, DE27IM)


class TestPerRegionInStructureSet:
    '''Tests that calculate_relationships() populates per_region_relations.

    Volume order for this fixture: ROI 2 (box) > ROI 1 (two cylinders).
    The directed edge in the graph is therefore roi_a=2 → roi_b=1.
    '''

    @pytest.fixture(scope='class')
    def two_in_box(self):
        return _two_disjoint_cylinders_vs_container()

    def test_per_region_relations_populated(self, two_in_box):
        rel = two_in_box.get_relationship(2, 1)
        assert rel is not None
        assert rel.per_region_relations is not None

    def test_has_multiple_regions_true(self, two_in_box):
        rel = two_in_box.get_relationship(2, 1)
        assert rel.has_multiple_regions() is True

    def test_region_relationship_types_non_empty(self, two_in_box):
        rel = two_in_box.get_relationship(2, 1)
        types = rel.region_relationship_types
        assert len(types) > 0
        for v in types.values():
            assert v is not None

    def test_describe_relationship_structure(self, two_in_box):
        desc = two_in_box.describe_relationship(2, 1)
        assert desc is not None
        assert 'consolidated_label' in desc
        assert 'has_multiple_regions' in desc
        assert 'per_region' in desc
        assert 'summary' in desc
        assert isinstance(desc['per_region'], list)
        assert isinstance(desc['summary'], list)

    def test_describe_relationship_missing_pair_returns_none(self, two_in_box):
        assert two_in_box.describe_relationship(99, 100) is None


class TestSingleRegionNoMulti:
    '''When both structures have one region, has_multiple_regions should be False.

    cyl_a (radius=2, ROI 1) is smaller, cyl_b (radius=5, ROI 2) is larger.
    Volume order: ROI 2 > ROI 1, so edge is 2→1 (CONTAINS).
    '''

    @pytest.fixture(scope='class')
    def simple_ss(self):
        cyl_a = make_vertical_cylinder(
            radius=2.0, length=2.0, spacing=0.5,
            offset_x=0.0, offset_y=0.0, offset_z=0.0,
            roi_num=1,
        )
        cyl_b = make_vertical_cylinder(
            radius=5.0, length=2.0, spacing=0.5,
            offset_x=0.0, offset_y=0.0, offset_z=0.0,
            roi_num=2,
        )
        return StructureSet(cyl_a + cyl_b)

    def test_not_multiple_regions(self, simple_ss):
        # Edge is roi_a=2 → roi_b=1 (larger contains smaller)
        rel = simple_ss.get_relationship(2, 1)
        assert rel is not None
        assert rel.has_multiple_regions() is False

    def test_display_label_no_brackets(self, simple_ss):
        rel = simple_ss.get_relationship(2, 1)
        label = rel.display_label(show_multi=True)
        assert '{' not in label
