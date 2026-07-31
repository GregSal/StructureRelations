'''Tests for metadata-driven diagram layout templates.'''

from pathlib import Path

import matplotlib
import networkx as nx
import pandas as pd
import pytest

matplotlib.use('Agg')

from diagram_layout import (
    GroupedGridLayout,
    GroupedGridLayoutConfig,
    LayoutTemplate,
    MetadataCondition,
    MetadataDisplayRule,
    RelationshipGraphPlanBuilder,
    SpringLayout,
    apply_layout_template,
    default_grouped_grid_template,
    evaluate_template_display_rules,
    get_layout_template,
    relationship_spring_template,
    register_layout_template,
)
from diagram_rendering import render_template_diagram
from structure_grouping import default_structure_group_definition_set


def make_template(*rules: MetadataDisplayRule) -> LayoutTemplate:
    '''Build a test template with the grouped-grid algorithm.'''
    return LayoutTemplate(
        name='test_template',
        display_by_default=False,
        display_rules=rules,
        grouping_definition_set=default_structure_group_definition_set(),
        algorithm=GroupedGridLayout(),
    )


def test_template_rules_preserve_selection_and_use_last_match() -> None:
    '''Template visibility should not alter original selection decisions.'''
    metadata = pd.DataFrame([
        {
            'ROINumber': 1,
            'Structure ID': 'PTV56',
            'DICOM Type': 'PTV',
            'SelectedByDefault': False,
            'IsFiltered': True,
            'DisplayedByDefault': True,
        },
    ]).set_index('ROINumber', drop=False)
    template = make_template(
        MetadataDisplayRule(
            rule_id='display-target',
            action='display',
            condition=MetadataCondition(
                field='DICOM Type',
                match_type='exact',
                value='PTV',
            ),
        ),
        MetadataDisplayRule(
            rule_id='hide-56',
            action='hide',
            condition=MetadataCondition(
                field='Structure ID',
                match_type='suffix',
                value='56',
            ),
        ),
    )

    result = evaluate_template_display_rules(metadata, template)

    assert not bool(result.loc[1, 'DisplayedByDefault'])
    assert bool(result.loc[1, 'IsHidden'])
    assert not bool(result.loc[1, 'SelectedByDefault'])
    assert bool(result.loc[1, 'IsFiltered'])
    assert result.loc[1, 'TemplateMatchedRules'] == [
        'display-target',
        'hide-56',
    ]
    assert result.loc[1, 'TemplateFinalMatch'] == 'hide-56'
    assert bool(metadata.loc[1, 'DisplayedByDefault'])


def test_grouped_grid_layout_computes_expected_positions() -> None:
    '''Grid coordinates should use all three placement indices.'''
    nodes = pd.DataFrame([
        {'ROI': 1, 'h_index': 2, 'v_index': 3, 'v_dup_index': 1},
        {'ROI': 2, 'h_index': 0, 'v_index': 1, 'v_dup_index': 0},
    ])
    algorithm = GroupedGridLayout(
        GroupedGridLayoutConfig(
            horizontal_spacing=3.0,
            vertical_spacing=2.0,
            duplicate_vertical_spread=0.5,
        ),
    )

    positions = algorithm.compute_positions(nodes, nx.Graph())

    assert positions == {1: (6.0, -6.5), 2: (0.0, -2.0)}


def test_default_template_is_available_by_registered_name() -> None:
    '''The built-in grouped grid should support stable name lookup.'''
    template = get_layout_template('grouped_grid')

    assert template.name == default_grouped_grid_template().name


def test_register_layout_template_rejects_duplicate_name() -> None:
    '''Registry names should be unique.'''
    template = make_template()
    register_layout_template(template)

    with pytest.raises(ValueError, match='already registered'):
        register_layout_template(template)


class FakeStructureSet:
    '''Minimal StructureSet contract for template application.'''

    def __init__(self) -> None:
        self.structure_metadata = pd.DataFrame()
        self.structure_filter_report = pd.DataFrame([
            {
                'ROINumber': 1,
                'Structure ID': 'PTV56',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': False,
                'IsFiltered': False,
            },
            {
                'ROINumber': 2,
                'Structure ID': 'Rectum',
                'DICOM Type': 'ORGAN',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
        ]).set_index('ROINumber', drop=False)
        self.relationship_graph = nx.DiGraph()
        self.relationship_graph.add_edge(1, 2)

    def summary(self) -> pd.DataFrame:
        '''Return the available rendered structures.'''
        return pd.DataFrame([
            {'ROI': 1, 'Name': 'PTV56'},
            {'ROI': 2, 'Name': 'Rectum'},
        ])


def test_apply_layout_template_filters_nodes_and_positions() -> None:
    '''Only template-displayed structures should receive positions.'''
    result = apply_layout_template(
        FakeStructureSet(),
        default_grouped_grid_template(),
    )

    assert result.template_name == 'grouped_grid'
    assert result.plot_nodes['ROI'].tolist() == [1]
    assert result.positions == {1: (0.0, -0.0)}
    assert bool(result.display_report.loc[1, 'DisplayedByDefault'])
    assert not bool(result.display_report.loc[2, 'DisplayedByDefault'])


def test_grouped_grid_template_hides_x_prefixed_targets() -> None:
    '''Resident contours should remain hidden even when target rules match.'''
    metadata = pd.DataFrame([
        {
            'ROINumber': 1,
            'Structure ID': 'x PTV56',
            'DICOM Type': 'PTV',
            'SelectedByDefault': False,
            'DisplayedByDefault': False,
        },
    ]).set_index('ROINumber', drop=False)

    result = evaluate_template_display_rules(
        metadata,
        default_grouped_grid_template(),
    )

    assert not bool(result.loc[1, 'DisplayedByDefault'])
    assert result.loc[1, 'TemplateFinalMatch'] == 'resident-contours'


def test_apply_layout_template_preserves_unparsed_visible_structures() -> None:
    '''Grouping preparation should not drop non-target metadata rows.'''
    template = LayoutTemplate(
        name='all_structures',
        display_by_default=True,
        display_rules=(),
        grouping_definition_set=default_structure_group_definition_set(),
        algorithm=GroupedGridLayout(),
    )

    result = apply_layout_template(FakeStructureSet(), template)

    assert set(result.plot_nodes['ROI']) == {1, 2}
    assert set(result.positions) == {1, 2}


def test_renderer_uses_required_layout_template() -> None:
    '''Renderer output should expose template-selected nodes and positions.'''
    settings_path = (
        Path(__file__).parents[1]
        / 'src'
        / 'webapp'
        / 'config'
        / 'diagram_settings.json'
    )

    result = render_template_diagram(
        structure_set=FakeStructureSet(),
        layout_template=default_grouped_grid_template(),
        diagram_settings_path=settings_path,
        show_plot=False,
    )

    assert result.layout_template_name == 'grouped_grid'
    assert set(result.relationship_graph.nodes) == {1}
    assert result.positions == {1: (0.0, -0.0)}
    assert result.plot_nodes['ROI'].tolist() == [1]
    assert bool(result.display_report.loc[1, 'DisplayedByDefault'])
    assert set(result.layout_graph.nodes) == {1}
    result.fig.clear()


class FakeRelationshipStructureSet(FakeStructureSet):
    '''StructureSet contract with typed relationship edges.'''

    def __init__(self) -> None:
        super().__init__()
        third_row = pd.DataFrame([
            {
                'ROINumber': 3,
                'Structure ID': 'CTV56',
                'DICOM Type': 'CTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
        ]).set_index('ROINumber', drop=False)
        self.structure_filter_report = pd.concat([
            self.structure_filter_report,
            third_row,
        ])
        self.relationship_graph = nx.DiGraph()
        self.relationship_graph.add_edge(1, 2, relation_type='CONTAINS')
        self.relationship_graph.add_edge(3, 1, relation_type='OVERLAPS')

    def summary(self) -> pd.DataFrame:
        '''Return three structures participating in relationships.'''
        return pd.DataFrame([
            {'ROI': 1, 'Name': 'PTV56'},
            {'ROI': 2, 'Name': 'Rectum'},
            {'ROI': 3, 'Name': 'CTV56'},
        ])


def test_relationship_spring_template_uses_graph_in_layout_plan() -> None:
    '''Relationship plans should expose graph features and stable positions.'''
    structure_set = FakeRelationshipStructureSet()
    template = relationship_spring_template()

    first_result = apply_layout_template(structure_set, template)
    second_result = apply_layout_template(structure_set, template)
    nodes = first_result.plot_nodes.set_index('ROI')

    assert set(first_result.layout_graph.edges) == {(1, 2), (3, 1)}
    assert nodes.loc[1, 'relationship_in_degree'] == 1
    assert nodes.loc[1, 'relationship_out_degree'] == 1
    assert nodes.loc[1, 'relationship_types'] == ('CONTAINS', 'OVERLAPS')
    assert first_result.positions == second_result.positions
    assert set(first_result.positions) == {1, 2, 3}


def test_relationship_plan_can_filter_layout_edges_by_type() -> None:
    '''A template may choose which relationship types influence placement.'''
    template = LayoutTemplate(
        name='contains_spring',
        display_by_default=True,
        display_rules=(),
        grouping_definition_set=None,
        algorithm=SpringLayout(),
        plan_builder=RelationshipGraphPlanBuilder(
            relationship_types=('CONTAINS',),
        ),
    )

    result = apply_layout_template(FakeRelationshipStructureSet(), template)
    nodes = result.plot_nodes.set_index('ROI')

    assert set(result.layout_graph.edges) == {(1, 2)}
    assert nodes.loc[1, 'relationship_types'] == ('CONTAINS',)
    assert nodes.loc[3, 'relationship_degree'] == 0


def test_renderer_accepts_relationship_plan_without_grid_columns() -> None:
    '''Rendering should not require metadata-grid placement columns.'''
    settings_path = (
        Path(__file__).parents[1]
        / 'src'
        / 'webapp'
        / 'config'
        / 'diagram_settings.json'
    )

    result = render_template_diagram(
        structure_set=FakeRelationshipStructureSet(),
        layout_template=relationship_spring_template(),
        diagram_settings_path=settings_path,
        show_plot=False,
    )

    assert result.layout_template_name == 'relationship_spring'
    assert set(result.layout_graph.edges) == {(1, 2), (3, 1)}
    assert set(result.positions) == {1, 2, 3}
    assert 'h_index' not in result.plot_nodes.columns
    result.fig.clear()
