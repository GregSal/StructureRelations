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
    PrincipleTargetSelectorConfig,
    RelationshipGraphPlanBuilder,
    SpringLayout,
    TargetOARBipartiteLayoutConfig,
    TargetOARPlanConfig,
    apply_layout_template,
    default_grouped_grid_template,
    evaluate_template_display_rules,
    get_layout_template,
    list_layout_templates,
    load_custom_template_from_file,
    relationship_spring_template,
    register_layout_template,
    target_oar_template,
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


def test_list_layout_templates_returns_registered_names() -> None:
    '''The template catalog should expose built-in templates by stable name.'''
    template_names = [item['name'] for item in list_layout_templates()]

    assert template_names[:4] == [
        'grouped_grid',
        'relationship_spring',
        'target_oar',
    ]


def test_load_optics_json_template_registers_and_lists_template() -> None:
    '''The shipped JSON template should load through the public loader.'''
    template_path = (
        Path(__file__).parents[1]
        / 'src'
        / 'layout_templates'
        / 'optics_template.json'
    )

    template = load_custom_template_from_file(template_path)

    assert template.name == 'Optics'
    assert get_layout_template('Optics') is template
    assert {'name': 'Optics'} in list_layout_templates()


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


class FakePrincipleTargetStructureSet:
    '''StructureSet with grouped targets for principle-target selection tests.'''

    def __init__(self) -> None:
        self.structure_metadata = pd.DataFrame()
        self.structure_filter_report = pd.DataFrame([
            {
                'ROINumber': 1,
                'Structure ID': 'PTV 56',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '56',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 2,
                'Structure ID': 'eval PTV 56 L a',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '56',
                'TargetType': 'PTV',
                'TargetLaterality': 'L',
                'TargetSubGroup': 'a',
                'Mod': 'eval',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 3,
                'Structure ID': 'opt PTV 56',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '56',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': 'opt',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 4,
                'Structure ID': 'CTV 66',
                'DICOM Type': 'CTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '66',
                'TargetType': 'CTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 5,
                'Structure ID': 'GTV 66',
                'DICOM Type': 'GTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '66',
                'TargetType': 'GTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 6,
                'Structure ID': 'PTV 70',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '70',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 7,
                'Structure ID': 'opt PTV 70',
                'DICOM Type': 'PTV',
                'Structure Code': '',
                'Coding Scheme': '',
                'Code Meaning': '',
                'TargetDose': '70',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': 'opt',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
        ]).set_index('ROINumber', drop=False)
        self.relationship_graph = nx.DiGraph()

    def summary(self) -> pd.DataFrame:
        '''Return target rows with explicit physical volumes for tie breaks.'''
        return pd.DataFrame([
            {'ROI': 1, 'Name': 'PTV 56', 'Physical_Volume': 25.0},
            {'ROI': 2, 'Name': 'eval PTV 56 L a', 'Physical_Volume': 20.0},
            {'ROI': 3, 'Name': 'opt PTV 56', 'Physical_Volume': 19.0},
            {'ROI': 4, 'Name': 'CTV 66', 'Physical_Volume': 15.0},
            {'ROI': 5, 'Name': 'GTV 66', 'Physical_Volume': 12.0},
            {'ROI': 6, 'Name': 'PTV 70', 'Physical_Volume': 24.0},
            {'ROI': 7, 'Name': 'opt PTV 70', 'Physical_Volume': 10.0},
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


def test_principle_targets_template_selects_one_per_group() -> None:
    '''One principle target should be selected per horizontal target group.'''
    structure_set = FakePrincipleTargetStructureSet()

    result = apply_layout_template(
        structure_set,
        target_oar_template(),
    )

    selected = result.plot_nodes.set_index('ROI')

    assert set(selected.index) == {2, 7}
    assert selected.loc[2, 'Name'] == 'eval PTV 56 L a'
    assert selected.loc[7, 'Name'] == 'opt PTV 70'
    assert selected['h_grouping'].tolist() == ['56', '70']


def test_principle_targets_template_uses_ctv_fallback_when_enabled() -> None:
    '''Fallback mode should pick CTV when a group has no PTV.'''
    structure_set = FakePrincipleTargetStructureSet()
    plan_config = TargetOARPlanConfig(
        selector_config=PrincipleTargetSelectorConfig(
            missing_ptv_mode='fallback_to_ctv_then_gtv'
            )
        )

    result = apply_layout_template(
        structure_set,
        target_oar_template(plan_config=plan_config),
    )

    selected = result.plot_nodes.set_index('ROI')
    assert set(selected.index) == {2, 4, 7}
    assert selected.loc[4, 'Name'] == 'CTV 66'


def test_grouped_grid_template_still_exposes_original_rules() -> None:
    '''Grouped-grid display behavior should remain backward compatible.'''
    result = apply_layout_template(
        FakePrincipleTargetStructureSet(),
        default_grouped_grid_template(),
    )

    assert set(result.plot_nodes['ROI']) == {1, 2, 3, 4, 5, 6, 7}


class FakeRelationshipType:
    '''Minimal relationship-type contract for Target-OAR tests.'''

    def __init__(self, relation_type: str, category: str, label: str) -> None:
        self.relation_type = relation_type
        self.category = category
        self.label = label


class FakeRelationship:
    '''Minimal relationship wrapper for graph edge metadata.'''

    def __init__(self, relation_type: str, category: str, label: str) -> None:
        self.relationship_type = FakeRelationshipType(
            relation_type=relation_type,
            category=category,
            label=label,
        )


class FakeTargetOARStructureSet:
    '''StructureSet with targets, OARs, and opt OARs for template tests.'''

    def __init__(self) -> None:
        self.structure_metadata = pd.DataFrame()
        self.structure_filter_report = pd.DataFrame([
            {
                'ROINumber': 1,
                'Structure ID': 'eval PTV 56',
                'DICOM Type': 'PTV',
                'TargetDose': '56',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': 'eval',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 2,
                'Structure ID': 'PTV 70',
                'DICOM Type': 'PTV',
                'TargetDose': '70',
                'TargetType': 'PTV',
                'TargetLaterality': '',
                'TargetSubGroup': '',
                'Mod': '',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 10,
                'Structure ID': 'Parotid L',
                'DICOM Type': 'ORGAN',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 11,
                'Structure ID': 'Mandible',
                'DICOM Type': 'ORGAN',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 12,
                'Structure ID': 'Larynx',
                'DICOM Type': 'ORGAN',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 13,
                'Structure ID': 'SpinalCord',
                'DICOM Type': 'ORGAN',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 20,
                'Structure ID': 'opt Parotid L',
                'DICOM Type': 'AVOIDANCE',
                'Mod': 'opt',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
            {
                'ROINumber': 21,
                'Structure ID': 'opt Larynx',
                'DICOM Type': 'AVOIDANCE',
                'Mod': 'opt',
                'SelectedByDefault': True,
                'DisplayedByDefault': True,
                'IsFiltered': False,
            },
        ]).set_index('ROINumber', drop=False)
        self.relationship_graph = nx.DiGraph()
        self.relationship_graph.add_edge(
            1,
            10,
            relationship=FakeRelationship('OVERLAPS', 'Shared', 'Overlaps with'),
        )
        self.relationship_graph.add_edge(
            2,
            11,
            relationship=FakeRelationship('OVERLAPS', 'Shared', 'Overlaps with'),
        )
        self.relationship_graph.add_edge(
            2,
            12,
            relationship=FakeRelationship('BORDERS', 'Adjoining', 'Borders'),
        )
        self.relationship_graph.add_edge(
            20,
            10,
            relationship=FakeRelationship('PARTITIONED', 'Adjoining', 'is Partitioned by'),
        )
        self.relationship_graph.add_edge(
            21,
            12,
            relationship=FakeRelationship('PARTITIONED', 'Adjoining', 'is Partitioned by'),
        )
        self.relationship_graph.add_edge(
            1,
            21,
            relationship=FakeRelationship('BORDERS', 'Adjoining', 'Borders'),
        )
        self.relationship_graph.add_edge(
            2,
            21,
            relationship=FakeRelationship('OVERLAPS', 'Shared', 'Overlaps with'),
        )

    def summary(self) -> pd.DataFrame:
        '''Return rows needed by Target-OAR template planning.'''
        return pd.DataFrame([
            {'ROI': 1, 'Name': 'eval PTV 56', 'Physical_Volume': 18.0},
            {'ROI': 2, 'Name': 'PTV 70', 'Physical_Volume': 25.0},
            {'ROI': 10, 'Name': 'Parotid L', 'Physical_Volume': 8.0},
            {'ROI': 11, 'Name': 'Mandible', 'Physical_Volume': 14.0},
            {'ROI': 12, 'Name': 'Larynx', 'Physical_Volume': 7.0},
            {'ROI': 13, 'Name': 'SpinalCord', 'Physical_Volume': 6.0},
            {'ROI': 20, 'Name': 'opt Parotid L', 'Physical_Volume': 4.0},
            {'ROI': 21, 'Name': 'opt Larynx', 'Physical_Volume': 3.0},
        ])


def test_target_oar_template_is_registered() -> None:
    '''The Target-OAR template should be retrievable by name.'''
    template = get_layout_template('target_oar')

    assert template.name == 'target_oar'


def test_target_oar_template_selects_targets_oars_and_opt_nodes() -> None:
    '''Template should select principle targets, ranked OARs, and matched opt nodes.'''
    result = apply_layout_template(
        FakeTargetOARStructureSet(),
        target_oar_template(),
    )

    nodes = result.plot_nodes.set_index('ROI')

    assert set(nodes.index) == {1, 2, 10, 11, 12, 20, 21}
    assert nodes.loc[1, 'node_side'] == 'target'
    assert nodes.loc[2, 'node_side'] == 'target'
    assert nodes.loc[10, 'node_side'] == 'oar'
    assert nodes.loc[20, 'node_side'] == 'opt'
    assert 13 not in nodes.index
    assert nodes.loc[10, 'weighted_oar_score'] == pytest.approx(-1.0)
    assert nodes.loc[11, 'weighted_oar_score'] == pytest.approx(1.0)
    assert nodes.loc[12, 'weighted_oar_score'] == pytest.approx(0.5)


def test_target_oar_bipartite_layout_positions_columns_and_midpoints() -> None:
    '''Target-OAR layout should place columns and opt-node midpoints deterministically.'''
    result = apply_layout_template(
        FakeTargetOARStructureSet(),
        target_oar_template(),
    )

    positions = result.positions

    assert positions[1][0] == pytest.approx(TargetOARBipartiteLayoutConfig().target_x)
    assert positions[2][0] == pytest.approx(TargetOARBipartiteLayoutConfig().target_x)
    assert positions[10][0] == pytest.approx(TargetOARBipartiteLayoutConfig().oar_x)
    assert positions[20][0] == pytest.approx(TargetOARBipartiteLayoutConfig().opt_x)
    assert positions[20][1] == pytest.approx(
        (positions[10][1] + positions[12][1]) / 2.0,
    )


def test_renderer_accepts_target_oar_template() -> None:
    '''Renderer should draw the Target-OAR template without special-case logic.'''
    settings_path = (
        Path(__file__).parents[1]
        / 'src'
        / 'webapp'
        / 'config'
        / 'diagram_settings.json'
    )

    result = render_template_diagram(
        structure_set=FakeTargetOARStructureSet(),
        layout_template=target_oar_template(),
        diagram_settings_path=settings_path,
        show_plot=False,
    )

    assert result.layout_template_name == 'target_oar'
    assert set(result.positions) == {1, 2, 10, 11, 12, 20, 21}
    result.fig.clear()
