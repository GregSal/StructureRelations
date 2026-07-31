'''Metadata-driven node selection and positioning for relationship diagrams.'''

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Protocol

import networkx as nx
import pandas as pd

from structure_grouping import (
    GroupDefinitionSet,
    build_structure_grouping_table,
    default_structure_group_definition_set,
    prepare_structure_grouping_source,
)


DisplayAction = Literal['hide', 'display']
MatchType = Literal['exact', 'prefix', 'suffix', 'regex', 'structure list']
RuleValue = str | tuple[str, ...]


@dataclass(frozen=True)
class MetadataCondition:
    '''Define one metadata value comparison.'''

    field: str
    match_type: MatchType
    value: RuleValue
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        '''Validate condition fields.'''
        if not self.field.strip():
            raise ValueError('Metadata condition field must not be blank')
        if self.match_type == 'structure list':
            if self.field != 'Structure ID':
                raise ValueError(
                    'structure list matching requires the Structure ID field',
                )
            if not isinstance(self.value, tuple):
                raise TypeError('structure list values must be a tuple')
        elif not isinstance(self.value, str):
            raise TypeError(f'{self.match_type} values must be strings')

    def matches(self, actual_value: object) -> bool:
        '''Return whether an actual metadata value satisfies the condition.'''
        actual_text = '' if pd.isna(actual_value) else str(actual_value).strip()
        if self.match_type == 'structure list':
            return actual_text in self.value

        expected_text = self.value.strip()
        if not actual_text or not expected_text:
            return False
        if self.match_type == 'regex':
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                return re.search(expected_text, actual_text, flags) is not None
            except re.error as exc:
                raise ValueError(
                    f'Invalid metadata rule regex {expected_text!r}',
                ) from exc

        if not self.case_sensitive:
            actual_text = actual_text.lower()
            expected_text = expected_text.lower()
        if self.match_type == 'exact':
            return actual_text == expected_text
        if self.match_type == 'prefix':
            return actual_text.startswith(expected_text)
        if self.match_type == 'suffix':
            return actual_text.endswith(expected_text)
        raise ValueError(f'Unsupported metadata match type: {self.match_type}')


@dataclass(frozen=True)
class MetadataDisplayRule:
    '''Set diagram visibility when metadata conditions match.'''

    rule_id: str
    action: DisplayAction
    condition: MetadataCondition
    companion: MetadataCondition | None = None
    description: str = ''

    def __post_init__(self) -> None:
        '''Validate rule identity.'''
        if not self.rule_id.strip():
            raise ValueError('Metadata display rule ID must not be blank')

    def matches(self, row: pd.Series) -> bool:
        '''Return whether all rule conditions match a metadata row.'''
        if self.condition.field not in row.index:
            raise KeyError(
                f'Missing metadata field: {self.condition.field}',
            )
        if not self.condition.matches(row[self.condition.field]):
            return False
        if self.companion is None:
            return True
        if self.companion.field not in row.index:
            raise KeyError(
                f'Missing metadata field: {self.companion.field}',
            )
        return self.companion.matches(row[self.companion.field])


@dataclass(frozen=True)
class GroupedGridLayoutConfig:
    '''Spacing parameters for deterministic grouped-grid placement.'''

    horizontal_spacing: float = 3.3
    vertical_spacing: float = 1.2
    duplicate_vertical_spread: float = 0.35

    def __post_init__(self) -> None:
        '''Require positive spacing values.'''
        if self.horizontal_spacing <= 0:
            raise ValueError('horizontal_spacing must be greater than zero')
        if self.vertical_spacing <= 0:
            raise ValueError('vertical_spacing must be greater than zero')
        if self.duplicate_vertical_spread < 0:
            raise ValueError(
                'duplicate_vertical_spread must be greater than or equal to zero',
            )


class LayoutAlgorithm(Protocol):
    '''Interface implemented by node-positioning algorithms.'''

    def compute_positions(
        self,
        nodes: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        '''Return an ``ROI -> (x, y)`` position mapping.'''


@dataclass(frozen=True)
class DiagramNodePlan:
    '''Nodes and relationship graph prepared for a layout algorithm.'''

    nodes: pd.DataFrame
    relationship_graph: nx.Graph


class LayoutPlanBuilder(Protocol):
    '''Interface for preparing template-specific node layout inputs.'''

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Return planned nodes and the graph that influences placement.'''


@dataclass(frozen=True)
class MetadataGroupingPlanBuilder:
    '''Prepare nodes using hierarchical metadata grouping definitions.'''

    grouping_definition_set: GroupDefinitionSet

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Build the current metadata-grouped node plan.'''
        grouping_source = prepare_structure_grouping_source(visible_metadata)
        grouping_table = build_structure_grouping_table(
            structures_df=grouping_source,
            grouping_definition_set=self.grouping_definition_set,
        )
        nodes = summary[['ROI', 'Name']].merge(
            grouping_table,
            left_on='ROI',
            right_index=True,
            how='inner',
        )
        sort_columns = [
            column
            for column in [
                'placement_order',
                'h_index',
                'v_index',
                'Name',
                'ROI',
            ]
            if column in nodes.columns
        ]
        nodes.sort_values(sort_columns, inplace=True)
        nodes.reset_index(drop=True, inplace=True)
        return DiagramNodePlan(
            nodes=nodes,
            relationship_graph=relationship_graph.copy(),
        )


def _relationship_type_name(edge_data: dict[str, Any]) -> str:
    '''Return the normalized relationship type stored on one graph edge.'''
    relationship = edge_data.get('relationship')
    relationship_type = getattr(relationship, 'relationship_type', None)
    return str(
        getattr(relationship_type, 'relation_type', None)
        or edge_data.get('relation_type')
        or 'UNKNOWN'
    ).upper()


def _incident_relationship_types(
    relationship_graph: nx.Graph,
    roi: int,
) -> tuple[str, ...]:
    '''Return relationship types on all edges incident to one node.'''
    if relationship_graph.is_directed():
        edge_items = [
            *relationship_graph.in_edges(roi, data=True),
            *relationship_graph.out_edges(roi, data=True),
        ]
    else:
        edge_items = list(relationship_graph.edges(roi, data=True))
    return tuple(sorted({
        _relationship_type_name(edge_data)
        for _, _, edge_data in edge_items
    }))


@dataclass(frozen=True)
class RelationshipGraphPlanBuilder:
    '''Prepare nodes and graph features from structure relationships.'''

    relationship_types: tuple[str, ...] = ()
    include_logical: bool = True

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Build a relationship-aware plan with optional edge filtering.'''
        normalized_types = {
            relationship_type.upper()
            for relationship_type in self.relationship_types
        }
        layout_graph = relationship_graph.__class__()
        layout_graph.add_nodes_from(relationship_graph.nodes(data=True))
        for source, target, edge_data in relationship_graph.edges(data=True):
            relationship = edge_data.get('relationship')
            if not self.include_logical and bool(
                getattr(relationship, 'is_logical', False)
            ):
                continue
            relationship_type = _relationship_type_name(edge_data)
            if normalized_types and relationship_type not in normalized_types:
                continue
            layout_graph.add_edge(source, target, **edge_data)

        metadata = visible_metadata.copy()
        if 'ROINumber' in metadata.columns:
            metadata.reset_index(drop=True, inplace=True)
        else:
            metadata['ROINumber'] = metadata.index
            metadata.reset_index(drop=True, inplace=True)
        nodes = summary[['ROI', 'Name']].merge(
            metadata,
            left_on='ROI',
            right_on='ROINumber',
            how='inner',
        )
        nodes['relationship_degree'] = nodes['ROI'].map(layout_graph.degree)
        if layout_graph.is_directed():
            nodes['relationship_in_degree'] = nodes['ROI'].map(
                layout_graph.in_degree,
            )
            nodes['relationship_out_degree'] = nodes['ROI'].map(
                layout_graph.out_degree,
            )
        else:
            nodes['relationship_in_degree'] = nodes['relationship_degree']
            nodes['relationship_out_degree'] = nodes['relationship_degree']
        nodes['relationship_types'] = nodes['ROI'].map(
            lambda roi: _incident_relationship_types(layout_graph, int(roi)),
        )
        nodes.sort_values(['ROI', 'Name'], inplace=True)
        nodes.reset_index(drop=True, inplace=True)
        return DiagramNodePlan(nodes=nodes, relationship_graph=layout_graph)


@dataclass(frozen=True)
class GroupedGridLayout:
    '''Position nodes from hierarchical horizontal and vertical indices.'''

    config: GroupedGridLayoutConfig = field(
        default_factory=GroupedGridLayoutConfig,
    )

    def compute_positions(
        self,
        nodes: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        '''Compute deterministic grouped-grid positions.'''
        del relationship_graph
        required_columns = {'ROI', 'h_index', 'v_index', 'v_dup_index'}
        missing_columns = required_columns.difference(nodes.columns)
        if missing_columns:
            missing_text = ', '.join(sorted(missing_columns))
            raise KeyError(f'Missing grouped-grid columns: {missing_text}')

        positions = {}
        for _, row in nodes.iterrows():
            roi = int(row['ROI'])
            positions[roi] = (
                float(row['h_index']) * self.config.horizontal_spacing,
                -(
                    float(row['v_index']) * self.config.vertical_spacing
                    + float(row['v_dup_index'])
                    * self.config.duplicate_vertical_spread
                ),
            )
        return positions


@dataclass(frozen=True)
class SpringLayoutConfig:
    '''Configuration for deterministic relationship-driven spring layout.'''

    seed: int = 42
    scale: float = 1.0
    iterations: int = 50
    optimal_distance: float | None = None

    def __post_init__(self) -> None:
        '''Validate spring-layout parameters.'''
        if self.scale <= 0:
            raise ValueError('scale must be greater than zero')
        if self.iterations < 1:
            raise ValueError('iterations must be greater than zero')
        if self.optimal_distance is not None and self.optimal_distance <= 0:
            raise ValueError('optimal_distance must be greater than zero')


@dataclass(frozen=True)
class SpringLayout:
    '''Position nodes using relationship edges as deterministic springs.'''

    config: SpringLayoutConfig = field(default_factory=SpringLayoutConfig)

    def compute_positions(
        self,
        nodes: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        '''Compute deterministic NetworkX spring-layout positions.'''
        node_rois = {int(roi) for roi in nodes['ROI']}
        layout_graph = relationship_graph.subgraph(node_rois)
        raw_positions = nx.spring_layout(
            layout_graph,
            seed=self.config.seed,
            scale=self.config.scale,
            iterations=self.config.iterations,
            k=self.config.optimal_distance,
        )
        return {
            int(roi): (float(position[0]), float(position[1]))
            for roi, position in raw_positions.items()
        }


@dataclass(frozen=True)
class LayoutTemplate:
    '''Bundle display rules, grouping policy, and a positioning algorithm.'''

    name: str
    display_by_default: bool
    display_rules: tuple[MetadataDisplayRule, ...]
    grouping_definition_set: GroupDefinitionSet | None
    algorithm: LayoutAlgorithm
    plan_builder: LayoutPlanBuilder | None = None

    def __post_init__(self) -> None:
        '''Validate template identity.'''
        if not self.name.strip():
            raise ValueError('Layout template name must not be blank')
        if self.plan_builder is None and self.grouping_definition_set is None:
            raise ValueError(
                'Layout templates require a plan builder or grouping definitions',
            )


@dataclass(frozen=True)
class DiagramLayoutResult:
    '''Renderer-neutral output from applying one layout template.'''

    template_name: str
    display_report: pd.DataFrame
    plot_nodes: pd.DataFrame
    layout_graph: nx.Graph
    positions: dict[int, tuple[float, float]]


def evaluate_template_display_rules(
    metadata: pd.DataFrame,
    template: LayoutTemplate,
) -> pd.DataFrame:
    '''Derive template display flags without mutating source metadata.'''
    report = metadata.copy(deep=True)
    if report.empty:
        report['DisplayedByDefault'] = pd.Series(dtype=bool)
        report['IsHidden'] = pd.Series(dtype=bool)
        report['TemplateMatchedRules'] = pd.Series(dtype=object)
        report['TemplateFinalMatch'] = pd.Series(dtype=object)
        return report

    displayed_values = []
    matched_values = []
    final_values = []
    for _, row in report.iterrows():
        displayed = template.display_by_default
        matched_rules: list[str] = []
        final_match = None
        for rule in template.display_rules:
            if not rule.matches(row):
                continue
            displayed = rule.action == 'display'
            matched_rules.append(rule.rule_id)
            final_match = rule.rule_id
        displayed_values.append(displayed)
        matched_values.append(matched_rules)
        final_values.append(final_match)

    report['DisplayedByDefault'] = displayed_values
    report['IsHidden'] = ~report['DisplayedByDefault']
    report['TemplateMatchedRules'] = matched_values
    report['TemplateFinalMatch'] = final_values
    return report


def apply_layout_template(
    structure_set: Any,
    template: LayoutTemplate,
) -> DiagramLayoutResult:
    '''Apply metadata visibility, grouping, and positioning to a StructureSet.'''
    source_report = structure_set.structure_filter_report
    if source_report.empty:
        source_report = structure_set.structure_metadata
    display_report = evaluate_template_display_rules(source_report, template)

    summary = structure_set.summary().copy()
    available_rois = {int(roi) for roi in summary.get('ROI', [])}
    displayed_rois = {
        int(roi)
        for roi in display_report.loc[
            display_report['DisplayedByDefault'],
            'ROINumber',
        ]
        if int(roi) in available_rois
    }
    visible_metadata = display_report[
        display_report['ROINumber'].isin(displayed_rois)
    ].copy()
    visible_summary = summary[summary['ROI'].isin(displayed_rois)].copy()
    visible_graph = structure_set.relationship_graph.subgraph(displayed_rois).copy()
    plan_builder = template.plan_builder
    if plan_builder is None:
        plan_builder = MetadataGroupingPlanBuilder(
            grouping_definition_set=template.grouping_definition_set,
        )
    node_plan = plan_builder.build_plan(
        summary=visible_summary,
        visible_metadata=visible_metadata,
        relationship_graph=visible_graph,
    )
    positions = template.algorithm.compute_positions(
        node_plan.nodes,
        node_plan.relationship_graph,
    )
    return DiagramLayoutResult(
        template_name=template.name,
        display_report=display_report,
        plot_nodes=node_plan.nodes,
        layout_graph=node_plan.relationship_graph,
        positions=positions,
    )


def default_grouped_grid_template() -> LayoutTemplate:
    '''Return the initial target-focused grouped-grid template.'''
    return LayoutTemplate(
        name='grouped_grid',
        display_by_default=False,
        display_rules=(
            MetadataDisplayRule(
                rule_id='target-label',
                action='display',
                condition=MetadataCondition(
                    field='Structure ID',
                    match_type='regex',
                    value=r'.*[GCPIH]+TV.*',
                ),
            ),
            MetadataDisplayRule(
                rule_id='target-type',
                action='display',
                condition=MetadataCondition(
                    field='DICOM Type',
                    match_type='regex',
                    value=r'[GCP]TV',
                ),
            ),
            MetadataDisplayRule(
                rule_id='resident-contours',
                action='hide',
                condition=MetadataCondition(
                    field='Structure ID',
                    match_type='prefix',
                    value='x',
                ),
            ),
        ),
        grouping_definition_set=default_structure_group_definition_set(),
        algorithm=GroupedGridLayout(),
    )


def relationship_spring_template(
    relationship_types: tuple[str, ...] = (),
) -> LayoutTemplate:
    '''Return a relationship-driven deterministic spring template.'''
    return LayoutTemplate(
        name='relationship_spring',
        display_by_default=True,
        display_rules=(),
        grouping_definition_set=None,
        algorithm=SpringLayout(),
        plan_builder=RelationshipGraphPlanBuilder(
            relationship_types=relationship_types,
        ),
    )


_LAYOUT_TEMPLATES: dict[str, LayoutTemplate] = {}


def register_layout_template(template: LayoutTemplate) -> None:
    '''Register a template under its stable name.'''
    if template.name in _LAYOUT_TEMPLATES:
        raise ValueError(f'Layout template already registered: {template.name}')
    _LAYOUT_TEMPLATES[template.name] = template


def get_layout_template(name: str) -> LayoutTemplate:
    '''Return a registered layout template by name.'''
    try:
        return _LAYOUT_TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(f'Unknown layout template: {name}') from exc


register_layout_template(default_grouped_grid_template())
register_layout_template(relationship_spring_template())


__all__ = [
    'DiagramLayoutResult',
    'DiagramNodePlan',
    'GroupedGridLayout',
    'GroupedGridLayoutConfig',
    'LayoutAlgorithm',
    'LayoutPlanBuilder',
    'LayoutTemplate',
    'MetadataGroupingPlanBuilder',
    'MetadataCondition',
    'MetadataDisplayRule',
    'RelationshipGraphPlanBuilder',
    'SpringLayout',
    'SpringLayoutConfig',
    'apply_layout_template',
    'default_grouped_grid_template',
    'evaluate_template_display_rules',
    'get_layout_template',
    'relationship_spring_template',
    'register_layout_template',
]
