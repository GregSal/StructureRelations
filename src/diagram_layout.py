'''Metadata-driven node selection and positioning for relationship diagrams.'''

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
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
MissingPTVMode = Literal['ignore_group', 'fallback_to_ctv_then_gtv']


def _normalize_metadata_text(value: object) -> str:
    '''Normalize optional metadata values to stripped lowercase text.'''
    if pd.isna(value):
        return ''
    return str(value).strip().lower()


def _is_missing_group_value(value: object) -> bool:
    '''Return whether one grouping/text value should be treated as missing.'''
    return _normalize_metadata_text(value) in {'', 'missing', 'blank', 'none'}


def _resolve_target_type(row: pd.Series) -> str:
    '''Resolve one row's target type using parsed and DICOM fallback fields.'''
    for column in ('TargetType', 'DICOM Type', 'Structure ID'):
        if column not in row.index:
            continue
        text = str(row[column]).strip().upper()
        for target_type in ('PTV', 'CTV', 'GTV'):
            if target_type in text:
                return target_type
    return ''


@dataclass(frozen=True)
class PrincipleTargetSelectorConfig:
    '''Configuration for selecting one principle target per target group.'''

    missing_ptv_mode: MissingPTVMode = 'ignore_group'
    volume_field: str = 'Physical_Volume'


@dataclass(frozen=True)
class TargetOARPlanConfig:
    '''Configuration for selecting and ranking Target-OAR layout nodes.'''

    selector_config: PrincipleTargetSelectorConfig = field(
        default_factory=PrincipleTargetSelectorConfig,
    )
    oar_dicom_type: str = 'ORGAN'
    opt_prefix: str = 'opt'


@dataclass(frozen=True)
class TargetOARBipartiteLayoutConfig:
    '''Column and spacing configuration for the Target-OAR template.'''

    target_x: float = 0.0
    opt_x: float = 4.35
    oar_x: float = 5.8
    vertical_spacing: float = 1.0
    opt_midpoint_jitter: float = 0.07
    opt_fallback_offset: float = 0.32

    def __post_init__(self) -> None:
        '''Validate column ordering and spacing values.'''
        if self.vertical_spacing <= 0:
            raise ValueError('vertical_spacing must be greater than zero')
        if not self.target_x < self.opt_x < self.oar_x:
            raise ValueError('Expected target_x < opt_x < oar_x')
        if self.opt_midpoint_jitter < 0:
            raise ValueError('opt_midpoint_jitter must be non-negative')
        if self.opt_fallback_offset < 0:
            raise ValueError('opt_fallback_offset must be non-negative')


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


def _sort_candidates_stably(candidates: pd.DataFrame) -> pd.DataFrame:
    '''Sort candidate rows deterministically using existing grouping order.'''
    sort_columns = [
        column
        for column in ['placement_order', 'ROI']
        if column in candidates.columns
    ]
    if not sort_columns:
        return candidates.copy()
    return candidates.sort_values(sort_columns, kind='stable').copy()


def _apply_principle_target_tiebreaks(
    candidates: pd.DataFrame,
    config: PrincipleTargetSelectorConfig,
) -> pd.DataFrame:
    '''Apply tie-break rules after group/type filtering.'''
    remaining = candidates.copy()

    if 'Mod' in remaining.columns:
        eval_index = remaining['Mod'].map(_normalize_metadata_text) == 'eval'
        if bool(eval_index.any()):
            remaining = remaining.loc[eval_index].copy()

    if 'TargetSubGroup' in remaining.columns:
        no_subgroup_index = remaining['TargetSubGroup'].map(_is_missing_group_value)
        if bool(no_subgroup_index.any()):
            remaining = remaining.loc[no_subgroup_index].copy()

    if 'TargetLaterality' in remaining.columns:
        no_laterality_index = remaining['TargetLaterality'].map(
            _is_missing_group_value,
        )
        if bool(no_laterality_index.any()):
            remaining = remaining.loc[no_laterality_index].copy()

    if 'Mod' in remaining.columns:
        mod_series = remaining['Mod'].map(_normalize_metadata_text)
        has_opt = bool((mod_series == 'opt').any())
        has_non_opt = bool((mod_series != 'opt').any())
        if has_opt and has_non_opt and config.volume_field in remaining.columns:
            numeric_volume = pd.to_numeric(
                remaining[config.volume_field],
                errors='coerce',
            )
            minimum_volume = numeric_volume.min(skipna=True)
            if pd.notna(minimum_volume):
                remaining = remaining.loc[numeric_volume == minimum_volume].copy()

    return _sort_candidates_stably(remaining)


def _select_group_principle_target(
    group_rows: pd.DataFrame,
    config: PrincipleTargetSelectorConfig,
) -> pd.Series | None:
    '''Select one principle target row from one target group.'''
    candidates = group_rows.copy()
    candidates['__target_type'] = candidates.apply(_resolve_target_type, axis=1)

    ptv_candidates = candidates.loc[candidates['__target_type'] == 'PTV']
    if not ptv_candidates.empty:
        filtered = ptv_candidates
    elif config.missing_ptv_mode == 'fallback_to_ctv_then_gtv':
        ctv_candidates = candidates.loc[candidates['__target_type'] == 'CTV']
        if not ctv_candidates.empty:
            filtered = ctv_candidates
        else:
            gtv_candidates = candidates.loc[candidates['__target_type'] == 'GTV']
            if gtv_candidates.empty:
                return None
            filtered = gtv_candidates
    else:
        return None

    ranked = _apply_principle_target_tiebreaks(filtered, config)
    if ranked.empty:
        return None
    return ranked.iloc[0]


def _select_principle_targets(
    nodes: pd.DataFrame,
    config: PrincipleTargetSelectorConfig,
) -> pd.DataFrame:
    '''Select exactly one principle target per horizontal target group.'''
    if nodes.empty:
        return nodes.copy()
    if 'h_grouping' not in nodes.columns:
        raise KeyError('Missing grouped-grid column: h_grouping')

    selected_rows: list[pd.Series] = []
    for _, group_rows in nodes.groupby('h_grouping', sort=False):
        selected = _select_group_principle_target(group_rows, config)
        if selected is not None:
            selected_rows.append(selected)

    if not selected_rows:
        return nodes.iloc[0:0].copy()

    selected_table = pd.DataFrame(selected_rows).reset_index(drop=True)
    return _sort_candidates_stably(selected_table)


def _relationship_edge_data_between(
    relationship_graph: nx.Graph,
    first_roi: int,
    second_roi: int,
) -> dict[str, Any] | None:
    '''Return edge metadata between two ROIs in either graph direction.'''
    if relationship_graph.has_edge(first_roi, second_roi):
        edge_data = relationship_graph.get_edge_data(first_roi, second_roi)
        if edge_data is not None:
            return edge_data
    if relationship_graph.has_edge(second_roi, first_roi):
        edge_data = relationship_graph.get_edge_data(second_roi, first_roi)
        if edge_data is not None:
            return edge_data
    return None


def _relationship_category_name(edge_data: dict[str, Any] | None) -> str:
    '''Return the relationship category text stored on one graph edge.'''
    if edge_data is None:
        return ''
    relationship = edge_data.get('relationship')
    relationship_type = getattr(relationship, 'relationship_type', None)
    category = getattr(relationship_type, 'category', None)
    if category is None:
        return ''
    return str(category)


def _relationship_category_rank(category: str) -> float:
    '''Return the Target-OAR ranking weight for one category label.'''
    category_ranks = {
        'shared': 1.0,
        'adjoining': 0.5,
        'separate': 0.0,
    }
    return category_ranks.get(category.strip().lower(), 0.0)


def _build_target_rank_values(num_targets: int) -> list[int]:
    '''Return centered target-rank values matching notebook behavior.'''
    if num_targets <= 0:
        return []
    if num_targets % 2 == 0:
        return [
            rank
            for rank in range(-(num_targets // 2), (num_targets // 2) + 1)
            if rank != 0
        ]
    return [
        rank
        for rank in range(-(num_targets // 2), (num_targets // 2) + 1)
    ]


def _prepare_target_oar_nodes(
    summary: pd.DataFrame,
    visible_metadata: pd.DataFrame,
    grouping_definition_set: GroupDefinitionSet,
) -> pd.DataFrame:
    '''Merge summary, metadata, and grouping data for Target-OAR planning.'''
    metadata = visible_metadata.copy()
    for column in ('Structure Code', 'Coding Scheme', 'Code Meaning'):
        if column not in metadata.columns:
            metadata[column] = ''
    if 'ROINumber' in metadata.columns:
        metadata.reset_index(drop=True, inplace=True)
    else:
        metadata['ROINumber'] = metadata.index
        metadata.reset_index(drop=True, inplace=True)

    grouping_source = prepare_structure_grouping_source(metadata)
    grouping_table = build_structure_grouping_table(
        structures_df=grouping_source,
        grouping_definition_set=grouping_definition_set,
    )
    nodes = summary.merge(
        metadata,
        left_on='ROI',
        right_on='ROINumber',
        how='inner',
    )
    nodes = nodes.merge(
        grouping_table,
        left_on='ROI',
        right_index=True,
        how='left',
    )
    canonical_columns = (
        'Structure ID',
        'DICOM Type',
        'Structure Code',
        'Coding Scheme',
        'Code Meaning',
        'TargetDose',
        'TargetType',
        'TargetLaterality',
        'TargetSubGroup',
        'Mod',
    )
    for column in canonical_columns:
        if column in nodes.columns:
            continue
        for candidate in (f'{column}_x', f'{column}_y'):
            if candidate in nodes.columns:
                nodes[column] = nodes[candidate]
                break
    return nodes


def _related_oar_rows_for_opt(
    opt_roi: int,
    oar_rows: pd.DataFrame,
    relationship_graph: nx.Graph,
) -> pd.DataFrame:
    '''Return OAR rows that have a non-separate relation to one opt ROI.'''
    related_rows: list[pd.Series] = []
    for _, oar_row in oar_rows.iterrows():
        edge_data = _relationship_edge_data_between(
            relationship_graph,
            opt_roi,
            int(oar_row['ROI']),
        )
        category = _relationship_category_name(edge_data)
        if category and category.strip().lower() != 'separate':
            related_rows.append(oar_row)
    if not related_rows:
        return oar_rows.iloc[0:0].copy()
    return pd.DataFrame(related_rows).reset_index(drop=True)


@dataclass(frozen=True)
class TargetOARPlanBuilder:
    '''Prepare principle targets, related OARs, and matched opt structures.'''

    grouping_definition_set: GroupDefinitionSet
    plan_config: TargetOARPlanConfig = field(
        default_factory=TargetOARPlanConfig,
    )

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Build one deterministic Target-OAR node plan.'''
        nodes = _prepare_target_oar_nodes(
            summary=summary,
            visible_metadata=visible_metadata,
            grouping_definition_set=self.grouping_definition_set,
        )
        selected_targets = _select_principle_targets(
            nodes=nodes,
            config=self.plan_config.selector_config,
        )
        if selected_targets.empty:
            return DiagramNodePlan(
                nodes=selected_targets.copy(),
                relationship_graph=relationship_graph.__class__(),
            )

        selected_targets = selected_targets.copy().reset_index(drop=True)
        selected_targets['target_rank'] = _build_target_rank_values(
            len(selected_targets),
        )
        selected_targets['node_side'] = 'target'
        selected_targets['display_order'] = range(len(selected_targets))

        oar_rows = nodes.loc[
            nodes['DICOM Type'].fillna('').astype(str).str.upper()
            == self.plan_config.oar_dicom_type.upper()
        ].copy()
        weighted_rows: list[pd.Series] = []
        for _, oar_row in oar_rows.iterrows():
            weighted_sum = 0.0
            has_non_separate = False
            for target_row in selected_targets.itertuples(index=False):
                edge_data = _relationship_edge_data_between(
                    relationship_graph,
                    int(oar_row['ROI']),
                    int(target_row.ROI),
                )
                category = _relationship_category_name(edge_data)
                category_rank = _relationship_category_rank(category)
                if category_rank > 0:
                    has_non_separate = True
                weighted_sum += category_rank * float(target_row.target_rank)
            if not has_non_separate:
                continue
            enriched_row = oar_row.copy()
            enriched_row['weighted_oar_score'] = weighted_sum
            weighted_rows.append(enriched_row)

        if weighted_rows:
            selected_oars = pd.DataFrame(weighted_rows)
            selected_oars.sort_values(
                ['weighted_oar_score', 'placement_order', 'ROI'],
                ascending=[False, True, True],
                kind='stable',
                inplace=True,
            )
            selected_oars.reset_index(drop=True, inplace=True)
        else:
            selected_oars = oar_rows.iloc[0:0].copy()

        selected_oars['node_side'] = 'oar'
        selected_oars['display_order'] = range(len(selected_oars))
        selected_oars['oar_rank'] = selected_oars['display_order']

        oar_structure_ids = selected_oars['Structure ID'].astype(str).tolist()
        oar_order_by_id = {
            str(row['Structure ID']): int(row['display_order'])
            for _, row in selected_oars.iterrows()
        }

        opt_candidates = nodes.loc[
            nodes['Structure ID']
            .fillna('')
            .astype(str)
            .str.lower()
            .str.startswith(self.plan_config.opt_prefix.lower())
        ].copy()

        matched_opt_rows: list[pd.Series] = []
        placement_count: dict[tuple[int, int | None], int] = {}
        oar_order_to_roi = {
            int(row['display_order']): int(row['ROI'])
            for _, row in selected_oars.iterrows()
        }

        for _, opt_row in opt_candidates.iterrows():
            opt_structure_id = str(opt_row['Structure ID'])
            matched_ids = [
                oar_id
                for oar_id in oar_structure_ids
                if oar_id in opt_structure_id
            ]
            if not matched_ids:
                continue

            anchor_id = matched_ids[0]
            anchor_order = oar_order_by_id[anchor_id]
            anchor_roi = oar_order_to_roi[anchor_order]

            related_oars = _related_oar_rows_for_opt(
                opt_roi=int(opt_row['ROI']),
                oar_rows=selected_oars,
                relationship_graph=relationship_graph,
            )
            related_orders = [
                int(row['display_order'])
                for _, row in related_oars.iterrows()
                if int(row['ROI']) != anchor_roi
            ]
            prefer_upper = True
            if related_orders:
                prefer_upper = (
                    sum(related_orders) / len(related_orders)
                    < float(anchor_order)
                )

            neighbor_order: int | None = None
            if prefer_upper and anchor_order > 0:
                neighbor_order = anchor_order - 1
            elif (not prefer_upper) and anchor_order < len(selected_oars) - 1:
                neighbor_order = anchor_order + 1
            elif anchor_order > 0:
                neighbor_order = anchor_order - 1
            elif anchor_order < len(selected_oars) - 1:
                neighbor_order = anchor_order + 1

            neighbor_roi = (
                oar_order_to_roi.get(neighbor_order)
                if neighbor_order is not None
                else None
            )
            slot_key = (anchor_roi, neighbor_roi)
            slot_index = placement_count.get(slot_key, 0)
            placement_count[slot_key] = slot_index + 1

            enriched_row = opt_row.copy()
            enriched_row['node_side'] = 'opt'
            enriched_row['display_order'] = float(anchor_order) + 0.5
            enriched_row['anchor_oar_roi'] = anchor_roi
            enriched_row['opt_neighbor_oar_roi'] = neighbor_roi
            enriched_row['opt_slot_index'] = slot_index
            enriched_row['prefer_upper_neighbor'] = prefer_upper
            matched_opt_rows.append(enriched_row)

        if matched_opt_rows:
            matched_opts = pd.DataFrame(matched_opt_rows)
            matched_opts.sort_values(
                ['display_order', 'Structure ID', 'ROI'],
                kind='stable',
                inplace=True,
            )
            matched_opts.reset_index(drop=True, inplace=True)
        else:
            matched_opts = opt_candidates.iloc[0:0].copy()

        selected_node_frames = [selected_targets, selected_oars, matched_opts]
        selected_nodes = pd.concat(selected_node_frames, ignore_index=True, sort=False)
        selected_nodes.sort_values(
            ['display_order', 'node_side', 'ROI'],
            kind='stable',
            inplace=True,
        )
        selected_nodes.reset_index(drop=True, inplace=True)

        selected_rois = {
            int(roi)
            for roi in selected_nodes['ROI'].tolist()
        }
        selected_graph = relationship_graph.subgraph(selected_rois).copy()
        return DiagramNodePlan(
            nodes=selected_nodes,
            relationship_graph=selected_graph,
        )


@dataclass(frozen=True)
class PrincipleTargetPlanBuilder:
    '''Prepare grouped nodes but keep one selected principle target per group.'''

    grouping_definition_set: GroupDefinitionSet
    selector_config: PrincipleTargetSelectorConfig = field(
        default_factory=PrincipleTargetSelectorConfig,
    )

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Build grouped nodes and reduce to principle targets only.'''
        grouping_source = prepare_structure_grouping_source(visible_metadata)
        grouping_table = build_structure_grouping_table(
            structures_df=grouping_source,
            grouping_definition_set=self.grouping_definition_set,
        )

        summary_columns = [
            column
            for column in ['ROI', 'Name', self.selector_config.volume_field]
            if column in summary.columns
        ]
        nodes = summary[summary_columns].merge(
            grouping_table,
            left_on='ROI',
            right_index=True,
            how='inner',
        )
        selected_nodes = _select_principle_targets(
            nodes=nodes,
            config=self.selector_config,
        )
        selected_rois = {
            int(roi)
            for roi in selected_nodes['ROI'].tolist()
        }
        selected_graph = relationship_graph.subgraph(selected_rois).copy()
        return DiagramNodePlan(
            nodes=selected_nodes,
            relationship_graph=selected_graph,
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


def _optional_int(value: object) -> int | None:
    '''Return one optional integer value from possibly-missing metadata.'''
    if pd.isna(value):
        return None
    return int(value)


@dataclass(frozen=True)
class TargetOARBipartiteLayout:
    '''Position principle targets, matched opt structures, and OARs in columns.'''

    config: TargetOARBipartiteLayoutConfig = field(
        default_factory=TargetOARBipartiteLayoutConfig,
    )

    def compute_positions(
        self,
        nodes: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        '''Compute deterministic left-middle-right Target-OAR positions.'''
        del relationship_graph
        required_columns = {'ROI', 'node_side', 'display_order'}
        missing_columns = required_columns.difference(nodes.columns)
        if missing_columns:
            missing_text = ', '.join(sorted(missing_columns))
            raise KeyError(f'Missing Target-OAR columns: {missing_text}')

        positions: dict[int, tuple[float, float]] = {}

        target_rows = nodes.loc[nodes['node_side'] == 'target'].copy()
        target_rows.sort_values(['display_order', 'ROI'], kind='stable', inplace=True)
        for _, row in target_rows.iterrows():
            positions[int(row['ROI'])] = (
                self.config.target_x,
                -float(row['display_order']) * self.config.vertical_spacing,
            )

        oar_rows = nodes.loc[nodes['node_side'] == 'oar'].copy()
        oar_rows.sort_values(['display_order', 'ROI'], kind='stable', inplace=True)
        for _, row in oar_rows.iterrows():
            positions[int(row['ROI'])] = (
                self.config.oar_x,
                -float(row['display_order']) * self.config.vertical_spacing,
            )

        opt_rows = nodes.loc[nodes['node_side'] == 'opt'].copy()
        opt_rows.sort_values(['display_order', 'ROI'], kind='stable', inplace=True)
        for _, row in opt_rows.iterrows():
            roi = int(row['ROI'])
            anchor_roi = _optional_int(row.get('anchor_oar_roi'))
            neighbor_roi = _optional_int(row.get('opt_neighbor_oar_roi'))
            slot_index = _optional_int(row.get('opt_slot_index')) or 0
            prefer_upper = bool(row.get('prefer_upper_neighbor', True))

            if anchor_roi is None or anchor_roi not in positions:
                positions[roi] = (self.config.opt_x, 0.0)
                continue

            anchor_y = positions[anchor_roi][1]
            if neighbor_roi is not None and neighbor_roi in positions:
                neighbor_y = positions[neighbor_roi][1]
                base_y = (anchor_y + neighbor_y) / 2.0
                direction = 1.0 if base_y >= anchor_y else -1.0
            else:
                direction = 1.0 if prefer_upper else -1.0
                base_y = anchor_y + (direction * self.config.opt_fallback_offset)

            positions[roi] = (
                self.config.opt_x,
                base_y + (direction * self.config.opt_midpoint_jitter * slot_index),
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
    target_display_rules = (
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
    )
    return LayoutTemplate(
        name='grouped_grid',
        display_by_default=False,
        display_rules=target_display_rules,
        grouping_definition_set=default_structure_group_definition_set(),
        algorithm=GroupedGridLayout(),
    )


def principle_targets_template(
    selector_config: PrincipleTargetSelectorConfig | None = None,
) -> LayoutTemplate:
    '''Return a grouped-grid template with one principle target per group.'''
    config = selector_config or PrincipleTargetSelectorConfig()
    return LayoutTemplate(
        name='principle_targets',
        display_by_default=False,
        display_rules=default_grouped_grid_template().display_rules,
        grouping_definition_set=default_structure_group_definition_set(),
        algorithm=GroupedGridLayout(),
        plan_builder=PrincipleTargetPlanBuilder(
            grouping_definition_set=default_structure_group_definition_set(),
            selector_config=config,
        ),
    )


def target_oar_template(
    plan_config: TargetOARPlanConfig | None = None,
    layout_config: TargetOARBipartiteLayoutConfig | None = None,
    ) -> LayoutTemplate:
    '''Return a bipartite principle-target/OAR layout template.'''
    resolved_plan_config = plan_config or TargetOARPlanConfig()
    resolved_layout_config = layout_config or TargetOARBipartiteLayoutConfig()
    display_rules = (
        *default_grouped_grid_template().display_rules,
        MetadataDisplayRule(
            rule_id='target-oar-organs',
            action='display',
            condition=MetadataCondition(
                field='DICOM Type',
                match_type='exact',
                value=resolved_plan_config.oar_dicom_type,
            ),
        ),
        MetadataDisplayRule(
            rule_id='target-oar-opt-prefix',
            action='display',
            condition=MetadataCondition(
                field='Structure ID',
                match_type='prefix',
                value=resolved_plan_config.opt_prefix,
            ),
        ),
    )
    grouping_definition_set = default_structure_group_definition_set()
    return LayoutTemplate(
        name='target_oar',
        display_by_default=False,
        display_rules=display_rules,
        grouping_definition_set=grouping_definition_set,
        algorithm=TargetOARBipartiteLayout(resolved_layout_config),
        plan_builder=TargetOARPlanBuilder(
            grouping_definition_set=grouping_definition_set,
            plan_config=resolved_plan_config,
        ),
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


@dataclass(frozen=True)
class CustomDiagramLayoutDefinition:
    '''Definition of a custom diagram layout loaded from a dictionary or JSON.'''

    template_name: str
    template_description: str
    template_version: str
    template_author: str
    template_date: str
    template_layout: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class CustomLayoutPlanBuilder:
    '''Build a plan by matching template Structure IDs to available ROIs.'''

    definition: CustomDiagramLayoutDefinition

    def build_plan(
        self,
        summary: pd.DataFrame,
        visible_metadata: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> DiagramNodePlan:
        '''Return nodes and graph restricted to structures named in the template.'''
        metadata = visible_metadata.copy()
        if 'ROINumber' in metadata.columns:
            metadata.reset_index(drop=True, inplace=True)
        else:
            metadata['ROINumber'] = metadata.index
            metadata.reset_index(drop=True, inplace=True)
        if 'Structure ID' not in metadata.columns:
            raise KeyError(
                'visible_metadata is missing required column: Structure ID',
            )
        nodes = summary[['ROI', 'Name']].merge(
            metadata,
            left_on='ROI',
            right_on='ROINumber',
            how='inner',
        )
        layout = self.definition.template_layout
        nodes = nodes[nodes['Structure ID'].isin(layout)].copy()
        nodes['custom_x'] = nodes['Structure ID'].map(
            lambda sid: layout[sid][0],
        )
        nodes['custom_y'] = nodes['Structure ID'].map(
            lambda sid: layout[sid][1],
        )
        nodes.reset_index(drop=True, inplace=True)
        selected_rois = {int(roi) for roi in nodes['ROI']}
        return DiagramNodePlan(
            nodes=nodes,
            relationship_graph=relationship_graph.subgraph(selected_rois).copy(),
        )


@dataclass(frozen=True)
class CustomLayoutAlgorithm:
    '''Position nodes using x/y coordinates stored in plan node columns.'''

    def compute_positions(
        self,
        nodes: pd.DataFrame,
        relationship_graph: nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        '''Return positions from the custom_x and custom_y node columns.'''
        del relationship_graph
        missing = {'custom_x', 'custom_y'}.difference(nodes.columns)
        if missing:
            raise KeyError(
                f'Missing custom layout columns: {", ".join(sorted(missing))}',
            )
        return {
            int(row['ROI']): (float(row['custom_x']), float(row['custom_y']))
            for _, row in nodes.iterrows()
        }


def _validate_custom_layout_dict(template_dict: dict) -> None:
    '''Raise ValueError/TypeError when template_dict is not a valid layout definition.'''
    if not isinstance(template_dict, dict):
        raise TypeError('Template definition must be a dictionary')
    name = template_dict.get('template_name', '')
    if not str(name).strip():
        raise ValueError('template_name must not be blank')
    layout = template_dict.get('template_layout')
    if not isinstance(layout, dict):
        raise ValueError('template_layout must be a dictionary')
    if not layout:
        raise ValueError('template_layout must not be empty')
    for structure_id, position in layout.items():
        if not isinstance(structure_id, str) or not structure_id.strip():
            raise ValueError('template_layout keys must be non-empty strings')
        if not isinstance(position, dict):
            raise ValueError(
                f'Position for {structure_id!r} must be a dictionary',
            )
        for axis in ('x', 'y'):
            if axis not in position:
                raise ValueError(
                    f'Position for {structure_id!r} must have an {axis!r} key',
                )
            try:
                float(position[axis])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Position {axis!r} for {structure_id!r} must be numeric',
                ) from exc


def _build_custom_layout_definition(
    template_dict: dict,
) -> CustomDiagramLayoutDefinition:
    '''Convert a validated template dictionary to a CustomDiagramLayoutDefinition.'''
    layout = {
        str(sid): (float(pos['x']), float(pos['y']))
        for sid, pos in template_dict['template_layout'].items()
    }
    return CustomDiagramLayoutDefinition(
        template_name=str(template_dict.get('template_name', '')),
        template_description=str(template_dict.get('template_description', '')),
        template_version=str(template_dict.get('template_version', '')),
        template_author=str(template_dict.get('template_author', '')),
        template_date=str(template_dict.get('template_date', '')),
        template_layout=layout,
    )


def add_custom_layout_template(template_dict: dict) -> LayoutTemplate:
    '''Validate, register and return a LayoutTemplate from a definition dict.'''
    _validate_custom_layout_dict(template_dict)
    definition = _build_custom_layout_definition(template_dict)
    template = LayoutTemplate(
        name=definition.template_name,
        display_by_default=True,
        display_rules=(),
        grouping_definition_set=None,
        algorithm=CustomLayoutAlgorithm(),
        plan_builder=CustomLayoutPlanBuilder(definition=definition),
    )
    register_layout_template(template)
    return template


def load_custom_template_from_file(file_path: str | Path) -> LayoutTemplate:
    '''Load a custom layout template from a JSON file and register it.'''
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as template_file:
        template_dict = json.load(template_file)
    return add_custom_layout_template(template_dict)


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


def list_layout_templates() -> list[dict[str, str]]:
    '''Return registered layout templates in stable registration order.'''
    return [
        {'name': template.name}
        for template in _LAYOUT_TEMPLATES.values()
    ]


register_layout_template(default_grouped_grid_template())
register_layout_template(relationship_spring_template())
register_layout_template(target_oar_template())


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
    'PrincipleTargetPlanBuilder',
    'PrincipleTargetSelectorConfig',
    'RelationshipGraphPlanBuilder',
    'SpringLayout',
    'SpringLayoutConfig',
    'TargetOARBipartiteLayout',
    'TargetOARBipartiteLayoutConfig',
    'TargetOARPlanBuilder',
    'TargetOARPlanConfig',
    'add_custom_layout_template',
    'apply_layout_template',
    'CustomDiagramLayoutDefinition',
    'CustomLayoutAlgorithm',
    'CustomLayoutPlanBuilder',
    'default_grouped_grid_template',
    'evaluate_template_display_rules',
    'get_layout_template',
    'load_custom_template_from_file',
    'list_layout_templates',
    'relationship_spring_template',
    'register_layout_template',
    'target_oar_template',
]
