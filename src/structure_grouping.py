'''Structure grouping and placement analysis for diagram layouts.'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd

from dicom import DicomStructureFile
from structure_id_parser import parse_structure_metadata


GroupAxis = Literal['horizontal', 'vertical']
SortMode = Literal['ordered_list', 'numeric', 'alphabetical', 'mixed']
NumericParseMode = Literal['text', 'float']
NumericPosition = Literal['before_text', 'after_text']
TextSortMode = Literal['alphabetical', 'ordered_list']

DEFAULT_VERTICAL_ORDER = (
    'GTV',
    'CTV',
    'PTV',
    'TREATED VOLUME',
    'SHELL',
)
_EMPTY_GROUP_VALUES = {'missing', 'blank'}
_EMPTY_HIERARCHY_VALUES = {*_EMPTY_GROUP_VALUES, 'None'}


def _normalize_group_value(
    value: object,
    case_sensitive: bool = False,
) -> str:
    '''Normalize one grouping value to a stable display token.'''
    if pd.isna(value):
        return 'missing'

    text_value = str(value).strip()
    if text_value == '':
        return 'blank'

    lowered = text_value.lower()
    if lowered == 'none':
        return 'None'
    if lowered in {'(ungrouped)', 'ungrouped', 'missing'}:
        return 'missing'

    return text_value if case_sensitive else text_value


def _sortable_text(value: str, case_sensitive: bool) -> str:
    '''Return text prepared for deterministic sorting.'''
    return value if case_sensitive else value.lower()


def _parse_numeric_value(value: str) -> float | None:
    '''Return a float value when the supplied text is numeric.'''
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _trim_horizontal_path(values: tuple[str, ...]) -> tuple[str, ...]:
    '''Remove trailing empty levels from one horizontal hierarchy path.'''
    path_length = len(values)
    while (
        path_length > 0
        and values[path_length - 1] in _EMPTY_HIERARCHY_VALUES
    ):
        path_length -= 1
    return values[:path_length]


def _centered_horizontal_indices(
    rows: pd.DataFrame,
    horizontal_columns: list[str],
) -> list[float]:
    '''Center parent paths over evenly spaced terminal descendants.'''
    if not horizontal_columns:
        return [0.0] * len(rows)

    row_paths = [
        _trim_horizontal_path(tuple(row[column] for column in horizontal_columns))
        for _, row in rows.iterrows()
    ]
    unique_paths = list(dict.fromkeys(row_paths))
    leaf_paths = [
        path
        for path in unique_paths
        if not any(
            len(candidate) > len(path)
            and candidate[:len(path)] == path
            for candidate in unique_paths
        )
    ]
    leaf_positions = {
        path: float(index)
        for index, path in enumerate(leaf_paths)
    }
    path_positions = {}
    for path in unique_paths:
        descendant_positions = [
            position
            for leaf_path, position in leaf_positions.items()
            if leaf_path[:len(path)] == path
        ]
        path_positions[path] = (
            sum(descendant_positions) / len(descendant_positions)
        )

    return [path_positions[path] for path in row_paths]


@dataclass(frozen=True)
class GroupFieldDefinition:
    '''Define one grouping field and its sort behavior.

    Attributes:
        axis: Whether the grouping is horizontal or vertical.
        hierarchy_rank: Order within the global refinement sequence.
        source_columns: One or more parsed or metadata columns used to derive
            the grouping value.
        level_name: Human-readable name for this hierarchy level.
        sort_mode: Primary sort strategy for this field.
        ordered_values: Explicit domain order when needed.
        numeric_parse_mode: Whether numeric-looking values stay as text or are
            parsed as floats.
        numeric_position: Position of numeric values relative to text in mixed
            sorting mode.
        text_sort_mode: Text sort strategy in mixed mode.
        case_sensitive: Whether text sorting should preserve case.
    '''

    axis: GroupAxis
    hierarchy_rank: int
    source_columns: tuple[str, ...]
    level_name: str
    sort_mode: SortMode = 'alphabetical'
    ordered_values: tuple[str, ...] = ()
    numeric_parse_mode: NumericParseMode = 'float'
    numeric_position: NumericPosition = 'before_text'
    text_sort_mode: TextSortMode = 'alphabetical'
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        '''Validate and normalize dataclass fields.'''
        if self.axis not in {'horizontal', 'vertical'}:
            raise ValueError(f'Unsupported group axis: {self.axis}')
        if self.hierarchy_rank < 1:
            raise ValueError('hierarchy_rank must be >= 1')
        if not self.source_columns:
            raise ValueError('source_columns must contain at least one column')

    def build_group_value(self, row: pd.Series) -> str:
        '''Build one grouping value from the configured source columns.'''
        normalized_values = [
            _normalize_group_value(row.get(column), self.case_sensitive)
            for column in self.source_columns
        ]

        for candidate in normalized_values:
            if candidate not in _EMPTY_GROUP_VALUES:
                return candidate

        if 'None' in normalized_values:
            return 'None'
        if 'blank' in normalized_values:
            return 'blank'
        return 'missing'

    def sort_key(self, value: object) -> tuple[Any, ...]:
        '''Build a deterministic sort key for one grouping value.'''
        normalized_value = _normalize_group_value(value, self.case_sensitive)
        text_value = _sortable_text(normalized_value, self.case_sensitive)

        if self.sort_mode == 'ordered_list':
            return self._ordered_list_sort_key(normalized_value)

        if self.sort_mode == 'numeric':
            return self._numeric_sort_key(normalized_value, text_value)

        if self.sort_mode == 'mixed':
            return self._mixed_sort_key(normalized_value, text_value)

        return (0, text_value)

    def _ordered_list_sort_key(self, value: str) -> tuple[Any, ...]:
        '''Return an ordered-list sort key with alphabetic fallback.'''
        order_lookup = {
            _sortable_text(item, self.case_sensitive): index
            for index, item in enumerate(self.ordered_values)
        }
        sortable_value = _sortable_text(value, self.case_sensitive)
        if sortable_value in order_lookup:
            return (0, order_lookup[sortable_value], sortable_value)
        return (1, sortable_value)

    def _numeric_sort_key(
        self,
        value: str,
        sortable_text: str,
    ) -> tuple[Any, ...]:
        '''Return a numeric sort key with text fallback.'''
        if self.numeric_parse_mode == 'float':
            numeric_value = _parse_numeric_value(value)
            if numeric_value is not None:
                return (0, numeric_value)
        return (1, sortable_text)

    def _text_sort_key(self, value: str) -> tuple[Any, ...]:
        '''Return the configured text sort key for mixed sorting.'''
        if self.text_sort_mode == 'ordered_list' and self.ordered_values:
            return self._ordered_list_sort_key(value)
        return (0, _sortable_text(value, self.case_sensitive))

    def _mixed_sort_key(
        self,
        value: str,
        sortable_text: str,
    ) -> tuple[Any, ...]:
        '''Return a sort key for fields that may contain text and numbers.'''
        numeric_value = None
        if self.numeric_parse_mode == 'float':
            numeric_value = _parse_numeric_value(value)

        if numeric_value is not None:
            bucket = 0 if self.numeric_position == 'before_text' else 1
            return (bucket, numeric_value, sortable_text)

        text_bucket = 1 if self.numeric_position == 'before_text' else 0
        return (text_bucket, *self._text_sort_key(value))


@dataclass(frozen=True)
class GroupDefinitionSet:
    '''Bundle ordered grouping definitions for one placement policy.'''

    name: str
    field_definitions: tuple[GroupFieldDefinition, ...]

    def definitions_in_hierarchy_order(
        self,
    ) -> tuple[GroupFieldDefinition, ...]:
        '''Return definitions ordered by rank, then declaration order.'''
        return tuple(sorted(
            self.field_definitions,
            key=lambda definition: definition.hierarchy_rank,
        ))

    def definitions_for_axis(self, axis: GroupAxis) -> tuple[GroupFieldDefinition, ...]:
        '''Return field definitions for one axis ordered by hierarchy rank.'''
        return tuple(
            definition
            for definition in self.definitions_in_hierarchy_order()
            if definition.axis == axis
        )


def default_structure_group_definition_set(
    vertical_order: Sequence[str] | None = None,
) -> GroupDefinitionSet:
    '''Return the default grouping policy for structure diagrams.'''
    vertical_values = tuple(vertical_order or DEFAULT_VERTICAL_ORDER)
    return GroupDefinitionSet(
        name='default_structure_grouping',
        field_definitions=(
            GroupFieldDefinition(
                axis='horizontal',
                hierarchy_rank=1,
                level_name='Target Groups',
                source_columns=(
                    'TargetNumber',
                    'TargetDose',
                    'Classifier',
                    'Combined',
                    'TargetOrgan',
                ),
                sort_mode='mixed',
                numeric_parse_mode='float',
                numeric_position='before_text',
                text_sort_mode='alphabetical',
            ),
            GroupFieldDefinition(
                axis='horizontal',
                hierarchy_rank=2,
                level_name='Target Laterality',
                source_columns=('TargetLaterality',),
                sort_mode='alphabetical',
            ),
            GroupFieldDefinition(
                axis='vertical',
                hierarchy_rank=3,
                level_name='Target Type',
                source_columns=(
                    'DICOM Type',
                    'TargetType',
                    'ExpansionSize',
                    'Structure Code',
                ),
                sort_mode='ordered_list',
                ordered_values=vertical_values,
            ),
            GroupFieldDefinition(
                axis='vertical',
                hierarchy_rank=2,
                level_name='Target Mods',
                source_columns=('Mod',),
                sort_mode='alphabetical',
            ),
            GroupFieldDefinition(
                axis='horizontal',
                hierarchy_rank=2,
                level_name='Target Subgroup',
                source_columns=('TargetSubGroup',),
                sort_mode='alphabetical',
            ),
        ),
    )


def load_structure_grouping_source(
    dicom_file: DicomStructureFile,
    apply_filter: bool = False,
) -> pd.DataFrame:
    '''Load parsed structure metadata for grouping analysis.

    Args:
        dicom_file: DICOM structure file used as the source.
        apply_filter: Whether to apply the notebook-style structure filter.

    Returns:
        pd.DataFrame: ROI-indexed metadata table containing parsed structure
            fields and selected DICOM metadata columns.
    '''
    metadata = dicom_file.get_structure_filter_metadata().copy()
    if metadata.empty:
        return pd.DataFrame()

    if apply_filter:
        filter_report = dicom_file.evaluate_structure_filters()
        selection = (
            filter_report['SelectedByDefault']
            & filter_report['DisplayedByDefault']
        )
        metadata = metadata.loc[selection].copy()

    return prepare_structure_grouping_source(metadata)


def prepare_structure_grouping_source(metadata: pd.DataFrame) -> pd.DataFrame:
    '''Prepare an ROI-indexed metadata snapshot for grouping analysis.

    Args:
        metadata: Per-ROI structure metadata containing ``Structure ID``.

    Returns:
        pd.DataFrame: Parsed metadata indexed by ROI when available.
    '''
    if metadata.empty:
        return metadata.copy()

    parsed = parse_structure_metadata(metadata)
    if parsed.empty:
        result = metadata.copy()
        if 'ROINumber' in result.columns:
            # Keep ROINumber available as both index and column in this branch:
            # downstream callers may inspect or export metadata directly before
            # it goes through the diagram-specific merge path.
            result.set_index('ROINumber', inplace=True, drop=False)
        return result

    parsed_columns = [
        column
        for column in parsed.columns
        if column not in metadata.columns
    ]
    merged = metadata.merge(
        parsed[parsed_columns],
        left_on='Structure ID',
        right_index=True,
        how='left',
    )

    if 'Structure ID' not in merged.columns:
        merged.insert(0, 'Structure ID', merged.index.astype(str))
    if 'ROINumber' in merged.columns:
        # Use ROI as the canonical index for grouping calculations. We drop
        # the duplicate column here to avoid pandas ambiguity when another
        # consumer also joins on a ROINumber label.
        merged.set_index('ROINumber', inplace=True, drop=True)

    return merged


def build_structure_grouping_table(
    structures_df: pd.DataFrame,
    grouping_definition_set: GroupDefinitionSet | None = None,
) -> pd.DataFrame:
    '''Build a placement table with hierarchical grouping metadata.

    Args:
        structures_df: Parsed structure metadata. It may be indexed by ROI or
            structure ID.
        grouping_definition_set: Optional grouping policy override.

    Returns:
        pd.DataFrame: Placement table including hierarchical group levels and
            final placement indices.
    '''
    if structures_df.empty:
        return structures_df.copy()

    grouping_set = (
        grouping_definition_set or default_structure_group_definition_set()
    )
    result = structures_df.copy()

    if 'Structure ID' not in result.columns:
        result.insert(0, 'Structure ID', result.index.astype(str))

    # Intentionally avoid re-adding ROINumber as a normal column when it is
    # already the index name. Keeping both in this table makes merge targets
    # ambiguous in pandas (index level + column label with the same name).

    axis_group_columns: dict[GroupAxis, list[str]] = {
        'horizontal': [],
        'vertical': [],
    }
    definition_columns: list[
        tuple[GroupFieldDefinition, str, str]
    ] = []

    for definition in grouping_set.definitions_in_hierarchy_order():
        prefix = 'h' if definition.axis == 'horizontal' else 'v'
        level_index = len(axis_group_columns[definition.axis]) + 1
        group_column = f'{prefix}_group_{level_index}'
        sort_column = f'_{group_column}_sort_key'
        result[group_column] = result.apply(definition.build_group_value, axis=1)
        result[sort_column] = result[group_column].map(definition.sort_key)
        axis_group_columns[definition.axis].append(group_column)
        definition_columns.append((definition, group_column, sort_column))

    horizontal_columns = axis_group_columns['horizontal']
    vertical_columns = axis_group_columns['vertical']

    result['h_grouping'] = (
        result[horizontal_columns[0]] if horizontal_columns else 'missing'
    )
    result['v_grouping'] = (
        result[vertical_columns[0]] if vertical_columns else 'missing'
    )
    result['h_key'] = result.apply(
        lambda row: tuple(row[column] for column in horizontal_columns),
        axis=1,
    )
    result['v_key'] = result.apply(
        lambda row: tuple(row[column] for column in vertical_columns),
        axis=1,
    )

    sort_records = []
    for row_index, row in result.iterrows():
        hierarchy_sort = tuple(
            row[sort_column]
            for _, _, sort_column in definition_columns
        )
        # Tie-break on ROI to keep ordering stable across runs even when two
        # structures share identical grouping and sort keys.
        roi_number = row.get('ROINumber', row_index)
        sort_records.append((
            row_index,
            hierarchy_sort,
            str(row.get('Structure ID', '')),
            str(roi_number),
        ))

    ordered_index = [
        row_index
        for row_index, *_ in sorted(
            sort_records,
            key=lambda item: (item[1], item[2], item[3]),
        )
    ]
    result = result.loc[ordered_index].copy()
    result['placement_order'] = range(len(result))

    duplicate_counts = Counter(zip(result['h_key'], result['v_key']))
    fallback_values = []
    v_slot_keys = []
    for _, row in result.iterrows():
        combined_key = (row['h_key'], row['v_key'])
        if duplicate_counts[combined_key] > 1:
            # Add a deterministic fallback token only when the semantic keys
            # still collide. This guarantees unique final placement slots.
            fallback_value = (
                f"{row.get('Structure ID', '')}|{row.get('ROINumber', '')}"
            )
            v_slot_key = (*row['v_key'], fallback_value)
        else:
            fallback_value = ''
            v_slot_key = row['v_key']
        fallback_values.append(fallback_value)
        v_slot_keys.append(v_slot_key)

    result['slot_fallback'] = fallback_values
    result['v_slot_key'] = v_slot_keys
    result['slot_key'] = list(zip(result['h_key'], result['v_slot_key']))

    vertical_index_maps: dict[tuple[str, ...], dict[tuple[str, ...], int]] = {}
    horizontal_indices = _centered_horizontal_indices(
        result,
        horizontal_columns,
    )
    vertical_indices = []

    vertical_definitions = [
        definition
        for definition, _, _ in definition_columns
        if definition.axis == 'vertical'
    ]
    if vertical_definitions:
        vertical_rank = min(
            definition.hierarchy_rank
            for definition in vertical_definitions
        )
        vertical_parent_columns = [
            group_column
            for definition, group_column, _ in definition_columns
            if definition.hierarchy_rank < vertical_rank
        ]
    else:
        vertical_parent_columns = horizontal_columns

    for _, row in result.iterrows():
        vertical_key = row['v_slot_key']
        vertical_parent_key = tuple(
            row[column] for column in vertical_parent_columns
        )

        vertical_index_map = vertical_index_maps.setdefault(
            vertical_parent_key,
            {},
        )
        if vertical_key not in vertical_index_map:
            vertical_index_map[vertical_key] = len(vertical_index_map)
        vertical_indices.append(vertical_index_map[vertical_key])

    result['h_index'] = horizontal_indices
    result['v_index'] = vertical_indices
    result['v_dup_index'] = 0
    result['is_unique_slot'] = ~result['slot_key'].duplicated(keep=False)

    sort_columns = [
        sort_column
        for _, _, sort_column in definition_columns
        if sort_column in result.columns
    ]
    if sort_columns:
        result.drop(columns=sort_columns, inplace=True)

    return result


def analyze_structure_grouping(
    dicom_file: DicomStructureFile,
    grouping_definition_set: GroupDefinitionSet | None = None,
    apply_filter: bool = False,
) -> pd.DataFrame:
    '''Return an ROI-indexed placement table for one DICOM structure file.'''
    source_df = load_structure_grouping_source(
        dicom_file=dicom_file,
        apply_filter=apply_filter,
    )
    return build_structure_grouping_table(
        structures_df=source_df,
        grouping_definition_set=grouping_definition_set,
    )


__all__ = [
    'DEFAULT_VERTICAL_ORDER',
    'GroupDefinitionSet',
    'GroupFieldDefinition',
    'analyze_structure_grouping',
    'build_structure_grouping_table',
    'default_structure_group_definition_set',
    'load_structure_grouping_source',
    'prepare_structure_grouping_source',
]
