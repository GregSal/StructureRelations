'''Parse DICOM structure IDs into target metadata columns.

This module extracts the notebook logic from Target Grouping into a reusable
parser that operates on the metadata table returned by
``DicomStructureFile.get_structure_filter_metadata()``.
'''

from __future__ import annotations

from collections.abc import Sequence
import logging
import re
from typing import Callable

import pandas as pd


logger = logging.getLogger(__name__)

RegexFragmentBuilder = Callable[[], str]


def _alternation(values: Sequence[str]) -> str:
    '''Return a regex alternation for the supplied literal values.'''
    ordered_values = list(dict.fromkeys(values))
    ordered_values.sort(key=len, reverse=True)
    return '|'.join(re.escape(value) for value in ordered_values)


def _optional_named_group(name: str, body: str, prefix: str = r'[ _]*') -> str:
    '''Build an optional named capture with an optional leading delimiter.'''
    return ''.join([prefix, fr'(?P<{name}>{body})?'])


TARGET_MODIFIERS = {
    'eval': 'Target volume explicitly intended for DVH evaluation',
    'opt': 'Target volume only intended for optimization',
    'mod': 'Target volume modified for unspecified reasons',
    'shell': 'Spherical shell used to optimize dose fall-off',
}

TARGET_TYPES = {
    'IGTV': 'Internal Gross Target Volume',
    'ICTV': 'Internal Clinical Target Volume',
    'PTV!': 'Partial Planning Target Volume',
    'Iliac Vessels': 'Nodal Volume',
    'Operative Bed': 'GTV Surrogate',
    'GTV': 'Gross Target Volume',
    'CTV': 'Clinical Target Volume',
    'ITV': 'Internal Target Volume',
    'PTV': 'Planning Target Volume',
    'Nodes': 'Nodal Volume',
    'Node': 'Nodal Volume',
    'LN': 'Nodal Volume',
    'Edema': 'CTV Surrogate',
    'Cavity': 'GTV Surrogate',
    'HTV': 'Clinical Target Volume',
    'HRV': 'Clinical Target Volume',
}

TARGET_CLASSIFIERS = {
    'PREOP': 'Pre-operative Target Volume',
    'Cavity': 'Target Expansion on the Cavity',
    'RES': 'Residual Disease Target Volume',
    'vas': 'vascular',
    'par': 'parenchyma',
    'sb': 'surgical bed',
    'low': 'Low Dose Target',
    'int': 'Intermediate Dose Target',
    'HR': 'High Risk Target Volume',
    'IR': 'Intermediate Risk Target Volume',
    'm': 'metastatic',
    'p': 'primary',
    'n': 'nodal',
    'v': 'venous thrombosis',
}

TARGET_MODALITIES = {
    'MRI': 'MRI',
    'PET': 'PET',
    'MR': 'MRI',
    'PT': 'PET',
    'CT': 'CT',
    'T1': 'MRI',
    'T2': 'MRI',
    'US': 'Ultrasound',
    'SP': 'SPECT',
}

TARGET_MOTIONS = {
    'MinIP': 'Minimum intensity projection',
    'MIP': 'Maximum intensity projection',
    'AIP': 'Average intensity projection',
    'AVE': 'Average intensity projection',
}

NODAL_REGIONS = {
    'Int iliac': 'Internal Iliac Nodes',
    'ext iliac': 'External Iliac Nodes',
    'com iliac': 'Common Iliac Nodes',
    'Para-Aortic': 'Para-Aortic Nodes',
    'Hepatoduod': 'Hepatoduodenal Nodes',
    'Hepatogastr': 'Hepatogastric Nodes',
    'Pancreatic': 'Pancreatic Nodes',
    'Subpyloric': 'Subpyloric Nodes',
    'Pyloric': 'Pyloric Nodes',
    'Obturator': 'Obturator Nodes',
    'Splenic': 'Splenic Nodes',
    'Gastric': 'Gastric Nodes',
    'Celiac': 'Celiac Nodes',
    'Hepatic': 'Hepatic Nodes',
    'Axilla': 'Axillary Nodes',
    'Sacral': 'Sacral Nodes',
    'IMC': 'Internal Mammary Chain Nodes',
    'SC': 'Supraclavicular Nodes',
    'Neck': 'Neck Nodes',
}

RELATIVE_DOSES = {
    'High': 'Highest target dose level',
    'Mid': 'Intermediate target dose level',
    'Low': 'Lowest target dose level',
}

COMBINED_TARGETS = {
    'BOTH': 'Combined target structure for all targets at the same dose level',
    'Total': 'Combined target structure for all targets at the same dose level',
    'All': 'Combined target structure for all targets at the same dose level',
}

LATERALITY_TYPES = {
    'RT': 'Right',
    'LT': 'Left',
    'MED': 'Medial',
    'LAT': 'Lateral',
    'POST': 'Posterior',
    'ANT': 'Anterior',
    'SUP': 'Superior',
    'INF': 'Inferior',
    'R': 'Right',
    'L': 'Left',
}


def build_target_modifier_fragment() -> str:
    '''Build the optional target modifier fragment.'''
    return ''.join([
        _optional_named_group('Mod', _alternation(TARGET_MODIFIERS), prefix=''),
        r'[ _]*',
    ])


def build_target_type_fragment() -> str:
    '''Build the required target type fragment.'''
    return fr'(?P<TargetType>{_alternation(TARGET_TYPES)})'


def build_target_classifier_fragment() -> str:
    '''Build the optional target classifier fragment.'''
    return ''.join([
        r'(?:',
        fr'(?P<Classifier>{_alternation(TARGET_CLASSIFIERS)})',
        r'(?![A-Za-z])',
        r')?',
    ])


def build_target_number_fragment() -> str:
    '''Build the optional target number fragment.'''
    return ''.join([
        r'(?:',
        r'[ _]*',
        r'(?<![0-9.+-])',
        r'(?P<TargetNumber>[1-9]|1[0-5])',
        r'(?![0-9.Dcm])',
        r')?',
    ])


def build_target_modality_fragment() -> str:
    '''Build the optional imaging modality fragment.'''
    return _optional_named_group('Modality', _alternation(TARGET_MODALITIES))


def build_target_motion_fragment() -> str:
    '''Build the optional motion-management fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?P<Motion>',
        _alternation(TARGET_MOTIONS),
        r')?',
        r'(?P<MotionPhase>4D[0-9][0-9]?)?',
    ])


def build_nodal_region_fragment() -> str:
    '''Build the optional nodal region fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?:',
        fr'(?P<Nodes>{_alternation(NODAL_REGIONS)})',
        r'[ _]*',
        r'(?P<NodalSubSection>[IVabc1-9]*)?',
        r')?',
    ])


def build_numeric_dose_fragment() -> str:
    '''Build the optional numeric dose fragment.'''
    return ''.join([
        r'(?:',
        r'[ _]*',
        r'(?P<TargetDose>[0-9]+[.p]?[0-9]*)',
        r'[ _]*',
        r'(?P<DoseUnits>(?:cGy|Gy))?',
        r'[ _]*',
        r'(?:',
        r'(?P<FractionDelimiter>/|in|x)',
        r'(?P<Fractions>[0-9]+)',
        r')?',
        r')?',
    ])


def build_relative_dose_fragment() -> str:
    '''Build the optional relative dose fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?:',
        fr'(?P<RelativeDose>{_alternation(RELATIVE_DOSES)})',
        r'(?P<RelativeDoseLevel>[0-9]{2})?',
        r')?',
    ])


def build_combined_target_fragment() -> str:
    '''Build the optional combined-target fragment.'''
    return _optional_named_group('Combined', _alternation(COMBINED_TARGETS))


def build_target_organ_fragment() -> str:
    '''Build the optional target-organ fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?P<TargetOrgan>[A-Za-z]{3,})?',
    ])


def build_target_expansion_fragment() -> str:
    '''Build the optional target-expansion fragment.'''
    expansion_one = ''.join([
        r'[ _]*',
        r'(?P<ExpansionSize1>[0-9.]+)',
        r'[ _]*',
        r'(?P<ExpansionUnit1>[cm]m)',
    ])

    expansion_two = ''.join([
        r' *',
        r'[+]',
        r' *',
        r'(?P<ExpansionSize2>[0-9.]+)',
        r'[ _]*',
        r'(?P<ExpansionUnit2>[cm]m)?',
    ])

    expansion_three = ''.join([
        r'[ _]*',
        r'Ev',
        r'[ _]*',
        r'(?P<ExpansionSize3>[0-9.]+)',
        r'[ _]*',
        r'(?P<ExpansionUnit3>[cm]m)?',
    ])

    return ''.join([
        r'(?:',
        expansion_one,
        r'|',
        expansion_two,
        r'|',
        expansion_three,
        r')?',
    ])


def build_target_minus_organ_fragment() -> str:
    '''Build the optional organ subtraction fragment.'''
    return ''.join([
        r'(?:',
        r'[ _]*',
        r'-',
        r' *',
        r'(?P<OrganSubtraction>[A-Za-z]*)',
        r')?',
    ])


def build_target_direction_fragment() -> str:
    '''Build the optional laterality/direction fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?P<TargetLaterality>',
        _alternation(LATERALITY_TYPES),
        r')?',
        r'[ _]*',
        r'(?P<TargetAP_Direction>ANT|POST)?',
        r'[ _]*',
        r'(?P<TargetSI_Direction>SUP|INF)?',
    ])


def build_target_subgroup_fragment() -> str:
    '''Build the optional subgroup fragment.'''
    return ''.join([
        r'[ _]*',
        r'(?P<TargetSubGroup>[abc])?',
    ])


def build_target_crop_fragment() -> str:
    '''Build the optional external crop fragment.'''
    return r'(?P<ExternalCrop>[0-9]{1,2})?'


def build_remainder_fragment() -> str:
    '''Build the remainder fragment.'''
    return r'(?P<Remainder>.+)?$'


STRUCTURE_ID_REGEX_FRAGMENT_BUILDERS: tuple[tuple[str, RegexFragmentBuilder], ...] = (
    ('target_mod', build_target_modifier_fragment),
    ('target_type', build_target_type_fragment),
    ('target_classifier', build_target_classifier_fragment),
    ('target_number', build_target_number_fragment),
    ('target_modality', build_target_modality_fragment),
    ('target_motion', build_target_motion_fragment),
    ('nodal_region', build_nodal_region_fragment),
    ('numeric_target_dose', build_numeric_dose_fragment),
    ('relative_target_dose', build_relative_dose_fragment),
    ('combined_target', build_combined_target_fragment),
    ('target_organ', build_target_organ_fragment),
    ('target_expansion', build_target_expansion_fragment),
    ('target_minus_organ', build_target_minus_organ_fragment),
    ('target_direction', build_target_direction_fragment),
    ('target_subgroup', build_target_subgroup_fragment),
    ('target_crop', build_target_crop_fragment),
    ('remainder', build_remainder_fragment),
)


def build_structure_id_regex_fragments() -> dict[str, str]:
    '''Build all regex fragments as an ordered mapping.'''
    return {
        name: builder()
        for name, builder in STRUCTURE_ID_REGEX_FRAGMENT_BUILDERS
    }


def build_structure_id_regex_pattern() -> str:
    '''Build the full structure ID regex pattern string.'''
    fragments = build_structure_id_regex_fragments()
    return ''.join([
        '^',
        *(fragments[name] for name, _ in STRUCTURE_ID_REGEX_FRAGMENT_BUILDERS),
    ])


def build_structure_id_regex() -> re.Pattern[str]:
    '''Compile the structure ID regex.'''
    return re.compile(build_structure_id_regex_pattern())


def merge_priority_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    '''Return the first non-empty value across a set of columns.'''
    if not columns:
        raise ValueError('columns must contain at least one column name')

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f'Missing columns: {missing}')

    temp = df[columns].copy()
    for column in columns:
        if pd.api.types.is_object_dtype(temp[column]) or pd.api.types.is_string_dtype(temp[column]):
            temp[column] = temp[column].replace(r'^\s*$', pd.NA, regex=True)

    return temp.bfill(axis=1).iloc[:, 0]


def parse_structure_ids(
    structure_ids: Sequence[str],
    regex: re.Pattern[str] | None = None,
    note_failures: bool = False,
    drop_blanks: bool = True,
) -> pd.DataFrame:
    '''Parse structure IDs into a DataFrame of named regex groups.'''
    compiled_regex = regex or build_structure_id_regex()
    matched_groups: list[dict[str, object]] = []

    for structure_id in structure_ids:
        structure_text = '' if structure_id is None else str(structure_id)
        match = compiled_regex.match(structure_text)
        if match:
            groups = match.groupdict()
            groups['Structure'] = structure_text
            matched_groups.append(groups)
        elif note_failures:
            logger.warning('Structure %s did not match the parser regex', structure_text)

    if not matched_groups:
        return pd.DataFrame(index=pd.Index([], name='Structure'))

    matched_groups_df = pd.DataFrame(matched_groups)
    if drop_blanks:
        matched_groups_df.dropna(axis=1, how='all', inplace=True)
    matched_groups_df.set_index('Structure', inplace=True)
    return matched_groups_df


def _collapse_expansion_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''Collapse expansion helper columns into the notebook-style output columns.'''
    result = df.copy()

    expansion_size_columns = [
        column for column in ['ExpansionSize1', 'ExpansionSize2', 'ExpansionSize3']
        if column in result.columns
    ]
    if expansion_size_columns:
        result['ExpansionSize'] = merge_priority_columns(result, expansion_size_columns)
        result.drop(columns=expansion_size_columns, inplace=True)

    expansion_unit_columns = [
        column for column in ['ExpansionUnit1', 'ExpansionUnit2', 'ExpansionUnit3']
        if column in result.columns
    ]
    if expansion_unit_columns:
        result['ExpansionUnit'] = merge_priority_columns(result, expansion_unit_columns)
        result.drop(columns=expansion_unit_columns, inplace=True)

    return result


def parse_structure_metadata(
    metadata: pd.DataFrame,
    regex: re.Pattern[str] | None = None,
    structure_column: str = 'Structure ID',
    note_failures: bool = False,
    drop_blanks: bool = True,
) -> pd.DataFrame:
    '''Parse a single-file structure metadata table into target columns.'''
    if structure_column not in metadata.columns:
        raise KeyError(f'Missing required column: {structure_column}')

    parsed = parse_structure_ids(
        metadata[structure_column].tolist(),
        regex=regex,
        note_failures=note_failures,
        drop_blanks=drop_blanks,
    )
    parsed.index.rename(structure_column, inplace=True)

    metadata_indexed = metadata.copy().set_index(structure_column)
    merged = parsed.merge(metadata_indexed, how='left', left_index=True, right_index=True)
    return _collapse_expansion_columns(merged)


__all__ = [
    'build_structure_id_regex',
    'build_structure_id_regex_fragments',
    'build_structure_id_regex_pattern',
    'merge_priority_columns',
    'parse_structure_ids',
    'parse_structure_metadata',
]
