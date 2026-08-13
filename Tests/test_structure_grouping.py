'''Tests for the structure_grouping module.'''

from pathlib import Path

from dicom import DicomStructureFile
import pandas as pd

from structure_grouping import (
    GroupDefinitionSet,
    GroupFieldDefinition,
    analyze_structure_grouping,
    build_structure_grouping_table,
    default_structure_group_definition_set,
)


def test_build_structure_grouping_table_splits_opt_ptv56_by_subgroups():
    '''opt PTV 56 rows should split by laterality and then subgroup.'''
    structures_df = pd.DataFrame(
        [
            {
                'ROINumber': 71,
                'Structure ID': 'opt PTV 56 L a',
                'Mod': 'opt',
                'DICOM Type': 'PTV',
                'TargetType': 'PTV',
                'TargetDose': '56',
                'TargetLaterality': 'L',
                'TargetSubGroup': 'a',
            },
            {
                'ROINumber': 72,
                'Structure ID': 'opt PTV 56 L b',
                'Mod': 'opt',
                'DICOM Type': 'PTV',
                'TargetType': 'PTV',
                'TargetDose': '56',
                'TargetLaterality': 'L',
                'TargetSubGroup': 'b',
            },
            {
                'ROINumber': 74,
                'Structure ID': 'opt PTV 56 R a',
                'Mod': 'opt',
                'DICOM Type': 'PTV',
                'TargetType': 'PTV',
                'TargetDose': '56',
                'TargetLaterality': 'R',
                'TargetSubGroup': 'a',
            },
            {
                'ROINumber': 75,
                'Structure ID': 'opt PTV 56 R b',
                'Mod': 'opt',
                'DICOM Type': 'PTV',
                'TargetType': 'PTV',
                'TargetDose': '56',
                'TargetLaterality': 'R',
                'TargetSubGroup': 'b',
            },
            {
                'ROINumber': 76,
                'Structure ID': 'opt PTV 56 R c',
                'Mod': 'opt',
                'DICOM Type': 'PTV',
                'TargetType': 'PTV',
                'TargetDose': '56',
                'TargetLaterality': 'R',
                'TargetSubGroup': 'c',
            },
        ]
    ).set_index('ROINumber', drop=False)

    result = build_structure_grouping_table(structures_df)

    assert list(result['h_group_1'].unique()) == ['56']
    assert result.loc[71, 'h_group_2'] == 'L'
    assert result.loc[74, 'h_group_2'] == 'R'
    assert result.loc[71, 'v_group_1'] == 'opt'
    assert result.loc[71, 'v_group_2'] == 'PTV'
    assert result.loc[71, 'h_group_3'] == 'a'
    assert result.loc[72, 'h_group_3'] == 'b'
    assert result.loc[76, 'h_group_3'] == 'c'
    assert result.loc[71, 'h_index'] < result.loc[72, 'h_index']
    assert result.loc[71, 'h_index'] < result.loc[74, 'h_index']
    assert result.loc[74, 'v_index'] == result.loc[75, 'v_index']
    assert result.loc[75, 'v_index'] == result.loc[76, 'v_index']
    assert bool(result['is_unique_slot'].all())


def test_build_structure_grouping_table_supports_mixed_numeric_sorting():
    '''Mixed sorting should place numeric values before text by float order.'''
    structures_df = pd.DataFrame(
        [
            {'ROINumber': 1, 'Structure ID': 'Ten', 'GroupValue': '10'},
            {'ROINumber': 2, 'Structure ID': 'Two', 'GroupValue': '2'},
            {'ROINumber': 3, 'Structure ID': 'Alpha', 'GroupValue': 'Alpha'},
        ]
    ).set_index('ROINumber', drop=False)
    grouping_definition_set = GroupDefinitionSet(
        name='mixed_sort_test',
        field_definitions=(
            GroupFieldDefinition(
                axis='horizontal',
                hierarchy_rank=1,
                level_name='Mixed Group',
                source_columns=('GroupValue',),
                sort_mode='mixed',
                numeric_parse_mode='float',
                numeric_position='before_text',
                text_sort_mode='alphabetical',
            ),
        ),
    )

    result = build_structure_grouping_table(
        structures_df=structures_df,
        grouping_definition_set=grouping_definition_set,
    )

    assert result['Structure ID'].tolist() == ['Two', 'Ten', 'Alpha']


def test_analyze_structure_grouping_uses_hn_example_as_acceptance_case():
    '''The H&N notebook example should split opt PTV 56 rows as planned.'''
    dicom_file = DicomStructureFile(
        top_dir=Path('tests'),
        file_path=Path('tests/RS.OROP2_3_dose_levels.dcm'),
    )

    result = analyze_structure_grouping(
        dicom_file=dicom_file,
        grouping_definition_set=default_structure_group_definition_set(),
    )
    opt_rows = result[result['Structure ID'].str.startswith('opt PTV 56')].copy()

    assert set(opt_rows['h_group_2']) == {'L', 'R'}
    assert set(opt_rows['h_group_3']) == {'a', 'b', 'c'}
    assert opt_rows.loc[71, 'h_index'] < opt_rows.loc[72, 'h_index']
    assert opt_rows.loc[74, 'h_index'] < opt_rows.loc[75, 'h_index']
    assert opt_rows.loc[71, 'h_index'] < opt_rows.loc[74, 'h_index']
    assert opt_rows.loc[74, 'v_index'] == opt_rows.loc[75, 'v_index']
    assert opt_rows.loc[75, 'v_index'] == opt_rows.loc[76, 'v_index']
    assert bool(opt_rows['is_unique_slot'].all())
