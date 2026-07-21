'''Tests for the structure_id_parser module.'''

import pandas as pd

from structure_id_parser import (
    build_structure_id_regex,
    parse_structure_ids,
    parse_structure_metadata,
)


def test_build_structure_id_regex_extracts_target_fields():
    '''The compiled regex should capture the expected target fields.'''
    regex = build_structure_id_regex()
    match = regex.match('eval PTV_50.4Gyx3')

    assert match is not None
    groups = match.groupdict()

    assert groups['Mod'] == 'eval'
    assert groups['TargetType'] == 'PTV'
    assert groups['TargetDose'] == '50.4'
    assert groups['DoseUnits'] == 'Gy'
    assert groups['Fractions'] == '3'


def test_parse_structure_ids_returns_indexed_dataframe():
    '''Parsing a list of IDs should return an indexed DataFrame.'''
    parsed = parse_structure_ids([
        'PTV_5040',
        'PTV +5mm',
        'Avoid a Rectum',
    ])

    assert list(parsed.index) == ['PTV_5040', 'PTV +5mm']
    assert parsed.index.name == 'Structure'
    assert 'TargetType' in parsed.columns
    assert 'ExpansionSize2' in parsed.columns or 'ExpansionSize' in parsed.columns


def test_parse_structure_metadata_merges_and_collapses_expansions():
    '''Parsing metadata should preserve original columns and collapse expansion fields.'''
    metadata = pd.DataFrame([
        {
            'ROINumber': 1,
            'Structure ID': 'PTV_5040',
            'Structure Name': 'Boost',
            'DICOM Type': 'PTV',
            'Structure Code': '',
            'Coding Scheme': '',
            'Code Meaning': '',
            'Density': '',
            'ROI Physical Property': '',
            'Generation Method': '',
            'Generation Description': '',
            'Contour Count': 2,
            'Has Contours': True,
            'File': 'sample.dcm',
        },
        {
            'ROINumber': 2,
            'Structure ID': 'PTV +5mm',
            'Structure Name': 'Boost Expansion',
            'DICOM Type': 'PTV',
            'Structure Code': '',
            'Coding Scheme': '',
            'Code Meaning': '',
            'Density': '',
            'ROI Physical Property': '',
            'Generation Method': '',
            'Generation Description': '',
            'Contour Count': 1,
            'Has Contours': True,
            'File': 'sample.dcm',
        },
        {
            'ROINumber': 3,
            'Structure ID': 'Avoid a Rectum',
            'Structure Name': 'Non-target',
            'DICOM Type': 'AVOIDANCE',
            'Structure Code': '',
            'Coding Scheme': '',
            'Code Meaning': '',
            'Density': '',
            'ROI Physical Property': '',
            'Generation Method': '',
            'Generation Description': '',
            'Contour Count': 4,
            'Has Contours': True,
            'File': 'sample.dcm',
        },
    ])

    parsed = parse_structure_metadata(metadata)

    assert list(parsed.index) == ['PTV_5040', 'PTV +5mm']
    assert parsed.index.name == 'Structure ID'
    assert parsed.loc['PTV_5040', 'ROINumber'] == 1
    assert parsed.loc['PTV_5040', 'Structure Name'] == 'Boost'
    assert parsed.loc['PTV_5040', 'File'] == 'sample.dcm'
    assert parsed.loc['PTV +5mm', 'ExpansionSize'] == '5'
    assert parsed.loc['PTV +5mm', 'ExpansionUnit'] == 'mm'

    for column in ['ExpansionSize1', 'ExpansionSize2', 'ExpansionSize3']:
        assert column not in parsed.columns
    for column in ['ExpansionUnit1', 'ExpansionUnit2', 'ExpansionUnit3']:
        assert column not in parsed.columns
    assert 'grouping' not in parsed.columns
