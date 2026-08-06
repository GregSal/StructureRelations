'''Tests for StructureSet metadata ownership.'''

import logging
from pathlib import Path

import pandas as pd

from structure_set import StructureSet


class FakeDicomStructureFile:
    '''Minimal DICOM source for metadata snapshot tests.'''

    def __init__(self, filter_report: pd.DataFrame | None = None) -> None:
        self.contour_points = []
        self.structure_filter_report = (
            filter_report.copy(deep=True)
            if filter_report is not None
            else pd.DataFrame()
        )
        self.structure_filter_config_path = Path('filter_rules.json')
        self.evaluate_count = 0
        self.metadata = pd.DataFrame([
            {
                'ROINumber': 1,
                'Structure ID': 'PTV56',
                'DICOM Type': 'PTV',
            },
        ]).set_index('ROINumber', drop=False)

    def get_structure_filter_metadata(self) -> pd.DataFrame:
        '''Return mutable source metadata.'''
        return self.metadata

    def evaluate_structure_filters(self) -> pd.DataFrame:
        '''Return a filter report and record evaluation.'''
        self.evaluate_count += 1
        self.structure_filter_report = self.metadata.assign(
            IsFiltered=False,
            IsHidden=False,
            SelectedByDefault=True,
            DisplayedByDefault=True,
        )
        return self.structure_filter_report


def test_structure_set_captures_metadata_before_empty_contour_return() -> None:
    '''DICOM metadata should remain available without contour geometry.'''
    dicom_file = FakeDicomStructureFile()

    structure_set = StructureSet(dicom_structure_file=dicom_file)

    assert structure_set.structure_metadata.loc[1, 'Structure ID'] == 'PTV56'
    assert bool(
        structure_set.structure_filter_report.loc[1, 'SelectedByDefault']
    )
    assert dicom_file.evaluate_count == 1
    assert structure_set.structure_filter_config_path == Path(
        'filter_rules.json',
    )


def test_structure_set_reuses_and_copies_existing_filter_report() -> None:
    '''An existing report should be copied without reevaluation.'''
    existing_report = pd.DataFrame([
        {
            'ROINumber': 1,
            'Structure ID': 'PTV56',
            'SelectedByDefault': True,
            'DisplayedByDefault': False,
        },
    ]).set_index('ROINumber', drop=False)
    dicom_file = FakeDicomStructureFile(filter_report=existing_report)

    structure_set = StructureSet(dicom_structure_file=dicom_file)
    dicom_file.metadata.loc[1, 'Structure ID'] = 'changed'
    dicom_file.structure_filter_report.loc[1, 'DisplayedByDefault'] = True

    assert dicom_file.evaluate_count == 0
    assert structure_set.structure_metadata.loc[1, 'Structure ID'] == 'PTV56'
    assert not bool(
        structure_set.structure_filter_report.loc[1, 'DisplayedByDefault']
    )


def test_non_dicom_structure_set_has_empty_metadata_tables() -> None:
    '''Non-DICOM construction should expose stable empty metadata state.'''
    structure_set = StructureSet()

    assert structure_set.structure_metadata.empty
    assert structure_set.structure_filter_report.empty
    assert structure_set.structure_filter_config_path is None


def test_structure_set_logging_can_be_disabled(caplog) -> None:
    '''Logging should be suppressible for one StructureSet instance.'''
    enabled = StructureSet(include_structures=['Included'])
    disabled = StructureSet(
        include_structures=['Included'],
        logging_enabled=False,
    )

    with caplog.at_level(logging.DEBUG, logger='structure_set'):
        assert not enabled._should_include_structure('Excluded enabled')
        assert not disabled._should_include_structure('Excluded disabled')

    messages = [record.getMessage() for record in caplog.records]
    assert any('Excluded enabled' in message for message in messages)
    assert all('Excluded disabled' not in message for message in messages)
