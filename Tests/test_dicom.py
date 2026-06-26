'''Test for the dicom.py module.

These tests verify that DicomStructureFile correctly loads DICOM RT Structure
files and extracts structure information without using deprecated DICOM fields.
'''
from pathlib import Path
import json
import shutil

import pytest
import pandas as pd

from dicom import DicomStructureFile
from types_and_classes import ROI_Type


class TestDicomStructureFile:
    '''Test the DicomStructureFile class.'''

    @pytest.fixture
    def test_dicom_file(self):
        '''Fixture to load the test DICOM file.'''
        tests_dir = Path(__file__).parent
        file_name = 'RS.HN_Struct.OROP.dcm'
        dicom_file = DicomStructureFile(
            top_dir=tests_dir,
            file_name=file_name
        )
        return dicom_file

    def test_file_loading(self, test_dicom_file):
        '''Test that the DICOM file loads successfully.'''
        assert test_dicom_file is not None
        assert test_dicom_file.dataset is not None
        assert test_dicom_file.is_structure_file()

    def test_structure_names(self, test_dicom_file):
        '''Test that structure names are extracted correctly.'''
        structure_names = test_dicom_file.structure_names

        assert structure_names is not None
        assert isinstance(structure_names, dict)
        assert len(structure_names) > 0

        # All keys should be ROI_Type (int)
        for roi_num in structure_names.keys():
            assert isinstance(roi_num, int)

        # All values should be strings
        for name in structure_names.values():
            assert isinstance(name, str)
            assert len(name) > 0  # Names should not be empty

    def test_get_roi_labels(self, test_dicom_file):
        '''Test that ROI labels are extracted correctly without using
        deprecated fields.

        This test verifies that:
        1. The method returns a DataFrame
        2. StructureName column exists and is populated
        3. StructureName values match those from get_structure_names()
        4. The deprecated ROIObservationLabel field is not used
        '''
        roi_labels = test_dicom_file.get_roi_labels()

        # Should return a DataFrame
        assert isinstance(roi_labels, pd.DataFrame)

        # DataFrame should not be empty
        assert len(roi_labels) > 0

        # Should have expected columns
        assert 'StructureName' in roi_labels.columns
        assert 'DICOM_Type' in roi_labels.columns

        # ROINumber should be the index
        assert roi_labels.index.name == 'ROINumber'

        # All structure names should be non-empty strings
        for idx, row in roi_labels.iterrows():
            assert isinstance(row['StructureName'], str)
            assert len(row['StructureName']) > 0

            # Verify structure name matches get_structure_names()
            expected_name = test_dicom_file.structure_names.get(idx)
            assert row['StructureName'] == expected_name, (
                f"ROI {idx}: StructureName '{row['StructureName']}' "
                f"doesn't match expected '{expected_name}'"
            )

    def test_roi_labels_consistency(self, test_dicom_file):
        '''Test that get_roi_labels() returns consistent data with
        get_structure_names().
        '''
        structure_names = test_dicom_file.structure_names
        roi_labels = test_dicom_file.get_roi_labels()

        # All ROI numbers in roi_labels should have corresponding entries
        # in structure_names
        for roi_num in roi_labels.index:
            assert roi_num in structure_names, (
                f"ROI {roi_num} in roi_labels but not in structure_names"
            )

            # Names should match exactly
            label_name = roi_labels.loc[roi_num, 'StructureName']
            struct_name = structure_names[roi_num]
            assert label_name == struct_name, (
                f"ROI {roi_num}: Names don't match - "
                f"roi_labels: '{label_name}', structure_names: '{struct_name}'"
            )

    def test_optional_code_fields(self, test_dicom_file):
        '''Test that optional code fields are handled correctly.'''
        roi_labels = test_dicom_file.get_roi_labels()

        # Optional fields may or may not be present
        optional_fields = ['Code', 'CodeScheme', 'CodeMeaning']

        for field in optional_fields:
            if field in roi_labels.columns:
                # If present, verify it's properly populated
                # (can be None, NaN, or string for individual rows)
                for idx, row in roi_labels.iterrows():
                    value = row.get(field)
                    assert (value is None or
                            isinstance(value, str) or
                            pd.isna(value))

    def test_contour_points_extraction(self, test_dicom_file):
        '''Test that contour points are extracted successfully.'''
        contour_points = test_dicom_file.contour_points

        assert contour_points is not None
        assert isinstance(contour_points, list)
        assert len(contour_points) > 0

    def test_structure_set_info(self, test_dicom_file):
        '''Test that structure set information is extracted.'''
        info = test_dicom_file.structure_set_info

        assert isinstance(info, dict)
        assert 'PatientID' in info
        assert 'StructureSet' in info
        assert 'File' in info

    def test_uploaded_session_prefix_is_removed_from_file_name(
        self,
        tmp_path,
    ):
        '''Uploaded temp prefixes should not leak into the internal file name.'''
        source_path = Path(__file__).parent / 'RS.HN_Struct.OROP.dcm'
        prefixed_name = '123e4567-e89b-12d3-a456-426614174000_RS.HN_Struct.OROP.dcm'
        copied_path = tmp_path / prefixed_name
        shutil.copy(source_path, copied_path)

        dicom_file = DicomStructureFile(
            top_dir=tmp_path,
            file_path=copied_path,
        )

        assert dicom_file.file_name == 'RS.HN_Struct.OROP.dcm'

    def test_no_deprecated_field_access(self, test_dicom_file):
        '''Verify that the deprecated ROIObservationLabel field is not
        accessed during ROI label extraction.

        This is an indirect test - we verify that the structure names
        come from StructureSetROISequence, not RTROIObservationsSequence.
        '''
        # Get structure names (from StructureSetROISequence)
        structure_names = test_dicom_file.structure_names

        # Get ROI labels (should use structure_names, not ROIObservationLabel)
        roi_labels = test_dicom_file.get_roi_labels()

        # If both succeed and match, we're using the correct source
        for roi_num in roi_labels.index:
            assert roi_num in structure_names
            assert roi_labels.loc[roi_num, 'StructureName'] == structure_names[roi_num]

    def test_json_structure_filters_record_matching_rules(self, tmp_path):
        '''JSON filters should track which rule deselected each structure.'''
        filter_path = tmp_path / 'structure_filter_rules.json'
        filter_path.write_text(
            json.dumps({
                'rules': [
                    {
                        'id': 'bowel-by-code-meaning',
                        'field': 'Code Meaning',
                        'match_type': 'exact',
                        'value': 'Intestine',
                        },
                    {
                        'id': 'artifact-z-prefix',
                        'field': 'Structure ID',
                        'match_type': 'prefix',
                        'value': 'Z',
                    },
                    {
                        'id': 'avoidance-rectum-suffix',
                        'field': 'Structure ID',
                        'match_type': 'suffix',
                        'value': 'Rectum',
                        'with': {
                            'field': 'DICOM Type',
                            'match_type': 'exact',
                            'value': 'AVOIDANCE',
                        },
                    },
                    {
                        'id': 'ptv-regex',
                        'field': 'Structure ID',
                        'match_type': 'regex',
                        'value': '^PTV 5[68]$',
                    },
                ]
            }),
            encoding='utf-8',
        )

        dicom_file = DicomStructureFile(
            top_dir=Path(__file__).parent,
            file_name='RS.Pros_Equal.dcm',
        )

        excluded = dicom_file.get_excluded_structures(filter_path)
        excluded_names = set(excluded['StructureID'])

        assert {'Bowel', 'Z1', 'Z2', 'Z3', 'PTV 56',
                'Avoid a Rectum','Avoid b Rectum'}.issubset(excluded_names)
        assert 'Rectum' not in excluded_names

        bowel_row = excluded.loc[excluded['StructureID'] == 'Bowel'].iloc[0]
        assert any(
            rule['id'] == 'bowel-by-code-meaning'
            for rule in bowel_row['MatchedRules']
        )

        rectum_row = excluded.loc[
            excluded['StructureID'] == 'Avoid a Rectum'
        ].iloc[0]
        assert any(
            rule['id'] == 'avoidance-rectum-suffix'
            for rule in rectum_row['MatchedRules']
        )

        bowel_roi = int(bowel_row['ROINumber'])
        filtered_contours = dicom_file.filter_exclusions(filter_path)
        remaining_rois = {int(cp['ROI']) for cp in filtered_contours}

        assert bowel_roi not in remaining_rois
        assert dicom_file.structure_filter_config_path == filter_path
        assert not dicom_file.structure_filter_report.empty


class TestDicomStructureFileWithDifferentFiles:
    '''Test DicomStructureFile with multiple DICOM files.'''

    @pytest.fixture(params=[
        'RS.HN_Struct.OROP.dcm',
        'RS.GJS_Struct_Tests.Relations.dcm',
    ])
    def dicom_file(self, request):
        '''Parametrized fixture to test multiple DICOM files.'''
        tests_dir = Path(__file__).parent
        file_name = request.param

        # Check if file exists before loading
        file_path = tests_dir / file_name
        if not file_path.exists():
            pytest.skip(f"Test file {file_name} not found")

        dicom_file = DicomStructureFile(
            top_dir=tests_dir,
            file_name=file_name
        )
        return dicom_file

    def test_structure_names_not_empty(self, dicom_file):
        '''Test that structure names are extracted for various DICOM files.'''
        structure_names = dicom_file.structure_names
        assert len(structure_names) > 0

    def test_roi_labels_match_structure_names(self, dicom_file):
        '''Test that ROI labels match structure names across different files.'''
        structure_names = dicom_file.structure_names
        roi_labels = dicom_file.get_roi_labels()

        for roi_num in roi_labels.index:
            if roi_num in structure_names:
                assert roi_labels.loc[roi_num, 'StructureName'] == structure_names[roi_num]


class TestStructureFilterRuleOptions:
    '''Unit tests for extended structure_filter_rules behavior.'''

    @staticmethod
    def _build_filter_test_file(metadata_rows, config):
        '''Create a lightweight DicomStructureFile test double.'''
        dicom_file = DicomStructureFile.__new__(DicomStructureFile)
        metadata = pd.DataFrame(metadata_rows)
        dicom_file.get_structure_filter_metadata = lambda: metadata
        dicom_file.load_structure_filter_rules = (
            lambda filter_file=None: (None, config)
        )
        dicom_file.structure_filter_report = pd.DataFrame()
        dicom_file.structure_filter_config_path = None
        return dicom_file

    def test_structure_list_match_type_uses_exact_structure_id(self):
        '''structure list should match exact Structure ID values only.'''
        dicom_file = self._build_filter_test_file(
            metadata_rows=[
                {'ROINumber': 1, 'Structure ID': 'PTV56'},
                {'ROINumber': 2, 'Structure ID': 'ptv56'},
                {'ROINumber': 3, 'Structure ID': 'Rectum'},
            ],
            config={
                'include by default': False,
                'display by default': True,
                'rules': [
                    {
                        'id': 'include-structure-list',
                        'action': 'include',
                        'field': 'Structure ID',
                        'match_type': 'structure list',
                        'value': ['PTV56', 'Rectum'],
                    }
                ],
            },
        )

        report = dicom_file.evaluate_structure_filters()

        assert not bool(report.loc[1, 'IsFiltered'])
        assert bool(report.loc[2, 'IsFiltered'])
        assert not bool(report.loc[3, 'IsFiltered'])

    def test_last_matching_rule_overrides_previous_matches(self):
        '''Later matching rules should override earlier matching rules.'''
        dicom_file = self._build_filter_test_file(
            metadata_rows=[
                {'ROINumber': 1, 'Structure ID': 'A'},
                {'ROINumber': 2, 'Structure ID': 'B'},
                {'ROINumber': 3, 'Structure ID': 'C'},
            ],
            config={
                'include by default': False,
                'display by default': False,
                'rules': [
                    {
                        'id': 'include-a',
                        'action': 'include',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'A',
                    },
                    {
                        'id': 'exclude-a',
                        'action': 'exclude',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'A',
                    },
                    {
                        'id': 'display-b',
                        'action': 'display',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'B',
                    },
                    {
                        'id': 'hide-b',
                        'action': 'hide',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'B',
                    },
                    {
                        'id': 'hide-c',
                        'action': 'hide',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'C',
                    },
                    {
                        'id': 'display-c',
                        'action': 'display',
                        'field': 'Structure ID',
                        'match_type': 'exact',
                        'value': 'C',
                    },
                ],
            },
        )

        report = dicom_file.evaluate_structure_filters()

        assert bool(report.loc[1, 'IsFiltered'])
        assert bool(report.loc[2, 'IsHidden'])
        assert not bool(report.loc[3, 'IsHidden'])
        assert report.loc[1, 'FinalMatch']['id'] == 'exclude-a'
        assert report.loc[2, 'FinalMatch']['id'] == 'hide-b'
        assert report.loc[3, 'FinalMatch']['id'] == 'display-c'

    def test_include_display_defaults_accept_boolean_like_strings(self):
        '''Top-level defaults should coerce common true/false string values.'''
        dicom_file = self._build_filter_test_file(
            metadata_rows=[
                {'ROINumber': 1, 'Structure ID': 'OnlyStructure'},
            ],
            config={
                'include by default': 'false',
                'display by default': 'true',
                'rules': [],
            },
        )

        report = dicom_file.evaluate_structure_filters()

        assert bool(report.loc[1, 'IsFiltered'])
        assert not bool(report.loc[1, 'IsHidden'])
        assert not bool(report.loc[1, 'SelectedByDefault'])
        assert bool(report.loc[1, 'DisplayedByDefault'])

    def test_invalid_regex_rule_is_skipped_gracefully(self, caplog):
        '''Invalid regex should not abort evaluation of other valid rules.'''
        dicom_file = self._build_filter_test_file(
            metadata_rows=[
                {'ROINumber': 1, 'Structure ID': 'PTV56'},
                {'ROINumber': 2, 'Structure ID': 'Rectum'},
            ],
            config={
                'include by default': False,
                'display by default': True,
                'rules': [
                    {
                        'id': 'invalid-regex-rule',
                        'action': 'include',
                        'field': 'Structure ID',
                        'match_type': 'regex',
                        'value': '(*bad',
                    },
                    {
                        'id': 'valid-ptv-regex',
                        'action': 'include',
                        'field': 'Structure ID',
                        'match_type': 'regex',
                        'value': r'^PTV\d+$',
                    },
                ],
            },
        )

        with caplog.at_level('WARNING', logger='dicom'):
            report = dicom_file.evaluate_structure_filters()

        # Evaluation continues and valid regex still applies.
        assert not bool(report.loc[1, 'IsFiltered'])
        assert bool(report.loc[2, 'IsFiltered'])
        assert report.loc[1, 'FinalMatch']['id'] == 'valid-ptv-regex'

        # Invalid regex is reported but does not raise.
        assert any(
            'Invalid regex in structure filter rule' in record.message
            and 'invalid-regex-rule' in record.message
            for record in caplog.records
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
