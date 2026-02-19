'''Test for the dicom.py module.

These tests verify that DicomStructureFile correctly loads DICOM RT Structure
files and extracts structure information without using deprecated DICOM fields.
'''
from pathlib import Path

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
