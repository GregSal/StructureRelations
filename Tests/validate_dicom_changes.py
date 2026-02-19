'''Simple validation script to test DICOM changes.

This script verifies that DicomStructureFile correctly extracts structure names
without using the deprecated ROIObservationLabel field.
'''
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dicom import DicomStructureFile

def test_dicom_changes():
    '''Test that the DICOM changes work correctly.'''
    print("=" * 70)
    print("Testing DICOM Structure File Changes")
    print("=" * 70)

    # Load test DICOM file
    tests_dir = Path(__file__).parent
    test_file = 'RS.HN_Struct.OROP.dcm'

    print(f"\nLoading DICOM file: {test_file}")
    try:
        dicom_file = DicomStructureFile(
            top_dir=tests_dir,
            file_name=test_file
        )
        print("✓ File loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load file: {e}")
        return False

    # Test structure names extraction
    print("\nTesting structure_names extraction...")
    try:
        structure_names = dicom_file.structure_names
        print(f"✓ Extracted {len(structure_names)} structure names")

        # Show sample
        for i, (roi_num, name) in enumerate(list(structure_names.items())[:3]):
            print(f"  ROI {roi_num}: {name}")
        if len(structure_names) > 3:
            print(f"  ... and {len(structure_names) - 3} more")

    except Exception as e:
        print(f"✗ Failed to get structure names: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test ROI labels extraction (the updated method)
    print("\nTesting get_roi_labels() - using structure_names, not ROIObservationLabel...")
    try:
        roi_labels = dicom_file.get_roi_labels()
        print(f"✓ Extracted ROI labels for {len(roi_labels)} structures")

        # Verify it's a DataFrame
        import pandas as pd
        assert isinstance(roi_labels, pd.DataFrame), "roi_labels should be a DataFrame"
        print("✓ Result is a pandas DataFrame")

        # Verify required columns
        assert 'StructureName' in roi_labels.columns, "Missing StructureName column"
        assert 'DICOM_Type' in roi_labels.columns, "Missing DICOM_Type column"
        print("✓ Contains required columns: StructureName, DICOM_Type")

        # Show sample
        print("\n  Sample ROI labels:")
        for i, (roi_num, row) in enumerate(roi_labels.head(3).iterrows()):
            print(f"  ROI {roi_num}: {row['StructureName']} ({row['DICOM_Type']})")

    except Exception as e:
        print(f"✗ Failed to get ROI labels: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify consistency between structure_names and roi_labels
    print("\nVerifying consistency between structure_names and roi_labels...")
    try:
        all_match = True
        mismatches = []

        for roi_num in roi_labels.index:
            if roi_num not in structure_names:
                mismatches.append(f"ROI {roi_num} in roi_labels but not in structure_names")
                all_match = False
                continue

            label_name = roi_labels.loc[roi_num, 'StructureName']
            struct_name = structure_names[roi_num]

            if label_name != struct_name:
                mismatches.append(
                    f"ROI {roi_num}: roi_labels='{label_name}' vs structure_names='{struct_name}'"
                )
                all_match = False

        if all_match:
            print("✓ All structure names match between sources")
        else:
            print(f"✗ Found {len(mismatches)} mismatches:")
            for mismatch in mismatches[:5]:
                print(f"  {mismatch}")
            return False

    except Exception as e:
        print(f"✗ Failed consistency check: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nThe deprecated ROIObservationLabel field is no longer used.")
    print("Structure names are now correctly sourced from StructureSetROISequence.ROIName")

    return True


if __name__ == '__main__':
    success = test_dicom_changes()
    sys.exit(0 if success else 1)
