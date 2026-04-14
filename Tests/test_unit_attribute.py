#!/usr/bin/env python
'''Test script to verify unit attribute functionality in StructureSet.'''

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dicom import DicomStructureFile
from structure_set import StructureSet

print("Testing StructureSet unit attribute")
print("=" * 50)

# Test 1: Default unit for empty StructureSet
print("\n1. Testing default unit for empty StructureSet:")
structure_set = StructureSet()
print(f"   Default unit: {structure_set.unit}")
assert structure_set.unit == 'cm', "Default unit should be 'cm'"
print("   ✓ Passed")

# Test 2: Custom unit for manually created StructureSet
print("\n2. Testing custom unit parameter:")
structure_set_mm = StructureSet(unit='mm')
print(f"   Custom unit: {structure_set_mm.unit}")
assert structure_set_mm.unit == 'mm', "Custom unit should be 'mm'"
print("   ✓ Passed")

# Test 3: Unit from DICOM data
print("\n3. Testing unit from DICOM file:")
test_file = Path(__file__).parent / 'RS.GJS_Struct_Tests.Relations.dcm'
if test_file.exists():
    dicom_file = DicomStructureFile(
        top_dir=test_file.parent,
        file_name=test_file.name
    )
    structure_set_dicom = StructureSet(
        dicom_structure_file=dicom_file,
        auto_calculate_relationships=False,
        auto_calculate_logical_flags=False
    )
    print(f"   DICOM unit: {structure_set_dicom.unit}")
    assert structure_set_dicom.unit == 'cm', "DICOM unit should be 'cm' (converted from mm)"
    print("   ✓ Passed")

    # Test 4: Unit in to_dict output
    print("\n4. Testing unit in to_dict output:")
    data_dict = structure_set_dicom.to_dict()
    print(f"   Unit in dict: {data_dict.get('unit')}")
    assert 'unit' in data_dict, "to_dict should include 'unit' field"
    assert data_dict['unit'] == 'cm', "Unit in dict should match structure_set.unit"
    print("   ✓ Passed")
else:
    print(f"   ⚠ Skipped (test file not found: {test_file})")

print("\n" + "=" * 50)
print("All tests passed! ✓")
