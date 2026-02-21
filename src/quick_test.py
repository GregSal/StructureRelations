#!/usr/bin/env python
'''Quick test to verify logical relationship functionality.'''

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Quick check
print("Test 1: Simple imports", flush=True)
from dicom import DicomStructureFile
from structure_set import StructureSet
print("Imports successful", flush=True)

# Load
print("\nTest 2: Loading DICOM", flush=True)
test_file = Path(__file__).parent / 'Tests' / 'RS.GJS_Struct_Tests.Relations.dcm'
print(f"File: {test_file}", flush=True)

try:
    dicom_file = DicomStructureFile(
        top_dir=test_file.parent,
        file_name=test_file.name
    )
    print("DICOM loaded", flush=True)

    # Create structure set
    print("\nTest 3: Creating StructureSet", flush=True)
    structure_set = StructureSet(dicom_structure_file=dicom_file)
    print("StructureSet created", flush=True)

    # Get a relationship
    print("\nTest 4: Getting a relationship", flush=True)
    rel = structure_set.get_relationship(1, 2)
    if rel:
        print(f"Relationship (1,2): {rel.relationship_type.label if rel.relationship_type else 'None'}", flush=True)
        print(f"  is_logical: {rel.is_logical}", flush=True)
        print(f"  intermediate_structures: {rel.intermediate_structures}", flush=True)
    else:
        print("No relationship found", flush=True)

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
