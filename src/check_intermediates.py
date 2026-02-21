#!/usr/bin/env python
'''Verify logical relationships and their intermediate structures.'''

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dicom import DicomStructureFile
from structure_set import StructureSet

# Load a test DICOM file
test_file = Path(__file__).parent / 'Tests' / 'RS.GJS_Struct_Tests.BRBL BH.dcm'
print(f'Loading DICOM file: {test_file.name}')

dicom_file = DicomStructureFile(
    top_dir=test_file.parent,
    file_name=test_file.name
)

# Create structure set
structure_set = StructureSet(dicom_structure_file=dicom_file)

# Check the specific relationships mentioned
rois_of_interest = [11, 12, 13, 14]
print(f'\nChecking relationships for ROIs: {rois_of_interest}')

for roi1 in rois_of_interest:
    for roi2 in rois_of_interest:
        if roi1 <= roi2:
            continue

        rel = structure_set.get_relationship(roi1, roi2)
        if rel and rel.relationship_type:
            print(f'\n({roi1}, {roi2}):')
            print(f'  Relationship: {rel.relationship_type.label}')
            print(f'  Is Logical: {rel.is_logical}')
            print(f'  Intermediate Structures: {rel.intermediate_structures}')
