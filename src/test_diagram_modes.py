#!/usr/bin/env python
'''Test diagram endpoint with different logical_relations_mode values.'''

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dicom import DicomStructureFile
from structure_set import StructureSet

# Load a test DICOM file
test_file = (
    Path(__file__).resolve().parent.parent
    / 'tests'
    / 'RS.GJS_Struct_Tests.BRBL BH.dcm'
)
print(f'Loading DICOM file: {test_file}')

dicom_file = DicomStructureFile(
    top_dir=test_file.parent,
    file_name=test_file.name
)
print(f'Loaded: {dicom_file}')

# Create structure set
print('\nCreating StructureSet...')
structure_set = StructureSet(dicom_structure_file=dicom_file)
print('StructureSet created')

# Check relationships
print('\n=== Checking relationships ===')
all_rois = [int(roi) for roi in structure_set.summary()['ROI'].tolist()]
print(f'All ROIs: {all_rois}')

# Count logical relationships
logical_count = 0
nonlogical_count = 0

for roi1 in all_rois:
    for roi2 in all_rois:
        if roi1 >= roi2:
            continue
        rel = structure_set.get_relationship(roi1, roi2)
        if rel and rel.relationship_type:
            if rel.is_logical:
                logical_count += 1
                print(f'  ({roi1},{roi2}): {rel.relationship_type.label} - LOGICAL (intermediates: {rel.intermediate_structures})')
            else:
                nonlogical_count += 1

print(f'\nTotal relationships: {logical_count + nonlogical_count}')
print(f'  Logical: {logical_count}')
print(f'  Non-logical: {nonlogical_count}')

# Test filtering with different modes
print('\n=== Testing filter logic ===')

# Simulate what should_include_logical does
test_rels = []
for roi1 in all_rois:
    for roi2 in all_rois:
        if roi1 >= roi2:
            continue
        rel = structure_set.get_relationship(roi1, roi2)
        if rel and rel.relationship_type:
            test_rels.append((roi1, roi2, rel))

visible_rois = set(all_rois)  # Assume all visible for now

for mode in ['show', 'hide', 'limited', 'faded']:
    included = 0
    for roi1, roi2, rel in test_rels:
        # Simulate should_include_logical logic
        if rel is None:
            continue

        if mode == 'show':
            should_include = True
        elif not rel.is_logical:
            should_include = True
        elif mode == 'hide':
            should_include = False
        elif mode == 'limited':
            intermediates = rel.intermediate_structures
            all_visible = all(roi in visible_rois for roi in intermediates)
            should_include = not all_visible
        else:  # faded
            should_include = True

        if should_include:
            included += 1

    print(f'Mode "{mode}": {included} edges shown')

print('\nDone!')
