"""
Debug script to isolate the NotImplementedError in Bowel_Full (ROI=11)
from RS.PROS_Test.dcm
"""
from pathlib import Path
import sys
import pandas as pd
import traceback

# Setup paths
repo_root = Path(__file__).parent
src_path = repo_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dicom import DicomStructureFile
from contours import ContourPoints, SliceSequence, build_contour_table
from contour_graph import build_contour_graph

# Load DICOM file
dicom_path = repo_root / 'Tests' / 'RS.PROS_Test.dcm'
print(f'Loading DICOM file: {dicom_path}')
dicom_file = DicomStructureFile(top_dir=repo_root / 'Tests',
                                file_path=dicom_path)

# Extract only ROI 11 (Bowel_Full) contours
all_contours = dicom_file.contour_points
roi_11_contours = [cp for cp in all_contours if cp['ROI'] == 11]

print(f'\nTotal contours in file: {len(all_contours)}')
print(f'Contours for ROI 11 (Bowel_Full): {len(roi_11_contours)}')

if len(roi_11_contours) == 0:
    print('\n⚠️ No contours found for ROI 11!')
    # Show available ROIs
    available_rois = sorted(set(cp['ROI'] for cp in all_contours))
    print(f'Available ROI numbers: {available_rois}')
    
    # Show structure names
    print('\nStructure names in file:')
    for seq_item in dicom_file.dicom_data.StructureSetROISequence:
        print(f'  ROI {seq_item.ROINumber}: {seq_item.ROIName}')
    sys.exit(1)

# Analyze contour distribution
slice_indices = sorted(set(cp['Slice'] for cp in roi_11_contours))
print(f'\nSlice indices with contours: {len(slice_indices)}')
print(f'Slice range: {min(slice_indices):.3f} to {max(slice_indices):.3f}')

# Show contour counts per slice
print('\nContours per slice:')
for slice_idx in slice_indices:
    count = sum(1 for cp in roi_11_contours if cp['Slice'] == slice_idx)
    print(f'  Slice {slice_idx:7.3f} cm: {count} contour(s)')

# Create contour table using the proper function
print('\n' + '='*80)
print('Building contour table...')
print('='*80)

contour_table, slice_sequence = build_contour_table(roi_11_contours)
roi = 11

print(f'\nContour table shape: {contour_table.shape}')
print(f'Contour table columns: {list(contour_table.columns)}')
print(f'\nSlice sequence has {len(slice_sequence.slices)} slices')

# Try to build contour graph and catch the specific error
print('\n' + '='*80)
print('Attempting to build contour graph...')
print('='*80)

try:
    contour_graph, slice_sequence = build_contour_graph(
        contour_table,
        slice_sequence,
        roi
    )
    print('✅ Success! Contour graph built without errors.')
    
except NotImplementedError as e:
    print(f'\n❌ NotImplementedError caught!')
    print(f'Error message: {e}')
    print('\nFull traceback:')
    traceback.print_exc()
    
    # Try to identify the problematic contour
    print('\n' + '='*80)
    print('ATTEMPTING TO IDENTIFY PROBLEMATIC CONTOUR')
    print('='*80)
    
    # Get the traceback details
    import sys
    exc_type, exc_value, exc_tb = sys.exc_info()
    
    # Walk through traceback to find local variables
    tb = exc_tb
    while tb.tb_next:
        tb = tb.tb_next
    
    frame = tb.tb_frame
    local_vars = frame.f_locals
    
    print('\nLocal variables at error point:')
    for var_name in ['poly', 'new_contour', 'contour_parameters', 
                      'interpolated_slice', 'starting_contour', 'contour',
                      'polygons_to_process']:
        if var_name in local_vars:
            var_value = local_vars[var_name]
            print(f'\n{var_name}:')
            print(f'  Type: {type(var_value)}')
            if hasattr(var_value, '__dict__'):
                attrs = [attr for attr in dir(var_value) if not attr.startswith('_')]
                print(f'  Main attributes: {attrs[:10]}...')  # Show first 10
            if hasattr(var_value, 'geom_type'):
                print(f'  Geometry type: {var_value.geom_type}')
                print(f'  Boundary type: {type(var_value.boundary)}')
                print(f'  Boundary geom_type: {var_value.boundary.geom_type}')
                if hasattr(var_value, 'interiors'):
                    print(f'  Number of holes: {len(list(var_value.interiors))}')
                if hasattr(var_value.boundary, 'geoms'):
                    print(f'  Boundary has {len(var_value.boundary.geoms)} parts')
            if hasattr(var_value, 'is_valid'):
                print(f'  Is valid: {var_value.is_valid}')
            if hasattr(var_value, 'is_empty'):
                print(f'  Is empty: {var_value.is_empty}')
            
            # For Contour objects
            if hasattr(var_value, 'slice_index'):
                print(f'  Slice index: {var_value.slice_index}')
            if hasattr(var_value, 'roi'):
                print(f'  ROI: {var_value.roi}')
            if hasattr(var_value, 'contour_index'):
                print(f'  Contour index: {var_value.contour_index}')
            # For dictionary-like objects (ContourPoints)
            if isinstance(var_value, dict):
                if 'Slice' in var_value:
                    print(f'  Slice: {var_value["Slice"]}')
                if 'ROI' in var_value:
                    print(f'  ROI: {var_value["ROI"]}')
                if 'Points' in var_value:
                    print(f'  Number of points: {len(var_value["Points"])}')
                
except Exception as e:
    print(f'\n❌ Unexpected error: {type(e).__name__}')
    print(f'Error message: {e}')
    print('\nFull traceback:')
    traceback.print_exc()

print('\n' + '='*80)
print('Debug script complete')
print('='*80)
