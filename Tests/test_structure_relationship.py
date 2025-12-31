'''Tests for StructureRelationship class and relationship matrix changes.

This test file verifies that the StructureRelationship abstraction works
correctly and that the relationship_summary and get_relationship_matrix
methods continue to return correct string outputs.
'''
from structure_set import StructureSet
from debug_tools import make_sphere, make_box


def test_relationship_summary_returns_labels():
    '''Test that relationship_summary returns correct label strings.'''
    slice_spacing = 0.5
    # Create two nested structures
    outer = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
    inner = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)
    slice_data = outer + inner

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get relationship summary
    summary = structure_set.relationship_summary()

    # Verify summary contains string labels, not objects
    assert not summary.empty
    # Check diagonal contains 'Equals'
    structure_names = list(structure_set.structures.values())
    for struct in structure_names:
        assert summary.loc[struct.name, struct.name] == 'Equals'

    # Check that CONTAINS or related string appears
    assert 'Contains' in summary.values or 'Equals' in summary.values


def test_get_relationship_matrix_with_symbols():
    '''Test that get_relationship_matrix returns correct symbols.'''
    slice_spacing = 0.5
    # Create two nested structures
    outer = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
    inner = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)
    slice_data = outer + inner

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get relationship matrix with symbols
    matrix = structure_set.get_relationship_matrix(use_symbols=True)

    # Verify matrix contains string symbols, not objects
    assert not matrix.empty
    # Check diagonal contains '=' symbol for EQUALS
    structure_names = list(structure_set.structures.values())
    for struct in structure_names:
        assert matrix.loc[struct.name, struct.name] == '='

    # Check that symbols (not objects) are in the matrix
    all_values = matrix.values.flatten()
    for val in all_values:
        assert isinstance(val, str), f'Expected string, got {type(val)}'


def test_get_relationship_matrix_with_labels():
    '''Test that get_relationship_matrix returns correct labels without symbols.'''
    slice_spacing = 0.5
    # Create two disjoint structures
    box1 = make_box(roi_num=1, width=4, offset_x=-5, spacing=slice_spacing)
    box2 = make_box(roi_num=2, width=4, offset_x=5, spacing=slice_spacing)
    slice_data = box1 + box2

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get relationship matrix with labels (no symbols)
    matrix = structure_set.get_relationship_matrix(use_symbols=False)

    # Verify matrix contains string labels, not objects
    assert not matrix.empty
    # Check diagonal contains 'Equals' label
    structure_names = list(structure_set.structures.values())
    for struct in structure_names:
        assert matrix.loc[struct.name, struct.name] == 'Equals'

    # Check that labels (not objects) are in the matrix
    all_values = matrix.values.flatten()
    for val in all_values:
        assert isinstance(val, str), f'Expected string, got {type(val)}'


def test_relationship_matrix_filtered():
    '''Test that filtered relationship matrix works correctly.'''
    slice_spacing = 0.5
    # Create three structures
    outer = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
    middle = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)
    inner = make_sphere(roi_num=3, radius=1, spacing=slice_spacing)
    slice_data = outer + middle + inner

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get filtered matrix - only ROI 1 and 2
    matrix = structure_set.get_relationship_matrix(
        row_rois=[1, 2],
        col_rois=[1, 2],
        use_symbols=True
    )

    # Verify matrix has correct dimensions
    assert matrix.shape == (2, 2)

    # Verify it contains strings
    all_values = matrix.values.flatten()
    for val in all_values:
        assert isinstance(val, str), f'Expected string, got {type(val)}'


def test_to_dict_serialization():
    '''Test that to_dict returns JSON-serializable data.'''
    slice_spacing = 0.5
    # Create two structures
    outer = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
    inner = make_sphere(roi_num=2, radius=3, spacing=slice_spacing)
    slice_data = outer + inner

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get dictionary representation
    data_dict = structure_set.to_dict()

    # Verify data dictionary has expected structure
    assert 'rows' in data_dict
    assert 'columns' in data_dict
    assert 'data' in data_dict
    assert 'row_names' in data_dict
    assert 'col_names' in data_dict

    # Verify data contains strings (from get_relationship_matrix)
    assert isinstance(data_dict['data'], list)
    for row in data_dict['data']:
        for val in row:
            assert isinstance(val, str), f'Expected string, got {type(val)}'


def test_empty_structure_set():
    '''Test that empty structure set handles relationship matrix correctly.'''
    # Create empty structure set
    structure_set = StructureSet()

    # Get relationship summary
    summary = structure_set.relationship_summary()
    assert summary.empty

    # Get relationship matrix
    matrix = structure_set.get_relationship_matrix()
    assert matrix.empty

    # Get dictionary
    data_dict = structure_set.to_dict()
    assert data_dict['rows'] == []
    assert data_dict['columns'] == []
    assert data_dict['data'] == []


def test_single_structure():
    '''Test that single structure creates correct self-relationship.'''
    slice_spacing = 0.5
    # Create one structure
    sphere = make_sphere(roi_num=1, radius=5, spacing=slice_spacing)
    slice_data = sphere

    # Build structure set
    structure_set = StructureSet(slice_data)

    # Get relationship summary
    summary = structure_set.relationship_summary()

    # Should have 1x1 matrix with 'Equals'
    assert summary.shape == (1, 1)
    struct_name = list(structure_set.structures.values())[0].name
    assert summary.loc[struct_name, struct_name] == 'Equals'

    # Get relationship matrix with symbols
    matrix = structure_set.get_relationship_matrix(use_symbols=True)
    assert matrix.shape == (1, 1)
    assert matrix.loc[struct_name, struct_name] == '='
