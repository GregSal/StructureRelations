'''Integration tests for web API job contracts and relationship result endpoints.'''

import asyncio
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd
from fastapi.testclient import TestClient

from relations import RELATION_SCHEMA_VERSION, RELATIONSHIP_TYPES
from relationships import StructureRelationship
from structure_set import StructureSet
from webapp.main import app
import webapp.main as web_main
from webapp.session_manager import SessionData, SessionManager


class FakeDiagramStructureSet:
    '''Pickle-safe minimal structure set for diagram endpoint tests.'''

    def __init__(self, relationship, tolerance: float = 0.1):
        self.relationship_graph = nx.DiGraph()
        self.relationship_graph.add_edge(1, 2, relationship=relationship)
        self.tolerance = tolerance
        self.dicom_structure_file = None
        self._summary_df = pd.DataFrame(
            [
                {
                    'ROI': 1,
                    'Name': 'Alpha',
                    'DICOM_Type': 'CTV',
                    'Physical_Volume': 1.0,
                    'Num_Regions': 1,
                },
                {
                    'ROI': 2,
                    'Name': 'Beta',
                    'DICOM_Type': 'GTV',
                    'Physical_Volume': 2.0,
                    'Num_Regions': 1,
                },
            ]
        )
        self._relationship = relationship

    def summary(self):
        return self._summary_df

    def get_relationship(self, from_roi, to_roi):
        if from_roi == 1 and to_roi == 2:
            return self._relationship
        return None


def _make_fake_structure_set(tolerance: float = 0.1):
    graph = nx.DiGraph()
    rel = StructureRelationship(
        de27im=None,
        is_identical=False,
        _override_type=RELATIONSHIP_TYPES['CONTAINS'],
    )
    graph.add_edge(1, 2, relationship=rel)
    return SimpleNamespace(relationship_graph=graph, tolerance=tolerance)


def _make_fake_diagram_structure_set(tolerance: float = 0.1):
    rel = StructureRelationship(
        de27im=None,
        is_identical=False,
        _override_type=RELATIONSHIP_TYPES['CONTAINS'],
    )
    return FakeDiagramStructureSet(relationship=rel, tolerance=tolerance)


def _make_fake_plot_structure_set():
    region_table = pd.DataFrame(
        {
            'SliceIndex': [1.0, 2.0],
            'Empty': [False, False],
            'Interpolated': [False, True],
        }
    )
    fake_structure = SimpleNamespace(region_table=region_table)
    return SimpleNamespace(
        structures={
            1: fake_structure,
            2: fake_structure,
            3: fake_structure,
            4: fake_structure,
        }
    )


def _prepare_client(monkeypatch, tmp_path):
    manager = SessionManager(sessions_dir=tmp_path / 'sessions')
    monkeypatch.setattr(web_main, 'session_manager', manager)
    monkeypatch.setattr(web_main, 'processing_tasks', {})
    monkeypatch.setattr(web_main, 'cancel_events', {})
    return TestClient(app), manager


def _create_session(manager: SessionManager, session_id: str):
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=None),
    )


def _wait_for_status(client: TestClient, session_id: str, expected: str, timeout_s: float = 3.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f'/api/jobs/{session_id}/status')
        assert response.status_code == 200
        payload = response.json()
        if payload['status'] == expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f'Expected status {expected} for {session_id}')


def test_job_submission_status_and_result_contract(monkeypatch, tmp_path):
    '''Verify submit, status, and result endpoints return typed job contracts.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'job-success'
    _create_session(manager, session_id)

    async def fake_process_structure_set(session_id, dicom_file_path, selected_rois, cancel_event=None):
        fake_set = _make_fake_structure_set(tolerance=0.2)
        manager.update_session_structure_set(session_id, fake_set)
        manager.update_session_job_state(
            session_id,
            job_status='completed',
            job_progress=100.0,
            job_message='complete',
            job_computed_at=web_main.datetime.now(),
            job_provenance={'source': 'test'},
        )

    monkeypatch.setattr(web_main, 'process_structure_set', fake_process_structure_set)

    submit_resp = client.post('/api/process', json={'session_id': session_id})
    assert submit_resp.status_code == 200
    submit_payload = submit_resp.json()
    assert submit_payload['job_id'] == session_id
    assert submit_payload['schema_version'] == RELATION_SCHEMA_VERSION

    status_payload = _wait_for_status(client, session_id, 'completed')
    assert status_payload['schema_version'] == RELATION_SCHEMA_VERSION

    result_resp = client.get(f'/api/jobs/{session_id}/result')
    assert result_resp.status_code == 200
    result_payload = result_resp.json()
    assert result_payload['schema_version'] == RELATION_SCHEMA_VERSION
    assert result_payload['pairs']
    first_pair = result_payload['pairs'][0]
    assert first_pair['schema_version'] == RELATION_SCHEMA_VERSION
    assert first_pair['relation_type'] == 'CONTAINS'


def test_job_cancellation_endpoint(monkeypatch, tmp_path):
    '''Verify cancellation endpoint transitions an active job to cancelled state.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'job-cancel'
    _create_session(manager, session_id)

    async def fake_process_structure_set(session_id, dicom_file_path, selected_rois, cancel_event=None):
        for _ in range(40):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError('cancelled in test')
            await asyncio.sleep(0.01)

    monkeypatch.setattr(web_main, 'process_structure_set', fake_process_structure_set)

    submit_resp = client.post('/api/process', json={'session_id': session_id})
    assert submit_resp.status_code == 200

    cancel_resp = client.post(f'/api/jobs/{session_id}/cancel')
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()['status'] == 'cancelled'

    status_payload = _wait_for_status(client, session_id, 'cancelled')
    assert status_payload['schema_version'] == RELATION_SCHEMA_VERSION


def test_error_recovery_allows_resubmission(monkeypatch, tmp_path):
    '''Verify a failed job can be resubmitted and complete successfully.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'job-recover'
    _create_session(manager, session_id)

    async def failing_process(session_id, dicom_file_path, selected_rois, cancel_event=None):
        manager.update_session_job_state(
            session_id,
            job_status='failed',
            job_progress=0.0,
            job_message='failed in test',
            job_error='test failure',
        )

    monkeypatch.setattr(web_main, 'process_structure_set', failing_process)

    first_submit = client.post('/api/process', json={'session_id': session_id})
    assert first_submit.status_code == 200
    failed_payload = _wait_for_status(client, session_id, 'failed')
    assert failed_payload['error'] == 'test failure'

    async def successful_process(session_id, dicom_file_path, selected_rois, cancel_event=None):
        fake_set = _make_fake_structure_set(tolerance=0.3)
        manager.update_session_structure_set(session_id, fake_set)
        manager.update_session_job_state(
            session_id,
            job_status='completed',
            job_progress=100.0,
            job_message='recovered',
            job_computed_at=web_main.datetime.now(),
            job_error=None,
            job_provenance={'source': 'test-recovery'},
        )

    monkeypatch.setattr(web_main, 'process_structure_set', successful_process)

    second_submit = client.post('/api/process', json={'session_id': session_id})
    assert second_submit.status_code == 200
    completed_payload = _wait_for_status(client, session_id, 'completed')
    assert completed_payload['error'] is None


def test_symbol_config_includes_diagram_style_sections(monkeypatch, tmp_path):
    '''Verify symbol config endpoint returns node and edge style metadata.'''
    client, _ = _prepare_client(monkeypatch, tmp_path)

    response = client.get('/api/config/symbols')
    assert response.status_code == 200

    payload = response.json()
    assert 'relationships' in payload
    assert 'node_shapes' in payload
    assert 'relationship_styles' in payload
    assert 'relationship_display_defaults' in payload
    assert 'diagram_options' in payload

    # Ensure key styles expected by diagram legend are present.
    assert 'shape_map' in payload['node_shapes']
    assert payload['node_shapes'].get('default_shape')
    assert 'CONTAINS' in payload['relationship_styles']
    assert payload['relationship_display_defaults'].get('show_edge_labels') is True
    assert payload['diagram_options'].get('interaction', {}).get('tooltip_delay') == 100
    diagram_layout = payload['diagram_options'].get('diagram_layout', {})
    assert diagram_layout.get('layout', {}).get('local_global_default') == 30
    assert 'local' in diagram_layout.get('physics', {})
    assert 'global' in diagram_layout.get('physics', {})


def test_diagram_endpoint_uses_human_relationship_label(monkeypatch, tmp_path):
    '''Verify diagram edge payload uses the human-readable relationship label.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'diagram-human-label'
    fake_set = _make_fake_diagram_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    response = client.post(
        '/api/diagram',
        json={
            'session_id': session_id,
            'logical_relations_mode': 'show',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['edges']
    edge = payload['edges'][0]
    assert edge['relation_type'] == 'CONTAINS'
    assert edge['label'] == 'Contains'


def test_preview_hides_uploaded_session_prefix_in_file_name(monkeypatch, tmp_path):
    '''Verify preview metadata strips the temporary upload prefix from file names.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'preview-clean-name'

    source_path = Path(__file__).parent / 'RS.GJS_Struct_Tests.Relations.dcm'
    prefixed_path = (
        tmp_path
        / '123e4567-e89b-12d3-a456-426614174000_RS.GJS_Struct_Tests.Relations.dcm'
    )
    shutil.copy(source_path, prefixed_path)

    manager.save_session(
        session_id,
        SessionData(dicom_file_path=str(prefixed_path), structure_set=None),
    )

    response = client.post('/api/preview', json={'session_id': session_id})

    assert response.status_code == 200
    assert response.json()['patient_info']['file_name'] == 'RS.GJS_Struct_Tests.Relations.dcm'


def test_plot_contours_excludes_interpolated_by_default(monkeypatch, tmp_path):
    '''Verify contour plotting blocks interpolated-only slices unless opted in.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-interp-default-off'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    monkeypatch.setattr(web_main, 'plot_roi_slice', lambda **kwargs: None)

    blocked = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1],
            'slice_index': 2.0,
        },
    )
    assert blocked.status_code == 400
    assert 'interpolated-slice setting' in blocked.json()['detail']

    allowed = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1],
            'slice_index': 2.0,
            'include_interpolated_slices': True,
        },
    )
    assert allowed.status_code == 200


def test_plot_contours_forwards_render_options(monkeypatch, tmp_path):
    '''Verify contour plot endpoint forwards mode/legend/tolerance options.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-render-options'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    captured = {}

    def fake_plot_roi_slice(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    response = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1, 2],
            'slice_index': 1.0,
            'plot_mode': 'relationship',
            'show_legend': False,
            'add_axis': True,
            'tolerance': 0.25,
        },
    )

    assert response.status_code == 200
    assert captured['plot_mode'] == 'relationship'
    assert captured['show_legend'] is False
    assert captured['add_axis'] is True
    assert captured['tolerance'] == 0.25


def test_plot_contours_supports_multi_roi_contour_mode(monkeypatch, tmp_path):
    '''Verify contour mode can forward more than two ordered structures.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-multi-roi-contour'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    captured = {}

    def fake_plot_roi_slice(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    response = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1, 2, 3],
            'slice_index': 1.0,
            'plot_mode': 'contour',
        },
    )

    assert response.status_code == 200
    assert captured['roi_list'] == [1, 2, 3]
    assert captured['plot_mode'] == 'contour'


def test_plot_contours_supports_third_structure_overlay(monkeypatch, tmp_path):
    '''Verify relationship mode can forward a selected third structure overlay.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-third-overlay'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    captured = {}

    def fake_plot_roi_slice(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    response = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1, 2, 3],
            'slice_index': 1.0,
            'plot_mode': 'relationship',
            'relationship_overlay': 'third_structure',
        },
    )

    assert response.status_code == 200
    assert captured['roi_list'] == [1, 2, 3]
    assert captured['relationship_overlay'] == 'third_structure'


def test_plot_contours_supports_single_structure_relationship_mode(monkeypatch, tmp_path):
    '''Verify relationship mode accepts the single-structure filled view.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-single-relationship'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    captured = {}

    def fake_plot_roi_slice(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    response = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1],
            'slice_index': 1.0,
            'plot_mode': 'relationship',
            'relationship_overlay': 'single_structure',
        },
    )

    assert response.status_code == 200
    assert captured['roi_list'] == [1]
    assert captured['relationship_overlay'] == 'single_structure'


def test_plot_contours_supports_all_outlines_overlay(monkeypatch, tmp_path):
    '''Verify relationship mode can forward all selected ROIs for outline overlays.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-all-outlines'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    captured = {}

    def fake_plot_roi_slice(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    response = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1, 2, 3, 4],
            'slice_index': 1.0,
            'plot_mode': 'relationship',
            'relationship_overlay': 'all_outlines',
        },
    )

    assert response.status_code == 200
    assert captured['roi_list'] == [1, 2, 3, 4]
    assert captured['relationship_overlay'] == 'all_outlines'


def test_plot_contours_reuses_cached_render_for_identical_requests(monkeypatch, tmp_path):
    '''Verify identical contour plot requests do not rerender more than once.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-cache-hit'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    call_count = {'count': 0}

    def fake_plot_roi_slice(**kwargs):
        call_count['count'] += 1

    monkeypatch.setattr(web_main, 'plot_roi_slice', fake_plot_roi_slice)

    payload = {
        'session_id': session_id,
        'roi_list': [1, 2],
        'slice_index': 1.0,
        'plot_mode': 'relationship',
        'relationship_overlay': 'none',
        'show_legend': True,
        'tolerance': 0.0,
    }

    response_a = client.post('/api/plot-contours', json=payload)
    response_b = client.post('/api/plot-contours', json=payload)
    response_c = client.post(
        '/api/plot-contours',
        json={**payload, 'slice_index': 2.0, 'include_interpolated_slices': True},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_c.status_code == 200
    assert call_count['count'] == 2


def test_plot_contours_rejects_invalid_render_options(monkeypatch, tmp_path):
    '''Verify plot option validation for mode and tolerance inputs.'''
    client, manager = _prepare_client(monkeypatch, tmp_path)
    session_id = 'plot-invalid-options'
    fake_set = _make_fake_plot_structure_set()
    manager.save_session(
        session_id,
        SessionData(dicom_file_path='dummy.dcm', structure_set=fake_set),
    )

    monkeypatch.setattr(web_main, 'plot_roi_slice', lambda **kwargs: None)

    bad_mode = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1],
            'slice_index': 1.0,
            'plot_mode': 'bad-mode',
        },
    )
    assert bad_mode.status_code == 400
    assert 'plot_mode' in bad_mode.json()['detail']

    bad_tolerance = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1],
            'slice_index': 1.0,
            'tolerance': -0.1,
        },
    )
    assert bad_tolerance.status_code == 400
    assert 'tolerance' in bad_tolerance.json()['detail']

    bad_overlay = client.post(
        '/api/plot-contours',
        json={
            'session_id': session_id,
            'roi_list': [1, 2],
            'slice_index': 1.0,
            'plot_mode': 'relationship',
            'relationship_overlay': 'third_structure',
        },
    )
    assert bad_overlay.status_code == 400
    assert 'third_structure' in bad_overlay.json()['detail']


def test_structure_set_serializes_slice_relationships():
    '''Verify slice relationship records are grouped by slice for the frontend.'''
    structure_set = StructureSet(auto_calculate_relationships=False)
    structure_set.structures = {
        1: SimpleNamespace(name='Alpha'),
        2: SimpleNamespace(name='Beta'),
    }
    structure_set.slice_relationship_records = {
        '1|2': [
            {
                'slice_index': 1.0,
                'relation_type': 'OVERLAPS',
                'relation_symbol': 'o',
                'is_interpolated': False,
                'has_boundary': True,
            }
        ]
    }

    serialized = structure_set._serialize_slice_relationships([1, 2])

    assert '1.0000' in serialized
    assert serialized['1.0000'][0]['rois'] == [1, 2]
    assert serialized['1.0000'][0]['label'] == 'Alpha / Beta: OVERLAPS'
    assert serialized['1.0000'][0]['has_boundary'] is True
