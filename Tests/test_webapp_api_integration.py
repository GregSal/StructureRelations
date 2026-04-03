'''Integration tests for web API job contracts and relationship result endpoints.'''

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
from fastapi.testclient import TestClient

from relations import RELATION_SCHEMA_VERSION, RELATIONSHIP_TYPES
from relationships import StructureRelationship
from webapp.main import app
import webapp.main as web_main
from webapp.session_manager import SessionData, SessionManager


def _make_fake_structure_set(tolerance: float = 0.1):
    graph = nx.DiGraph()
    rel = StructureRelationship(
        de27im=None,
        is_identical=False,
        _override_type=RELATIONSHIP_TYPES['CONTAINS'],
    )
    graph.add_edge(1, 2, relationship=rel)
    return SimpleNamespace(relationship_graph=graph, tolerance=tolerance)


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
