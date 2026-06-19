'''Tests for session manager runtime configuration and expiry behavior.'''

from datetime import datetime, timedelta
import json

from webapp.session_manager import SessionData, SessionManager


def _write_settings(settings_path, payload):
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(payload), encoding='utf-8')


def test_session_manager_uses_defaults_when_settings_missing(tmp_path):
    sessions_dir = tmp_path / 'sessions'
    settings_path = tmp_path / 'missing_settings.json'

    manager = SessionManager(
        sessions_dir=sessions_dir,
        settings_file=settings_path,
    )

    assert manager.max_disk_size == 2048 * 1024 * 1024
    assert manager.warn_disk_size == 500 * 1024 * 1024
    assert manager.session_expiry == timedelta(hours=8)


def test_session_manager_loads_valid_runtime_settings(tmp_path):
    sessions_dir = tmp_path / 'sessions'
    settings_path = tmp_path / 'webapp_settings.json'
    _write_settings(
        settings_path,
        {
            'session_storage': {
                'max_disk_size_mb': 1024,
                'warn_disk_size_mb': 256,
                'session_expiry_hours': 10,
            }
        },
    )

    manager = SessionManager(
        sessions_dir=sessions_dir,
        settings_file=settings_path,
    )

    assert manager.max_disk_size == 1024 * 1024 * 1024
    assert manager.warn_disk_size == 256 * 1024 * 1024
    assert manager.session_expiry == timedelta(hours=10)


def test_session_manager_invalid_settings_fall_back_to_defaults(tmp_path):
    sessions_dir = tmp_path / 'sessions'
    settings_path = tmp_path / 'webapp_settings.json'
    _write_settings(
        settings_path,
        {
            'session_storage': {
                'max_disk_size_mb': 100,
                'warn_disk_size_mb': 200,
                'session_expiry_hours': -1,
            }
        },
    )

    manager = SessionManager(
        sessions_dir=sessions_dir,
        settings_file=settings_path,
    )

    assert manager.max_disk_size == 2048 * 1024 * 1024
    assert manager.warn_disk_size == 500 * 1024 * 1024
    assert manager.session_expiry == timedelta(hours=8)


def test_session_expiry_uses_configured_eight_hours(tmp_path):
    sessions_dir = tmp_path / 'sessions'
    settings_path = tmp_path / 'webapp_settings.json'
    _write_settings(
        settings_path,
        {
            'session_storage': {
                'max_disk_size_mb': 2048,
                'warn_disk_size_mb': 500,
                'session_expiry_hours': 8,
            }
        },
    )

    manager = SessionManager(
        sessions_dir=sessions_dir,
        settings_file=settings_path,
    )

    session_id = 'stale-session'
    stale_data = SessionData(dicom_file_path='dummy.dcm')
    stale_data.last_accessed = datetime.now() - timedelta(hours=9)
    manager.save_session(session_id, stale_data)

    loaded = manager.load_session(session_id)
    assert loaded is None
    assert session_id not in dict(manager.list_sessions())
