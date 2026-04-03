'''Session management for the StructureRelations web application.

Handles persistent storage, expiration, and disk usage management of user sessions.
'''
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    '''Data stored for each user session.

    Attributes:
        dicom_file_path (str): Path to the uploaded DICOM file.
        structure_set: The built StructureSet object (after processing).
        created_at (datetime): When the session was created.
        last_accessed (datetime): When the session was last accessed.
        app_version (str): Application version for compatibility checking.
        job_status (str): Processing status
            (idle/running/completed/failed/cancelled).
        job_progress (float): Processing percentage from 0.0 to 100.0.
        job_message (str): Human-readable processing status.
        job_error (Optional[str]): Last processing error message, if any.
        job_computed_at (Optional[datetime]): Timestamp of latest completed job.
        job_provenance (dict): Additional job metadata for API responses.
    '''
    dicom_file_path: str
    structure_set: Optional[object] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    app_version: str = '1.0.0'
    job_status: str = 'idle'
    job_progress: float = 0.0
    job_message: str = ''
    job_error: Optional[str] = None
    job_computed_at: Optional[datetime] = None
    job_provenance: dict = field(default_factory=dict)


class SessionManager:
    '''Manages persistent sessions with disk usage limits and automatic expiration.

    Sessions are stored as pickle files in the sessions_dir directory.
    Implements hybrid deletion strategy: expired sessions first, then oldest by LRU.

    Attributes:
        sessions_dir (Path): Directory for storing session pickle files.
        MAX_DISK_SIZE (int): Maximum total size in bytes (250 MB).
        WARN_DISK_SIZE (int): Warning threshold in bytes (100 MB).
        SESSION_EXPIRY (timedelta): Session expiration time (2 hours).
    '''

    MAX_DISK_SIZE = 250 * 1024 * 1024  # 250 MB
    WARN_DISK_SIZE = 100 * 1024 * 1024  # 100 MB
    SESSION_EXPIRY = timedelta(hours=2)

    def __init__(self, sessions_dir: Path = None):
        '''Initialize the SessionManager.

        Args:
            sessions_dir (Path, optional): Directory for session storage.
                Defaults to './webapp_sessions'.
        '''
        if sessions_dir is None:
            sessions_dir = Path('webapp_sessions')

        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)

        # In-memory cache of session metadata
        self._sessions: Dict[str, SessionData] = {}

        # Load existing sessions on startup
        self._load_existing_sessions()

    def _load_existing_sessions(self):
        '''Load existing session metadata from disk on startup.'''
        session_files = list(self.sessions_dir.glob('*.pkl'))

        for session_file in session_files:
            session_id = session_file.stem
            try:
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                    self._sessions[session_id] = session_data
            except Exception as e:
                logger.error(f'Failed to load session {session_id}: {e}')
                # Delete corrupted session file
                session_file.unlink(missing_ok=True)

        disk_usage = self.get_disk_usage()
        num_expired = sum(1 for s in self._sessions.values()
                         if datetime.now() - s.last_accessed > self.SESSION_EXPIRY)

        logger.info(
            f'Loaded {len(self._sessions)} sessions, '
            f'{disk_usage / (1024*1024):.1f} MB disk usage, '
            f'{num_expired} expired'
        )

    def get_disk_usage(self) -> int:
        '''Calculate total disk usage of all session files.

        Returns:
            int: Total size in bytes.
        '''
        total_size = 0
        for session_file in self.sessions_dir.glob('*.pkl'):
            try:
                total_size += session_file.stat().st_size
            except OSError:
                pass
        return total_size

    def enforce_disk_limit(self):
        '''Enforce disk usage limits using hybrid deletion strategy.

        Strategy:
        1. Delete all expired sessions (>2 hours old)
        2. If still over limit, delete oldest sessions by last_accessed

        Each deletion is logged at INFO level.
        '''
        disk_usage = self.get_disk_usage()

        if disk_usage <= self.MAX_DISK_SIZE:
            return

        logger.info(f'Disk usage {disk_usage/(1024*1024):.1f} MB exceeds limit, enforcing cleanup')

        now = datetime.now()

        # Step 1: Delete expired sessions
        expired_sessions = [
            (session_id, session_data)
            for session_id, session_data in self._sessions.items()
            if now - session_data.last_accessed > self.SESSION_EXPIRY
        ]

        for session_id, session_data in expired_sessions:
            self.delete_session(session_id)
            logger.info(f'Deleted expired session {session_id}')

        # Check if we're now under the limit
        disk_usage = self.get_disk_usage()
        if disk_usage <= self.MAX_DISK_SIZE:
            logger.info(f'Disk usage now {disk_usage/(1024*1024):.1f} MB after deleting expired sessions')
            return

        # Step 2: Delete oldest sessions by LRU until under limit
        sessions_by_age = sorted(
            self._sessions.items(),
            key=lambda item: item[1].last_accessed
        )

        for session_id, session_data in sessions_by_age:
            if disk_usage <= self.MAX_DISK_SIZE:
                break

            file_size = self._get_session_file_size(session_id)
            self.delete_session(session_id)
            disk_usage -= file_size
            logger.info(
                f'Deleted oldest session {session_id} '
                f'(last accessed {session_data.last_accessed}), '
                f'disk usage now {disk_usage/(1024*1024):.1f} MB'
            )

    def _get_session_file_size(self, session_id: str) -> int:
        '''Get the file size of a session pickle file.

        Args:
            session_id (str): The session ID.

        Returns:
            int: File size in bytes, or 0 if file doesn't exist.
        '''
        session_file = self.sessions_dir / f'{session_id}.pkl'
        try:
            return session_file.stat().st_size
        except OSError:
            return 0

    def update_session_structure_set(self, session_id: str, structure_set) -> bool:
        '''Update session with structure_set without modifying last_accessed.

        This is used during processing to avoid race conditions with the
        timestamp update that load_session performs.

        Args:
            session_id (str): Unique session identifier.
            structure_set: The built StructureSet object.

        Returns:
            bool: True if successful, False if session not found.
        '''
        if session_id not in self._sessions:
            logger.error(f'Session {session_id} not found in cache')
            return False

        session_data = self._sessions[session_id]
        session_data.structure_set = structure_set

        # Save to disk immediately
        session_file = self.sessions_dir / f'{session_id}.pkl'
        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
            logger.info(f'Updated structure_set for session {session_id}')
            return True
        except OSError as e:
            logger.error(f'Failed to save structure_set for session {session_id}: {e}')
            return False

    def update_session_job_state(self, session_id: str, **state_updates) -> bool:
        '''Update job-state fields for a session without touching timestamps.

        Args:
            session_id (str): Unique session identifier.
            **state_updates: SessionData fields to update.

        Returns:
            bool: True if saved successfully, False otherwise.
        '''
        if session_id not in self._sessions:
            logger.error(f'Session {session_id} not found in cache')
            return False

        session_data = self._sessions[session_id]
        for key, value in state_updates.items():
            if hasattr(session_data, key):
                setattr(session_data, key, value)

        session_file = self.sessions_dir / f'{session_id}.pkl'
        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
            return True
        except OSError as e:
            logger.error(f'Failed to save job state for session {session_id}: {e}')
            return False

    def save_session(self, session_id: str, session_data: SessionData):
        '''Save a session to disk.

        Enforces disk limits before saving. Raises OSError if filesystem is full.

        Args:
            session_id (str): Unique session identifier.
            session_data (SessionData): Session data to save.

        Raises:
            OSError: If unable to save due to disk space issues.
        '''
        # Enforce disk limits before saving
        self.enforce_disk_limit()

        session_file = self.sessions_dir / f'{session_id}.pkl'

        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)

            # Update in-memory cache
            self._sessions[session_id] = session_data

        except OSError as e:
            logger.error(f'Failed to save session {session_id}: {e}')
            raise OSError(f'Storage full, cannot save session. Contact administrator.') from e

    def load_session(self, session_id: str) -> Optional[SessionData]:
        '''Load a session from disk.

        Updates last_accessed timestamp and re-saves the session.

        Args:
            session_id (str): The session ID to load.

        Returns:
            SessionData: The session data, or None if not found or expired.
        '''
        # Check in-memory cache first
        if session_id not in self._sessions:
            return None

        session_data = self._sessions[session_id]

        # Check if expired
        if datetime.now() - session_data.last_accessed > self.SESSION_EXPIRY:
            logger.info(f'Session {session_id} has expired')
            self.delete_session(session_id)
            return None

        # Update last_accessed
        session_data.last_accessed = datetime.now()

        # Re-save with updated timestamp
        try:
            self.save_session(session_id, session_data)
        except OSError:
            # If save fails, still return the session data
            logger.warning(f'Failed to update last_accessed for session {session_id}')

        return session_data

    def delete_session(self, session_id: str):
        '''Delete a session from disk and memory.

        Args:
            session_id (str): The session ID to delete.
        '''
        session_file = self.sessions_dir / f'{session_id}.pkl'
        session_file.unlink(missing_ok=True)

        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[tuple]:
        '''List all sessions sorted by last_accessed.

        Returns:
            List[tuple]: List of (session_id, session_data) tuples.
        '''
        return sorted(
            self._sessions.items(),
            key=lambda item: item[1].last_accessed,
            reverse=True
        )

    def get_disk_usage_info(self) -> dict:
        '''Get disk usage information.

        Returns:
            dict: Dictionary with disk usage statistics:
                {
                    'usage_bytes': int,
                    'usage_mb': float,
                    'limit_mb': float,
                    'warn_mb': float,
                    'is_warning': bool,
                    'is_over_limit': bool
                }
        '''
        usage = self.get_disk_usage()

        return {
            'usage_bytes': usage,
            'usage_mb': usage / (1024 * 1024),
            'limit_mb': self.MAX_DISK_SIZE / (1024 * 1024),
            'warn_mb': self.WARN_DISK_SIZE / (1024 * 1024),
            'is_warning': usage > self.WARN_DISK_SIZE,
            'is_over_limit': usage > self.MAX_DISK_SIZE
        }
