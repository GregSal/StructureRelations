'''WebSocket connection management for real-time progress updates.'''
import logging
import json
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    '''Manages WebSocket connections for real-time progress updates.

    Supports one WebSocket connection per session, with a maximum of 2 concurrent connections.
    '''

    def __init__(self):
        '''Initialize the ConnectionManager.'''
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        '''Accept a new WebSocket connection.

        Args:
            session_id (str): The session ID for this connection.
            websocket (WebSocket): The WebSocket connection.
        '''
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f'WebSocket connected for session {session_id}')

    def disconnect(self, session_id: str):
        '''Disconnect a WebSocket connection.

        Args:
            session_id (str): The session ID to disconnect.
        '''
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f'WebSocket disconnected for session {session_id}')

    async def send_progress(self, session_id: str, stage: str, progress: float,
                          current_structure: str = '', message: str = '',
                          disk_usage_mb: float = 0.0):
        '''Send a progress update to a connected client.

        Args:
            session_id (str): The session ID.
            stage (str): Current processing stage
                (e.g., 'parsing_dicom', 'building_graphs', 'calculating_relationships').
            progress (float): Progress percentage (0-100).
            current_structure (str, optional): Name of structure currently being processed.
            message (str, optional): Human-readable status message.
            disk_usage_mb (float, optional): Current disk usage in MB.
        '''
        if session_id not in self.active_connections:
            return

        websocket = self.active_connections[session_id]

        data = {
            'type': 'progress',
            'stage': stage,
            'progress': progress,
            'current_structure': current_structure,
            'message': message,
            'disk_usage_mb': round(disk_usage_mb, 1)
        }

        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f'Failed to send progress to session {session_id}: {e}')
            self.disconnect(session_id)

    async def send_error(self, session_id: str, error_message: str):
        '''Send an error message to a connected client.

        Args:
            session_id (str): The session ID.
            error_message (str): The error message to send.
        '''
        if session_id not in self.active_connections:
            return

        websocket = self.active_connections[session_id]

        data = {
            'type': 'error',
            'message': error_message
        }

        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f'Failed to send error to session {session_id}: {e}')
            self.disconnect(session_id)

    async def send_status_line(self, session_id: str, stage: str, source: str, message: str):
        '''Send a single status-line update to a connected client.

        Args:
            session_id (str): The session ID.
            stage (str): Current processing stage (mirrors send_progress stage values).
            source (str): Origin of the message, e.g. 'backend' or 'frontend'.
            message (str): Human-readable single-line status text.
        '''
        if session_id not in self.active_connections:
            return

        websocket = self.active_connections[session_id]

        data = {
            'type': 'status_line',
            'stage': stage,
            'source': source,
            'message': message,
        }

        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error('Failed to send status_line to session %s: %s', session_id, e)
            self.disconnect(session_id)

    async def send_complete(self, session_id: str, message: str = 'Processing complete'):
        '''Send a completion message to a connected client.

        Args:
            session_id (str): The session ID.
            message (str, optional): Completion message.
        '''
        if session_id not in self.active_connections:
            return

        websocket = self.active_connections[session_id]

        data = {
            'type': 'complete',
            'message': message
        }

        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f'Failed to send completion to session {session_id}: {e}')
            self.disconnect(session_id)

    def is_connected(self, session_id: str) -> bool:
        '''Check if a session has an active WebSocket connection.

        Args:
            session_id (str): The session ID to check.

        Returns:
            bool: True if connected, False otherwise.
        '''
        return session_id in self.active_connections
