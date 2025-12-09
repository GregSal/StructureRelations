"""
StructureRelations Web Application

FastAPI-based web interface for analyzing spatial relationships
between DICOM RT structures using DE-27IM classification.

Main components:
- main.py: FastAPI application with REST API and WebSocket endpoints
- session_manager.py: Session persistence with disk usage management
- websocket_manager.py: Real-time progress updates
- static/: Frontend HTML, CSS, and JavaScript

Run with: uvicorn main:app --reload
"""

__version__ = '1.0.0'
