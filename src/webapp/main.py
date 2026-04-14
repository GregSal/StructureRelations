'''FastAPI web application for StructureRelations DICOM analysis.

This application provides a web interface for uploading DICOM RT files,
selecting structures, and analyzing spatial relationships with customizable
matrix visualization.
'''
import sys
import os
import logging
import uuid
import asyncio
import tempfile
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import io

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dicom import DicomStructureFile, clean_uploaded_file_name
from structure_set import StructureSet
from contour_plotting import plot_roi_slice
from relations import RELATION_SCHEMA_VERSION
from webapp.session_manager import SessionManager, SessionData
from webapp.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title='StructureRelations', version='1.0.3')

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.debug(f"=== REQUEST: {request.method} {request.url.path} ===")
    try:
        response = await call_next(request)
        logger.debug(f"=== RESPONSE: {request.method} {request.url.path} - Status: {response.status_code} ===")
        return response
    except Exception as e:
        logger.error(f"=== ERROR in {request.method} {request.url.path}: {e} ===", exc_info=True)
        raise

# Initialize managers
session_manager = SessionManager()
connection_manager = ConnectionManager()
processing_tasks: Dict[str, asyncio.Task] = {}
cancel_events: Dict[str, asyncio.Event] = {}
_plot_response_cache: OrderedDict = OrderedDict()
PLOT_RESPONSE_CACHE_MAX_ENTRIES = 128

# Mount static files
static_dir = Path(__file__).parent / 'static'
static_dir.mkdir(exist_ok=True)
app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')


# Pydantic models for request/response validation
class UploadResponse(BaseModel):
    session_id: str
    disk_usage_mb: float
    disk_warning: bool


class PreviewResponse(BaseModel):
    session_id: str
    structures: List[dict]
    patient_info: dict
    disk_usage_mb: float


class SessionRequest(BaseModel):
    session_id: str


class MatrixRequest(BaseModel):
    session_id: str
    row_rois: Optional[List[int]] = None
    col_rois: Optional[List[int]] = None
    use_symbols: bool = True
    show_disjoint: bool = False
    show_unknown: bool = False
    logical_relations_mode: str = 'limited'  # 'hide', 'limited', 'show', 'faded'


class MatrixResponse(BaseModel):
    rows: List[int]
    columns: List[int]
    data: List[List[str]]
    row_names: List[str]
    col_names: List[str]
    colors: dict
    dicom_types: dict
    code_meanings: dict
    volumes: dict
    num_regions: dict
    slice_ranges: dict
    slice_indices: List[float]
    slice_indices_original: List[float] = Field(default_factory=list)
    structure_slices: dict
    structure_slices_original: dict = Field(default_factory=dict)
    structure_slices_interpolated: dict = Field(default_factory=dict)
    slice_relationships: dict = Field(default_factory=dict)
    faded_relationships: Optional[dict] = None


class DiagramNode(BaseModel):
    id: int
    label: str
    color: str
    shape: str
    title: str  # Tooltip


class DiagramEdge(BaseModel):
    from_node: int
    to_node: int
    relation_type: str
    symbol: Optional[str] = None
    label: str
    title: str
    color: str
    width: int
    dashes: bool
    arrows: Optional[str] = None
    is_logical: bool = False


class DiagramResponse(BaseModel):
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]


_diagram_settings_state = {'data': {}, 'mtime': None}
_relationship_definitions_state = {'data': {}, 'mtime': None}

# Micro-timing log control: log only when total_ms exceeds threshold,
# or once every N requests (0 = threshold-only, no sampling).
_TIMING_THRESHOLD_MS: int = int(os.getenv('SR_TIMING_THRESHOLD_MS', '1500'))
_TIMING_SAMPLE_N: int = int(os.getenv('SR_TIMING_SAMPLE_N', '0'))
_timing_request_counter: int = 0


def _load_diagram_settings() -> dict:
    """Load diagram settings configuration from disk.

    Returns:
        dict: Diagram settings dictionary or an empty dictionary on failure.
    """
    diagram_file = Path(__file__).parent / 'config' / 'diagram_settings.json'
    if not diagram_file.exists():
        logger.warning('Diagram settings file not found: %s', diagram_file)
        return {}

    try:
        current_mtime = diagram_file.stat().st_mtime
        cached_data = _diagram_settings_state['data']
        if cached_data and _diagram_settings_state['mtime'] == current_mtime:
            return cached_data

        with open(diagram_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            _diagram_settings_state['data'] = settings
            _diagram_settings_state['mtime'] = current_mtime
            return settings
    except (IOError, OSError, json.JSONDecodeError) as exc:
        logger.warning('Failed to load diagram settings: %s', exc)
        return {}


def _load_relationship_definitions() -> dict:
    """Load relationship definitions from disk.

    Returns:
        dict: Relationship definitions dictionary or an empty dictionary on failure.
    """
    definitions_file = Path(__file__).parent.parent / 'relationship_definitions.json'
    if not definitions_file.exists():
        logger.warning('Relationship definitions file not found: %s', definitions_file)
        return {}

    try:
        current_mtime = definitions_file.stat().st_mtime
        cached_data = _relationship_definitions_state['data']
        if cached_data and _relationship_definitions_state['mtime'] == current_mtime:
            return cached_data

        with open(definitions_file, 'r', encoding='utf-8') as f:
            definitions = json.load(f)
            _relationship_definitions_state['data'] = definitions
            _relationship_definitions_state['mtime'] = current_mtime
            return definitions
    except (IOError, OSError, json.JSONDecodeError) as exc:
        logger.warning('Failed to load relationship definitions: %s', exc)
        return {}


class ProcessRequest(BaseModel):
    session_id: str
    selected_rois: Optional[List[int]] = None


class RelationshipContractMixin(BaseModel):
    relation_type: Optional[str] = None
    label: Optional[str] = None
    symbol: Optional[str] = None
    schema_version: int = RELATION_SCHEMA_VERSION
    tolerance: float = 0.0
    computed_at: Optional[datetime] = None
    provenance: dict = Field(default_factory=dict)


class JobSubmitResponse(RelationshipContractMixin):
    job_id: str
    session_id: str
    status: str
    message: str


class JobStatusResponse(RelationshipContractMixin):
    job_id: str
    session_id: str
    status: str
    stage: str = 'idle'
    progress: float
    message: str
    error: Optional[str] = None


class StructurePairResult(RelationshipContractMixin):
    roi_a: int
    roi_b: int
    is_logical: bool = False


class RelationshipResultResponse(RelationshipContractMixin):
    job_id: str
    session_id: str
    status: str
    pairs: List[StructurePairResult]


# Startup event
@app.on_event('startup')
async def startup_event():
    '''Initialize the application and start cleanup task.'''
    logger.info('Starting StructureRelations web application')

    # Start background cleanup task
    asyncio.create_task(periodic_cleanup())


async def periodic_cleanup():
    '''Periodically clean up expired sessions every 30 minutes.'''
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        logger.info('Running periodic session cleanup')
        session_manager.enforce_disk_limit()


@app.get('/', response_class=HTMLResponse)
async def root():
    '''Serve the main application page.'''
    html_file = Path(__file__).parent / 'static' / 'index.html'
    if html_file.exists():
        return HTMLResponse(
            content=html_file.read_text(encoding='utf-8'),
            status_code=200
        )
    return HTMLResponse(
        content='<h1>StructureRelations</h1><p>Static files not found</p>',
        status_code=200
    )


@app.get('/api/config/symbols')
async def get_symbol_config():
    '''Get the relationship symbol and color configuration.

    Returns:
        dict: Configuration for relationship symbols, labels, descriptions, and colors.

    Raises:
        HTTPException: If relationship_definitions.json or diagram_settings.json not found.
    '''
    definitions = _load_relationship_definitions()
    diagram_settings = _load_diagram_settings()

    if not definitions:
        logger.error('Unable to load relationship_definitions.json')
        raise HTTPException(
            status_code=500,
            detail='Configuration file not found: relationship_definitions.json'
        )
    if not diagram_settings:
        logger.error('Unable to load diagram_settings.json')
        raise HTTPException(
            status_code=500,
            detail='Configuration file not found: diagram_settings.json'
        )

    try:
        relationship_styles = diagram_settings.get('relationship_styles', {})
        node_shapes = diagram_settings.get('node_shapes', {})
        relationship_defaults = diagram_settings.get(
            'relationship_display_defaults',
            diagram_settings.get('Relationship_Display_Defaults', {})
        )
        diagram_options = diagram_settings.get(
            'diagram_options',
            diagram_settings.get('DiagramOptions', {})
        )

        # Transform relationship definitions to symbol config format
        relationships = {}
        for rel in definitions.get('Relationships', []):
            rel_type = rel.get('relation_type')
            if rel_type:
                relationships[rel_type] = {
                    'symbol': rel.get('symbol', '?'),
                    'label': rel.get('label', rel_type),
                    'description': rel.get('description', ''),
                    'color': relationship_styles.get(rel_type, {}).get(
                        'color', '#9ca3af'
                    )
                }
        # Include logical relationships config
        config = {
            'description': definitions.get('Introduction', ''),
            'version': '1.0',
            'logical_relationships': definitions.get('logical_relationships', {
                'transparency': 20,
                'description': 'Settings for logical relationship display'
            }),
            'relationships': relationships,
            'node_shapes': node_shapes,
            'relationship_styles': relationship_styles,
            'relationship_display_defaults': relationship_defaults,
            'diagram_options': diagram_options,
            'tooltips': diagram_settings.get('tooltips', {}),
        }
        return config
    except json.JSONDecodeError as e:
        logger.error('Invalid JSON in configuration files: %s', e)
        raise HTTPException(
            status_code=500,
            detail=f'Invalid JSON in configuration files: {str(e)}'
        ) from e
    except IOError as e:
        logger.error('Error reading configuration files: %s', e)
        raise HTTPException(
            status_code=500,
            detail=f'Error reading configuration files: {str(e)}'
        ) from e


@app.post('/api/upload', response_model=UploadResponse)
async def upload_dicom(file: UploadFile = File(...)):
    '''Upload a DICOM RT Structure Set file and create a new session.

    Args:
        file (UploadFile): The DICOM file to upload.

    Returns:
        UploadResponse: Session ID and disk usage information.
    '''
    # Validate file extension
    if not file.filename.lower().endswith('.dcm'):
        raise HTTPException(status_code=400, detail='Only .dcm files are allowed')

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Save uploaded file to temporary location
    temp_dir = Path(tempfile.gettempdir()) / 'structurerelations_uploads'
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / f'{session_id}_{file.filename}'

    try:
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info('Uploaded DICOM file %s for session %s', file.filename, session_id)

        # Create session data
        session_data = SessionData(
            dicom_file_path=str(file_path),
            structure_set=None,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )

        # Save session
        session_manager.save_session(session_id, session_data)

        # Get disk usage info
        disk_info = session_manager.get_disk_usage_info()

        return UploadResponse(
            session_id=session_id,
            disk_usage_mb=disk_info['usage_mb'],
            disk_warning=disk_info['is_warning']
        )

    except Exception as e:
        logger.error('Error uploading file: %s', e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/preview', response_model=PreviewResponse)
async def preview_structures(request: SessionRequest):
    '''Get DICOM structure metadata without full processing.

    Args:
        request (SessionRequest): Request containing session_id.

    Returns:
        PreviewResponse: Structure metadata and patient information.
    '''
    # Load session
    session_data = session_manager.load_session(request.session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    try:
        # Load DICOM file
        file_path = Path(session_data.dicom_file_path)
        dicom_file = DicomStructureFile(
            top_dir=file_path.parent,
            file_path=file_path
        )

        # Extract structure metadata
        structures = []
        structure_names = dicom_file.get_structure_names()

        # Get ROI labels (DICOM Type and Code Meaning)
        roi_labels = dicom_file.get_roi_labels()

        # Get colors from DICOM
        colors = {}
        try:
            for roi_contour in dicom_file.dataset.ROIContourSequence:
                roi_num = roi_contour.ReferencedROINumber
                if hasattr(roi_contour, 'ROIDisplayColor'):
                    colors[roi_num] = list(roi_contour.ROIDisplayColor)
        except AttributeError:
            pass

        # Build structure list
        for roi, name in structure_names.items():
            # Get DICOM Type and Code Meaning from roi_labels
            dicom_type = ''
            code_meaning = ''
            if not roi_labels.empty and roi in roi_labels.index:
                dicom_type = roi_labels.loc[roi].get('DICOM_Type', '')
                code_meaning = roi_labels.loc[roi].get('CodeMeaning', '')

            structure_info = {
                'roi': roi,
                'name': name,
                'dicom_type': dicom_type,
                'code_meaning': code_meaning,
                'color': colors.get(roi, [128, 128, 128]),  # Default gray
                'num_contours': sum(
                    1 for cp in dicom_file.contour_points if cp['ROI'] == roi
                )
            }
            # Only include structures with contours
            if structure_info['num_contours'] > 0:
                structures.append(structure_info)

        # Sort by ROI number
        structures.sort(key=lambda s: s['roi'])

        # Get patient and structure-set metadata from existing DICOM info
        structure_set_info = dicom_file.get_structure_set_info()
        file_path_value = structure_set_info.get('File')
        file_name = ''
        if file_path_value:
            try:
                file_name = clean_uploaded_file_name(file_path_value)
            except (TypeError, ValueError):
                file_name = clean_uploaded_file_name(str(file_path_value))

        resolution_value = dicom_file.resolution
        resolution_cm_per_pixel = (
            float(resolution_value) if resolution_value is not None else None
        )

        patient_info = {
            'patient_id': structure_set_info.get('PatientID', ''),
            'patient_name': structure_set_info.get('PatientName', ''),
            'structure_set': structure_set_info.get('StructureSet', ''),
            'study_id': structure_set_info.get('StudyID', ''),
            'series_number': structure_set_info.get('SeriesNumber', ''),
            'series_description': structure_set_info.get('SeriesDescription', ''),
            'file_name': file_name,
            'file_path': str(file_path_value) if file_path_value else '',
            'resolution_cm_per_pixel': resolution_cm_per_pixel,
            'structure_count': len(structures),
        }

        # Get disk usage
        disk_info = session_manager.get_disk_usage_info()

        return PreviewResponse(
            session_id=request.session_id,
            structures=structures,
            patient_info=patient_info,
            disk_usage_mb=disk_info['usage_mb']
        )

    except Exception as e:
        logger.error('Error previewing structures for session %s: %s', request.session_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/process', response_model=JobSubmitResponse)
async def process_structures(request: ProcessRequest):
    '''Process selected structures and calculate relationships.

    This endpoint starts background processing and sends progress updates via WebSocket.

    Args:
        request (ProcessRequest): Processing request with session ID and selected ROIs.

    Returns:
        dict: Confirmation message.
    '''
    session_data = session_manager.load_session(request.session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    if request.session_id in processing_tasks:
        task = processing_tasks[request.session_id]
        if not task.done():
            return JobSubmitResponse(
                job_id=request.session_id,
                session_id=request.session_id,
                status='running',
                message='Processing already running',
                tolerance=session_data.structure_set.tolerance if session_data.structure_set else 0.0,
                computed_at=session_data.job_computed_at,
                provenance={
                    'api': '/api/process',
                    'event': 'already_running',
                },
            )

    cancel_event = asyncio.Event()
    cancel_events[request.session_id] = cancel_event
    session_manager.update_session_job_state(
        request.session_id,
        job_status='running',
        job_stage='queued',
        job_progress=0.0,
        job_message='Processing started',
        job_error=None,
        job_provenance={
            'api': '/api/process',
            'selected_rois': request.selected_rois or [],
            'started_at': datetime.now().isoformat(),
        },
    )

    processing_tasks[request.session_id] = asyncio.create_task(
        process_structure_set(
            request.session_id,
            session_data.dicom_file_path,
            request.selected_rois,
            cancel_event=cancel_event,
        )
    )

    return JobSubmitResponse(
        job_id=request.session_id,
        session_id=request.session_id,
        status='running',
        message='Processing started',
        tolerance=session_data.structure_set.tolerance if session_data.structure_set else 0.0,
        computed_at=None,
        provenance={
            'api': '/api/process',
            'selected_rois': request.selected_rois or [],
        },
    )


@app.get('/api/jobs/{session_id}/status', response_model=JobStatusResponse)
async def get_job_status(session_id: str):
    '''Get status for a relationship processing job.'''
    session_data = session_manager.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    return JobStatusResponse(
        job_id=session_id,
        session_id=session_id,
        status=session_data.job_status,
        stage=session_data.job_stage,
        progress=session_data.job_progress,
        message=session_data.job_message,
        error=session_data.job_error,
        tolerance=session_data.structure_set.tolerance if session_data.structure_set else 0.0,
        computed_at=session_data.job_computed_at,
        provenance={
            **session_data.job_provenance,
            'api': f'/api/jobs/{session_id}/status',
        },
    )


@app.get('/api/jobs/{session_id}/result', response_model=RelationshipResultResponse)
async def get_job_result(session_id: str):
    '''Return relationship pair results for a completed session job.'''
    session_data = session_manager.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')
    if session_data.structure_set is None:
        raise HTTPException(status_code=400, detail='Structures not yet processed')
    if session_data.job_status not in {'completed', 'cancelled'}:
        raise HTTPException(status_code=409, detail='Processing not complete')

    pairs: List[StructurePairResult] = []
    structure_set = session_data.structure_set
    for roi_a, roi_b, edge_data in structure_set.relationship_graph.edges(data=True):
        relationship = edge_data.get('relationship')
        rel_type = relationship.relationship_type if relationship else None
        pairs.append(
            StructurePairResult(
                roi_a=int(roi_a),
                roi_b=int(roi_b),
                relation_type=rel_type.relation_type if rel_type else 'UNKNOWN',
                label=rel_type.label if rel_type else 'Unknown',
                symbol=rel_type.symbol if rel_type else '?',
                schema_version=RELATION_SCHEMA_VERSION,
                tolerance=structure_set.tolerance,
                computed_at=session_data.job_computed_at,
                provenance={
                    'api': f'/api/jobs/{session_id}/result',
                    'logical': bool(getattr(relationship, 'is_logical', False)),
                },
                is_logical=bool(getattr(relationship, 'is_logical', False)),
            )
        )

    pairs.sort(key=lambda pair: (pair.roi_a, pair.roi_b))
    return RelationshipResultResponse(
        job_id=session_id,
        session_id=session_id,
        status=session_data.job_status,
        relation_type='BATCH',
        label='Relationship Result Set',
        symbol='*',
        schema_version=RELATION_SCHEMA_VERSION,
        tolerance=structure_set.tolerance,
        computed_at=session_data.job_computed_at,
        provenance={
            **session_data.job_provenance,
            'api': f'/api/jobs/{session_id}/result',
            'pair_count': len(pairs),
        },
        pairs=pairs,
    )


@app.post('/api/jobs/{session_id}/cancel', response_model=JobStatusResponse)
async def cancel_job(session_id: str):
    '''Request cancellation for an in-flight processing job.'''
    session_data = session_manager.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    cancel_event = cancel_events.get(session_id)
    task = processing_tasks.get(session_id)
    if cancel_event is None or task is None or task.done():
        if session_data.job_status == 'running':
            session_manager.update_session_job_state(
                session_id,
                job_status='cancelled',
                job_stage='cancelled',
                job_message='Cancellation requested',
                job_provenance={
                    **session_data.job_provenance,
                    'cancel_requested_at': datetime.now().isoformat(),
                },
            )
            session_data = session_manager.load_session(session_id)
        return JobStatusResponse(
            job_id=session_id,
            session_id=session_id,
            status=session_data.job_status,
            stage=session_data.job_stage,
            progress=session_data.job_progress,
            message='No active job to cancel',
            error=session_data.job_error,
            tolerance=session_data.structure_set.tolerance if session_data.structure_set else 0.0,
            computed_at=session_data.job_computed_at,
            provenance={
                **session_data.job_provenance,
                'api': f'/api/jobs/{session_id}/cancel',
            },
        )

    cancel_event.set()
    session_manager.update_session_job_state(
        session_id,
        job_status='cancelled',
        job_stage='cancelled',
        job_message='Cancellation requested',
        job_provenance={
            **session_data.job_provenance,
            'cancel_requested_at': datetime.now().isoformat(),
        },
    )
    return JobStatusResponse(
        job_id=session_id,
        session_id=session_id,
        status='cancelled',
        stage='cancelled',
        progress=session_data.job_progress,
        message='Cancellation requested',
        error=None,
        tolerance=session_data.structure_set.tolerance if session_data.structure_set else 0.0,
        computed_at=session_data.job_computed_at,
        provenance={
            **session_data.job_provenance,
            'api': f'/api/jobs/{session_id}/cancel',
        },
    )


async def process_structure_set(
    session_id: str,
    dicom_file_path: str,
    selected_rois: Optional[List[int]],
    cancel_event: Optional[asyncio.Event] = None,
):
    '''Background task to process DICOM file and calculate relationships.

    Args:
        session_id (str): The session ID.
        dicom_file_path (str): Path to the DICOM file.
        selected_rois (List[int], optional): List of ROI numbers to process.
    '''
    try:
        def _was_cancelled() -> bool:
            return bool(cancel_event is not None and cancel_event.is_set())

        ws_loop = asyncio.get_running_loop()
        ws_last_emit_time = 0.0
        ws_last_emit_progress = -1.0

        def _queue_progress_update(
            stage: str,
            progress: float,
            current_structure: str,
            message: str,
            force_emit: bool = False,
        ) -> None:
            nonlocal ws_last_emit_time, ws_last_emit_progress
            bounded_progress = max(0.0, min(100.0, float(progress)))
            session_manager.update_session_job_state(
                session_id,
                job_stage=stage,
                job_progress=bounded_progress,
                job_message=message,
            )
            now = time.monotonic()
            time_since_last = now - ws_last_emit_time
            should_emit = (
                bounded_progress in {0.0, 100.0}
                or (force_emit and time_since_last >= 0.1)
                or (not force_emit and (
                    (bounded_progress - ws_last_emit_progress) >= 0.5
                    or time_since_last >= 0.4
                ))
            )
            if not should_emit:
                return

            ws_last_emit_time = now
            ws_last_emit_progress = bounded_progress
            disk_info = session_manager.get_disk_usage_info()
            # Use run_coroutine_threadsafe so this is safe to call from either
            # the event-loop thread or a run_in_executor worker thread.
            asyncio.run_coroutine_threadsafe(
                connection_manager.send_progress(
                    session_id,
                    stage,
                    bounded_progress,
                    current_structure,
                    message,
                    disk_info['usage_mb'],
                ),
                ws_loop,
            )

        def _queue_status_line(stage: str, message: str) -> None:
            ws_loop.create_task(
                connection_manager.send_status_line(
                    session_id, stage, 'backend', message
                )
            )

        # Send initial progress
        session_manager.update_session_job_state(
            session_id,
            job_status='running',
            job_stage='parsing_dicom',
            job_progress=0.0,
            job_message='Loading DICOM file',
        )
        _queue_progress_update(
            'parsing_dicom',
            0.0,
            '',
            'Loading DICOM file...',
            force_emit=True,
        )
        _queue_status_line('parsing_dicom', 'Loading DICOM file...')
        await asyncio.sleep(0)

        if _was_cancelled():
            raise asyncio.CancelledError('Processing cancelled before parsing')

        # Load DICOM file with error handling
        try:
            file_path = Path(dicom_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f'DICOM file not found: {dicom_file_path}')

            dicom_file = DicomStructureFile(
                top_dir=file_path.parent,
                file_path=file_path
            )
        except FileNotFoundError as e:
            logger.error('File error in session %s: %s', session_id, e)
            await connection_manager.send_error(session_id, f'File not found: {str(e)}')
            return
        except Exception as e:
            logger.error('DICOM parsing error in session %s: %s', session_id, e, exc_info=True)
            await connection_manager.send_error(session_id, f'Failed to parse DICOM file: {str(e)}')
            return

        _queue_progress_update(
            'parsing_dicom',
            20.0,
            '',
            'Parsing structures...',
            force_emit=True,
        )
        _queue_status_line(
            'parsing_dicom',
            f'DICOM loaded \u2014 {len(dicom_file.contour_points)} contour slice(s) found.',
        )
        await asyncio.sleep(0)

        # Filter contour points to only include selected ROIs
        if selected_rois:
            original_count = len(dicom_file.contour_points)
            dicom_file.contour_points = [
                cp for cp in dicom_file.contour_points
                if cp['ROI'] in selected_rois
            ]
            logger.info('Filtered contours from %d to %d for selected ROIs: %s', original_count, len(dicom_file.contour_points), selected_rois)

        # Create structure set with filtered contours
        try:
            structure_set = StructureSet(
                dicom_structure_file=dicom_file,
                auto_calculate_relationships=False,
                auto_calculate_logical_flags=False,
            )
        except Exception as e:
            logger.error('Structure set creation error in session %s: %s', session_id, e, exc_info=True)
            await connection_manager.send_error(session_id, f'Failed to create structure set: {str(e)}')
            session_manager.update_session_job_state(
                session_id,
                job_status='failed',
                job_stage='failed',
                job_message='Failed to create structure set',
                job_error=str(e),
            )
            return

        _queue_progress_update(
            'building_graphs',
            40.0,
            '',
            'Building contour graphs...',
            force_emit=True,
        )
        _queue_status_line(
            'building_graphs',
            f'Structure set created with {len(structure_set.structures)} structure(s).'
            ' Building contour graphs...',
        )
        await asyncio.sleep(0)

        # Process each structure
        total_structures = len(structure_set.structures)
        for idx, (roi, structure) in enumerate(structure_set.structures.items()):
            if selected_rois and roi not in selected_rois:
                continue

            progress = 40.0 + ((idx / total_structures) * 30.0)
            _queue_progress_update(
                'building_graphs',
                progress,
                structure.name,
                f'Preparing {structure.name}...',
            )

            if _was_cancelled():
                raise asyncio.CancelledError('Processing cancelled during graph build')

        _queue_progress_update(
            'calculating_relationships',
            70.0,
            '',
            'Calculating relationships...',
            force_emit=True,
        )
        _n_structures = len(structure_set.structures)
        _queue_status_line(
            'calculating_relationships',
            f'Contour graphs built. Computing relationships for '
            f'{_n_structures * (_n_structures - 1) // 2} structure pair(s)...',
        )
        await asyncio.sleep(0)

        # Calculate relationships with error handling
        try:
            def pair_progress(completed_pairs: int, total_pairs: int) -> None:
                if total_pairs <= 0:
                    progress = 95.0
                else:
                    progress = 70.0 + (completed_pairs / total_pairs) * 25.0
                pair_status = structure_set.relationship_progress.get(
                    'status',
                    f'pair {completed_pairs}/{total_pairs}',
                )
                current_slice = structure_set.relationship_progress.get('current_slice', 0)
                total_slices = structure_set.relationship_progress.get('total_slices', 0)
                if isinstance(current_slice, int) and isinstance(total_slices, int):
                    message = (
                        f'Computing {pair_status} '
                        f'(slice {current_slice}/{total_slices})'
                    )
                else:
                    message = f'Computing {pair_status}'

                _queue_progress_update(
                    'calculating_relationships',
                    progress,
                    '',
                    message,
                    force_emit=True,
                )
                if _was_cancelled():
                    raise RuntimeError('Processing cancelled by user')

            await ws_loop.run_in_executor(
                None,
                lambda: structure_set.calculate_relationships(
                    force=True, progress_callback=pair_progress
                ),
            )
            _queue_progress_update(
                'calculating_relationships',
                97.5,
                '',
                'Resolving logical relationships...',
                force_emit=True,
            )
            _queue_status_line(
                'calculating_relationships',
                'Spatial relationships calculated. Resolving logical flags...',
            )
            await asyncio.sleep(0)
            structure_set.calculate_logical_flags()
            _queue_status_line('calculating_relationships', 'Logical flags resolved.')
            await asyncio.sleep(0)
        except Exception as e:
            if 'cancelled' in str(e).lower():
                raise asyncio.CancelledError(str(e)) from e
            logger.error('Relationship calculation error in session %s: %s', session_id, e, exc_info=True)
            await connection_manager.send_error(session_id, f'Failed to calculate relationships: {str(e)}')
            session_manager.update_session_job_state(
                session_id,
                job_status='failed',
                job_stage='failed',
                job_message='Failed to calculate relationships',
                job_error=str(e),
            )
            return

        _queue_progress_update(
            'calculating_relationships',
            100.0,
            '',
            'Complete!',
            force_emit=True,
        )

        # Save completed session BEFORE sending complete message
        # Use dedicated method to avoid race conditions with last_accessed updates
        if not session_manager.update_session_structure_set(session_id, structure_set):
            logger.error('Failed to save structure set to session %s', session_id)
            await connection_manager.send_error(session_id, 'Failed to save results')
            session_manager.update_session_job_state(
                session_id,
                job_status='failed',
                job_stage='failed',
                job_message='Failed to save results',
                job_error='Failed to save structure set',
            )
            return

        session_manager.update_session_job_state(
            session_id,
            job_status='completed',
            job_stage='completed',
            job_progress=100.0,
            job_message='Processing complete',
            job_error=None,
            job_computed_at=datetime.now(),
            job_provenance={
                'api': '/api/process',
                'selected_rois': selected_rois or [],
                'result_schema_version': RELATION_SCHEMA_VERSION,
            },
        )

        # Small delay to ensure file system flushes the pickle file
        await asyncio.sleep(0.1)

        # Send completion message (frontend will immediately request matrix)
        await connection_manager.send_complete(session_id, 'Processing complete')

        logger.info('Completed processing for session %s', session_id)

    except asyncio.CancelledError as e:
        logger.info('Processing cancelled for session %s: %s', session_id, e)
        session_manager.update_session_job_state(
            session_id,
            job_status='cancelled',
            job_stage='cancelled',
            job_message='Processing cancelled',
            job_error=None,
            job_provenance={
                'api': '/api/process',
                'cancelled_at': datetime.now().isoformat(),
            },
        )
        await connection_manager.send_error(session_id, 'Processing cancelled')

    except Exception as e:
        logger.error('Unexpected error processing session %s: %s', session_id, e, exc_info=True)
        await connection_manager.send_error(session_id, f'Unexpected processing error: {str(e)}')
        session_manager.update_session_job_state(
            session_id,
            job_status='failed',
            job_stage='failed',
            job_message='Unexpected processing error',
            job_error=str(e),
        )
    finally:
        processing_tasks.pop(session_id, None)
        cancel_events.pop(session_id, None)


@app.post('/api/matrix', response_model=MatrixResponse)
async def get_relationship_matrix(request: MatrixRequest):
    '''Get a filtered relationship matrix.

    Args:
        request (MatrixRequest): Matrix request with session ID and filter parameters.

    Returns:
        MatrixResponse: The filtered relationship matrix.
    '''
    logger.info(f'=== MATRIX REQUEST RECEIVED === session: {request.session_id}')
    logger.info(f'Matrix request for session {request.session_id}: '
                f'row_rois={request.row_rois}, col_rois={request.col_rois}, '
                f'use_symbols={request.use_symbols}, '
                f'logical_relations_mode={request.logical_relations_mode}')

    request_start = time.perf_counter()
    await connection_manager.send_status_line(
        request.session_id,
        'loading_results',
        'backend',
        'Loading processed session data...',
    )
    session_load_start = time.perf_counter()
    session_data = session_manager.load_session(request.session_id)
    session_load_ms = round((time.perf_counter() - session_load_start) * 1000.0)
    if session_data is None:
        logger.error(f'Session {request.session_id} not found or expired')
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    if session_data.structure_set is None:
        logger.error(f'Session {request.session_id} has no structure_set')
        raise HTTPException(status_code=400, detail='Structures not yet processed')

    try:
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Preparing matrix filters...',
        )

        # Build visible_rois list from row and col rois for limited mode
        filter_start = time.perf_counter()
        visible_rois = None
        if request.logical_relations_mode == 'limited':
            visible_rois = []
            if request.row_rois:
                visible_rois.extend(request.row_rois)
            if request.col_rois:
                visible_rois.extend(request.col_rois)
            if visible_rois:
                visible_rois = list(set(visible_rois))  # Remove duplicates
        filter_ms = round((time.perf_counter() - filter_start) * 1000.0)

        # Get matrix as dictionary
        logger.info(f'Generating matrix for session {request.session_id}')
        matrix_generation_start = time.perf_counter()
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Generating relationship matrix payload...',
        )
        matrix_dict = session_data.structure_set.to_dict(
            row_rois=request.row_rois,
            col_rois=request.col_rois,
            use_symbols=request.use_symbols,
            logical_relations_mode=request.logical_relations_mode,
            visible_rois=visible_rois
        )
        matrix_generation_ms = round(
            (time.perf_counter() - matrix_generation_start) * 1000.0
        )
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            f'Matrix payload generated ({matrix_generation_ms} ms).',
        )

        logger.info(f'Matrix dict keys: {matrix_dict.keys()}')
        logger.info(f'Matrix dict types: {[(k, type(v).__name__) for k, v in matrix_dict.items()]}')

        # Create response
        try:
            validation_start = time.perf_counter()
            await connection_manager.send_status_line(
                request.session_id,
                'loading_results',
                'backend',
                'Validating matrix response...',
            )
            response = MatrixResponse(**matrix_dict)
            validation_ms = round(
                (time.perf_counter() - validation_start) * 1000.0
            )
            total_ms = round((time.perf_counter() - request_start) * 1000.0)
            await connection_manager.send_status_line(
                request.session_id,
                'loading_results',
                'backend',
                f'Matrix ready ({total_ms} ms total).',
            )
            global _timing_request_counter
            _timing_request_counter += 1
            _should_log_timing = (
                total_ms > _TIMING_THRESHOLD_MS
                or (_TIMING_SAMPLE_N > 0
                    and (_timing_request_counter % _TIMING_SAMPLE_N) == 0)
            )
            if _should_log_timing:
                logger.info(
                    'MATRIX_TIMING session=%s session_load_ms=%s filter_ms=%s '
                    'payload_ms=%s validation_ms=%s total_ms=%s rows=%s cols=%s',
                    request.session_id,
                    session_load_ms,
                    filter_ms,
                    matrix_generation_ms,
                    validation_ms,
                    total_ms,
                    len(matrix_dict.get('rows', [])),
                    len(matrix_dict.get('columns', [])),
                )
            logger.info(f'Matrix generated successfully for session {request.session_id}')
            return response
        except Exception as validation_error:
            logger.error(f'Error validating MatrixResponse: {validation_error}', exc_info=True)
            logger.error(f'Matrix dict content: {matrix_dict}')
            raise HTTPException(status_code=500, detail=f'Response validation error: {str(validation_error)}')

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error generating matrix for session %s: %s', request.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/export/{export_format}/{session_id}')
async def export_matrix(export_format: str, session_id: str, row_rois: Optional[str] = None,
                       col_rois: Optional[str] = None, use_symbols: bool = True):
    '''Export relationship matrix in various formats.

    Args:
        export_format (str): Export format ('csv', 'excel', 'json').
        session_id (str): The session ID.
        row_rois (str, optional): Comma-separated list of row ROI numbers.
        col_rois (str, optional): Comma-separated list of column ROI numbers.
        use_symbols (bool, optional): Use symbols instead of labels.

    Returns:
        FileResponse or StreamingResponse: The exported file.
    '''
    session_data = session_manager.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    if session_data.structure_set is None:
        raise HTTPException(status_code=400, detail='Structures not yet processed')

    # Parse ROI lists
    row_rois_list = [int(r) for r in row_rois.split(',')] if row_rois else None
    col_rois_list = [int(c) for c in col_rois.split(',')] if col_rois else None

    # Get matrix
    matrix_df = session_data.structure_set.get_relationship_matrix(
        row_rois=row_rois_list,
        col_rois=col_rois_list,
        use_symbols=use_symbols
    )

    if export_format == 'csv':
        output = io.StringIO()
        matrix_df.to_csv(output)
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename=relationships.csv'}
        )

    elif export_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            matrix_df.to_excel(writer, sheet_name='Relationships')
        output.seek(0)
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename=relationships.xlsx'}
        )

    elif export_format == 'json':
        matrix_dict = session_data.structure_set.to_dict(
            row_rois=row_rois_list,
            col_rois=col_rois_list,
            use_symbols=use_symbols
        )
        return matrix_dict

    else:
        raise HTTPException(status_code=400, detail='Invalid format. Use csv, excel, or json')


@app.post('/api/diagram', response_model=DiagramResponse)
async def get_diagram_data(request: MatrixRequest):
    '''Generate network diagram data for relationship visualization.

    Args:
        request (MatrixRequest): Contains session_id and optional ROI filters.

    Returns:
        DiagramResponse: Nodes and edges for network visualization.
    '''
    logger.info(f'=== DIAGRAM REQUEST RECEIVED === session: {request.session_id}')
    logger.info(f'    logical_relations_mode: {request.logical_relations_mode}')
    logger.info(f'    row_rois: {request.row_rois}')
    logger.info(f'    col_rois: {request.col_rois}')
    try:
        request_start = time.perf_counter()
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Loading data for relationship diagram...',
        )
        session_load_start = time.perf_counter()
        session_data = session_manager.load_session(request.session_id)
        session_load_ms = round((time.perf_counter() - session_load_start) * 1000.0)
        if session_data is None:
            raise HTTPException(status_code=404, detail='Session expired, please re-upload')

        if session_data.structure_set is None:
            raise HTTPException(status_code=400, detail='Structures not yet processed')

        structure_set = session_data.structure_set

        # Load node shape definitions and edge styles from diagram settings
        shape_map = {}
        default_shape = 'ellipse'
        edge_styles = {}
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Loading diagram configuration...',
        )
        config_start = time.perf_counter()
        diagram_settings = _load_diagram_settings()
        node_shapes = diagram_settings.get('node_shapes', {})
        shape_map = node_shapes.get('shape_map', {})
        default_shape = node_shapes.get('default_shape', 'ellipse')
        rel_styles = diagram_settings.get('relationship_styles', {})
        for rel_type, style in rel_styles.items():
            edge_styles[rel_type] = style
        tooltips_cfg = diagram_settings.get('tooltips', {})

        # Load transparency settings from relationship definitions
        logical_opacity = 0.2  # Default 20% opacity
        definitions = _load_relationship_definitions()
        transparency = definitions.get('logical_relationships', {}).get(
            'transparency', 20
        )
        logical_opacity = max(0.0, min(1.0, transparency / 100.0))
        relationship_metadata = {
            rel.get('relation_type'): {
                'symbol': rel.get('symbol', '?'),
                'label': rel.get('label', rel.get('relation_type', 'Unknown')),
                'description': rel.get('description', ''),
            }
            for rel in definitions.get('Relationships', [])
            if rel.get('relation_type')
        }
        config_ms = round((time.perf_counter() - config_start) * 1000.0)
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Building diagram nodes...',
        )

        def hex_to_rgba(color_hex: str, alpha: float) -> str:
            color_hex = color_hex.lstrip('#')
            if len(color_hex) != 6:
                return color_hex
            red = int(color_hex[0:2], 16)
            green = int(color_hex[2:4], 16)
            blue = int(color_hex[4:6], 16)
            return f'rgba({red}, {green}, {blue}, {alpha:.3f})'

        def get_arrow_description(arrows: Optional[str]) -> str:
            if arrows == 'to':
                return 'Directed from source to target'
            if arrows == 'to;from':
                return 'Bidirectional'
            if arrows == 'from':
                return 'Directed from target to source'
            return 'Undirected'

        def build_edge_tooltip(
            from_name: str,
            to_name: str,
            relation_type: str,
            is_logical: bool,
            style: dict,
        ) -> tuple[str, Optional[str]]:
            metadata = relationship_metadata.get(relation_type, {})
            symbol = metadata.get('symbol')
            label = metadata.get('label', relation_type)
            description = metadata.get('description', '')

            line_style = 'dashed' if style.get('dashes') else 'solid'
            direction = get_arrow_description(style.get('arrows'))
            logical_text = 'yes' if is_logical else 'no'

            edge_cfg = tooltips_cfg.get('edges', {})
            tooltip_lines = []
            if edge_cfg.get('show_source', True):
                tooltip_lines.append(f'Source: {from_name}')
            if edge_cfg.get('show_target', True):
                tooltip_lines.append(f'Target: {to_name}')
            if edge_cfg.get('show_relationship', True):
                tooltip_lines.append(f'Relationship: {label} ({relation_type})')
            if edge_cfg.get('show_symbol', True):
                tooltip_lines.append(f'Symbol: {symbol or "n/a"}')
            if edge_cfg.get('show_direction', True):
                tooltip_lines.append(f'Direction: {direction}')
            if edge_cfg.get('show_style', False):
                tooltip_lines.append(
                    f'Edge style: {line_style}, width {style.get("width", 2)}'
                )
            if edge_cfg.get('show_logical', True):
                tooltip_lines.append(f'Logical relation: {logical_text}')
            if description and edge_cfg.get('show_description', True):
                tooltip_lines.append(f'Description: {description}')
            return '\n'.join(tooltip_lines), symbol

        # Get summary data
        summary_start = time.perf_counter()
        summary_df = structure_set.summary()
        summary_ms = round((time.perf_counter() - summary_start) * 1000.0)

        # Extract colors from DICOM file
        color_start = time.perf_counter()
        colors = {}
        if structure_set.dicom_structure_file and hasattr(structure_set.dicom_structure_file, 'dataset'):
            try:
                for roi_contour in structure_set.dicom_structure_file.dataset.ROIContourSequence:
                    roi_num = int(roi_contour.ReferencedROINumber)
                    if hasattr(roi_contour, 'ROIDisplayColor'):
                        colors[roi_num] = [int(c) for c in roi_contour.ROIDisplayColor]
            except AttributeError:
                pass
        color_extract_ms = round((time.perf_counter() - color_start) * 1000.0)

        # Build edges from relationship matrix
        edges = []
        row_rois = request.row_rois if request.row_rois else [int(roi) for roi in summary_df['ROI'].tolist()]
        col_rois = request.col_rois if request.col_rois else [int(roi) for roi in summary_df['ROI'].tolist()]

        # Get union of all ROIs that should be displayed
        visible_rois = set(row_rois) | set(col_rois)

        def should_include_logical(rel_obj) -> bool:
            if rel_obj is None:
                return False

            is_logical = getattr(rel_obj, 'is_logical', False)
            intermediates = getattr(rel_obj, 'intermediate_structures', [])

            logger.debug(f'should_include: is_logical={is_logical}, intermediates={intermediates}, mode={request.logical_relations_mode}, visible={sorted(visible_rois)}')

            if request.logical_relations_mode == 'show':
                logger.debug('Mode is "show" - including all relationships')
                return True

            if not is_logical:
                logger.debug('Not a logical relationship - including')
                return True

            if request.logical_relations_mode == 'hide':
                logger.debug('Mode is "hide" and relationship is logical - excluding')
                return False

            if request.logical_relations_mode == 'limited':
                # Convert numpy types to Python ints for comparison
                intermediates_set = {int(roi) for roi in intermediates}
                all_visible = intermediates_set.issubset(visible_rois)
                should_include = not all_visible
                logger.debug(f'Limited mode: all_visible={all_visible}, should_include={should_include}, intermediates={intermediates_set}')
                return should_include

            logger.debug(f'Default include for mode: {request.logical_relations_mode}')
            return True

        # Build nodes only for visible structures
        nodes = []
        node_build_start = time.perf_counter()
        roi_to_name = {}
        for _, row in summary_df.iterrows():
            roi = int(row['ROI'])

            # Skip structures not in either From or To list
            if roi not in visible_rois:
                continue

            name = row['Name']
            dicom_type = row.get('DICOM_Type', 'NONE')

            # Get color from extracted colors or use default
            color_rgb = colors.get(roi, [200, 200, 200])  # Default gray if no color
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)

            # Create tooltip with structure info
            volume = row.get('Physical_Volume', 0)
            num_regions = row.get('Num_Regions', 0)
            node_cfg = tooltips_cfg.get('nodes', {})
            name_label = (
                f'{name} (ROI {roi})'
                if node_cfg.get('show_roi_number', True)
                else name
            )
            tooltip_lines = [name_label]
            if node_cfg.get('show_type', True):
                tooltip_lines.append(f'Type: {dicom_type}')
            if node_cfg.get('show_volume', True):
                tooltip_lines.append(f'Volume: {volume:.2f} cm³')
            if node_cfg.get('show_regions', True):
                tooltip_lines.append(f'Regions: {num_regions}')
            tooltip = '\n'.join(tooltip_lines)

            nodes.append(DiagramNode(
                id=roi,
                label=name,
                color=color_hex,
                shape=shape_map.get(dicom_type, default_shape),
                title=tooltip
            ))
            roi_to_name[roi] = name
        node_build_ms = round((time.perf_counter() - node_build_start) * 1000.0)

        # Build edges from relationship matrix
        edges = []
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            'Building diagram edges...',
        )

        # Define symmetric relationships (no direction)
        symmetric_relations = {'OVERLAPS', 'BORDERS', 'DISJOINT', 'EQUAL'}

        # For symmetric relationships: check all visible pairs once
        edge_build_start = time.perf_counter()
        symmetric_start = time.perf_counter()
        symmetric_pairs_checked = 0
        symmetric_edges_added = 0
        all_visible_rois = sorted(visible_rois)
        for i, roi1 in enumerate(all_visible_rois):
            for roi2 in all_visible_rois[i+1:]:
                symmetric_pairs_checked += 1
                # Check both directions since edges are only stored once
                rel = structure_set.get_relationship(roi1, roi2)
                if rel is None:
                    rel = structure_set.get_relationship(roi2, roi1)
                if rel is None:
                    continue

                try:
                    rel_type_obj = rel.relationship_type
                    if rel_type_obj is None:
                        logger.warning(f'Relationship type is None for {roi1}-{roi2}')
                        continue
                    rel_type = rel_type_obj.relation_type
                except (AttributeError, KeyError) as e:
                    logger.error(f'Error getting relationship type for {roi1}-{roi2}: {e}')
                    continue

                logger.debug(f'Symmetric pair ({roi1},{roi2}): type={rel_type}, is_logical={rel.is_logical}')
                if not should_include_logical(rel):
                    logger.debug(f'  Filtering out {rel_type} relationship ({roi1},{roi2})')
                    continue

                if rel_type == 'EQUAL':
                    continue
                if rel_type == 'DISJOINT' and not request.show_disjoint:
                    continue
                if rel_type == 'UNKNOWN' and not request.show_unknown:
                    continue

                if rel_type in symmetric_relations:
                    # Symmetric relationships: show between any two visible structures
                    style = edge_styles.get(rel_type, {'color': '#999999', 'width': 2, 'dashes': False, 'arrows': None})
                    edge_color = style['color']
                    if request.logical_relations_mode == 'faded' and rel.is_logical:
                        edge_color = hex_to_rgba(edge_color, logical_opacity)
                    relation_label = relationship_metadata.get(rel_type, {}).get(
                        'label', rel_type
                    )
                    edge_label = (
                        f'[{relation_label}]' if rel.is_logical else relation_label
                    )
                    edge_title, edge_symbol = build_edge_tooltip(
                        roi_to_name.get(roi1, f'ROI {roi1}'),
                        roi_to_name.get(roi2, f'ROI {roi2}'),
                        rel_type,
                        rel.is_logical,
                        style,
                    )
                    edges.append(DiagramEdge(
                        from_node=roi1,
                        to_node=roi2,
                        relation_type=rel_type,
                        symbol=edge_symbol,
                        label=edge_label,
                        title=edge_title,
                        color=edge_color,
                        width=style['width'],
                        dashes=style['dashes'],
                        arrows=style['arrows'],
                        is_logical=rel.is_logical,
                    ))
                    symmetric_edges_added += 1
        symmetric_ms = round((time.perf_counter() - symmetric_start) * 1000.0)

        # For directional relationships: check From->To combinations only
        directional_start = time.perf_counter()
        directional_pairs_checked = 0
        directional_edges_added = 0
        for from_roi in col_rois:  # From = Source structures
            for to_roi in row_rois:  # To = Target structures
                directional_pairs_checked += 1
                if from_roi == to_roi:
                    continue  # Skip self-loops

                # Query the relationship in the From->To direction
                # get_relationship(from_roi, to_roi) checks if there's an edge from_roi -> to_roi
                # representing "from_roi [relationship] to_roi"
                rel = structure_set.get_relationship(from_roi, to_roi)
                if rel is None:
                    continue

                try:
                    rel_type_obj = rel.relationship_type
                    if rel_type_obj is None:
                        logger.warning(f'Relationship type is None for {from_roi}->{to_roi}')
                        continue
                    rel_type = rel_type_obj.relation_type
                except (AttributeError, KeyError) as e:
                    logger.error(f'Error getting relationship type for {from_roi}->{to_roi}: {e}')
                    continue

                logger.debug(f'Directional ({from_roi}->{to_roi}): type={rel_type}, is_logical={rel.is_logical}')
                if not should_include_logical(rel):
                    logger.debug(f'  Filtering out {rel_type} relationship ({from_roi}->{to_roi})')
                    continue

                logger.debug(f'  INCLUDING {rel_type} relationship ({from_roi}->{to_roi})')

                # Debug logging
                logger.debug('Checking from_roi=%s to to_roi=%s: relationship=%s', from_roi, to_roi, rel_type)

                if rel_type == 'EQUAL':
                    continue
                if rel_type == 'DISJOINT' and not request.show_disjoint:
                    continue
                if rel_type == 'UNKNOWN' and not request.show_unknown:
                    continue

                if rel_type not in symmetric_relations:
                    # Directional relationship: from_roi [rel_type] to_roi
                    style = edge_styles.get(rel_type, {'color': '#999999', 'width': 2, 'dashes': False, 'arrows': None})
                    edge_color = style['color']
                    if request.logical_relations_mode == 'faded' and rel.is_logical:
                        edge_color = hex_to_rgba(edge_color, logical_opacity)
                    relation_label = relationship_metadata.get(rel_type, {}).get(
                        'label', rel_type
                    )
                    edge_label = (
                        f'[{relation_label}]' if rel.is_logical else relation_label
                    )
                    edge_title, edge_symbol = build_edge_tooltip(
                        roi_to_name.get(from_roi, f'ROI {from_roi}'),
                        roi_to_name.get(to_roi, f'ROI {to_roi}'),
                        rel_type,
                        rel.is_logical,
                        style,
                    )
                    edges.append(DiagramEdge(
                        from_node=from_roi,
                        to_node=to_roi,
                        relation_type=rel_type,
                        symbol=edge_symbol,
                        label=edge_label,
                        title=edge_title,
                        color=edge_color,
                        width=style['width'],
                        dashes=style['dashes'],
                        arrows=style['arrows'],
                        is_logical=rel.is_logical,
                    ))
                    directional_edges_added += 1
        directional_ms = round((time.perf_counter() - directional_start) * 1000.0)
        edge_build_ms = round((time.perf_counter() - edge_build_start) * 1000.0)

        total_ms = round((time.perf_counter() - request_start) * 1000.0)
        await connection_manager.send_status_line(
            request.session_id,
            'loading_results',
            'backend',
            f'Diagram data ready ({len(nodes)} nodes, {len(edges)} edges, {total_ms} ms).',
        )
        global _timing_request_counter
        _timing_request_counter += 1
        _should_log_timing = (
            total_ms > _TIMING_THRESHOLD_MS
            or (_TIMING_SAMPLE_N > 0
                and (_timing_request_counter % _TIMING_SAMPLE_N) == 0)
        )
        if _should_log_timing:
            logger.info(
                'DIAGRAM_TIMING session=%s session_load_ms=%s config_ms=%s '
                'summary_ms=%s color_extract_ms=%s node_build_ms=%s '
                'edge_build_ms=%s symmetric_ms=%s directional_ms=%s '
                'symmetric_pairs=%s directional_pairs=%s '
                'symmetric_edges=%s directional_edges=%s total_ms=%s '
                'nodes=%s edges=%s',
                request.session_id,
                session_load_ms,
                config_ms,
                summary_ms,
                color_extract_ms,
                node_build_ms,
                edge_build_ms,
                symmetric_ms,
                directional_ms,
                symmetric_pairs_checked,
                directional_pairs_checked,
                symmetric_edges_added,
                directional_edges_added,
                total_ms,
                len(nodes),
                len(edges),
            )
        logger.info(f'Diagram response: {len(nodes)} nodes, {len(edges)} edges (mode={request.logical_relations_mode})')
        return DiagramResponse(nodes=nodes, edges=edges)

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error generating diagram: %s', e)
        raise HTTPException(status_code=500, detail=f'Failed to generate diagram: {str(e)}')


class PlotRequest(BaseModel):
    session_id: str
    roi_list: List[int]
    slice_index: float
    include_interpolated_slices: bool = False
    plot_mode: str = 'contour'
    relationship_overlay: str = 'none'
    show_legend: bool = True
    add_axis: bool = False
    tolerance: float = 0.0


def _get_plottable_slices_for_roi(
    structure_set: StructureSet,
    roi: int,
    include_interpolated_slices: bool,
) -> List[float]:
    """Return valid plotting slices for one ROI.

    Args:
        structure_set (StructureSet): The loaded structure set.
        roi (int): ROI number.
        include_interpolated_slices (bool): Whether interpolated slices are allowed.

    Returns:
        List[float]: Sorted slice indices that can be plotted for the ROI.
    """
    structure = structure_set.structures[roi]
    if structure.region_table.empty:
        return []

    region_table = structure.region_table
    valid_rows = region_table.loc[~region_table.Empty]

    if not include_interpolated_slices and 'Interpolated' in valid_rows.columns:
        valid_rows = valid_rows.loc[~valid_rows.Interpolated]

    if valid_rows.empty:
        return []

    return sorted(valid_rows['SliceIndex'].unique().tolist())


def _slice_matches_any(candidate: float, available_slices: List[float]) -> bool:
    """Check if candidate slice index matches an available value.

    Args:
        candidate (float): Requested slice index.
        available_slices (List[float]): Valid slice indexes.

    Returns:
        bool: True when the candidate is found in available slices.
    """
    return any(abs(candidate - item) < 1e-6 for item in available_slices)


def _make_plot_cache_key(request: PlotRequest, structure_set) -> tuple:
    '''Build a stable cache key for identical contour plot requests.'''
    return (
        request.session_id,
        id(structure_set),
        tuple(int(roi) for roi in request.roi_list),
        round(float(request.slice_index), 4),
        bool(request.include_interpolated_slices),
        request.plot_mode,
        request.relationship_overlay,
        bool(request.show_legend),
        bool(request.add_axis),
        round(float(request.tolerance), 4),
    )


def _get_cached_plot_bytes(cache_key: tuple) -> Optional[bytes]:
    '''Return cached plot bytes when available.'''
    cached_bytes = _plot_response_cache.get(cache_key)
    if cached_bytes is None:
        return None
    _plot_response_cache.move_to_end(cache_key)
    return cached_bytes


def _store_cached_plot_bytes(cache_key: tuple, image_bytes: bytes) -> None:
    '''Store plot bytes in a small bounded LRU cache.'''
    _plot_response_cache[cache_key] = image_bytes
    _plot_response_cache.move_to_end(cache_key)
    while len(_plot_response_cache) > PLOT_RESPONSE_CACHE_MAX_ENTRIES:
        _plot_response_cache.popitem(last=False)


@app.post('/api/plot-contours')
async def plot_contours(request: PlotRequest):
    '''Generate a contour plot for selected structures on a specific slice.

    Args:
        request (PlotRequest): Contains session_id, roi_list (one or more
            ROIs), slice_index, and interpolated-slice preference.

    Returns:
        StreamingResponse: PNG image of the contour plot.
    '''
    try:
        session_data = session_manager.load_session(request.session_id, touch=False)
        if session_data is None:
            raise HTTPException(status_code=404, detail='Session expired, please re-upload')

        if session_data.structure_set is None:
            raise HTTPException(status_code=400, detail='Structures not yet processed')

        structure_set = session_data.structure_set

        # Validate inputs
        if request.plot_mode not in {'contour', 'relationship'}:
            raise HTTPException(status_code=400, detail='plot_mode must be contour or relationship')

        allowed_overlays = {
            'none',
            'single_structure',
            'third_structure',
            'all_outlines',
            'structure_1',
            'structure_2',
            'intersection_ab',
            'a_minus_b',
            'a_xor_b',
            'intersection_vs_c',
            'union_vs_c',
            'xor_vs_c',
            'difference_vs_c',
        }
        if request.relationship_overlay not in allowed_overlays:
            raise HTTPException(status_code=400, detail='relationship_overlay is invalid')

        if request.tolerance < 0:
            raise HTTPException(status_code=400, detail='tolerance must be >= 0')

        if request.plot_mode == 'contour':
            if len(request.roi_list) == 0:
                raise HTTPException(status_code=400, detail='Must provide at least 1 ROI number')
        else:
            if len(request.roi_list) == 0:
                raise HTTPException(
                    status_code=400,
                    detail='Relationship mode requires at least 1 ROI number',
                )

            if request.relationship_overlay == 'single_structure':
                if len(request.roi_list) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail='single_structure overlay requires exactly 1 ROI number',
                    )
            else:
                if len(request.roi_list) < 2:
                    raise HTTPException(
                        status_code=400,
                        detail='Relationship mode requires at least 2 ROI numbers',
                    )

            if (
                request.relationship_overlay != 'all_outlines'
                and len(request.roi_list) > 3
            ):
                raise HTTPException(
                    status_code=400,
                    detail='Relationship mode supports more than 3 ROIs only for all_outlines',
                )

            if request.relationship_overlay in {
                'third_structure',
                'all_outlines',
                'intersection_vs_c',
                'union_vs_c',
                'xor_vs_c',
                'difference_vs_c',
            } and len(request.roi_list) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f'{request.relationship_overlay} overlay requires a third ROI',
                )

        for roi in request.roi_list:
            if roi not in structure_set.structures:
                raise HTTPException(status_code=400, detail=f'Invalid ROI selected: {roi}')

        # Default behavior excludes interpolated slices unless explicitly enabled.
        slice_pool = set()
        for roi in request.roi_list:
            slice_pool.update(
                _get_plottable_slices_for_roi(
                    structure_set=structure_set,
                    roi=roi,
                    include_interpolated_slices=request.include_interpolated_slices,
                )
            )
        available_slices = sorted(slice_pool)

        if not available_slices:
            detail = (
                'No slices are available for the selected structures with the '
                'current interpolated-slice setting.'
            )
            raise HTTPException(status_code=400, detail=detail)

        if not _slice_matches_any(request.slice_index, available_slices):
            detail = (
                'Selected slice is not available for plotting with the current '
                'interpolated-slice setting.'
            )
            raise HTTPException(status_code=400, detail=detail)

        cache_key = _make_plot_cache_key(request, structure_set)
        cached_image = _get_cached_plot_bytes(cache_key)
        if cached_image is not None:
            return StreamingResponse(io.BytesIO(cached_image), media_type='image/png')

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_roi_slice(
            structure_set=structure_set,
            slice_index=request.slice_index,
            roi_list=request.roi_list,
            axes=ax,
            add_axis=request.add_axis,
            tolerance=request.tolerance,
            plot_mode=request.plot_mode,
            relationship_overlay=request.relationship_overlay,
            show_legend=request.show_legend,
        )

        # Save plot to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        image_bytes = buf.getvalue()
        _store_cached_plot_bytes(cache_key, image_bytes)
        plt.close(fig)

        return StreamingResponse(io.BytesIO(image_bytes), media_type='image/png')

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error generating contour plot: %s', e)
        raise HTTPException(status_code=500, detail=f'Failed to generate plot: {str(e)}')


@app.websocket('/ws/{session_id}')
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    '''WebSocket endpoint for real-time progress updates.

    Args:
        websocket (WebSocket): The WebSocket connection.
        session_id (str): The session ID.
    '''
    # Verify session exists
    session_data = session_manager.load_session(session_id)
    if session_data is None:
        await websocket.accept()
        await connection_manager.send_error(session_id, 'Session expired, please re-upload')
        await websocket.close()
        return

    await connection_manager.connect(session_id, websocket)

    try:
        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Client can send ping messages to keep connection alive
            if data == 'ping':
                await websocket.send_text('pong')

    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
        logger.info('Client disconnected from session %s', session_id)
    except Exception as e:
        logger.error('WebSocket error for session %s: %s', session_id, e)
        connection_manager.disconnect(session_id)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
