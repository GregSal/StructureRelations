'''FastAPI web application for StructureRelations DICOM analysis.

This application provides a web interface for uploading DICOM RT files,
selecting structures, and analyzing spatial relationships with customizable
matrix visualization.
'''
import sys
import logging
import uuid
import asyncio
import tempfile
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dicom import DicomStructureFile
from structure_set import StructureSet
from contour_plotting import plot_roi_slice
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
    structure_slices: dict
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
    label: str
    color: str
    width: int
    dashes: bool
    arrows: Optional[str] = None


class DiagramResponse(BaseModel):
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]


class ProcessRequest(BaseModel):
    session_id: str
    selected_rois: Optional[List[int]] = None


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
    '''
    config_file = Path(__file__).parent.parent / 'relationship_symbols.json'

    try:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        else:
            # Return default config if file doesn't exist
            logger.warning('Config file not found: %s', config_file)
            return get_default_symbol_config()
    except (IOError, json.JSONDecodeError) as e:
        logger.error('Error loading symbol config: %s', e)
        return get_default_symbol_config()


def get_default_symbol_config():
    '''Get the default relationship symbol configuration.

    Returns:
        dict: Default configuration for relationship symbols.
    '''
    return {
        "description": "Default relationship symbols and colors",
        "version": "1.0",
        "relationships": {
            "CONTAINS": {"symbol": "⊂", "label": "Contains", "description": "Structure A fully encloses structure B", "color": "#10b981"},
            "OVERLAPS": {"symbol": "∩", "label": "Overlaps", "description": "Structures share common volume", "color": "#ef4444"},
            "BORDERS": {"symbol": "|", "label": "Borders", "description": "Structures touch at boundaries", "color": "#3b82f6"},
            "SURROUNDS": {"symbol": "○", "label": "Surrounds", "description": "Structure B is within a hole in A", "color": "#8b5cf6"},
            "SHELTERS": {"symbol": "△", "label": "Shelters", "description": "B within convex hull of A, not touching", "color": "#f59e0b"},
            "PARTITION": {"symbol": "⊕", "label": "Partition", "description": "Structures partition space between them", "color": "#ec4899"},
            "CONFINES": {"symbol": "⊏", "label": "Confines", "description": "B contacts inner surface of A", "color": "#06b6d4"},
            "DISJOINT": {"symbol": "∅", "label": "Disjoint", "description": "Structures are completely separated", "color": "#6b7280"},
            "EQUALS": {"symbol": "=", "label": "Equals", "description": "Same structure", "color": "#000000"},
            "UNKNOWN": {"symbol": "?", "label": "Unknown", "description": "Relationship not determined", "color": "#9ca3af"}
        }
    }


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

        # Get patient info from structure set info
        structure_set_info = dicom_file.get_structure_set_info()
        patient_info = {
            'patient_id': structure_set_info.get('PatientID', ''),
            'patient_name': structure_set_info.get('PatientName', ''),
            'structure_set': structure_set_info.get('StructureSet', ''),
            'study_id': structure_set_info.get('StudyID', '')
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


@app.post('/api/process')
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

    # Start processing task asynchronously
    asyncio.create_task(
        process_structure_set(
            request.session_id,
            session_data.dicom_file_path,
            request.selected_rois
        )
    )

    return {'message': 'Processing started', 'session_id': request.session_id}


async def process_structure_set(session_id: str, dicom_file_path: str, selected_rois: Optional[List[int]]):
    '''Background task to process DICOM file and calculate relationships.

    Args:
        session_id (str): The session ID.
        dicom_file_path (str): Path to the DICOM file.
        selected_rois (List[int], optional): List of ROI numbers to process.
    '''
    try:
        # Send initial progress
        disk_info = session_manager.get_disk_usage_info()
        await connection_manager.send_progress(
            session_id, 'parsing_dicom', 0, '', 'Loading DICOM file...', disk_info['usage_mb']
        )

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

        await connection_manager.send_progress(
            session_id, 'parsing_dicom', 20, '', 'Parsing structures...', disk_info['usage_mb']
        )

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
            structure_set = StructureSet(dicom_structure_file=dicom_file)
        except Exception as e:
            logger.error('Structure set creation error in session %s: %s', session_id, e, exc_info=True)
            await connection_manager.send_error(session_id, f'Failed to create structure set: {str(e)}')
            return

        await connection_manager.send_progress(
            session_id, 'building_graphs', 40, '', 'Building contour graphs...', disk_info['usage_mb']
        )

        # Process each structure
        total_structures = len(structure_set.structures)
        for idx, (roi, structure) in enumerate(structure_set.structures.items()):
            if selected_rois and roi not in selected_rois:
                continue

            progress = 40 + int((idx / total_structures) * 30)
            await connection_manager.send_progress(
                session_id, 'building_graphs', progress,
                structure.name, f'Processing {structure.name}...', disk_info['usage_mb']
            )

        await connection_manager.send_progress(
            session_id, 'calculating_relationships', 70, '', 'Calculating relationships...', disk_info['usage_mb']
        )

        # Calculate relationships with error handling
        try:
            structure_set.finalize()
        except Exception as e:
            logger.error('Relationship calculation error in session %s: %s', session_id, e, exc_info=True)
            await connection_manager.send_error(session_id, f'Failed to calculate relationships: {str(e)}')
            return

        await connection_manager.send_progress(
            session_id, 'calculating_relationships', 100, '', 'Complete!', disk_info['usage_mb']
        )

        # Save completed session BEFORE sending complete message
        # Use dedicated method to avoid race conditions with last_accessed updates
        if not session_manager.update_session_structure_set(session_id, structure_set):
            logger.error('Failed to save structure set to session %s', session_id)
            await connection_manager.send_error(session_id, 'Failed to save results')
            return

        # Small delay to ensure file system flushes the pickle file
        await asyncio.sleep(0.1)

        # Send completion message (frontend will immediately request matrix)
        await connection_manager.send_complete(session_id, 'Processing complete')

        logger.info('Completed processing for session %s', session_id)

    except Exception as e:
        logger.error('Unexpected error processing session %s: %s', session_id, e, exc_info=True)
        await connection_manager.send_error(session_id, f'Unexpected processing error: {str(e)}')


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

    session_data = session_manager.load_session(request.session_id)
    if session_data is None:
        logger.error(f'Session {request.session_id} not found or expired')
        raise HTTPException(status_code=404, detail='Session expired, please re-upload')

    if session_data.structure_set is None:
        logger.error(f'Session {request.session_id} has no structure_set')
        raise HTTPException(status_code=400, detail='Structures not yet processed')

    try:
        # Build visible_rois list from row and col rois for limited mode
        visible_rois = None
        if request.logical_relations_mode == 'limited':
            visible_rois = []
            if request.row_rois:
                visible_rois.extend(request.row_rois)
            if request.col_rois:
                visible_rois.extend(request.col_rois)
            if visible_rois:
                visible_rois = list(set(visible_rois))  # Remove duplicates

        # Get matrix as dictionary
        logger.info(f'Generating matrix for session {request.session_id}')
        matrix_dict = session_data.structure_set.to_dict(
            row_rois=request.row_rois,
            col_rois=request.col_rois,
            use_symbols=request.use_symbols,
            logical_relations_mode=request.logical_relations_mode,
            visible_rois=visible_rois
        )

        logger.info(f'Matrix dict keys: {matrix_dict.keys()}')
        logger.info(f'Matrix dict types: {[(k, type(v).__name__) for k, v in matrix_dict.items()]}')

        # Create response
        try:
            response = MatrixResponse(**matrix_dict)
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
        session_data = session_manager.load_session(request.session_id)
        if session_data is None:
            raise HTTPException(status_code=404, detail='Session expired, please re-upload')

        if session_data.structure_set is None:
            raise HTTPException(status_code=400, detail='Structures not yet processed')

        structure_set = session_data.structure_set

        # Define node shapes by DICOM type
        shape_map = {
            'GTV': 'star',
            'CTV': 'hexagon',
            'PTV': 'diamond',
            'EXTERNAL': 'box',
            'ORGAN': 'ellipse',
            'AVOIDANCE': 'triangle',
            'BOLUS': 'dot',
            'SUPPORT': 'square',
            'FIXATION': 'triangleDown'
        }

        # Define edge styles by relationship type
        edge_styles = {
            'CONTAINS': {'color': '#00CED1', 'width': 4, 'dashes': False, 'arrows': 'to'},
            'WITHIN': {'color': '#00CED1', 'width': 4, 'dashes': False, 'arrows': 'to'},
            'OVERLAPS': {'color': '#FF6347', 'width': 5, 'dashes': False, 'arrows': None},
            'BORDERS': {'color': '#32CD32', 'width': 3, 'dashes': True, 'arrows': None},
            'SURROUNDS': {'color': '#4169E1', 'width': 3, 'dashes': False, 'arrows': 'to'},
            'ENCLOSED': {'color': '#4169E1', 'width': 3, 'dashes': False, 'arrows': 'to'},
            'SHELTERS': {'color': '#9370DB', 'width': 2, 'dashes': True, 'arrows': 'to'},
            'SHELTERED': {'color': '#9370DB', 'width': 2, 'dashes': True, 'arrows': 'to'},
            'PARTITIONED': {'color': '#FFD700', 'width': 4, 'dashes': False, 'arrows': 'to'},
            'PARTITIONS': {'color': '#FFD700', 'width': 4, 'dashes': False, 'arrows': 'to'},
            'CONFINES': {'color': '#FF1493', 'width': 3, 'dashes': False, 'arrows': 'to'},
            'CONFINED': {'color': '#FF1493', 'width': 3, 'dashes': False, 'arrows': 'to'},
            'DISJOINT': {'color': '#808080', 'width': 1, 'dashes': True, 'arrows': None},
            'EQUALS': {'color': '#FF0000', 'width': 5, 'dashes': False, 'arrows': 'to;from'},
            'UNKNOWN': {'color': '#999999', 'width': 1, 'dashes': True, 'arrows': None}
        }

        config_file = Path(__file__).parent.parent / 'relationship_symbols.json'
        logical_opacity = 0.5
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    transparency = config.get('logical_relationships', {}).get(
                        'transparency', 50
                    )
                    logical_opacity = max(0.0, min(1.0, transparency / 100.0))
            except (IOError, json.JSONDecodeError):
                logger.warning('Failed to load logical relationship config')

        def hex_to_rgba(color_hex: str, alpha: float) -> str:
            color_hex = color_hex.lstrip('#')
            if len(color_hex) != 6:
                return color_hex
            red = int(color_hex[0:2], 16)
            green = int(color_hex[2:4], 16)
            blue = int(color_hex[4:6], 16)
            return f'rgba({red}, {green}, {blue}, {alpha:.3f})'

        # Get summary data
        summary_df = structure_set.summary()

        # Extract colors from DICOM file
        colors = {}
        if structure_set.dicom_structure_file and hasattr(structure_set.dicom_structure_file, 'dataset'):
            try:
                for roi_contour in structure_set.dicom_structure_file.dataset.ROIContourSequence:
                    roi_num = int(roi_contour.ReferencedROINumber)
                    if hasattr(roi_contour, 'ROIDisplayColor'):
                        colors[roi_num] = [int(c) for c in roi_contour.ROIDisplayColor]
            except AttributeError:
                pass

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
            tooltip = f"{name} (ROI {roi})\\nType: {dicom_type}\\nVolume: {volume:.2f} cm³\\nRegions: {num_regions}"

            nodes.append(DiagramNode(
                id=roi,
                label=name,
                color=color_hex,
                shape=shape_map.get(dicom_type, 'ellipse'),
                title=tooltip
            ))

        # Build edges from relationship matrix
        edges = []

        # Define symmetric relationships (no direction)
        symmetric_relations = {'OVERLAPS', 'BORDERS', 'DISJOINT', 'EQUALS'}

        # For symmetric relationships: check all visible pairs once
        all_visible_rois = sorted(visible_rois)
        for i, roi1 in enumerate(all_visible_rois):
            for roi2 in all_visible_rois[i+1:]:
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

                if rel_type == 'EQUALS':
                    continue
                if rel_type == 'DISJOINT' and not request.show_disjoint:
                    continue

                if rel_type in symmetric_relations:
                    # Symmetric relationships: show between any two visible structures
                    style = edge_styles.get(rel_type, {'color': '#999999', 'width': 2, 'dashes': False, 'arrows': None})
                    edge_color = style['color']
                    if request.logical_relations_mode == 'faded' and rel.is_logical:
                        edge_color = hex_to_rgba(edge_color, logical_opacity)
                    edges.append(DiagramEdge(
                        from_node=roi1,
                        to_node=roi2,
                        label=rel_type,
                        color=edge_color,
                        width=style['width'],
                        dashes=style['dashes'],
                        arrows=style['arrows']
                    ))

        # For directional relationships: check From->To combinations only
        for from_roi in col_rois:  # From = Source structures
            for to_roi in row_rois:  # To = Target structures
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

                if rel_type == 'EQUALS':
                    continue
                if rel_type == 'DISJOINT' and not request.show_disjoint:
                    continue

                if rel_type not in symmetric_relations:
                    # Directional relationship: from_roi [rel_type] to_roi
                    style = edge_styles.get(rel_type, {'color': '#999999', 'width': 2, 'dashes': False, 'arrows': None})
                    edge_color = style['color']
                    if request.logical_relations_mode == 'faded' and rel.is_logical:
                        edge_color = hex_to_rgba(edge_color, logical_opacity)
                    edges.append(DiagramEdge(
                        from_node=from_roi,
                        to_node=to_roi,
                        label=rel_type,
                        color=edge_color,
                            width=style['width'],
                            dashes=style['dashes'],
                            arrows=style['arrows']
                        ))

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


@app.post('/api/plot-contours')
async def plot_contours(request: PlotRequest):
    '''Generate a contour plot for selected structures on a specific slice.

    Args:
        request (PlotRequest): Contains session_id, roi_list (1-2 ROIs), and slice_index.

    Returns:
        StreamingResponse: PNG image of the contour plot.
    '''
    try:
        session_data = session_manager.load_session(request.session_id)
        if session_data is None:
            raise HTTPException(status_code=404, detail='Session expired, please re-upload')

        if session_data.structure_set is None:
            raise HTTPException(status_code=400, detail='Structures not yet processed')

        structure_set = session_data.structure_set

        # Validate inputs
        if len(request.roi_list) == 0 or len(request.roi_list) > 2:
            raise HTTPException(status_code=400, detail='Must provide 1 or 2 ROI numbers')

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_roi_slice(
            structure_set=structure_set,
            slice_index=request.slice_index,
            roi_list=request.roi_list,
            axes=ax,
            add_axis=False,
            tolerance=0.1
        )

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return StreamingResponse(buf, media_type='image/png')

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
