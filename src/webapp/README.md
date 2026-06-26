# StructureRelations Web Application

## Quick Start

### Prerequisites
- Anaconda/Miniconda with `StructureRelations` environment configured
- Python 3.10+

### Installation

1. Activate the conda environment:
```bash
conda activate StructureRelations
```

2. Install web application dependencies:
```bash
pip install -r src/webapp/requirements.txt
```

### Running the Application

#### Windows
Double-click `start_webapp.bat` or run:
```bash
start_webapp.bat
```

#### Manual Start
```bat
rem set WORKSPACE_FOLDER="..."
cd "%WORKSPACE_FOLDER%"
cd src/webapp

REM Activate conda environment
CALL ...\anaconda3\Scripts\activate.bat D:\anaconda3
CALL conda activate StructureRelations

REM Start FastAPI server
CALL uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
```

The application will be available at: http://localhost:8000

### Usage Workflow

1. **Upload DICOM File**
   - Drag and drop or click to select a DICOM RT Structure Set file (.dcm)
   - Monitor disk usage indicator in header

2. **Select Structures**
   - Review patient information
   - Check/uncheck structures to include in analysis
   - Use "Select All" / "Select None" buttons for convenience

3. **Process**
   - Click "Start Processing"
   - Monitor real-time progress via WebSocket
   - Processing stages: Parsing DICOM → Building Graphs → Calculating Relationships

4. **View Results**
   - Relationship matrix displays with default symbol notation
   - Drag structures between "Available" and "Selected" lists to customize matrix axes
   - Rows and columns can display different structures
   - Toggle between symbols (?, =, ⊂, etc.) and labels (UNKNOWN, EQUAL, CONTAINS, etc.)
   - Click "Update Matrix" to refresh display

5. **Export**
   - Export relationship matrix as CSV, Excel, or JSON
   - Files include complete relationship data

### URL Automation (Stage 3)

The web app supports URL-driven automation for direct diagram generation:

- Endpoint: `/api/automation/diagram-image`
- Required query parameter:
   - `dicom_path`: Full path to a DICOM RT structure file
- Optional query parameters:
   - `filter_path`: Full path to a custom `structure_filter_rules.json`
   - `output_path`: Full path to save the resulting image file
   - `output_dir`: Directory path to save the resulting image file
   - `image_format`: `png` (default), `jpeg`, `jpg`, or `pdf`

Example:

```text
http://localhost:8000/api/automation/diagram-image?dicom_path=D:/data/RS.case.dcm&filter_path=D:/data/filters.json&output_dir=D:/exports&image_format=png
```

You can also call the root URL with `direct_image=true` to return the image
directly from the main web app route:

```text
http://localhost:8000/?direct_image=true&dicom_path=D:/data/RS.case.dcm&filter_path=D:/data/filters.json&output_path=D:/exports/case_diagram.png
```

### Automated vis-network Layout Tuning

For iterative layout tuning without manual UI interaction, use the headless
layout tuner:

```bash
python src/webapp/layout_tuner.py --dicom-path tests/RS.GJS_Struct_Tests.Relations.dcm
```

Run fully unattended (starts/stops uvicorn automatically):

```bash
python src/webapp/layout_tuner.py --start-server --dicom-path tests/RS.GJS_Struct_Tests.Relations.dcm
```

What it does automatically:
- opens the web app in headless Chrome
- uploads the DICOM file and runs processing
- applies a set of layout-rule variants to vis-network
- scores each variant with deterministic layout metrics
   (edge-crossing count, edge-length variance, node spacing penalty)
- writes screenshots and a JSON report under
   `src/webapp/layout_tuner_output/`

Optional flags:
- `--apply-best`: write best discovered `layout_rules` back to
   `src/webapp/config/diagram_settings.json`
- `--start-server`: start/stop uvicorn automatically for a single tuning run
- `--base-url http://127.0.0.1:8000`: target an existing server
- `--no-headless`: run with visible browser window
- `--timeout-seconds 300`: override processing timeout

### Architecture

**Backend (FastAPI)**
- `main.py`: REST API endpoints and WebSocket handler
- `session_manager.py`: Session persistence with disk usage enforcement
- `websocket_manager.py`: Real-time progress updates

**Frontend**
- `static/index.html`: Multi-stage workflow interface
- `static/css/styles.css`: Responsive styling
- `static/js/app.js`: WebSocket client and UI logic
- SortableJS: Drag-and-drop structure ordering

**Session Management**
- Expiration: 2 hours of inactivity
- Disk limit: 250 MB (warning at 100 MB)
- Cleanup: Expired sessions deleted first, then oldest by LRU
- In-memory session cache: configurable max cached sessions (`max_cached_sessions`)
- Persistence: Sessions saved as pickle files in `sessions/` directory

### Testing

Run Selenium tests (requires Chrome/Chromium):
```bash
pytest tests/test_webapp_selenium.py -v
```

Test coverage includes:
- File upload and preview
- Structure selection
- Processing workflow
- Matrix display and configuration
- Symbol/label toggling
- Export functionality
- Session management
- Error handling

### Symbol Mappings

| Symbol | Relationship Type | Description |
|--------|------------------|-------------|
| ?      | UNKNOWN          | Relationship not determined |
| =      | EQUAL            | Identical structures |
| ⊂      | CONTAINS         | One inside the other |
| ∩      | OVERLAPS         | Partial intersection |
| ⊕      | PARTITION        | Adjacent without overlap |
| \|     | BORDERS          | Touching boundaries |
| ○      | SURROUNDS        | Inside a hole |
| △      | SHELTERS         | Within convex hull |
| ∅      | DISJOINT         | No intersection |
| ⊏      | CONFINES         | Confined relationship |

### Troubleshooting

**Port already in use:**
```bash
# Find process using port 8000
netstat -ano | findstr :8000
# Kill process by PID
taskkill /PID <pid> /F
```

**WebSocket connection fails:**
- Ensure firewall allows localhost connections
- Check browser console for errors
- Verify uvicorn started successfully

**Session not persisting:**
- Check `src/webapp/sessions/` directory exists
- Verify write permissions
- Review disk usage (may exceed 250 MB limit)

**Selenium tests fail:**
- Ensure Chrome/Chromium installed
- Check ChromeDriver version matches Chrome version
- Run tests with `--tb=short` for debugging

### Configuration

Edit `src/webapp/main.py` to customize:
- `SESSION_EXPIRATION`: Session timeout (default: 2 hours)
- `MAX_DISK_USAGE_MB`: Maximum disk usage (default: 250 MB)
- `DISK_WARNING_THRESHOLD_MB`: Warning threshold (default: 100 MB)
- `CLEANUP_INTERVAL_MINUTES`: Background cleanup frequency (default: 30 minutes)

Session storage runtime settings can also be configured in
`src/webapp/config/webapp_settings.json` under `session_storage`:
- `max_disk_size_mb`
- `warn_disk_size_mb`
- `session_expiry_hours`
- `max_cached_sessions`

`max_cached_sessions` controls how many sessions stay resident in memory at
once (lower values reduce RAM usage on constrained machines).

### Security Notes

- Application designed for secure internal environments
- Pickle serialization used for session data (not suitable for untrusted data)
- No authentication implemented (add if exposing externally)
- CORS disabled by default (enable if needed for API access)

### Development

**Hot reload enabled:** Code changes automatically restart server

**Debug mode:** Set `DEBUG=True` in `main.py` for verbose logging

**Add new endpoints:** Follow FastAPI patterns in `main.py`

**Modify frontend:** Edit HTML/CSS/JS in `static/` directory
