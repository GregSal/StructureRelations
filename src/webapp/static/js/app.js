class WebAppClient {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.selectedStructures = new Set();
        this.sortableRows = null;
        this.sortableColumns = null;
        this.symbolConfig = null;  // Will store loaded config
        this.network = null;  // vis-network instance
        this.plotAbortController = null;  // Track current plot request

        this.loadSymbolConfig();  // Load config on initialization
        this.initializeEventListeners();
    }

    async loadSymbolConfig() {
        try {
            const response = await fetch('/api/config/symbols');
            if (response.ok) {
                this.symbolConfig = await response.json();
                console.log('Loaded symbol configuration:', this.symbolConfig);
                this.applySymbolColors();  // Apply custom colors
            } else {
                console.warn('Failed to load symbol config, using defaults');
            }
        } catch (error) {
            console.error('Error loading symbol config:', error);
        }
    }

    applySymbolColors() {
        // Apply custom colors from config to CSS custom properties
        if (!this.symbolConfig || !this.symbolConfig.relationships) return;

        const root = document.documentElement;
        for (const [relType, config] of Object.entries(this.symbolConfig.relationships)) {
            if (config.color) {
                root.style.setProperty(`--rel-${relType.toLowerCase()}-color`, config.color);
            }
        }
    }

    // Helper to convert RGB array to CSS color string
    rgbToColor(colorValue) {
        if (!colorValue) return '#cccccc';
        if (typeof colorValue === 'string') return colorValue;
        if (Array.isArray(colorValue) && colorValue.length >= 3) {
            return `rgb(${colorValue[0]}, ${colorValue[1]}, ${colorValue[2]})`;
        }
        return '#cccccc';
    }

    initializeEventListeners() {
        // Upload area
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Structure selection - use correct IDs from HTML
        document.getElementById('selectAllBtn').addEventListener('click', () => {
            this.selectAllStructures(true);
        });
        document.getElementById('selectNoneBtn').addEventListener('click', () => {
            this.selectAllStructures(false);
        });
        document.getElementById('processBtn').addEventListener('click', () => {
            this.startProcessing();
        });

        // Matrix controls - use correct IDs from HTML
        document.getElementById('useSymbolsToggle').addEventListener('change', () => {
            this.updateMatrix();
        });
        document.getElementById('updateMatrixBtn').addEventListener('click', () => {
            this.updateMatrix();
        });

        // Export - use correct IDs from HTML
        document.getElementById('exportCsvBtn').addEventListener('click', () => {
            this.exportMatrix('csv');
        });
        document.getElementById('exportExcelBtn').addEventListener('click', () => {
            this.exportMatrix('excel');
        });
        document.getElementById('exportJsonBtn').addEventListener('click', () => {
            this.exportMatrix('json');
        });

        // Legend toggle
        document.getElementById('legendToggle').addEventListener('click', () => {
            this.toggleLegend();
        });

        // Analysis config toggle
        document.getElementById('analysisConfigToggle').addEventListener('click', () => {
            this.toggleAnalysisConfig();
        });

        // DICOM Type filters
        document.getElementById('rowDicomFilter').addEventListener('change', (e) => {
            this.filterStructuresByType('rows', e.target.value);
        });
        document.getElementById('colDicomFilter').addEventListener('change', (e) => {
            this.filterStructuresByType('columns', e.target.value);
        });

        // Diagram controls
        document.getElementById('refreshDiagramBtn').addEventListener('click', () => {
            this.refreshDiagram();
        });
        document.getElementById('showDisjointToggle').addEventListener('change', () => {
            this.refreshDiagram();
        });
        document.getElementById('showLabelsToggle').addEventListener('change', () => {
            this.toggleEdgeLabels();
        });

        // Select All/None buttons
        document.getElementById('selectAllRowsBtn').addEventListener('click', () => {
            this.selectAllAxisStructures('rows');
        });
        document.getElementById('selectNoneRowsBtn').addEventListener('click', () => {
            this.selectNoneAxisStructures('rows');
        });
        document.getElementById('selectAllColsBtn').addEventListener('click', () => {
            this.selectAllAxisStructures('columns');
        });
        document.getElementById('selectNoneColsBtn').addEventListener('click', () => {
            this.selectNoneAxisStructures('columns');
        });

        // Contour plotting controls
        document.getElementById('plotContoursBtn').addEventListener('click', () => {
            this.plotContours();
        });
        document.getElementById('sliceSlider').addEventListener('input', (e) => {
            this.updateSliceValue(e.target.value);
            this.updateSliderArrows();
            // Auto-update plot when slider moves
            const structure1 = document.getElementById('structure1Select').value;
            if (structure1) {
                this.plotContours();
            }
        });
        document.getElementById('slicePrevBtn').addEventListener('click', () => {
            this.stepSlice(-1);
        });
        document.getElementById('sliceNextBtn').addEventListener('click', () => {
            this.stepSlice(1);
        });

        // Navigation buttons
        document.getElementById('backToUploadBtn').addEventListener('click', () => {
            this.showStage('upload');
        });
        document.getElementById('newAnalysisBtn').addEventListener('click', () => {
            this.showStage('upload');
            this.resetApp();
        });
    }

    async handleFileSelect(file) {
        if (!file.name.endsWith('.dcm')) {
            alert('Please select a DICOM (.dcm) file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            this.sessionId = data.session_id;

            // Update disk warning
            const diskWarning = document.getElementById('diskWarning');
            if (data.disk_warning) {
                diskWarning.style.display = 'block';
                diskWarning.querySelector('p').textContent = data.disk_warning;
            } else {
                diskWarning.style.display = 'none';
            }

            // Connect websocket
            this.connectWebSocket();

            // Load preview
            await this.loadPreview();

            // Show selection stage
            this.showStage('selection');

        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file. Please try again.');
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            this.updateConnectionStatus(true);
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.websocket.onclose = () => {
            this.updateConnectionStatus(false);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }

    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'progress') {
            this.updateProgress(data.stage, data.progress, data.message, data.current_structure);
            if (data.disk_usage_mb !== undefined) {
                document.getElementById('diskUsage').textContent =
                    `Disk: ${data.disk_usage_mb.toFixed(1)} MB`;
            }
        } else if (data.type === 'complete') {
            this.onProcessingComplete();
        } else if (data.type === 'error') {
            alert(data.message);
            this.showStage('upload');
        }
    }

    async loadPreview() {
        try {
            const response = await fetch('/api/preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error('Preview failed');
            }

            const data = await response.json();

            // Update patient info
            const patientInfo = document.getElementById('patientInfo');
            const pInfo = data.patient_info || {};
            patientInfo.innerHTML = `
                <div><strong>Patient ID:</strong> ${pInfo.patient_id || 'N/A'}</div>
                <div><strong>Patient Name:</strong> ${pInfo.patient_name || 'N/A'}</div>
                <div><strong>Structure Set:</strong> ${pInfo.structure_set || 'N/A'}</div>
            `;

            // Populate structures list
            const structuresList = document.getElementById('structuresList');
            structuresList.innerHTML = '';

            data.structures.forEach(struct => {
                // Convert RGB array to CSS color string
                const colorArr = struct.color || [128, 128, 128];
                const colorStr = `rgb(${colorArr[0]}, ${colorArr[1]}, ${colorArr[2]})`;

                // Build display with name, label, and DICOM type
                const name = struct.name || '';
                const label = struct.code_meaning || struct.name;
                const dicomType = struct.dicom_type || '';

                const item = document.createElement('div');
                item.className = 'structure-item';
                item.innerHTML = `
                    <input type="checkbox"
                           data-roi="${struct.roi}"
                           checked>
                    <div class="structure-color"
                         style="background-color: ${colorStr}"></div>
                    <div class="structure-info">
                        <span class="structure-id">${struct.roi}</span>
                        <span class="structure-name">${name}</span>
                        <span class="structure-type" style="color: #666;">${dicomType}</span>
                        <span class="structure-label" style="color: #888; font-size: 0.9em;">${label}</span>
                        <span class="structure-contours">${struct.num_contours} contour${struct.num_contours !== 1 ? 's' : ''}</span>
                    </div>
                `;
                structuresList.appendChild(item);
                this.selectedStructures.add(struct.roi);
            });

            // Add checkbox event listeners
            structuresList.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.addEventListener('change', (e) => {
                    const roi = parseInt(e.target.dataset.roi);
                    if (e.target.checked) {
                        this.selectedStructures.add(roi);
                    } else {
                        this.selectedStructures.delete(roi);
                    }
                });
            });

        } catch (error) {
            console.error('Preview error:', error);
            alert('Failed to load structure preview.');
        }
    }

    selectAllStructures(select) {
        const checkboxes = document.querySelectorAll('#structuresList input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.checked = select;
            const roi = parseInt(cb.dataset.roi);
            if (select) {
                this.selectedStructures.add(roi);
            } else {
                this.selectedStructures.delete(roi);
            }
        });
    }

    async startProcessing() {
        if (this.selectedStructures.size === 0) {
            alert('Please select at least one structure');
            return;
        }

        this.showStage('processing');

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    selected_rois: Array.from(this.selectedStructures)
                })
            });

            if (!response.ok) {
                throw new Error('Processing failed to start');
            }

            // Progress updates will come via WebSocket

        } catch (error) {
            console.error('Processing error:', error);
            alert('Failed to start processing.');
        }
    }

    updateProgress(stage, progress, message, currentStructure) {
        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.getElementById('progressText');
        const progressDetail = document.getElementById('progressDetail');

        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}% - ${stage.replace('_', ' ')}`;

        let detailText = message;
        if (currentStructure) {
            detailText += ` (${currentStructure})`;
        }
        progressDetail.textContent = detailText;
    }

    async onProcessingComplete() {
        // Load structure summary
        try {
            const response = await fetch('/api/matrix', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    row_rois: null,
                    col_rois: null,
                    use_symbols: true
                })
            });

            if (!response.ok) {
                throw new Error('Failed to load matrix');
            }

            const data = await response.json();

            // Store data for sorting and checkbox management
            this.summaryData = data;

            // Populate structure summary
            this.populateStructureSummary(data);

            // Initialize sortable lists
            this.initializeSortableLists(data);

            // Populate contour plot controls
            this.populateContourPlotControls(data);

            // Display initial matrix
            this.displayMatrix(data);

            // Refresh diagram with initial data
            this.refreshDiagram();

            // Show results stage
            this.showStage('results');

        } catch (error) {
            console.error('Results error:', error);
            alert('Failed to load results.');
        }
    }

    populateStructureSummary(data) {
        const summaryBody = document.getElementById('structuresSummaryBody');
        summaryBody.innerHTML = '';

        const allStructures = [...new Set([...data.rows, ...data.columns])];
        allStructures.sort((a, b) => a - b);

        allStructures.forEach(roi => {
            const idx = data.rows.indexOf(roi);
            const name = idx >= 0 ? data.row_names[idx] :
                         data.col_names[data.columns.indexOf(roi)];

            // Get data from dictionaries using ROI as key
            const dicomType = data.dicom_types ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '') : '';
            const codeMeaning = data.code_meanings ? (data.code_meanings[roi] || data.code_meanings[String(roi)] || '') : '';
            const volume = data.volumes ? (data.volumes[roi] || data.volumes[String(roi)] || 0) : 0;
            const numRegions = data.num_regions ? (data.num_regions[roi] || data.num_regions[String(roi)] || 0) : 0;
            const sliceRange = data.slice_ranges ? (data.slice_ranges[roi] || data.slice_ranges[String(roi)] || '') : '';

            // Colors object has string keys in JSON
            const color = this.rgbToColor(data.colors[roi] || data.colors[String(roi)]);

            // Build label with code meaning or name
            let label = codeMeaning || name;

            const row = document.createElement('tr');
            row.dataset.roi = roi;
            row.dataset.name = name;
            row.dataset.type = dicomType;
            row.dataset.label = label;
            row.dataset.volume = volume;
            row.dataset.regions = numRegions;

            row.innerHTML = `
                <td class="checkbox-cell">
                    <input type="checkbox" class="row-checkbox" data-roi="${roi}" checked>
                </td>
                <td class="checkbox-cell">
                    <input type="checkbox" class="col-checkbox" data-roi="${roi}" checked>
                </td>
                <td class="name-cell">${name}</td>
                <td class="roi-cell">${roi}</td>
                <td class="type-cell">${dicomType}</td>
                <td class="label-cell">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="structure-color" style="background-color: ${color}"></div>
                        ${label}
                    </div>
                </td>
                <td class="number-cell">${numRegions}</td>
                <td class="number-cell">${volume.toFixed(2)}</td>
                <td class="slice-cell">${sliceRange}</td>
            `;
            summaryBody.appendChild(row);
        });

        // Add checkbox event listeners
        this.initializeSummaryCheckboxes();

        // Add sort listeners
        this.initializeSorting();
    }

    initializeSummaryCheckboxes() {
        // Row checkboxes
        document.querySelectorAll('.row-checkbox').forEach(cb => {
            cb.addEventListener('change', (e) => {
                this.updateListFromCheckbox(e.target, 'row');
            });
        });

        // Column checkboxes
        document.querySelectorAll('.col-checkbox').forEach(cb => {
            cb.addEventListener('change', (e) => {
                this.updateListFromCheckbox(e.target, 'col');
            });
        });
    }

    updateListFromCheckbox(checkbox, type) {
        const roi = parseInt(checkbox.dataset.roi);
        const isChecked = checkbox.checked;

        const listId = type === 'row' ? 'selectedRowsList' : 'selectedColsList';
        const availableListId = type === 'row' ? 'availableRowsList' : 'availableColsList';

        const selectedList = document.getElementById(listId);
        const availableList = document.getElementById(availableListId);

        // Find the item in either list
        let item = Array.from(selectedList.children).find(child =>
            parseInt(child.dataset.roi) === roi
        );

        if (!item) {
            item = Array.from(availableList.children).find(child =>
                parseInt(child.dataset.roi) === roi
            );
        }

        if (item) {
            if (isChecked && item.parentElement === availableList) {
                // Move from available to selected
                selectedList.appendChild(item);
            } else if (!isChecked && item.parentElement === selectedList) {
                // Move from selected to available
                availableList.appendChild(item);
            }
        }
    }

    initializeSorting() {
        document.querySelectorAll('.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.column;
                this.sortTable(column);
            });
        });
    }

    sortTable(column) {
        const tbody = document.getElementById('structuresSummaryBody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        // Determine sort direction
        const currentSort = this.currentSort || {};
        const ascending = currentSort.column === column ? !currentSort.ascending : true;
        this.currentSort = { column, ascending };

        // Sort rows
        rows.sort((a, b) => {
            let aVal, bVal;

            switch(column) {
                case 'roi':
                    aVal = parseInt(a.dataset.roi);
                    bVal = parseInt(b.dataset.roi);
                    break;
                case 'type':
                    aVal = a.dataset.type.toLowerCase();
                    bVal = b.dataset.type.toLowerCase();
                    break;
                case 'label':
                    aVal = a.dataset.label.toLowerCase();
                    bVal = b.dataset.label.toLowerCase();
                    break;
                case 'volume':
                    aVal = parseFloat(a.dataset.volume);
                    bVal = parseFloat(b.dataset.volume);
                    break;
                case 'regions':
                    aVal = parseInt(a.dataset.regions);
                    bVal = parseInt(b.dataset.regions);
                    break;
                default:
                    return 0;
            }

            if (aVal < bVal) return ascending ? -1 : 1;
            if (aVal > bVal) return ascending ? 1 : -1;
            return 0;
        });

        // Re-append rows in sorted order
        rows.forEach(row => tbody.appendChild(row));

        // Update sort indicators
        document.querySelectorAll('.sortable').forEach(th => {
            th.classList.remove('sorted-asc', 'sorted-desc');
        });
        const sortedHeader = document.querySelector(`.sortable[data-column="${column}"]`);
        sortedHeader.classList.add(ascending ? 'sorted-asc' : 'sorted-desc');
    }

    initializeSortableLists(data) {
        // Populate selected lists (start with all structures selected)
        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');
        const availableRowsList = document.getElementById('availableRowsList');
        const availableColsList = document.getElementById('availableColsList');

        // Clear all lists
        selectedRowsList.innerHTML = '';
        selectedColsList.innerHTML = '';
        availableRowsList.innerHTML = '';
        availableColsList.innerHTML = '';

        // Collect unique DICOM types and create items
        const dicomTypes = new Set();
        const items = [];

        data.rows.forEach((roi, idx) => {
            // Colors object has string keys in JSON
            const color = this.rgbToColor(data.colors[roi] || data.colors[String(roi)]);
            const name = data.row_names[idx];

            // Get data from dictionaries using ROI as key
            const dicomType = data.dicom_types ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '') : '';
            const codeMeaning = data.code_meanings ? (data.code_meanings[roi] || data.code_meanings[String(roi)] || '') : '';

            if (dicomType) {
                dicomTypes.add(dicomType);
            }

            items.push({ roi, name, color, dicomType, codeMeaning });
        });

        // Populate DICOM type filter dropdowns
        this.populateDicomTypeFilters(Array.from(dicomTypes).sort());

        // Sort items by DICOM Type, then by name
        items.sort((a, b) => {
            if (a.dicomType !== b.dicomType) {
                return a.dicomType.localeCompare(b.dicomType);
            }
            return a.name.localeCompare(b.name);
        });

        // Add sorted items to selected lists
        items.forEach(item => {
            const rowItem = this.createSortableItem(item.roi, item.name, item.color, item.dicomType, item.codeMeaning);
            const colItem = this.createSortableItem(item.roi, item.name, item.color, item.dicomType, item.codeMeaning);

            selectedRowsList.appendChild(rowItem);
            selectedColsList.appendChild(colItem);
        });

        // Initialize SortableJS
        this.sortableRows = {
            available: new Sortable(availableRowsList, {
                group: 'rows',
                animation: 150,
                ghostClass: 'dragging'
            }),
            selected: new Sortable(selectedRowsList, {
                group: 'rows',
                animation: 150,
                ghostClass: 'dragging'
            })
        };

        this.sortableColumns = {
            available: new Sortable(availableColsList, {
                group: 'columns',
                animation: 150,
                ghostClass: 'dragging'
            }),
            selected: new Sortable(selectedColsList, {
                group: 'columns',
                animation: 150,
                ghostClass: 'dragging'
            })
        };
    }

    createSortableItem(roi, name, color, dicomType = '', codeMeaning = '') {
        const item = document.createElement('div');
        item.className = 'sortable-item';
        item.dataset.roi = roi;
        item.dataset.dicomType = dicomType;

        // Build label with structure name and optional DICOM Type
        let label = name;
        if (dicomType) {
            label = `${label} (${dicomType})`;
        }

        item.innerHTML = `
            <div class="item-color" style="background-color: ${color}"></div>
            <div class="item-id">${roi}</div>
            <div class="item-name">${label}</div>
        `;
        return item;
    }

    populateDicomTypeFilters(types) {
        const rowFilter = document.getElementById('rowDicomFilter');
        const colFilter = document.getElementById('colDicomFilter');

        // Clear existing options except "All Types"
        rowFilter.innerHTML = '<option value="">All Types</option>';
        colFilter.innerHTML = '<option value="">All Types</option>';

        types.forEach(type => {
            const rowOption = document.createElement('option');
            rowOption.value = type;
            rowOption.textContent = type;
            rowFilter.appendChild(rowOption);

            const colOption = document.createElement('option');
            colOption.value = type;
            colOption.textContent = type;
            colFilter.appendChild(colOption);
        });
    }

    filterStructuresByType(axis, selectedType) {
        const availableList = axis === 'rows'
            ? document.getElementById('availableRowsList')
            : document.getElementById('availableColsList');
        const selectedList = axis === 'rows'
            ? document.getElementById('selectedRowsList')
            : document.getElementById('selectedColsList');

        // Get all items from both lists
        const allItems = [
            ...Array.from(availableList.children),
            ...Array.from(selectedList.children)
        ];

        allItems.forEach(item => {
            const itemType = item.dataset.dicomType || '';
            if (selectedType === '' || itemType === selectedType) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    }

    async updateMatrix() {
        console.log('updateMatrix called');

        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');

        const rowRois = Array.from(selectedRowsList.children)
            .map(item => parseInt(item.dataset.roi));
        const colRois = Array.from(selectedColsList.children)
            .map(item => parseInt(item.dataset.roi));

        if (rowRois.length === 0 || colRois.length === 0) {
            alert('Please select at least one structure for both rows and columns');
            return;
        }

        const useSymbols = document.getElementById('useSymbolsToggle').checked;

        try {
            const response = await fetch('/api/matrix', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    row_rois: rowRois,
                    col_rois: colRois,
                    use_symbols: useSymbols
                })
            });

            if (!response.ok) {
                throw new Error('Failed to update matrix');
            }

            const data = await response.json();
            this.displayMatrix(data);

            // Also refresh the diagram if it's visible
            this.refreshDiagram();

        } catch (error) {
            console.error('Matrix update error:', error);
            alert('Failed to update matrix.');
        }
    }

    displayMatrix(data) {
        const matrixInfo = document.getElementById('matrixInfo');
        matrixInfo.innerHTML = `
            Displaying ${data.rows.length} × ${data.columns.length} relationship matrix
            (${data.rows.length} row structures, ${data.columns.length} column structures)
        `;

        const thead = document.getElementById('matrixHead');
        const tbody = document.getElementById('matrixBody');

        // Build header row
        thead.innerHTML = '<tr><th>Structure</th></tr>';
        const headerRow = thead.querySelector('tr');
        data.col_names.forEach(name => {
            const th = document.createElement('th');
            th.textContent = name;
            headerRow.appendChild(th);
        });

        // Build body rows
        tbody.innerHTML = '';
        data.data.forEach((row, rowIdx) => {
            const tr = document.createElement('tr');

            // Row header
            const th = document.createElement('th');
            th.textContent = data.row_names[rowIdx];
            tr.appendChild(th);

            // Data cells
            row.forEach(value => {
                const td = document.createElement('td');
                td.textContent = value;

                // Add relationship-specific class for color coding
                const relClass = this.getRelationshipClass(value);
                if (relClass) {
                    td.classList.add(relClass);
                }

                // Add tooltip with relationship name
                const relLabel = this.getRelationshipLabel(value);
                if (relLabel) {
                    td.title = relLabel;
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
    }

    getRelationshipClass(symbol) {
        // Use config if available, otherwise fall back to hardcoded map
        if (this.symbolConfig && this.symbolConfig.relationships) {
            for (const [relType, config] of Object.entries(this.symbolConfig.relationships)) {
                if (config.symbol === symbol) {
                    return `rel-${relType.toLowerCase()}`;
                }
            }
        }

        // Fallback to hardcoded map
        const symbolMap = {
            '⊂': 'rel-contains',
            '∩': 'rel-overlaps',
            '|': 'rel-borders',
            '○': 'rel-surrounds',
            '△': 'rel-shelters',
            '⊕': 'rel-partition',
            '⊏': 'rel-confines',
            '∅': 'rel-disjoint',
            '=': 'rel-equals',
            '?': 'rel-unknown'
        };
        return symbolMap[symbol] || null;
    }

    getRelationshipLabel(symbol) {
        // Use config if available
        if (this.symbolConfig && this.symbolConfig.relationships) {
            for (const config of Object.values(this.symbolConfig.relationships)) {
                if (config.symbol === symbol) {
                    return `${config.label} - ${config.description}`;
                }
            }
        }

        // Fallback to hardcoded map
        const labelMap = {
            '⊂': 'Contains - Structure A fully encloses structure B',
            '∩': 'Overlaps - Structures share common volume',
            '|': 'Borders - Structures touch at boundaries',
            '○': 'Surrounds - Structure B is within a hole in A',
            '△': 'Shelters - B within convex hull of A, not touching',
            '⊕': 'Partition - Structures partition space between them',
            '⊏': 'Confines - B contacts inner surface of A',
            '∅': 'Disjoint - Structures are completely separated',
            '=': 'Equals - Same structure',
            '?': 'Unknown - Relationship not determined'
        };
        return labelMap[symbol] || null;
    }

    async exportMatrix(format) {
        try {
            const response = await fetch(`/api/export/${format}/${this.sessionId}`);

            if (!response.ok) {
                throw new Error('Export failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;

            const extension = format === 'excel' ? 'xlsx' : format;
            a.download = `relationships_${this.sessionId}.${extension}`;

            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Export error:', error);
            alert('Failed to export matrix.');
        }
    }

    toggleLegend() {
        const content = document.getElementById('legendContent');
        const toggle = document.getElementById('legendToggle');

        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggle.textContent = '▼';
        } else {
            content.style.display = 'none';
            toggle.textContent = '►';
        }
    }

    toggleAnalysisConfig() {
        const content = document.getElementById('analysisConfigContent');
        const toggle = document.getElementById('analysisConfigToggle');

        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggle.textContent = '▼';
        } else {
            content.style.display = 'none';
            toggle.textContent = '►';
        }
    }

    async refreshDiagram() {
        if (!this.sessionId) {
            console.warn('No session ID available for diagram');
            return;
        }

        try {
            const selectedRowsList = document.getElementById('selectedRowsList');
            const selectedColsList = document.getElementById('selectedColsList');

            const rowRois = Array.from(selectedRowsList.children)
                .map(item => parseInt(item.dataset.roi));
            const colRois = Array.from(selectedColsList.children)
                .map(item => parseInt(item.dataset.roi));

            const response = await fetch('/api/diagram', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    row_rois: rowRois,
                    col_rois: colRois
                })
            });

            if (!response.ok) {
                throw new Error('Failed to fetch diagram data');
            }

            const data = await response.json();
            this.renderDiagram(data);

        } catch (error) {
            console.error('Error refreshing diagram:', error);
            alert('Failed to generate relationship diagram');
        }
    }

    renderDiagram(data) {
        const container = document.getElementById('networkDiagram');
        const showLabels = document.getElementById('showLabelsToggle').checked;

        // Prepare nodes for vis-network
        const nodes = data.nodes.map(node => ({
            id: node.id,
            label: node.label,
            color: {
                background: node.color,
                border: this.darkenColor(node.color),
                highlight: {
                    background: node.color,
                    border: this.darkenColor(node.color)
                }
            },
            shape: node.shape,
            title: node.title,
            font: {
                color: this.getTextColor(node.color),
                size: 14,
                face: 'Arial'
            },
            borderWidth: 2
        }));

        // Prepare edges for vis-network
        const edges = data.edges.map(edge => ({
            from: edge.from_node,
            to: edge.to_node,
            label: showLabels ? edge.label : '',
            color: edge.color,
            width: edge.width,
            dashes: edge.dashes,
            arrows: edge.arrows ? edge.arrows : undefined,
            font: {
                size: 12,
                color: edge.color,
                strokeWidth: 0
            },
            smooth: {
                type: 'continuous',
                roundness: 0.5
            }
        }));

        // Network options
        const options = {
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 4
            },
            edges: {
                smooth: {
                    type: 'continuous'
                }
            },
            physics: {
                enabled: true,
                stabilization: {
                    iterations: 200
                },
                barnesHut: {
                    gravitationalConstant: -2000,
                    springConstant: 0.04,
                    springLength: 150
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        };

        // Destroy existing network if present
        if (this.network) {
            this.network.destroy();
        }

        // Create new network
        this.network = new vis.Network(container, { nodes, edges }, options);

        // Add event listeners
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                console.log('Clicked node:', params.nodes[0]);
            }
        });
    }

    toggleEdgeLabels() {
        if (!this.network) return;

        const showLabels = document.getElementById('showLabelsToggle').checked;
        const edges = this.network.body.data.edges.get();

        edges.forEach(edge => {
            this.network.body.data.edges.update({
                id: edge.id,
                label: showLabels && edge.originalLabel ? edge.originalLabel : ''
            });
        });
    }

    selectAllAxisStructures(axis) {
        const availableList = document.getElementById(axis === 'rows' ? 'availableRowsList' : 'availableColsList');
        const selectedList = document.getElementById(axis === 'rows' ? 'selectedRowsList' : 'selectedColsList');

        // Move all items from available to selected
        while (availableList.firstChild) {
            selectedList.appendChild(availableList.firstChild);
        }
    }

    selectNoneAxisStructures(axis) {
        const availableList = document.getElementById(axis === 'rows' ? 'availableRowsList' : 'availableColsList');
        const selectedList = document.getElementById(axis === 'rows' ? 'selectedRowsList' : 'selectedColsList');

        // Move all items from selected to available
        while (selectedList.firstChild) {
            availableList.appendChild(selectedList.firstChild);
        }
    }

    copyRowsToColumns() {
        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');
        const availableColsList = document.getElementById('availableColsList');

        // Clear columns lists
        selectedColsList.innerHTML = '';
        availableColsList.innerHTML = '';

        // Copy all row items to selected columns
        Array.from(selectedRowsList.children).forEach(rowItem => {
            const colItem = rowItem.cloneNode(true);
            selectedColsList.appendChild(colItem);
        });
    }

    darkenColor(hex) {
        // Remove # if present
        hex = hex.replace('#', '');

        // Parse RGB
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);

        // Darken by 30%
        const darkenedR = Math.floor(r * 0.7);
        const darkenedG = Math.floor(g * 0.7);
        const darkenedB = Math.floor(b * 0.7);

        return `#${darkenedR.toString(16).padStart(2, '0')}${darkenedG.toString(16).padStart(2, '0')}${darkenedB.toString(16).padStart(2, '0')}`;
    }

    getTextColor(bgColor) {
        // Remove # if present
        const hex = bgColor.replace('#', '');

        // Parse RGB
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);

        // Calculate brightness
        const brightness = (r * 299 + g * 587 + b * 114) / 1000;

        return brightness > 128 ? '#000000' : '#FFFFFF';
    }

    populateContourPlotControls(data) {
        // Store all slice data
        this.allSliceIndices = data.slice_indices || [];
        this.structureSlices = data.structure_slices || {};
        this.sliceIndices = this.allSliceIndices;  // Start with all slices

        // Configure slider with discrete steps
        const slider = document.getElementById('sliceSlider');
        if (this.sliceIndices.length > 0) {
            slider.min = 0;
            slider.max = this.sliceIndices.length - 1;
            slider.value = Math.floor(this.sliceIndices.length / 2);
            slider.step = 1;

            // Set initial slice value
            const midIndex = Math.floor(this.sliceIndices.length / 2);
            document.getElementById('sliceValue').textContent = this.sliceIndices[midIndex].toFixed(2);
        } else {
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            document.getElementById('sliceValue').textContent = '0.0';
        }

        // Update arrow button states
        this.updateSliderArrows();

        // Populate structure dropdowns
        const select1 = document.getElementById('structure1Select');
        const select2 = document.getElementById('structure2Select');

        // Clear existing options except first
        select1.innerHTML = '<option value="">Select structure...</option>';
        select2.innerHTML = '<option value="">None</option>';

        // Add structures
        data.rows.forEach((roi, idx) => {
            const name = data.row_names[idx];
            const dicomType = data.dicom_types ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '') : '';
            const label = dicomType ? `${name} (${dicomType})` : name;

            const option1 = document.createElement('option');
            option1.value = roi;
            option1.textContent = label;
            select1.appendChild(option1);

            const option2 = document.createElement('option');
            option2.value = roi;
            option2.textContent = label;
            select2.appendChild(option2);
        });

        // Add event listeners for structure selection changes
        select1.addEventListener('change', () => this.updateSliceRangeForStructures());
        select2.addEventListener('change', () => this.updateSliceRangeForStructures());
    }

    updateSliceRangeForStructures() {
        const structure1 = document.getElementById('structure1Select').value;
        const structure2 = document.getElementById('structure2Select').value;

        let filteredSlices = [];

        if (structure1 || structure2) {
            // Get slices for selected structures
            const slices1 = structure1 ? (this.structureSlices[parseInt(structure1)] || []) : [];
            const slices2 = structure2 ? (this.structureSlices[parseInt(structure2)] || []) : [];

            // Combine slices (union of both structures)
            const slicesSet = new Set([...slices1, ...slices2]);
            filteredSlices = Array.from(slicesSet).sort((a, b) => a - b);
        } else {
            // No structures selected, use all slices
            filteredSlices = [...this.allSliceIndices];
        }

        // Get current slice value BEFORE updating the indices
        const slider = document.getElementById('sliceSlider');
        const oldSliceIndices = this.sliceIndices || [];
        let currentSliceValue;
        const currentIndex = parseInt(slider.value);
        if (oldSliceIndices.length > currentIndex) {
            currentSliceValue = oldSliceIndices[currentIndex];
        } else {
            currentSliceValue = filteredSlices[Math.floor(filteredSlices.length / 2)];
        }

        // Update the active slice indices
        this.sliceIndices = filteredSlices;

        // Reconfigure slider
        if (filteredSlices.length > 0) {
            // Update slider range
            slider.min = 0;
            slider.max = filteredSlices.length - 1;

            // Find closest slice in new range
            let closestIndex = 0;
            let minDiff = Infinity;
            filteredSlices.forEach((slice, idx) => {
                const diff = Math.abs(slice - currentSliceValue);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIndex = idx;
                }
            });

            slider.value = closestIndex;
            document.getElementById('sliceValue').textContent = filteredSlices[closestIndex].toFixed(2);
        } else {
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            document.getElementById('sliceValue').textContent = '0.0';
        }

        this.updateSliderArrows();

        // Auto-update plot if a structure is selected
        if (structure1) {
            this.plotContours();
        }
    }

    updateSliceValue(sliderValue) {
        // Use discrete slice index from stored array
        if (this.sliceIndices && this.sliceIndices.length > 0) {
            const index = parseInt(sliderValue);
            const sliceValue = this.sliceIndices[index];
            document.getElementById('sliceValue').textContent = sliceValue.toFixed(2);
        }
    }

    updateSliderArrows() {
        const slider = document.getElementById('sliceSlider');
        const prevBtn = document.getElementById('slicePrevBtn');
        const nextBtn = document.getElementById('sliceNextBtn');

        const currentValue = parseInt(slider.value);
        const minValue = parseInt(slider.min);
        const maxValue = parseInt(slider.max);

        prevBtn.disabled = (currentValue <= minValue);
        nextBtn.disabled = (currentValue >= maxValue);
    }

    stepSlice(direction) {
        const slider = document.getElementById('sliceSlider');
        const currentValue = parseInt(slider.value);
        const newValue = currentValue + direction;

        const minValue = parseInt(slider.min);
        const maxValue = parseInt(slider.max);

        if (newValue >= minValue && newValue <= maxValue) {
            slider.value = newValue;
            this.updateSliceValue(newValue);
            this.updateSliderArrows();

            // Auto-update plot if structure is selected
            const structure1 = document.getElementById('structure1Select').value;
            if (structure1) {
                this.plotContours();
            }
        }
    }

    async plotContours() {
        const structure1 = document.getElementById('structure1Select').value;
        const structure2 = document.getElementById('structure2Select').value;
        const sliderValue = document.getElementById('sliceSlider').value;

        if (!structure1) {
            alert('Please select at least one structure to plot');
            return;
        }

        // Get actual slice value from discrete indices
        const sliceIndex = this.sliceIndices[parseInt(sliderValue)];

        // Cancel any in-flight request
        if (this.plotAbortController) {
            this.plotAbortController.abort();
        }

        // Create new abort controller for this request
        this.plotAbortController = new AbortController();
        const signal = this.plotAbortController.signal;

        // Show loading indicator
        const loadingOverlay = document.getElementById('plotLoading');
        loadingOverlay.classList.add('active');

        try {
            const response = await fetch('/api/plot-contours', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    roi_list: structure2 ? [parseInt(structure1), parseInt(structure2)] : [parseInt(structure1)],
                    slice_index: sliceIndex
                }),
                signal: signal
            });

            if (!response.ok) {
                throw new Error('Failed to generate plot');
            }

            // Get image blob and display it
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            const plotImg = document.getElementById('contourPlot');
            plotImg.src = imageUrl;
            plotImg.style.display = 'block';

            // Hide loading indicator
            loadingOverlay.classList.remove('active');

        } catch (error) {
            // Hide loading indicator
            loadingOverlay.classList.remove('active');

            // Don't show error if request was aborted (this is expected)
            if (error.name === 'AbortError') {
                console.log('Plot request cancelled');
                return;
            }

            console.error('Plot error:', error);
            alert('Failed to generate contour plot');
        } finally {
            // Clear the abort controller
            this.plotAbortController = null;
        }
    }

    showStage(stageName) {
        // Hide all stages
        document.querySelectorAll('.stage').forEach(stage => {
            stage.style.display = 'none';
        });

        // Show selected stage
        document.getElementById(`stage-${stageName}`).style.display = 'block';
    }

    resetApp() {
        // Reset application state for new analysis
        this.sessionId = null;
        this.selectedStructures.clear();
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
        this.updateConnectionStatus(false);

        // Clear file input
        document.getElementById('fileInput').value = '';
    }
}

// Initialize app when DOM is ready
// Tab Switching
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    console.log('Initializing tabs, found', tabButtons.length, 'buttons and', tabContents.length, 'content divs');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            console.log('Tab clicked:', targetTab);

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const targetContent = document.getElementById(`tab-${targetTab}`);
            if (targetContent) {
                targetContent.classList.add('active');
                console.log('Activated tab:', targetTab);
            } else {
                console.error('Tab content not found for:', targetTab);
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new WebAppClient();
    initializeTabs();
});
