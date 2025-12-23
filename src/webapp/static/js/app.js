class WebAppClient {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.selectedStructures = new Set();
        this.sortableRows = null;
        this.sortableColumns = null;

        this.initializeEventListeners();
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

        // Collapsible summary
        document.getElementById('summaryToggle').addEventListener('click', () => {
            this.toggleSummary();
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

                // Build label: prefer Code Meaning, fallback to name
                let label = struct.code_meaning || struct.name;
                if (struct.dicom_type) {
                    label = `${label} <span style="color: #666;">(${struct.dicom_type})</span>`;
                }

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
                        <span class="structure-label">${label}</span>
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

            // Display initial matrix
            this.displayMatrix(data);

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

    toggleSummary() {
        const content = document.getElementById('summaryContent');
        const toggle = document.getElementById('summaryToggle');

        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggle.textContent = '▼';
        } else {
            content.style.display = 'none';
            toggle.textContent = '▶';
        }
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

        data.rows.forEach((roi, idx) => {
            // Colors object has string keys in JSON
            const color = this.rgbToColor(data.colors[roi] || data.colors[String(roi)]);
            const name = data.row_names[idx];

            // Get data from dictionaries using ROI as key
            const dicomType = data.dicom_types ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '') : '';
            const codeMeaning = data.code_meanings ? (data.code_meanings[roi] || data.code_meanings[String(roi)] || '') : '';

            const rowItem = this.createSortableItem(roi, name, color, dicomType, codeMeaning);
            const colItem = this.createSortableItem(roi, name, color, dicomType, codeMeaning);

            // Add to selected lists by default
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

        // Build label with ROI, Type, and Code Meaning
        let label = codeMeaning || name;
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
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
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

    showStage(stageName) {
        // Hide all stages
        document.querySelectorAll('.stage').forEach(stage => {
            stage.style.display = 'none';
        });

        // Show selected stage
        document.getElementById(`stage-${stageName}`).style.display = 'block';
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new WebAppClient();
});
