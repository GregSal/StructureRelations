class WebAppClient {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.selectedStructures = new Set();
        this.sortableRows = null;
        this.sortableColumns = null;

        this.initializeEventListeners();
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

        // Structure selection
        document.getElementById('selectAll').addEventListener('click', () => {
            this.selectAllStructures(true);
        });
        document.getElementById('selectNone').addEventListener('click', () => {
            this.selectAllStructures(false);
        });
        document.getElementById('processBtn').addEventListener('click', () => {
            this.startProcessing();
        });

        // Matrix controls
        document.getElementById('symbolToggle').addEventListener('change', () => {
            this.updateMatrix();
        });
        document.getElementById('updateMatrixBtn').addEventListener('click', () => {
            this.updateMatrix();
        });

        // Export
        document.getElementById('exportCsv').addEventListener('click', () => {
            this.exportMatrix('csv');
        });
        document.getElementById('exportExcel').addEventListener('click', () => {
            this.exportMatrix('excel');
        });
        document.getElementById('exportJson').addEventListener('click', () => {
            this.exportMatrix('json');
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
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('connectionStatus');

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
            patientInfo.innerHTML = `
                <div><strong>Patient ID:</strong> ${data.patient_id}</div>
                <div><strong>Patient Name:</strong> ${data.patient_name}</div>
                <div><strong>Study Date:</strong> ${data.study_date}</div>
            `;

            // Populate structures list
            const structuresList = document.getElementById('structuresList');
            structuresList.innerHTML = '';

            data.structures.forEach(struct => {
                const item = document.createElement('div');
                item.className = 'structure-item';
                item.innerHTML = `
                    <input type="checkbox"
                           data-roi="${struct.roi_number}"
                           checked>
                    <div class="structure-color"
                         style="background-color: ${struct.color}"></div>
                    <div class="structure-info">
                        <div class="structure-name">${struct.name}</div>
                        <div class="structure-details">ROI ${struct.roi_number}</div>
                    </div>
                `;
                structuresList.appendChild(item);
                this.selectedStructures.add(struct.roi_number);
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

            // Populate structure summary
            const summaryBody = document.getElementById('structureSummaryBody');
            summaryBody.innerHTML = '';

            const allStructures = [...new Set([...data.rows, ...data.columns])];
            allStructures.sort((a, b) => a - b);

            allStructures.forEach(roi => {
                const idx = data.rows.indexOf(roi);
                const name = idx >= 0 ? data.row_names[idx] :
                             data.col_names[data.columns.indexOf(roi)];
                const color = data.colors[roi] || '#cccccc';

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${roi}</td>
                    <td>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div class="structure-color"
                                 style="background-color: ${color}"></div>
                            ${name}
                        </div>
                    </td>
                `;
                summaryBody.appendChild(row);
            });

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

    initializeSortableLists(data) {
        // Populate available lists
        const availableRowsList = document.getElementById('availableRowsList');
        const availableColsList = document.getElementById('availableColsList');

        availableRowsList.innerHTML = '';
        availableColsList.innerHTML = '';

        data.rows.forEach((roi, idx) => {
            const color = data.colors[roi] || '#cccccc';
            const name = data.row_names[idx];

            const rowItem = this.createSortableItem(roi, name, color);
            const colItem = this.createSortableItem(roi, name, color);

            availableRowsList.appendChild(rowItem);
            availableColsList.appendChild(colItem);
        });

        // Initialize SortableJS
        this.sortableRows = {
            available: new Sortable(availableRowsList, {
                group: 'rows',
                animation: 150,
                ghostClass: 'dragging'
            }),
            selected: new Sortable(document.getElementById('selectedRowsList'), {
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
            selected: new Sortable(document.getElementById('selectedColsList'), {
                group: 'columns',
                animation: 150,
                ghostClass: 'dragging'
            })
        };
    }

    createSortableItem(roi, name, color) {
        const item = document.createElement('div');
        item.className = 'sortable-item';
        item.dataset.roi = roi;
        item.innerHTML = `
            <div class="item-color" style="background-color: ${color}"></div>
            <div class="item-name">${name}</div>
        `;
        return item;
    }

    async updateMatrix() {
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

        const useSymbols = document.getElementById('symbolToggle').checked;

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
        document.querySelectorAll('.card').forEach(card => {
            card.style.display = 'none';
        });

        // Show selected stage
        document.getElementById(`${stageName}Stage`).style.display = 'block';
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new WebAppClient();
});
