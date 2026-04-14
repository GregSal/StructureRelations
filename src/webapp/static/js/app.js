class WebAppClient {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.selectedStructures = new Set();
        this.sortableRows = null;
        this.sortableColumns = null;
        this.sortableListsInitialized = false;
        this.sortableInitScheduled = false;
        this.summarySortable = null;
        this.summaryRowOrder = [];
        this.summaryDefaultOrder = [];
        this.summaryHiddenRows = new Set();
        this.summaryDragSortEnabled = false;
        this.summaryColumnOrder = this.getDefaultSummaryColumnOrder();
        this.summaryColumnSettings = this.getDefaultSummaryColumnSettings();
        this.currentSort = { column: 'roi', ascending: true };
        this._activeInputModal = null;
        this.symbolConfig = null;  // Will store loaded config
        this.network = null;  // vis-network instance
        this.plotAbortController = null;  // Track current plot request
        this.diagramLogicalRelationsMode = 'limited';  // Default: limited mode
        this.diagramLogicalRelationsModeApplied = 'limited';
        this.diagramShowDisjointApplied = false;
        this.diagramShowLabelsApplied = true;
        this.diagramShowUnknownApplied = false;
        this.diagramSelection = new Set();
        this.diagramAppliedSelection = new Set();
        this.diagramSelectionPending = false;
        this.diagramSelectionModalOpen = false;
        this.patientInfo = null;
        this.structureItems = [];
        this.structureItemsByRoi = new Map();
        this.statusPollIntervalId = null;
        this.latestDiagramData = null;
        this.hiddenNodes = new Set();    // ROI ids hidden via context menu
        this.hiddenLabels = new Set();   // ROI ids with label hidden
        this.fixedNodes = new Set();     // ROI ids pinned via context menu
        this.manualLayoutActive = false;
        this._dragFrozen = [];
        this._contextMenu = null;        // active context menu DOM element
        this._invalidTooltipRelationshipModesWarned = new Set();
        this.diagramOptions = {
            font: {
                face: 'Arial',
                node_size: 14,
                edge_size: 12,
                dark_color: '#000000',
                light_color: '#FFFFFF'
            },
            Background: {
                color: '#000000'
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: false,
                local_global_default: 30,
                local_global_min: 0,
                local_global_max: 100
            },
            physics: {
                enabled: true,
                stabilization_iterations: 200,
                local: {
                    gravitationalConstant: -700,
                    springConstant: 0.008,
                    springLength: 80,
                    damping: 0.55,
                    centralGravity: 0.02
                },
                global: {
                    gravitationalConstant: -2000,
                    springConstant: 0.04,
                    springLength: 150,
                    damping: 0.28,
                    centralGravity: 0.15
                }
            }
        };
        this.layoutInfluence = 30;

        // Plot transform state
        this.plotScale = 1.0;
        this.plotTranslateX = 0;
        this.plotTranslateY = 0;
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.contourSortables = null;
        this.allSliceIndices = [];
        this.allSliceIndicesOriginal = [];
        this.sliceIndices = [];
        this.structureSlices = {};
        this.structureSlicesOriginal = {};
        this.structureSlicesInterpolated = {};
        this.sliceRelationships = {};
        this.contourStructureLabels = {};
        this.contourStructureColors = {};
        this.plotImageCache = new Map();
        this.maxPlotCacheEntries = 48;
        this.plotDebounceTimer = null;
        this.plotDebounceMs = 120;
        this.lastRenderedPlotKey = null;

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
                this.applyDiagramSettingsFromConfig();
                if (this.latestDiagramData) {
                    this.renderDiagramLegend(this.latestDiagramData);
                }
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

    applyDiagramSettingsFromConfig() {
        if (!this.symbolConfig) return;

        const configOptions = this.symbolConfig.diagram_options || {};
        const configFont = configOptions.font || {};
        const configBackground = configOptions.background || configOptions.Background || {};
        const configInteractionRaw = configOptions.interaction || {};
        const configInteraction = {
            ...configInteractionRaw,
            tooltipDelay: configInteractionRaw.tooltip_delay ?? configInteractionRaw.tooltipDelay,
            navigationButtons: configInteractionRaw.navigation_buttons ?? configInteractionRaw.navigationButtons,
        };
        const groupedDiagramLayout = configOptions.diagram_layout || configOptions['Diagram Layout'] || {};
        const configLayoutRaw = groupedDiagramLayout.layout || configOptions.layout || {};
        const configLayout = {
            ...configLayoutRaw,
            improvedLayout: configLayoutRaw.improved_layout ?? configLayoutRaw.improvedLayout,
        };
        const configPhysicsRaw = groupedDiagramLayout.physics || configOptions.physics || {};
        const configPhysics = {
            ...configPhysicsRaw,
            local: {
                ...(configPhysicsRaw.local || {}),
                gravitationalConstant: configPhysicsRaw.local?.gravitational_constant ?? configPhysicsRaw.local?.gravitationalConstant,
                springConstant: configPhysicsRaw.local?.spring_constant ?? configPhysicsRaw.local?.springConstant,
                springLength: configPhysicsRaw.local?.spring_length ?? configPhysicsRaw.local?.springLength,
                centralGravity: configPhysicsRaw.local?.central_gravity ?? configPhysicsRaw.local?.centralGravity,
            },
            global: {
                ...(configPhysicsRaw.global || {}),
                gravitationalConstant: configPhysicsRaw.global?.gravitational_constant ?? configPhysicsRaw.global?.gravitationalConstant,
                springConstant: configPhysicsRaw.global?.spring_constant ?? configPhysicsRaw.global?.springConstant,
                springLength: configPhysicsRaw.global?.spring_length ?? configPhysicsRaw.global?.springLength,
                centralGravity: configPhysicsRaw.global?.central_gravity ?? configPhysicsRaw.global?.centralGravity,
            }
        };

        this.diagramOptions = {
            font: {
                ...this.diagramOptions.font,
                ...configFont,
            },
            Background: {
                ...this.diagramOptions.Background,
                ...configBackground,
            },
            interaction: {
                ...this.diagramOptions.interaction,
                ...configInteraction,
            },
            layout: {
                ...this.diagramOptions.layout,
                ...configLayout,
            },
            physics: {
                ...this.diagramOptions.physics,
                ...configPhysics,
                local: {
                    ...this.diagramOptions.physics.local,
                    ...(configPhysics.local || {}),
                },
                global: {
                    ...this.diagramOptions.physics.global,
                    ...(configPhysics.global || {}),
                }
            }
        };

        const defaults = this.symbolConfig.relationship_display_defaults || {};
        const showDisjoint = defaults.show_disjoint ?? defaults.Show_Disjoint;
        const showEdgeLabels = defaults.show_edge_labels ?? defaults.Show_Edge_Labels;
        const showUnknown = defaults.show_unknown ?? defaults.Show_Unknown;
        const logicalModeRaw = defaults.logical_relations ?? defaults.Logical_Relations;

        if (typeof showDisjoint === 'boolean') {
            document.getElementById('showDisjointToggle').checked = showDisjoint;
            this.diagramShowDisjointApplied = showDisjoint;
        }
        if (typeof showEdgeLabels === 'boolean') {
            document.getElementById('showLabelsToggle').checked = showEdgeLabels;
            this.diagramShowLabelsApplied = showEdgeLabels;
        }
        if (typeof showUnknown === 'boolean') {
            this.diagramShowUnknownApplied = showUnknown;
        }

        if (typeof logicalModeRaw === 'string') {
            const normalized = logicalModeRaw.toLowerCase();
            const validModes = new Set(['hide', 'limited', 'show', 'faded']);
            if (validModes.has(normalized)) {
                this.diagramLogicalRelationsMode = normalized;
                this.diagramLogicalRelationsModeApplied = normalized;
                document.getElementById('diagramLogicalRelationsMode').value = normalized;
            }
        }

        const defaultInfluence = Number(this.diagramOptions.layout.local_global_default);
        this.setLayoutInfluence(
            Number.isFinite(defaultInfluence) ? defaultInfluence : this.layoutInfluence,
            false
        );

        this.tooltipsConfig = this.symbolConfig.tooltips || {};
    }

    clampNumber(value, minValue, maxValue) {
        return Math.max(minValue, Math.min(maxValue, value));
    }

    interpolateNumber(localValue, globalValue, weight) {
        return localValue + (globalValue - localValue) * weight;
    }

    getLayoutInfluenceWeight() {
        const minValue = Number(this.diagramOptions.layout.local_global_min ?? 0);
        const maxValue = Number(this.diagramOptions.layout.local_global_max ?? 100);
        const clamped = this.clampNumber(this.layoutInfluence, minValue, maxValue);
        if (maxValue <= minValue) {
            return 0;
        }
        return (clamped - minValue) / (maxValue - minValue);
    }

    getPhysicsFromInfluence() {
        const localPhysics = this.diagramOptions.physics.local || {};
        const globalPhysics = this.diagramOptions.physics.global || {};
        const weight = this.getLayoutInfluenceWeight();

        return {
            enabled: this.diagramOptions.physics.enabled !== false,
            stabilization: {
                iterations: Number(this.diagramOptions.physics.stabilization_iterations || 200)
            },
            barnesHut: {
                gravitationalConstant: this.interpolateNumber(
                    Number(localPhysics.gravitationalConstant ?? -700),
                    Number(globalPhysics.gravitationalConstant ?? -2000),
                    weight
                ),
                springConstant: this.interpolateNumber(
                    Number(localPhysics.springConstant ?? 0.008),
                    Number(globalPhysics.springConstant ?? 0.04),
                    weight
                ),
                springLength: this.interpolateNumber(
                    Number(localPhysics.springLength ?? 80),
                    Number(globalPhysics.springLength ?? 150),
                    weight
                ),
                damping: this.interpolateNumber(
                    Number(localPhysics.damping ?? 0.55),
                    Number(globalPhysics.damping ?? 0.28),
                    weight
                ),
                centralGravity: this.interpolateNumber(
                    Number(localPhysics.centralGravity ?? 0.02),
                    Number(globalPhysics.centralGravity ?? 0.15),
                    weight
                )
            }
        };
    }

    setLayoutInfluence(value, shouldUpdateNetwork) {
        const minValue = Number(this.diagramOptions.layout.local_global_min ?? 0);
        const maxValue = Number(this.diagramOptions.layout.local_global_max ?? 100);
        const safeValue = Number.isFinite(value) ? value : this.layoutInfluence;
        this.layoutInfluence = this.clampNumber(safeValue, minValue, maxValue);

        const slider = document.getElementById('layoutInfluenceSlider');
        const label = document.getElementById('layoutInfluenceLabel');
        if (slider) {
            slider.min = String(minValue);
            slider.max = String(maxValue);
            slider.value = String(this.layoutInfluence);
        }

        if (label) {
            const localWeight = 100 - this.getLayoutInfluenceWeight() * 100;
            label.textContent = `Local ${Math.round(localWeight)}%`;
        }

        if (shouldUpdateNetwork && this.network) {
            this.network.setOptions({ physics: this.getPhysicsFromInfluence() });
            this.network.startSimulation();
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
        document.getElementById('copyFromDiagramBtn').addEventListener('click', () => {
            this.copyFromDiagramToMatrix();
        });
        document.getElementById('summaryResetBtn').addEventListener('click', () => {
            this.resetSummaryView();
        });
        document.getElementById('summaryDragToggleBtn').addEventListener('click', () => {
            this.toggleSummaryDragSorting();
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

        // Analysis config toggle
        document.getElementById('analysisConfigToggle').addEventListener('click', () => {
            this.toggleAnalysisConfig();
        });
        document.getElementById('contourConfigToggle').addEventListener('click', () => {
            this.toggleContourConfig();
        });

        // Structure-set badge and selector modal
        const structureSetBadge = document.getElementById('structureSetBadge');
        const structureSetTooltip = document.getElementById('structureSetBadgeTooltip');
        if (structureSetBadge) {
            structureSetBadge.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                this.openDiagramStructureModal();
            });
            structureSetBadge.addEventListener('click', () => {
                this.openDiagramStructureModal();
            });
            structureSetBadge.addEventListener('mouseenter', () => {
                if (structureSetTooltip && structureSetTooltip.innerHTML.trim()) {
                    structureSetTooltip.classList.add('visible');
                }
            });
            structureSetBadge.addEventListener('mouseleave', () => {
                structureSetTooltip?.classList.remove('visible');
            });
        }

        const modalBackdrop = document.getElementById('diagramStructureModalBackdrop');
        if (modalBackdrop) {
            modalBackdrop.addEventListener('click', () => {
                this.closeDiagramStructureModal(true);
            });
        }

        const modalClose = document.getElementById('diagramStructureModalCloseBtn');
        if (modalClose) {
            modalClose.addEventListener('click', () => {
                this.closeDiagramStructureModal(true);
            });
        }

        const modalCancel = document.getElementById('diagramStructureModalCancelBtn');
        if (modalCancel) {
            modalCancel.addEventListener('click', () => {
                this.closeDiagramStructureModal(true);
            });
        }

        const modalApply = document.getElementById('diagramStructureModalApplyBtn');
        if (modalApply) {
            modalApply.addEventListener('click', () => {
                this.applyDiagramSelection();
                this.closeDiagramStructureModal(false);
            });
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.diagramSelectionModalOpen) {
                this.closeDiagramStructureModal(true);
            }
        });

        // DICOM Type filters
        document.getElementById('rowDicomFilter').addEventListener('change', (e) => {
            this.filterStructuresByType('rows', e.target.value);
        });
        document.getElementById('colDicomFilter').addEventListener('change', (e) => {
            this.filterStructuresByType('columns', e.target.value);
        });

        // Diagram controls
        document.getElementById('showDisjointToggle').addEventListener('change', () => {
            this.updateDiagramPendingState();
        });
        document.getElementById('showLabelsToggle').addEventListener('change', () => {
            this.updateDiagramPendingState();
        });
        document.getElementById('diagramLogicalRelationsMode').addEventListener('change', (e) => {
            console.log('Diagram logical relations mode changed to:', e.target.value);
            this.diagramLogicalRelationsMode = e.target.value;
            this.updateDiagramPendingState();
        });
        document.getElementById('layoutInfluenceSlider').addEventListener('input', (e) => {
            this.setLayoutInfluence(Number(e.target.value), true);
        });
        document.getElementById('applyDiagramBtn').addEventListener('click', () => {
            this.applyDiagramSelection();
        });
        document.getElementById('manualLayoutBtn').addEventListener('click', () => {
            this.enableManualLayout();
        });
        document.getElementById('resetLayoutBtn').addEventListener('click', () => {
            this.resetDiagramLayout();
        });
        document.getElementById('copyFromMatrixBtn').addEventListener('click', () => {
            this.copyFromMatrixToDiagram();
        });
        document.getElementById('diagramSearchInput').addEventListener('input', (e) => {
            this.filterDiagramSelectionList(e.target.value);
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
        document.getElementById('plotModeSelect').addEventListener('change', () => {
            this.updateRelationshipOverlayState();
            this.updateSliceRangeForStructures();
        });
        document.getElementById('relationshipOverlaySelect').addEventListener('change', () => {
            this.updateRelationshipOverlayState();
        });
        document.getElementById('selectAllContourBtn').addEventListener('click', () => {
            this.selectAllContourStructures();
        });
        document.getElementById('clearContourBtn').addEventListener('click', () => {
            this.clearContourStructures();
        });

        // Zoom and pan controls
        this.initializePlotControls();
        document.getElementById('sliceSlider').addEventListener('input', (e) => {
            this.updateSliceValue(e.target.value);
            this.updateSliderArrows();
            this.schedulePlotContours({ suppressAlerts: true });
        });
        document.getElementById('sliceDropdown').addEventListener('change', (e) => {
            const slider = document.getElementById('sliceSlider');
            slider.value = e.target.value;
            this.updateSliceValue(e.target.value);
            this.updateSliderArrows();
            this.schedulePlotContours({ suppressAlerts: true, debounceMs: 0 });
        });
        document.getElementById('includeInterpolatedSlicesToggle').addEventListener('change', () => {
            this.updateSliceRangeForStructures();
        });
        document.getElementById('showAxisToggle').addEventListener('change', () => {
            this.schedulePlotContours({ suppressAlerts: true, debounceMs: 0 });
        });
        document.getElementById('showToleranceToggle').addEventListener('change', (e) => {
            const toleranceInput = document.getElementById('toleranceInput');
            toleranceInput.disabled = !e.target.checked;
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
        document.getElementById('statusLogClearBtn').addEventListener('click', () => {
            this.clearStatusLog();
        });

        this.setLayoutInfluence(this.layoutInfluence, false);
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
            const warningParagraph = diskWarning?.querySelector('p');
            const defaultDiskWarning =
                '⚠️ Session storage usage is high. Older sessions may be automatically deleted.';
            if (data.disk_warning) {
                diskWarning.style.display = 'block';
                if (warningParagraph) {
                    warningParagraph.textContent =
                        typeof data.disk_warning === 'string' && data.disk_warning.trim()
                            ? data.disk_warning
                            : defaultDiskWarning;
                }
            } else {
                diskWarning.style.display = 'none';
                if (warningParagraph) {
                    warningParagraph.textContent = defaultDiskWarning;
                }
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
            this.stopStatusPolling();
            this.onProcessingComplete();
        } else if (data.type === 'status_line') {
            this.appendStatusLogLine(data.source || 'backend', data.message || '');
        } else if (data.type === 'error') {
            this.stopStatusPolling();
            alert(data.message);
            this.showStage('upload');
        }
    }

    startStatusPolling() {
        this.stopStatusPolling();
        this.statusPollIntervalId = window.setInterval(async () => {
            if (!this.sessionId) {
                return;
            }
            try {
                const response = await fetch(`/api/jobs/${this.sessionId}/status`);
                if (!response.ok) {
                    return;
                }
                const status = await response.json();
                if (typeof status.progress === 'number') {
                    this.updateProgress(
                        status.stage || 'processing',
                        status.progress,
                        status.message || 'Processing...',
                        ''
                    );
                }

                if (status.status === 'completed') {
                    this.stopStatusPolling();
                }
                if (status.status === 'failed' || status.status === 'cancelled') {
                    this.stopStatusPolling();
                }
            } catch (error) {
                console.debug('Status polling skipped:', error);
            }
        }, 1200);
    }

    stopStatusPolling() {
        if (this.statusPollIntervalId) {
            window.clearInterval(this.statusPollIntervalId);
            this.statusPollIntervalId = null;
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
            this.patientInfo = pInfo;
            patientInfo.innerHTML = `
                <div><strong>Patient ID:</strong> ${pInfo.patient_id || 'N/A'}</div>
                <div><strong>Patient Name:</strong> ${pInfo.patient_name || 'N/A'}</div>
                <div><strong>Structure Set:</strong> ${pInfo.structure_set || 'N/A'}</div>
                <div><strong>Resolution:</strong> ${this.formatResolution(pInfo.resolution_cm_per_pixel)}</div>
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

            this.updateStructureSetBadge();

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
        this.clearStatusLog();
        this.appendStatusLogLine('frontend', 'Processing started...');
        this.updateProgress('initializing', 0, 'Starting processing...', '');
        this.startStatusPolling();

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
            this.stopStatusPolling();
            console.error('Processing error:', error);
            alert('Failed to start processing.');
        }
    }

    formatStageLabel(stage) {
        const normalized = String(stage || '').trim();
        if (!normalized) {
            return 'processing';
        }

        const stageLabels = {
            parsing_dicom: 'parsing DICOM',
            building_graphs: 'building structures',
            calculating_relationships: 'calculating relationships',
            rendering_results: 'rendering results',
            completed: 'completed',
        };

        return stageLabels[normalized] || normalized.replaceAll('_', ' ');
    }

    updateProgress(stage, progress, message, currentStructure) {
        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.getElementById('progressText');
        const progressDetail = document.getElementById('progressDetail');
        if (!progressFill || !progressText || !progressDetail) {
            return;
        }

        const boundedProgress = Math.max(0, Math.min(100, Number(progress) || 0));

        progressFill.style.width = `${boundedProgress}%`;
        progressText.textContent = `${boundedProgress.toFixed(1)}% - ${this.formatStageLabel(stage)}`;

        let detailText = typeof message === 'string' ? message.trim() : '';
        if (!detailText) {
            detailText = 'Processing...';
        }
        if (currentStructure && !detailText.includes(currentStructure)) {
            detailText += ` (${currentStructure})`;
        }
        progressDetail.textContent = detailText;
    }

    clearStatusLog() {
        const content = document.getElementById('statusLogContent');
        if (content) {
            content.innerHTML = '';
        }
    }

    appendStatusLogLine(source, message) {
        const content = document.getElementById('statusLogContent');
        if (!content) return;
        const line = document.createElement('div');
        line.className = `status-log-line source-${source}`;
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] ${message}`;
        content.appendChild(line);
        content.scrollTop = content.scrollHeight;
    }

    async onProcessingComplete() {
        this.stopStatusPolling();
        this.updateProgress('rendering_results', 80.0, 'Loading processed session data...', '');
        this.appendStatusLogLine('frontend', 'Fetching results from server...');

        try {
            const matrixFetchStart = performance.now();
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

            this.updateProgress('rendering_results', 82.5, 'Loading processed session data...', '');
            const data = await response.json();
            const matrixFetchMs = Math.round(performance.now() - matrixFetchStart);
            this.appendStatusLogLine(
                'frontend',
                `Loaded ${data.rows?.length ?? 0} structure(s) in ${matrixFetchMs} ms. Building views...`
            );

            const runViewBuildStep = (label, progressValue, callback) => {
                this.updateProgress('rendering_results', progressValue, `${label}...`, '');
                this.appendStatusLogLine('frontend', `${label}...`);
                const stepStart = performance.now();
                callback();
                const stepMs = Math.round(performance.now() - stepStart);
                this.appendStatusLogLine('frontend', `${label} complete (${stepMs} ms).`);
            };

            this.summaryData = data;
            this.buildStructureItems(data);

            runViewBuildStep('Building structure summary', 85.0, () => {
                this.populateStructureSummary(data);
            });
            runViewBuildStep('Configuring diagram selection', 88.0, () => {
                this.initializeDiagramSelection(data);
            });

            this.updateProgress('rendering_results', 90.0, 'Requesting relationship diagram...', '');
            this.appendStatusLogLine('frontend', 'Requesting relationship diagram...');
            const diagramPromise = this.applyDiagramSelection();

            runViewBuildStep('Populating contour controls', 93.0, () => {
                this.populateContourPlotControls(data);
            });
            runViewBuildStep('Rendering relationship matrix', 96.0, () => {
                this.displayMatrix(data);
            });
            runViewBuildStep('Rendering structure set details', 98.0, () => {
                this.renderStructureSetInfo();
            });

            this.scheduleDeferredSortableInitialization();

            if (diagramPromise && typeof diagramPromise.then === 'function') {
                await diagramPromise;
            }

            this.updateProgress('completed', 100.0, 'Processing complete', '');
            this.appendStatusLogLine('frontend', 'All views ready.');
            this.showStage('results');

        } catch (error) {
            console.error('Results error:', error);
            alert('Failed to load results.');
        }
    }

    getDefaultSummaryColumnOrder() {
        return ['name', 'roi', 'type', 'label', 'regions', 'volume', 'slices'];
    }

    getDefaultSummaryColumnSettings() {
        return {
            name: { hidden: false, align: 'left', width: '' },
            roi: { hidden: false, align: 'right', width: '' },
            type: { hidden: false, align: 'left', width: '' },
            label: { hidden: false, align: 'left', width: '' },
            regions: { hidden: false, align: 'right', width: '' },
            volume: { hidden: false, align: 'right', decimals: 2, width: '' },
            slices: { hidden: false, align: 'center', decimals: 2, width: '' },
        };
    }

    initializeSummaryState(data) {
        const allStructures = [...new Set([...(data.rows || []), ...(data.columns || [])])]
            .map(roi => Number(roi))
            .sort((a, b) => a - b);
        const signature = allStructures.join(',');

        if (this._summarySignature !== signature) {
            this._summarySignature = signature;
            this.summaryDefaultOrder = [...allStructures];
            this.summaryRowOrder = [...allStructures];
            this.summaryHiddenRows.clear();
            this.summaryDragSortEnabled = false;
            if (this.summarySortable) {
                this.summarySortable.destroy();
                this.summarySortable = null;
            }
            this.summaryColumnOrder = this.getDefaultSummaryColumnOrder();
            this.summaryColumnSettings = this.getDefaultSummaryColumnSettings();
            this.currentSort = { column: 'roi', ascending: true };
            return;
        }

        const validRois = new Set(allStructures);
        this.summaryDefaultOrder = [...allStructures];
        this.summaryRowOrder = this.summaryRowOrder.filter(roi => validRois.has(Number(roi)));
        allStructures.forEach(roi => {
            if (!this.summaryRowOrder.includes(roi)) {
                this.summaryRowOrder.push(roi);
            }
        });
        this.summaryHiddenRows = new Set(
            Array.from(this.summaryHiddenRows).filter(roi => validRois.has(Number(roi)))
        );
    }

    getSummaryRowDetails(data, roi) {
        const normalizedRoi = Number(roi);
        const rowIndex = (data.rows || []).indexOf(normalizedRoi);
        const colIndex = (data.columns || []).indexOf(normalizedRoi);
        const name = rowIndex >= 0
            ? data.row_names?.[rowIndex]
            : data.col_names?.[colIndex] || '';
        const dicomType = data.dicom_types?.[normalizedRoi]
            || data.dicom_types?.[String(normalizedRoi)]
            || '';
        const codeMeaning = data.code_meanings?.[normalizedRoi]
            || data.code_meanings?.[String(normalizedRoi)]
            || '';
        const volume = Number(
            data.volumes?.[normalizedRoi]
            || data.volumes?.[String(normalizedRoi)]
            || 0
        );
        const numRegions = Number(
            data.num_regions?.[normalizedRoi]
            || data.num_regions?.[String(normalizedRoi)]
            || 0
        );
        const sliceRange = data.slice_ranges?.[normalizedRoi]
            || data.slice_ranges?.[String(normalizedRoi)]
            || '';
        const color = this.rgbToColor(
            data.colors?.[normalizedRoi] || data.colors?.[String(normalizedRoi)]
        );

        return {
            roi: normalizedRoi,
            name: name || '',
            dicomType,
            label: codeMeaning || name || '',
            volume,
            numRegions,
            sliceRange,
            color,
        };
    }

    formatSummarySliceRange(sliceRange) {
        if (sliceRange === null || sliceRange === undefined || sliceRange === '') {
            return '';
        }

        const decimals = Number(this.summaryColumnSettings?.slices?.decimals ?? 2);
        let start;
        let end;
        let unit = '';
        let rawText = '';

        if (Array.isArray(sliceRange) && sliceRange.length >= 2) {
            [start, end] = sliceRange;
            rawText = `${start} to ${end}`;
        } else if (typeof sliceRange === 'object') {
            start = sliceRange.start ?? sliceRange.min ?? sliceRange.lower;
            end = sliceRange.end ?? sliceRange.max ?? sliceRange.upper;
            unit = sliceRange.unit || '';
            rawText = `${start} to ${end}${unit ? ` ${unit}` : ''}`;
        } else {
            rawText = String(sliceRange).trim();
            const match = rawText.match(
                /(-?\d+(?:\.\d+)?)\s*(?:cm|mm)?\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)(?:\s*(cm|mm))?/i
            );
            if (match) {
                start = match[1];
                end = match[2];
                unit = match[3] || (/(cm|mm)/i.exec(rawText)?.[1] || '');
            }
        }

        const startNumber = Number(start);
        const endNumber = Number(end);
        if (!Number.isFinite(startNumber) || !Number.isFinite(endNumber)) {
            return `<span>${this.escapeHtml(rawText)}</span>`;
        }

        return `
            <div class="slice-range-display">
                <span class="slice-range-value slice-range-start">${startNumber.toFixed(decimals)}</span>
                <span class="slice-range-separator">to</span>
                <span class="slice-range-value slice-range-end">${endNumber.toFixed(decimals)}</span>
                <span class="slice-range-unit">${this.escapeHtml(unit)}</span>
            </div>
        `;
    }

    getSummarySliceSortValue(sliceRange) {
        if (Array.isArray(sliceRange) && sliceRange.length > 0) {
            return Number(sliceRange[0]) || 0;
        }
        if (sliceRange && typeof sliceRange === 'object') {
            return Number(sliceRange.start ?? sliceRange.min ?? sliceRange.lower) || 0;
        }
        const text = String(sliceRange || '');
        const match = text.match(/-?\d+(?:\.\d+)?/);
        return match ? Number(match[0]) : 0;
    }

    populateStructureSummary(data) {
        const summaryBody = document.getElementById('structuresSummaryBody');
        if (!summaryBody) return;

        this.initializeSummaryState(data);
        summaryBody.innerHTML = '';

        this.summaryRowOrder
            .filter(roi => !this.summaryHiddenRows.has(Number(roi)))
            .forEach(roi => {
                const details = this.getSummaryRowDetails(data, roi);
                const row = document.createElement('tr');
                row.dataset.roi = String(details.roi);
                row.dataset.name = details.name;
                row.dataset.type = details.dicomType;
                row.dataset.label = details.label;
                row.dataset.volume = String(details.volume);
                row.dataset.regions = String(details.numRegions);
                row.dataset.slices = String(this.getSummarySliceSortValue(details.sliceRange));

                row.innerHTML = `
                    <td class="name-cell" data-column="name">${this.escapeHtml(details.name)}</td>
                    <td class="roi-cell number-cell" data-column="roi">${details.roi}</td>
                    <td class="type-cell" data-column="type">${this.escapeHtml(details.dicomType)}</td>
                    <td class="label-cell" data-column="label">
                        <div class="summary-label-wrap">
                            <div class="structure-color" style="background-color: ${details.color}"></div>
                            <span>${this.escapeHtml(details.label)}</span>
                        </div>
                    </td>
                    <td class="number-cell" data-column="regions">${details.numRegions}</td>
                    <td class="number-cell" data-column="volume">${details.volume.toFixed(Number(this.summaryColumnSettings?.volume?.decimals ?? 2))}</td>
                    <td class="slice-cell" data-column="slices">${this.formatSummarySliceRange(details.sliceRange)}</td>
                `;
                row.addEventListener('contextmenu', (event) => {
                    event.preventDefault();
                    this.showSummaryRowContextMenu(details.roi, event);
                });
                summaryBody.appendChild(row);
            });

        this.initializeSorting();
        this.applySummaryColumnSettings();
        this.refreshSummarySortable();
        this.updateSummaryToolbar();
    }

    renderStructureSetInfo() {
        const container = document.getElementById('structureSetInfo');
        if (!container) return;

        const pInfo = this.patientInfo || {};
        const lines = [];

        if (pInfo.patient_name) {
            lines.push({ label: 'Patient Name', value: pInfo.patient_name });
        }
        if (pInfo.patient_id) {
            lines.push({ label: 'Patient ID', value: pInfo.patient_id });
        }
        if (pInfo.structure_set) {
            lines.push({ label: 'Structure Set', value: pInfo.structure_set });
        }

        if (lines.length === 0) {
            container.style.display = 'none';
            container.innerHTML = '';
            return;
        }

        container.innerHTML = lines.map(line => (
            `<div class="structure-set-line"><span class="structure-set-label">${line.label}:</span> ${line.value}</div>`
        )).join('');
        container.style.display = 'block';
    }

    initializeSorting() {
        document.querySelectorAll('#structuresSummaryTable .sortable').forEach(th => {
            if (!th.dataset.summaryBound) {
                th.dataset.summaryBound = 'true';
                th.addEventListener('click', () => {
                    const column = th.dataset.column;
                    this.sortTable(column);
                });
                th.addEventListener('contextmenu', (event) => {
                    event.preventDefault();
                    const column = th.dataset.column;
                    this.showSummaryHeaderContextMenu(column, event);
                });
            }

            th.classList.remove('sorted-asc', 'sorted-desc');
            if (this.currentSort?.column === th.dataset.column) {
                th.classList.add(this.currentSort.ascending ? 'sorted-asc' : 'sorted-desc');
            }
        });
    }

    sortTable(column, forcedAscending = null) {
        if (!this.summaryData) return;

        const currentSort = this.currentSort || {};
        const ascending = typeof forcedAscending === 'boolean'
            ? forcedAscending
            : (currentSort.column === column ? !currentSort.ascending : true);
        this.currentSort = { column, ascending };

        const sortValueFor = (details) => {
            switch (column) {
                case 'name':
                    return details.name.toLowerCase();
                case 'roi':
                    return Number(details.roi);
                case 'type':
                    return details.dicomType.toLowerCase();
                case 'label':
                    return details.label.toLowerCase();
                case 'regions':
                    return Number(details.numRegions);
                case 'volume':
                    return Number(details.volume);
                case 'slices':
                    return this.getSummarySliceSortValue(details.sliceRange);
                default:
                    return Number(details.roi);
            }
        };

        this.summaryRowOrder.sort((aRoi, bRoi) => {
            const aValue = sortValueFor(this.getSummaryRowDetails(this.summaryData, aRoi));
            const bValue = sortValueFor(this.getSummaryRowDetails(this.summaryData, bRoi));

            if (aValue < bValue) return ascending ? -1 : 1;
            if (aValue > bValue) return ascending ? 1 : -1;
            return Number(aRoi) - Number(bRoi);
        });

        this.populateStructureSummary(this.summaryData);
    }

    updateSummaryToolbar() {
        const dragButton = document.getElementById('summaryDragToggleBtn');
        const resetButton = document.getElementById('summaryResetBtn');
        const hasSummary = !!this.summaryData && this.summaryDefaultOrder.length > 0;

        if (dragButton) {
            dragButton.disabled = !hasSummary;
            dragButton.textContent = this.summaryDragSortEnabled
                ? 'Disable Drag Sorting'
                : 'Enable Drag Sorting';
        }

        if (resetButton) {
            resetButton.disabled = !hasSummary;
        }
    }

    toggleSummaryDragSorting() {
        if (!this.summaryData) return;

        this.summaryDragSortEnabled = !this.summaryDragSortEnabled;
        this.refreshSummarySortable();
        this.updateSummaryToolbar();
        this.appendStatusLogLine(
            'frontend',
            this.summaryDragSortEnabled
                ? 'Drag sorting enabled in the structure summary.'
                : 'Drag sorting disabled in the structure summary.'
        );
    }

    refreshSummarySortable() {
        const tbody = document.getElementById('structuresSummaryBody');
        if (!tbody) return;

        if (this.summarySortable) {
            this.summarySortable.destroy();
            this.summarySortable = null;
        }

        tbody.classList.toggle('summary-drag-enabled', this.summaryDragSortEnabled);

        if (this.summaryDragSortEnabled && typeof Sortable !== 'undefined') {
            this.summarySortable = new Sortable(tbody, {
                animation: 150,
                ghostClass: 'dragging',
                onEnd: () => {
                    this.captureSummaryRowOrderFromDom();
                    this.currentSort = { column: 'custom', ascending: true };
                    this.initializeSorting();
                },
            });
        }
    }

    captureSummaryRowOrderFromDom() {
        const tbody = document.getElementById('structuresSummaryBody');
        if (!tbody) return;

        const visibleOrder = Array.from(tbody.querySelectorAll('tr')).map((row) => (
            Number(row.dataset.roi)
        ));
        const hiddenOrder = this.summaryRowOrder.filter((roi) => (
            this.summaryHiddenRows.has(Number(roi))
        ));
        this.summaryRowOrder = [...visibleOrder, ...hiddenOrder];
    }

    moveSummaryRow(roi, direction) {
        const normalizedRoi = Number(roi);
        const currentIndex = this.summaryRowOrder.indexOf(normalizedRoi);
        if (currentIndex < 0) return;

        let targetIndex = currentIndex + direction;
        while (
            targetIndex >= 0
            && targetIndex < this.summaryRowOrder.length
            && this.summaryHiddenRows.has(Number(this.summaryRowOrder[targetIndex]))
        ) {
            targetIndex += direction;
        }

        if (targetIndex < 0 || targetIndex >= this.summaryRowOrder.length) {
            return;
        }

        const reordered = [...this.summaryRowOrder];
        reordered.splice(currentIndex, 1);
        reordered.splice(targetIndex, 0, normalizedRoi);
        this.summaryRowOrder = reordered;
        this.currentSort = { column: 'custom', ascending: true };
        this.populateStructureSummary(this.summaryData);
    }

    hideSummaryRow(roi) {
        this.summaryHiddenRows.add(Number(roi));
        this.populateStructureSummary(this.summaryData);
    }

    resetSummaryView() {
        if (!this.summaryData) return;

        this.summaryHiddenRows.clear();
        this.summaryRowOrder = [...this.summaryDefaultOrder];
        this.summaryDragSortEnabled = false;
        this.summaryColumnOrder = this.getDefaultSummaryColumnOrder();
        this.summaryColumnSettings = this.getDefaultSummaryColumnSettings();
        this.currentSort = { column: 'roi', ascending: true };
        this.populateStructureSummary(this.summaryData);
        this.appendStatusLogLine('frontend', 'Structure summary reset to the default view.');
    }

    applySummaryColumnOrder() {
        const table = document.getElementById('structuresSummaryTable');
        if (!table) return;

        const headerRow = table.querySelector('thead tr');
        if (!headerRow) return;

        const availableColumns = Array.from(headerRow.querySelectorAll('th[data-column]'))
            .map((header) => header.dataset.column);
        const orderedColumns = (this.summaryColumnOrder || this.getDefaultSummaryColumnOrder())
            .filter((column) => availableColumns.includes(column));

        availableColumns.forEach((column) => {
            if (!orderedColumns.includes(column)) {
                orderedColumns.push(column);
            }
        });
        this.summaryColumnOrder = orderedColumns;

        orderedColumns.forEach((column) => {
            const header = headerRow.querySelector(`th[data-column="${column}"]`);
            if (header) {
                headerRow.appendChild(header);
            }
        });

        table.querySelectorAll('tbody tr').forEach((row) => {
            orderedColumns.forEach((column) => {
                const cell = row.querySelector(`td[data-column="${column}"]`);
                if (cell) {
                    row.appendChild(cell);
                }
            });
        });
    }

    moveSummaryColumn(column, direction) {
        const currentOrder = [...(this.summaryColumnOrder || this.getDefaultSummaryColumnOrder())];
        const currentIndex = currentOrder.indexOf(column);
        if (currentIndex < 0) return;

        const targetIndex = currentIndex + direction;
        if (targetIndex < 0 || targetIndex >= currentOrder.length) {
            return;
        }

        [currentOrder[currentIndex], currentOrder[targetIndex]] = [
            currentOrder[targetIndex],
            currentOrder[currentIndex],
        ];
        this.summaryColumnOrder = currentOrder;
        this.applySummaryColumnOrder();
        this.applySummaryColumnSettings();
        this.appendStatusLogLine('frontend', `Moved ${column} column.`);
    }

    applySummaryColumnSettings() {
        const table = document.getElementById('structuresSummaryTable');
        if (!table) return;

        this.applySummaryColumnOrder();

        Object.entries(this.summaryColumnSettings).forEach(([column, settings]) => {
            const header = table.querySelector(`th[data-column="${column}"]`);
            const cells = table.querySelectorAll(`td[data-column="${column}"]`);
            const display = settings.hidden ? 'none' : '';
            const alignment = settings.align || 'left';
            const width = settings.width || '';

            if (header) {
                header.style.display = display;
                header.style.textAlign = alignment;
                header.style.width = width;
                header.style.minWidth = width;
            }

            cells.forEach(cell => {
                cell.style.display = display;
                cell.style.textAlign = alignment;
                if (width) {
                    cell.style.width = width;
                    cell.style.minWidth = width;
                } else {
                    cell.style.removeProperty('width');
                    cell.style.removeProperty('min-width');
                }
            });
        });
    }

    setSummaryColumnAlignment(column, alignment) {
        if (!this.summaryColumnSettings[column]) return;
        this.summaryColumnSettings[column].align = alignment;
        this.applySummaryColumnSettings();
    }

    setSummaryColumnHidden(column, hidden = true) {
        if (!this.summaryColumnSettings[column]) return;
        this.summaryColumnSettings[column].hidden = hidden;
        this.applySummaryColumnSettings();
    }

    autoFitSummaryColumn(column) {
        const table = document.getElementById('structuresSummaryTable');
        if (!table || !this.summaryColumnSettings[column]) return;

        const header = table.querySelector(`th[data-column="${column}"]`);
        const cells = table.querySelectorAll(`td[data-column="${column}"]`);
        let maxWidth = header ? header.scrollWidth : 80;
        cells.forEach(cell => {
            maxWidth = Math.max(maxWidth, cell.scrollWidth);
        });

        this.summaryColumnSettings[column].width = `${Math.min(Math.max(maxWidth + 24, 80), 420)}px`;
        this.applySummaryColumnSettings();
    }

    showNumericInputDialog(title, message, initialValue, minValue = 0, maxValue = 6) {
        if (this._activeInputModal) {
            this._activeInputModal.remove();
            this._activeInputModal = null;
        }

        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'diagram-structure-modal';
            modal.style.display = 'block';
            modal.innerHTML = `
                <div class="diagram-structure-modal-backdrop"></div>
                <div class="diagram-structure-modal-dialog summary-input-dialog" role="dialog" aria-modal="true" aria-labelledby="summaryInputDialogTitle">
                    <div class="diagram-structure-modal-header">
                        <h3 id="summaryInputDialogTitle">${this.escapeHtml(title)}</h3>
                    </div>
                    <p class="summary-input-help">${this.escapeHtml(message)}</p>
                    <label for="summaryNumericInput" class="summary-input-label">Decimal places</label>
                    <input
                        id="summaryNumericInput"
                        class="summary-input-field"
                        type="number"
                        min="${minValue}"
                        max="${maxValue}"
                        step="1"
                        value="${Number(initialValue)}"
                    >
                    <p class="summary-input-error" id="summaryNumericInputError"></p>
                    <div class="button-group">
                        <button class="btn btn-secondary" type="button" id="summaryNumericCancelBtn">Cancel</button>
                        <button class="btn btn-primary" type="button" id="summaryNumericApplyBtn">Apply</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
            this._activeInputModal = modal;

            const input = modal.querySelector('#summaryNumericInput');
            const error = modal.querySelector('#summaryNumericInputError');
            const cancelButton = modal.querySelector('#summaryNumericCancelBtn');
            const applyButton = modal.querySelector('#summaryNumericApplyBtn');
            const backdrop = modal.querySelector('.diagram-structure-modal-backdrop');

            const close = (result) => {
                document.removeEventListener('keydown', handleKeyDown);
                modal.remove();
                if (this._activeInputModal === modal) {
                    this._activeInputModal = null;
                }
                resolve(result);
            };

            const apply = () => {
                const parsed = Number.parseInt(input.value, 10);
                if (!Number.isFinite(parsed) || parsed < minValue || parsed > maxValue) {
                    error.textContent = `Enter a whole number from ${minValue} to ${maxValue}.`;
                    error.style.display = 'block';
                    input.focus();
                    input.select();
                    return;
                }
                close(parsed);
            };

            const handleKeyDown = (event) => {
                if (event.key === 'Escape') {
                    close(null);
                } else if (event.key === 'Enter') {
                    event.preventDefault();
                    apply();
                }
            };

            cancelButton.addEventListener('click', () => close(null));
            applyButton.addEventListener('click', apply);
            backdrop.addEventListener('click', () => close(null));
            document.addEventListener('keydown', handleKeyDown);

            error.style.display = 'none';
            input.focus();
            input.select();
        });
    }

    async setSummaryColumnDecimals(column) {
        if (!this.summaryColumnSettings[column] || !this.summaryData) return;

        const currentValue = Number(this.summaryColumnSettings[column].decimals ?? 2);
        const parsed = await this.showNumericInputDialog(
            `Set decimal places for ${column}`,
            'Choose a value between 0 and 6.',
            currentValue,
            0,
            6
        );
        if (parsed === null) return;

        this.summaryColumnSettings[column].decimals = parsed;
        this.populateStructureSummary(this.summaryData);
        this.appendStatusLogLine(
            'frontend',
            `${column} decimal places updated to ${parsed}.`
        );
    }

    _showSimpleContextMenu(items, event) {
        this._dismissContextMenu();

        const menu = document.createElement('div');
        menu.className = 'node-context-menu';
        menu.style.left = `${event.clientX}px`;
        menu.style.top = `${event.clientY}px`;

        for (const item of items) {
            if (item.separator) {
                const sep = document.createElement('div');
                sep.className = 'node-context-menu-separator';
                menu.appendChild(sep);
                continue;
            }

            const el = document.createElement('div');
            el.className = 'node-context-menu-item';
            if (item.active) {
                el.classList.add('is-active');
            }
            if (item.disabled) {
                el.classList.add('is-disabled');
            }
            el.textContent = item.label;
            if (!item.disabled) {
                el.addEventListener('mousedown', (mouseEvent) => {
                    mouseEvent.stopPropagation();
                    item.action?.();
                    this._dismissContextMenu();
                });
            }
            menu.appendChild(el);
        }

        document.body.appendChild(menu);
        this._contextMenu = menu;

        const rect = menu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            menu.style.left = `${Math.max(event.clientX - rect.width, 8)}px`;
        }
        if (rect.bottom > window.innerHeight) {
            menu.style.top = `${Math.max(event.clientY - rect.height, 8)}px`;
        }

        const dismiss = (mouseEvent) => {
            if (!menu.contains(mouseEvent.target)) {
                this._dismissContextMenu();
                document.removeEventListener('mousedown', dismiss, true);
            }
        };
        document.addEventListener('mousedown', dismiss, true);
    }

    showSummaryRowContextMenu(roi, event) {
        const rows = Array.from(document.querySelectorAll('#structuresSummaryBody tr'));
        const visibleOrder = rows.map(row => Number(row.dataset.roi));
        const index = visibleOrder.indexOf(Number(roi));

        this._showSimpleContextMenu([
            {
                label: 'Move Up',
                disabled: index <= 0,
                action: () => this.moveSummaryRow(roi, -1),
            },
            {
                label: 'Move Down',
                disabled: index < 0 || index >= visibleOrder.length - 1,
                action: () => this.moveSummaryRow(roi, 1),
            },
            { separator: true },
            {
                label: 'Hide Structure',
                action: () => this.hideSummaryRow(roi),
            },
        ], event);
    }

    showSummaryHeaderContextMenu(column, event) {
        const supportsDecimals = column === 'volume' || column === 'slices';
        const alignment = this.summaryColumnSettings[column]?.align || 'left';
        const columnOrder = this.summaryColumnOrder || this.getDefaultSummaryColumnOrder();
        const columnIndex = columnOrder.indexOf(column);

        this._showSimpleContextMenu([
            {
                label: 'Move Left',
                disabled: columnIndex <= 0,
                action: () => this.moveSummaryColumn(column, -1),
            },
            {
                label: 'Move Right',
                disabled: columnIndex < 0 || columnIndex >= columnOrder.length - 1,
                action: () => this.moveSummaryColumn(column, 1),
            },
            { separator: true },
            {
                label: 'Sort Ascending',
                action: () => this.sortTable(column, true),
            },
            {
                label: 'Sort Descending',
                action: () => this.sortTable(column, false),
            },
            { separator: true },
            {
                label: 'Auto Fit Width',
                action: () => this.autoFitSummaryColumn(column),
            },
            {
                label: 'Hide Column',
                action: () => this.setSummaryColumnHidden(column, true),
            },
            { separator: true },
            {
                label: 'Justify Left',
                active: alignment === 'left',
                action: () => this.setSummaryColumnAlignment(column, 'left'),
            },
            {
                label: 'Justify Centre',
                active: alignment === 'center',
                action: () => this.setSummaryColumnAlignment(column, 'center'),
            },
            {
                label: 'Justify Right',
                active: alignment === 'right',
                action: () => this.setSummaryColumnAlignment(column, 'right'),
            },
            { separator: true },
            {
                label: 'Decimal Places...',
                disabled: !supportsDecimals,
                action: () => this.setSummaryColumnDecimals(column),
            },
        ], event);
    }

    initializeSortableLists(data) {
        if (data) {
            this.buildStructureItems(data);
        }

        // Populate selected lists (start with all structures selected)
        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');
        const availableRowsList = document.getElementById('availableRowsList');
        const availableColsList = document.getElementById('availableColsList');

        if (this.sortableRows) {
            this.sortableRows.available.destroy();
            this.sortableRows.selected.destroy();
            this.sortableRows = null;
        }
        if (this.sortableColumns) {
            this.sortableColumns.available.destroy();
            this.sortableColumns.selected.destroy();
            this.sortableColumns = null;
        }

        // Clear all lists
        selectedRowsList.innerHTML = '';
        selectedColsList.innerHTML = '';
        availableRowsList.innerHTML = '';
        availableColsList.innerHTML = '';

        const items = this.structureItems;

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

        this.sortableListsInitialized = true;
        this.sortableInitScheduled = false;
    }

    buildStructureItems(data) {
        if (!data || !data.rows || !data.row_names) {
            return;
        }

        // Collect unique DICOM types and create items.
        const dicomTypes = new Set();
        const items = [];

        data.rows.forEach((roi, idx) => {
            const color = this.rgbToColor(data.colors[roi] || data.colors[String(roi)]);
            const name = data.row_names[idx];
            const dicomType = data.dicom_types ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '') : '';
            const codeMeaning = data.code_meanings ? (data.code_meanings[roi] || data.code_meanings[String(roi)] || '') : '';

            if (dicomType) {
                dicomTypes.add(dicomType);
            }

            items.push({ roi, name, color, dicomType, codeMeaning });
        });

        items.sort((a, b) => {
            if (a.dicomType !== b.dicomType) {
                return a.dicomType.localeCompare(b.dicomType);
            }
            return a.name.localeCompare(b.name);
        });

        this.populateDicomTypeFilters(Array.from(dicomTypes).sort());
        this.structureItems = items;
        this.structureItemsByRoi = new Map();
        items.forEach(item => {
            this.structureItemsByRoi.set(parseInt(item.roi), item);
        });
    }

    ensureSortableListsInitialized() {
        if (this.sortableListsInitialized) {
            return true;
        }
        if (this.structureItems.length === 0 && this.summaryData) {
            this.buildStructureItems(this.summaryData);
        }
        if (this.structureItems.length === 0) {
            return false;
        }
        this.initializeSortableLists();
        return true;
    }

    scheduleDeferredSortableInitialization() {
        if (this.sortableListsInitialized || this.sortableInitScheduled) {
            return;
        }
        this.sortableInitScheduled = true;
        const initAction = () => {
            if (this.sortableListsInitialized) {
                this.sortableInitScheduled = false;
                return;
            }
            this.appendStatusLogLine('frontend', 'Initializing matrix selectors...');
            const stepStart = performance.now();
            this.ensureSortableListsInitialized();
            const stepMs = Math.round(performance.now() - stepStart);
            this.appendStatusLogLine(
                'frontend',
                `Matrix selectors ready (${stepMs} ms).`
            );
        };

        if (typeof window.requestIdleCallback === 'function') {
            window.requestIdleCallback(initAction, { timeout: 1200 });
            return;
        }

        window.setTimeout(initAction, 0);
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

        // Add double-click handler to move between selected and available lists
        item.addEventListener('dblclick', () => {
            const parentList = item.parentElement;
            if (parentList && parentList.id === 'selectedRowsList') {
                // Move from From list to Available From list
                const availableList = document.getElementById('availableRowsList');
                availableList.appendChild(item);
            } else if (parentList && parentList.id === 'selectedColsList') {
                // Move from To list to Available To list
                const availableList = document.getElementById('availableColsList');
                availableList.appendChild(item);
            } else if (parentList && parentList.id === 'availableRowsList') {
                // Move from Available From list to From list
                const selectedList = document.getElementById('selectedRowsList');
                selectedList.appendChild(item);
            } else if (parentList && parentList.id === 'availableColsList') {
                // Move from Available To list to To list
                const selectedList = document.getElementById('selectedColsList');
                selectedList.appendChild(item);
            }
        });

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

        if (!this.ensureSortableListsInitialized()) {
            alert('Matrix selectors are not ready yet. Please try again.');
            return;
        }

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
        const configRelationships = this.symbolConfig?.relationships || {};

        // Build quick lookup maps once per render to avoid per-cell scans.
        const relationshipClassBySymbol = {};
        const relationshipLabelBySymbol = {};
        Object.entries(configRelationships).forEach(([relType, config]) => {
            if (!config?.symbol) {
                return;
            }
            relationshipClassBySymbol[config.symbol] = `rel-${relType.toLowerCase()}`;
            relationshipLabelBySymbol[config.symbol] =
                `${config.label} - ${config.description}`;
        });

        // Build header row
        thead.innerHTML = '<tr><th>Structure</th></tr>';
        const headerRow = thead.querySelector('tr');
        const headerFragment = document.createDocumentFragment();
        data.col_names.forEach(name => {
            const th = document.createElement('th');
            th.textContent = name;
            headerFragment.appendChild(th);
        });
        headerRow.appendChild(headerFragment);

        // Build body rows
        tbody.innerHTML = '';
        const bodyFragment = document.createDocumentFragment();
        data.data.forEach((row, rowIdx) => {
            const tr = document.createElement('tr');

            // Row header
            const th = document.createElement('th');
            th.textContent = data.row_names[rowIdx];
            tr.appendChild(th);

            // Data cells
            row.forEach((value, colIdx) => {
                const td = document.createElement('td');
                td.textContent = value;

                // Add relationship-specific class for color coding
                const relClass = relationshipClassBySymbol[value]
                    || this.getRelationshipClass(value);
                if (relClass) {
                    td.classList.add(relClass);
                }

                // Apply faded class if this relationship should be faded
                if (data.faded_relationships) {
                    const fadeKey = `${rowIdx}_${colIdx}`;
                    if (data.faded_relationships[fadeKey]) {
                        td.classList.add('rel-faded');
                    }
                }

                // Add tooltip with relationship name
                const relLabel = relationshipLabelBySymbol[value]
                    || this.getRelationshipLabel(value);
                if (relLabel) {
                    td.title = relLabel;
                }

                tr.appendChild(td);
            });

            bodyFragment.appendChild(tr);
        });
        tbody.appendChild(bodyFragment);
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

    escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    getArrowMeaning(arrows) {
        if (arrows === 'to') return 'directed';
        if (arrows === 'to;from') return 'bidirectional';
        if (arrows === 'from') return 'reverse directed';
        return 'undirected';
    }

    wrapTooltipText(text, maxChars = 80) {
        const clean = String(text || '').trim().replace(/\s+/g, ' ');
        if (!clean) return '';
        const words = clean.split(' ');
        const lines = [];
        let current = '';
        for (const word of words) {
            if (!current) {
                current = word;
                continue;
            }
            if ((current.length + 1 + word.length) <= maxChars) {
                current += ` ${word}`;
            } else {
                lines.push(current);
                current = word;
            }
        }
        if (current) {
            lines.push(current);
        }
        return lines.join('\n');
    }

    getTooltipRelationshipMode() {
        const edgeCfg = this.tooltipsConfig?.edges || {};
        const rawConfigured = edgeCfg.relationship;
        const configured = String(rawConfigured || '').trim().toLowerCase();
        const validModes = new Set(['label', 'symbol', 'type']);

        if (validModes.has(configured)) {
            return configured;
        }

        if (rawConfigured !== undefined && rawConfigured !== null && configured !== '') {
            const warningKey = String(rawConfigured);
            if (!this._invalidTooltipRelationshipModesWarned.has(warningKey)) {
                this._invalidTooltipRelationshipModesWarned.add(warningKey);
                const warningMessage =
                    `Invalid tooltips.edges.relationship value `
                    + `'${warningKey}'. Falling back to 'label'. `
                    + `Valid options: label, symbol, type.`;
                console.warn(warningMessage);
                this.appendStatusLogLine('frontend', warningMessage);
            }
            return 'label';
        }

        // Backward compatibility with legacy booleans.
        const showSymbol = edgeCfg.show_symbol === true;
        const showRelationship = edgeCfg.show_relationship !== false;
        if (showSymbol && !showRelationship) {
            return 'symbol';
        }
        return 'label';
    }

    isDirectionalEdge(arrows) {
        return arrows === 'to' || arrows === 'from';
    }

    buildEdgeTooltip(edge, nodes = []) {
        const edgeCfg = this.tooltipsConfig?.edges || {};
        const nodeNameById = new Map(nodes.map(node => [node.id, node.label]));
        const sourceLabel = nodeNameById.get(edge.from_node) || `ROI ${edge.from_node}`;
        const targetLabel = nodeNameById.get(edge.to_node) || `ROI ${edge.to_node}`;
        const relationType = String(
            edge.relation_type || String(edge.label || '').replace(/\[|\]/g, '')
        ).toUpperCase();
        const relationConfig = this.symbolConfig?.relationships?.[relationType] || {};
        const baseRelationLabel = relationConfig.label
            || String(edge.label || '').replace(/\[|\]/g, '')
            || relationType;
        const relationLabel = edge.is_logical
            ? `[${baseRelationLabel}]`
            : baseRelationLabel;
        const relationSymbol = edge.symbol || relationConfig.symbol || '?';
        const relationDescription = relationConfig.description || '';
        const relationshipMode = this.getTooltipRelationshipMode();

        let relationToken = relationLabel;
        if (relationshipMode === 'symbol') {
            relationToken = relationSymbol;
        } else if (relationshipMode === 'type') {
            relationToken = relationType;
        }

        const showStructures = edgeCfg.show_structures !== false
            && edgeCfg.show_source !== false
            && edgeCfg.show_target !== false;
        const firstLine = showStructures
            ? `${sourceLabel} ${relationToken} ${targetLabel}`
            : `${relationToken}`;

        const lines = [firstLine];

        if (edgeCfg.show_direction !== false) {
            lines.push(
                `Relationship is ${this.isDirectionalEdge(edge.arrows) ? 'directional' : 'non-directional'}`
            );
        }

        if (edgeCfg.show_logical !== false) {
            lines.push(`Relationship is ${edge.is_logical ? 'logical' : 'not logical'}`);
        }

        if (edgeCfg.show_description !== false && relationDescription) {
            const wrappedDescription = this.wrapTooltipText(relationDescription, 80);
            lines.push(`Description: ${wrappedDescription}`);
        }

        return lines.join('\n');
    }

    updateDiagramLayoutButtons() {
        const manualButton = document.getElementById('manualLayoutBtn');
        const resetButton = document.getElementById('resetLayoutBtn');
        const nodeIds = this.network?.body?.data?.nodes?.getIds?.() || [];
        const hasDiagram = nodeIds.length > 0;

        if (manualButton) {
            manualButton.disabled = !hasDiagram;
            manualButton.textContent = this.manualLayoutActive
                ? 'Manual Layout On'
                : 'Manual Layout';
        }

        if (resetButton) {
            resetButton.disabled = !hasDiagram;
        }
    }

    enableManualLayout() {
        if (!this.network) return;

        const nodeIds = this.network.body.data.nodes.getIds();
        if (!nodeIds.length) return;

        const positions = this.network.getPositions(nodeIds);
        const pinnedNodes = nodeIds.map((id) => {
            const pos = positions[id] || { x: 0, y: 0 };
            this.fixedNodes.add(Number(id));
            return {
                id,
                x: pos.x,
                y: pos.y,
                physics: false,
            };
        });

        this.manualLayoutActive = true;
        this.network.body.data.nodes.update(pinnedNodes);
        this.network.stopSimulation();
        this.updateDiagramLayoutButtons();
        this.appendStatusLogLine('frontend', 'Manual layout enabled for the diagram.');
    }

    resetDiagramLayout() {
        if (!this.latestDiagramData) return;

        this.fixedNodes.clear();
        this.manualLayoutActive = false;
        this._dragFrozen = [];
        this.renderDiagram(this.latestDiagramData);

        if (this.network) {
            this.network.setOptions({ physics: this.getPhysicsFromInfluence() });
            this.network.fit();
            this.network.startSimulation();
        }

        this.appendStatusLogLine('frontend', 'Diagram layout reset to automatic positioning.');
    }

    renderDiagramLegend(data) {
        // Remove any existing inset legend
        const existing = document.getElementById('diagramInsetLegend');
        if (existing) existing.remove();

        // Only show inset legend when edge labels are hidden
        if (this.diagramShowLabelsApplied) return;
        if (!data || !data.edges || data.edges.length === 0) return;

        // Collect unique edge types present in this diagram
        const edgeTypes = new Map();
        data.edges.forEach(edge => {
            const relType = edge.relation_type || String(edge.label || '').replace(/\[|\]/g, '');
            if (!edgeTypes.has(relType)) {
                edgeTypes.set(relType, {
                    color: edge.color,
                    width: edge.width,
                    dashes: edge.dashes,
                    symbol: edge.symbol
                        || this.symbolConfig?.relationships?.[relType]?.symbol
                        || '',
                });
            }
        });

        if (edgeTypes.size === 0) return;

        // Determine text/background colors based on diagram background
        const background = this.diagramOptions.Background || {};
        const bgColor = background.color || '#ffffff';
        const lightColor = this.diagramOptions?.font?.light_color || '#FFFFFF';
        const isDark = this.getTextColor(bgColor) === lightColor;
        const textColor = isDark
            ? lightColor
            : (this.diagramOptions?.font?.dark_color || '#000000');
        const legendBg = isDark ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.80)';
        const borderColor = isDark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.15)';

        // Build a row per unique edge type: swatch | symbol | label
        const items = Array.from(edgeTypes.entries()).map(([relType, meta]) => {
            const symbol = meta.symbol && meta.symbol !== '?'
                ? this.escapeHtml(meta.symbol) : '';
            const swatchClass = meta.dashes ? 'is-dashed' : 'is-solid';
            const symbolHtml = symbol
                ? `<span class="inset-legend-symbol" style="color:${this.escapeHtml(meta.color)}">${symbol}</span>`
                : `<span class="inset-legend-symbol"></span>`;
            return `
                <div class="inset-legend-row">
                    <span class="legend-edge-swatch ${swatchClass}"
                          style="--edge-color:${this.escapeHtml(meta.color)};--edge-width:${Math.max(meta.width || 2, 1)}px;"></span>
                    ${symbolHtml}
                    <span class="inset-legend-label">${this.escapeHtml(relType)}</span>
                </div>
            `;
        }).join('');

        const legend = document.createElement('div');
        legend.id = 'diagramInsetLegend';
        legend.className = 'diagram-inset-legend';
        legend.style.cssText =
            `background:${legendBg};border-color:${borderColor};color:${textColor};`;
        legend.innerHTML = items;

        const container = document.getElementById('networkDiagram');
        if (container) {
            container.appendChild(legend);
        }
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

    toggleContourConfig() {
        const content = document.getElementById('contourConfigContent');
        const toggle = document.getElementById('contourConfigToggle');

        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggle.textContent = '▼';
        } else {
            content.style.display = 'none';
            toggle.textContent = '►';
        }
    }

    formatResolution(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || numeric <= 0) {
            return 'N/A';
        }
        return `${numeric.toFixed(3)} cm/pixel`;
    }

    updateStructureSetBadge() {
        const badgeName = document.getElementById('structureSetBadgeName');
        const tooltip = document.getElementById('structureSetBadgeTooltip');
        if (!badgeName || !tooltip) return;

        const pInfo = this.patientInfo || {};
        const ssCfg = this.tooltipsConfig?.structure_set || {};
        const structureSetName = pInfo.structure_set || 'Structure Set';
        const structureCount = Number(pInfo.structure_count) || this.structureItems.length;
        const tooltipLines = [];
        if (ssCfg.show_structure_set !== false) {
            tooltipLines.push(['Structure Set', structureSetName]);
        }
        if (ssCfg.show_structure_count !== false) {
            tooltipLines.push(['Structures', structureCount ? String(structureCount) : 'N/A']);
        }
        if (ssCfg.show_file_name !== false) {
            tooltipLines.push(['File Name', pInfo.file_name || 'N/A']);
        }
        if (ssCfg.show_series_label !== false) {
            tooltipLines.push(['Series Label', pInfo.series_description || pInfo.series_number || 'N/A']);
        }
        if (ssCfg.show_patient_name !== false) {
            tooltipLines.push(['Patient Name', pInfo.patient_name || 'N/A']);
        }
        if (ssCfg.show_patient_id !== false) {
            tooltipLines.push(['Patient ID', pInfo.patient_id || 'N/A']);
        }
        if (ssCfg.show_resolution !== false) {
            tooltipLines.push(['Resolution', this.formatResolution(pInfo.resolution_cm_per_pixel)]);
        }

        badgeName.textContent = structureSetName;
        tooltip.innerHTML = tooltipLines.map(([label, value]) => (
            `<div class="structure-set-tooltip-line"><span class="structure-set-tooltip-label">${this.escapeHtml(label)}:</span>${this.escapeHtml(value)}</div>`
        )).join('');
    }

    openDiagramStructureModal() {
        const modal = document.getElementById('diagramStructureModal');
        if (!modal) return;

        const searchInput = document.getElementById('diagramSearchInput');
        if (searchInput) {
            searchInput.value = '';
            this.filterDiagramSelectionList('');
        }

        modal.style.display = 'block';
        this.diagramSelectionModalOpen = true;
        document.getElementById('structureSetBadgeTooltip')?.classList.remove('visible');
    }

    closeDiagramStructureModal(revertSelection = false) {
        const modal = document.getElementById('diagramStructureModal');
        if (!modal) return;

        if (revertSelection) {
            this.syncDiagramSelection(this.diagramAppliedSelection);
        }

        modal.style.display = 'none';
        this.diagramSelectionModalOpen = false;
    }

    initializeDiagramSelection(data) {
        const list = document.getElementById('diagramStructureList');
        list.innerHTML = '';

        this.diagramSelection = new Set();
        this.diagramAppliedSelection = new Set();

        this.structureItems.forEach(item => {
            const listItem = this.createDiagramListItem(item);
            list.appendChild(listItem);
            this.diagramSelection.add(parseInt(item.roi));
        });

        this.diagramAppliedSelection = new Set(this.diagramSelection);
        this.diagramShowDisjointApplied = document.getElementById('showDisjointToggle').checked;
        this.diagramShowLabelsApplied = document.getElementById('showLabelsToggle').checked;
        this.diagramLogicalRelationsModeApplied = this.diagramLogicalRelationsMode;
        this.updateDiagramPendingState();
        this.updateStructureSetBadge();

        if (data && data.rows && data.rows.length > 0) {
            this.updateDiagramPendingState();
        }
    }

    createDiagramListItem(item) {
        const wrapper = document.createElement('label');
        wrapper.className = 'diagram-structure-item';
        wrapper.dataset.roi = item.roi;
        wrapper.dataset.name = item.name.toLowerCase();
        wrapper.dataset.dicomType = (item.dicomType || '').toLowerCase();
        wrapper.dataset.codeMeaning = (item.codeMeaning || '').toLowerCase();

        const metaParts = [];
        if (item.dicomType) {
            metaParts.push(item.dicomType);
        }
        if (item.codeMeaning && item.codeMeaning !== item.name) {
            metaParts.push(item.codeMeaning);
        }

        const metaText = metaParts.join(' • ');

        wrapper.innerHTML = `
            <input type="checkbox" data-roi="${item.roi}" checked>
            <span class="item-color" style="background-color: ${item.color}"></span>
            <span class="item-id">${item.roi}</span>
            <span class="diagram-structure-text">
                <span class="item-name">${item.name}</span>
                ${metaText ? `<span class="diagram-structure-meta">${metaText}</span>` : ''}
            </span>
        `;

        const checkbox = wrapper.querySelector('input[type="checkbox"]');
        checkbox.addEventListener('change', (e) => {
            const roi = parseInt(e.target.dataset.roi);
            if (e.target.checked) {
                this.diagramSelection.add(roi);
            } else {
                this.diagramSelection.delete(roi);
            }
            this.updateDiagramPendingState();
        });

        return wrapper;
    }

    filterDiagramSelectionList(query) {
        const normalized = query.trim().toLowerCase();
        const tokens = normalized.split(/\s+/).filter(Boolean);
        const items = document.querySelectorAll('.diagram-structure-item');

        items.forEach(item => {
            if (!normalized) {
                item.style.display = '';
                return;
            }

            const roiText = item.dataset.roi || '';
            const nameText = item.dataset.name || '';
            const typeText = item.dataset.dicomType || '';
            const labelText = item.dataset.codeMeaning || '';
            const haystack = [roiText, nameText, typeText, labelText].join(' ');
            const matchesAny = tokens.some(token => haystack.includes(token));

            item.style.display = matchesAny ? '' : 'none';
        });
    }

    applyDiagramSelection() {
        this.diagramAppliedSelection = new Set(this.diagramSelection);
        this.hiddenNodes = new Set(
            this.structureItems
                .map(item => parseInt(item.roi))
                .filter(roi => !this.diagramAppliedSelection.has(roi))
        );
        this.diagramShowDisjointApplied = document.getElementById('showDisjointToggle').checked;
        this.diagramShowLabelsApplied = document.getElementById('showLabelsToggle').checked;
        this.diagramLogicalRelationsModeApplied = this.diagramLogicalRelationsMode;
        this.updateDiagramPendingState();
        return this.refreshDiagram();
    }

    setDiagramPending(isPending) {
        this.diagramSelectionPending = isPending;
        const applyButton = document.getElementById('applyDiagramBtn');
        applyButton.disabled = !isPending;

        const badge = document.getElementById('structureSetBadge');
        if (badge) {
            badge.classList.toggle('pending', isPending);
        }
    }

    updateDiagramPendingState() {
        const selectionChanged = !this.areSetsEqual(
            this.diagramSelection,
            this.diagramAppliedSelection
        );
        const showDisjoint = document.getElementById('showDisjointToggle').checked;
        const showLabels = document.getElementById('showLabelsToggle').checked;
        const togglesChanged =
            showDisjoint !== this.diagramShowDisjointApplied ||
            showLabels !== this.diagramShowLabelsApplied;
        const logicalChanged =
            this.diagramLogicalRelationsMode !== this.diagramLogicalRelationsModeApplied;

        this.setDiagramPending(selectionChanged || togglesChanged || logicalChanged);
    }

    getDiagramAppliedRois() {
        const applied = this.diagramAppliedSelection;
        return this.structureItems
            .map(item => parseInt(item.roi))
            .filter(roi => applied.has(roi));
    }

    areSetsEqual(a, b) {
        if (a.size !== b.size) return false;
        for (const value of a) {
            if (!b.has(value)) return false;
        }
        return true;
    }

    syncDiagramSelection(newSelection) {
        this.diagramSelection = new Set(newSelection);
        this.updateDiagramPendingState();

        const checkboxes = document.querySelectorAll('#diagramStructureList input[type="checkbox"]');
        checkboxes.forEach(cb => {
            const roi = parseInt(cb.dataset.roi);
            cb.checked = this.diagramSelection.has(roi);
        });
    }

    async refreshDiagram() {
        if (!this.sessionId) {
            console.warn('No session ID available for diagram');
            return;
        }

        try {
            const selectedRois = this.getDiagramAppliedRois();
            if (selectedRois.length === 0) {
                alert('Please select at least one structure for the diagram');
                return;
            }

            const showDisjoint = this.diagramShowDisjointApplied;

            const diagramRequest = {
                session_id: this.sessionId,
                row_rois: selectedRois,
                col_rois: selectedRois,
                show_disjoint: showDisjoint,
                show_unknown: this.diagramShowUnknownApplied,
                logical_relations_mode: this.diagramLogicalRelationsModeApplied
            };
            console.log('Sending diagram request:', diagramRequest);

            this.appendStatusLogLine('frontend', 'Fetching relationship diagram...');
            const diagramFetchStart = performance.now();
            const response = await fetch('/api/diagram', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(diagramRequest)
            });

            if (!response.ok) {
                throw new Error('Failed to fetch diagram data');
            }

            const data = await response.json();
            const diagramFetchMs = Math.round(performance.now() - diagramFetchStart);
            console.log('Diagram response received:', { nodes: data.nodes.length, edges: data.edges.length });
            console.log('Diagram edges:', data.edges.map(e => `(${e.from_node}->${e.to_node}): ${e.label}`));
            this.appendStatusLogLine(
                'frontend',
                `Diagram data received (${diagramFetchMs} ms): ${data.nodes.length} node(s), ${data.edges.length} edge(s). Rendering...`
            );

            const diagramRenderStart = performance.now();
            this.renderDiagram(data);
            const diagramRenderMs = Math.round(performance.now() - diagramRenderStart);
            this.appendStatusLogLine(
                'frontend',
                `Diagram rendered in ${diagramRenderMs} ms.`
            );

        } catch (error) {
            console.error('Error refreshing diagram:', error);
            alert('Failed to generate relationship diagram');
        }
    }

    renderDiagram(data) {
        const container = document.getElementById('networkDiagram');
        const showLabels = this.diagramShowLabelsApplied;
        const nodeFont = this.diagramOptions.font || {};
        const interaction = this.diagramOptions.interaction || {};
        const background = this.diagramOptions.Background || {};
        const layoutSettings = this.diagramOptions.layout || {};
        this.latestDiagramData = data;

        if (background.color) {
            container.style.backgroundColor = background.color;
            const textColor = this.getTextColor(background.color);
            const lightColor = this.diagramOptions?.font?.light_color || '#FFFFFF';
            const isDark = textColor === lightColor;
            container.style.setProperty(
                '--vis-nav-filter',
                isDark ? 'invert(1) brightness(1.8)' : 'none'
            );
        }

        // Keep state sets aligned with currently rendered nodes.
        const renderedNodeIds = new Set(data.nodes.map(node => Number(node.id)));
        this.hiddenNodes = new Set(Array.from(this.hiddenNodes).filter(id => renderedNodeIds.has(Number(id))));
        this.hiddenLabels = new Set(Array.from(this.hiddenLabels).filter(id => renderedNodeIds.has(Number(id))));
        this.fixedNodes = new Set(Array.from(this.fixedNodes).filter(id => renderedNodeIds.has(Number(id))));
        this.manualLayoutActive = renderedNodeIds.size > 0
            && Array.from(renderedNodeIds).every(id => this.fixedNodes.has(Number(id)));

        // Prepare nodes for vis-network
        const nodes = data.nodes.map(node => {
            const roi = Number(node.id);
            const originalLabel = node.label;
            const labelHidden = this.hiddenLabels.has(roi);
            const isPhysicsFixed = this.fixedNodes.has(roi);
            return {
                id: roi,
                _originalLabel: originalLabel,
                label: labelHidden ? '' : originalLabel,
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
                size: Number(nodeFont.node_size || 14),
                face: nodeFont.face || 'Arial'
            },
                borderWidth: 2,
                physics: !isPhysicsFixed
            };
        });

        // Prepare edges for vis-network
        const edges = data.edges.map(edge => {
            const relationType = String(edge.relation_type || '').toUpperCase();
            const relationConfig = this.symbolConfig?.relationships?.[relationType] || {};
            const baseLabel = relationConfig.label || edge.label || relationType;
            const displayLabel = edge.is_logical && baseLabel && !String(baseLabel).startsWith('[')
                ? `[${baseLabel}]`
                : baseLabel;
            return ({
            from: edge.from_node,
            to: edge.to_node,
            label: showLabels ? displayLabel : '',
            originalLabel: displayLabel,
            title: this.buildEdgeTooltip(edge, data.nodes),
            color: edge.color,
            width: edge.width,
            dashes: edge.dashes,
            arrows: edge.arrows ? edge.arrows : undefined,
            font: {
                size: Number(nodeFont.edge_size || 12),
                color: edge.color,
                strokeWidth: 0
            },
            smooth: {
                type: 'continuous',
                roundness: 0.5
            }
        });
        });

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
                ...this.getPhysicsFromInfluence()
            },
            interaction: {
                hover: interaction.hover !== false,
                tooltipDelay: Number(interaction.tooltipDelay || 100),
                navigationButtons: interaction.navigationButtons !== false,
                keyboard: interaction.keyboard !== false
            },
            layout: {
                improvedLayout: layoutSettings.improvedLayout !== false,
                hierarchical: layoutSettings.hierarchical === true
            }
        };

        // Destroy existing network if present
        if (this.network) {
            this.network.destroy();
        }

        // Create new network
        this.network = new vis.Network(container, { nodes, edges }, options);

        // Add event listeners
        this.network.on('click', () => this._dismissContextMenu());

        // Right-click: show node context menu
        this.network.on('oncontext', (params) => {
            params.event.preventDefault();
            this._dismissContextMenu();
            const domPoint = params.pointer?.DOM;
            const hitNode = domPoint ? this.network.getNodeAt(domPoint) : null;
            const targetNode = hitNode ?? (params.nodes.length > 0 ? params.nodes[0] : null);
            if (targetNode !== null && targetNode !== undefined) {
                this._showNodeContextMenu(Number(targetNode), params.event);
            }
        });

        // Double-click: remove structure from diagram
        this.network.on('doubleClick', (params) => {
            if (params.nodes.length > 0) {
                const roi = params.nodes[0];
                this.removeStructureFromDiagram(roi);
            }
        });

        // Drag-start: when Local is 100%, freeze all OTHER nodes so only the
        // dragged node moves.  Fixed nodes stay fixed regardless.
        this.network.on('dragStart', (params) => {
            if (params.nodes.length === 0) return;
            const min = Number(this.diagramOptions.layout.local_global_min ?? 0);
            if (this.layoutInfluence !== min) return;   // only at pure-local
            const draggedId = params.nodes[0];
            const allIds = this.network.body.data.nodes.getIds();
            const toFreeze = [];
            for (const id of allIds) {
                if (id === draggedId) continue;
                if (this.fixedNodes.has(id)) continue;  // already fixed
                const pos = this.network.getPositions([id])[id];
                toFreeze.push({ id, x: pos.x, y: pos.y, physics: false });
            }
            this._dragFrozen = toFreeze.map(n => n.id);
            this.network.body.data.nodes.update(toFreeze);
        });

        // Drag-end: unfreeze nodes that were only frozen for this drag
        this.network.on('dragEnd', () => {
            if (!this._dragFrozen || this._dragFrozen.length === 0) return;
            const toThaw = this._dragFrozen
                .filter(id => !this.fixedNodes.has(id))
                .map(id => {
                    const pos = this.network.getPositions([id])[id];
                    return { id, x: pos.x, y: pos.y, physics: true };
                });
            if (toThaw.length > 0) {
                this.network.body.data.nodes.update(toThaw);
            }
            this._dragFrozen = [];
        });

        this.renderDiagramLegend(data);
        this.updateDiagramLayoutButtons();
    }

    removeStructureFromDiagram(roi) {
        const checkbox = document.querySelector(
            `#diagramStructureList input[type="checkbox"][data-roi="${roi}"]`
        );
        if (checkbox) {
            checkbox.checked = false;
            this.diagramSelection.delete(parseInt(roi));
            this.updateDiagramPendingState();
        }
    }

    _dismissContextMenu() {
        if (this._contextMenu) {
            this._contextMenu.remove();
            this._contextMenu = null;
        }
    }

    _showNodeContextMenu(roi, event) {
        const normalizedRoi = Number(roi);
        const node = this.network?.body?.data?.nodes?.get(normalizedRoi);
        if (!node) return;

        const isDisplayed = this.diagramAppliedSelection.has(normalizedRoi);
        const isFixed = this.fixedNodes.has(roi);
        const isLabelHidden = this.hiddenLabels.has(normalizedRoi);
        const isHidden = !isDisplayed;

        const menu = document.createElement('div');
        menu.className = 'node-context-menu';
        menu.style.left = `${event.clientX}px`;
        menu.style.top = `${event.clientY}px`;

        const items = [
            {
                label: isFixed ? 'Unfix Position' : 'Fix Position',
                active: isFixed,
                action: () => this._ctxToggleFixed(normalizedRoi),
            },
            { separator: true },
            {
                label: isLabelHidden ? 'Show Label' : 'Hide Label',
                active: isLabelHidden,
                action: () => this._ctxToggleLabel(normalizedRoi),
            },
            {
                label: isHidden ? 'Show Structure' : 'Hide Structure',
                active: isHidden,
                action: () => this._ctxToggleVisibility(normalizedRoi),
            },
        ];

        for (const item of items) {
            if (item.separator) {
                const sep = document.createElement('div');
                sep.className = 'node-context-menu-separator';
                menu.appendChild(sep);
                continue;
            }
            const el = document.createElement('div');
            el.className = 'node-context-menu-item' + (item.active ? ' is-active' : '');
            el.textContent = item.label;
            el.addEventListener('mousedown', (e) => {
                e.stopPropagation();
                item.action();
                this._dismissContextMenu();
            });
            menu.appendChild(el);
        }

        document.body.appendChild(menu);
        this._contextMenu = menu;

        // Flip menu if it would overflow the viewport
        const rect = menu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            menu.style.left = `${event.clientX - rect.width}px`;
        }
        if (rect.bottom > window.innerHeight) {
            menu.style.top = `${event.clientY - rect.height}px`;
        }

        // Dismiss on next outside click
        const dismiss = (e) => {
            if (!menu.contains(e.target)) {
                this._dismissContextMenu();
                document.removeEventListener('mousedown', dismiss, true);
            }
        };
        document.addEventListener('mousedown', dismiss, true);
    }

    _ctxToggleFixed(roi) {
        if (!this.network) return;
        if (this.fixedNodes.has(roi)) {
            this.fixedNodes.delete(roi);
            const pos = this.network.getPositions([roi])[roi];
            this.network.body.data.nodes.update([
                { id: roi, x: pos.x, y: pos.y, physics: true },
            ]);
        } else {
            this.fixedNodes.add(roi);
            const pos = this.network.getPositions([roi])[roi];
            this.network.body.data.nodes.update([
                { id: roi, x: pos.x, y: pos.y, physics: false },
            ]);
        }

        const nodeIds = this.network.body.data.nodes.getIds();
        this.manualLayoutActive = nodeIds.length > 0
            && nodeIds.every(id => this.fixedNodes.has(Number(id)));
        this.updateDiagramLayoutButtons();
    }

    _ctxToggleLabel(roi) {
        if (!this.network) return;
        const node = this.network.body.data.nodes.get(roi);
        if (!node) return;
        if (this.hiddenLabels.has(roi)) {
            this.hiddenLabels.delete(roi);
            this.network.body.data.nodes.update([
                { id: roi, label: node._originalLabel || node.label || String(roi) },
            ]);
        } else {
            this.hiddenLabels.add(roi);
            const original = node._originalLabel || node.label || String(roi);
            this.network.body.data.nodes.update([
                { id: roi, _originalLabel: original, label: '' },
            ]);
        }
    }

    _ctxToggleVisibility(roi) {
        if (!this.network) return;

        const shouldShow = !this.diagramAppliedSelection.has(roi);

        if (shouldShow) {
            this.diagramSelection.add(roi);
            this.diagramAppliedSelection.add(roi);
            this.hiddenNodes.delete(roi);
        } else {
            this.diagramSelection.delete(roi);
            this.diagramAppliedSelection.delete(roi);
            this.hiddenNodes.add(roi);
        }

        const checkbox = document.querySelector(
            `#diagramStructureList input[type="checkbox"][data-roi="${roi}"]`
        );
        if (checkbox) {
            checkbox.checked = shouldShow;
        }

        this.updateDiagramPendingState();
        this.refreshDiagram();
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
        if (!this.ensureSortableListsInitialized()) {
            return;
        }
        const availableList = document.getElementById(axis === 'rows' ? 'availableRowsList' : 'availableColsList');
        const selectedList = document.getElementById(axis === 'rows' ? 'selectedRowsList' : 'selectedColsList');

        // Move all items from available to selected
        while (availableList.firstChild) {
            selectedList.appendChild(availableList.firstChild);
        }
    }

    selectNoneAxisStructures(axis) {
        if (!this.ensureSortableListsInitialized()) {
            return;
        }
        const availableList = document.getElementById(axis === 'rows' ? 'availableRowsList' : 'availableColsList');
        const selectedList = document.getElementById(axis === 'rows' ? 'selectedRowsList' : 'selectedColsList');

        // Move all items from selected to available
        while (selectedList.firstChild) {
            availableList.appendChild(selectedList.firstChild);
        }
    }

    copyRowsToColumns() {
        if (!this.ensureSortableListsInitialized()) {
            return;
        }
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

    copyFromMatrixToDiagram() {
        if (!this.ensureSortableListsInitialized()) {
            return;
        }
        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');

        const rowRois = Array.from(selectedRowsList.children)
            .map(item => parseInt(item.dataset.roi));
        const colRois = Array.from(selectedColsList.children)
            .map(item => parseInt(item.dataset.roi));

        const combined = new Set([...rowRois, ...colRois]);
        this.syncDiagramSelection(combined);
    }

    copyFromDiagramToMatrix() {
        if (!this.ensureSortableListsInitialized()) {
            return;
        }
        const selectedRowsList = document.getElementById('selectedRowsList');
        const selectedColsList = document.getElementById('selectedColsList');
        const availableRowsList = document.getElementById('availableRowsList');
        const availableColsList = document.getElementById('availableColsList');

        selectedRowsList.innerHTML = '';
        selectedColsList.innerHTML = '';
        availableRowsList.innerHTML = '';
        availableColsList.innerHTML = '';

        const selectionSet = new Set(this.diagramSelection);
        this.structureItems.forEach(item => {
            const rowItem = this.createSortableItem(item.roi, item.name, item.color, item.dicomType, item.codeMeaning);
            const colItem = this.createSortableItem(item.roi, item.name, item.color, item.dicomType, item.codeMeaning);

            if (selectionSet.has(parseInt(item.roi))) {
                selectedRowsList.appendChild(rowItem);
                selectedColsList.appendChild(colItem);
            } else {
                availableRowsList.appendChild(rowItem);
                availableColsList.appendChild(colItem);
            }
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

        const darkColor = this.diagramOptions?.font?.dark_color || '#000000';
        const lightColor = this.diagramOptions?.font?.light_color || '#FFFFFF';
        return brightness > 128 ? darkColor : lightColor;
    }

    populateContourPlotControls(data) {
        this.clearContourPlotCache();
        this.allSliceIndices = data.slice_indices || [];
        this.allSliceIndicesOriginal = data.slice_indices_original || this.allSliceIndices;
        this.structureSlices = data.structure_slices || {};
        this.structureSlicesOriginal = data.structure_slices_original || this.structureSlices;
        this.structureSlicesInterpolated =
            data.structure_slices_interpolated || {};
        this.sliceRelationships = this.normalizeSliceRelationships(
            data.slice_relationships
        );

        const includeInterpolatedToggle = document.getElementById(
            'includeInterpolatedSlicesToggle'
        );
        includeInterpolatedToggle.checked = false;
        document.getElementById('plotModeSelect').value = 'relationship';
        document.getElementById('relationshipOverlaySelect').value = 'none';
        document.getElementById('showLegendToggle').checked = true;
        document.getElementById('showAxisToggle').checked = false;
        document.getElementById('showToleranceToggle').checked = true;
        document.getElementById('toleranceInput').value = '0.10';
        document.getElementById('toleranceInput').disabled = false;

        this.contourStructureLabels = {};
        this.contourStructureColors = {};
        data.rows.forEach((roi, idx) => {
            const name = data.row_names[idx];
            const dicomType = data.dicom_types
                ? (data.dicom_types[roi] || data.dicom_types[String(roi)] || '')
                : '';
            const label = dicomType ? `${name} (${dicomType})` : name;
            this.contourStructureLabels[String(roi)] = label;
            this.contourStructureColors[String(roi)] = this.rgbToColor(
                data.colors[roi] || data.colors[String(roi)]
            );
        });

        this.initializeContourSortableLists(data);
        this.updateRelationshipOverlayState();

        this.sliceIndices = this.getDefaultSliceIndices();
        this.updateSliceRangeForStructures();
    }

    initializeContourSortableLists(data) {
        if (data) {
            this.buildStructureItems(data);
        }

        const availableList = document.getElementById('availableContourList');
        const selectedList = document.getElementById('selectedContourList');

        if (this.contourSortables) {
            this.contourSortables.available.destroy();
            this.contourSortables.selected.destroy();
            this.contourSortables = null;
        }

        availableList.innerHTML = '';
        selectedList.innerHTML = '';

        this.structureItems.forEach((item) => {
            const contourItem = this.createContourSortableItem(
                item.roi,
                item.name,
                item.color,
                item.dicomType,
                item.codeMeaning
            );
            availableList.appendChild(contourItem);
        });

        const sortableConfig = {
            group: 'contour-structures',
            animation: 150,
            ghostClass: 'dragging',
            onSort: () => this.updateContourSelectionState(),
            onAdd: () => this.updateContourSelectionState(),
            onRemove: () => this.updateContourSelectionState(),
            onEnd: () => this.updateContourSelectionState(),
        };

        this.contourSortables = {
            available: new Sortable(availableList, sortableConfig),
            selected: new Sortable(selectedList, sortableConfig),
        };
    }

    createContourSortableItem(roi, name, color, dicomType = '', codeMeaning = '') {
        const item = this.createSortableItem(roi, name, color, dicomType, codeMeaning);

        item.addEventListener('dblclick', () => {
            const parentList = item.parentElement;
            if (parentList && parentList.id === 'selectedContourList') {
                document.getElementById('availableContourList').appendChild(item);
                this.updateContourSelectionState();
            } else if (parentList && parentList.id === 'availableContourList') {
                document.getElementById('selectedContourList').appendChild(item);
                this.updateContourSelectionState();
            }
        });

        return item;
    }

    updateContourSelectionState() {
        this.updateRelationshipOverlayState();
        this.updateSliceRangeForStructures();
    }

    getSelectedContourRois() {
        const selectedList = document.getElementById('selectedContourList');
        if (!selectedList) {
            return [];
        }
        return Array.from(selectedList.children)
            .map((item) => Number.parseInt(item.dataset.roi, 10))
            .filter((roi) => Number.isInteger(roi));
    }

    selectAllContourStructures() {
        const availableList = document.getElementById('availableContourList');
        const selectedList = document.getElementById('selectedContourList');
        Array.from(availableList.children).forEach((item) => {
            selectedList.appendChild(item);
        });
        this.updateContourSelectionState();
    }

    clearContourStructures() {
        const availableList = document.getElementById('availableContourList');
        const selectedList = document.getElementById('selectedContourList');
        Array.from(selectedList.children).forEach((item) => {
            availableList.appendChild(item);
        });
        this.updateContourSelectionState();
    }

    getRelationshipOverlayOptions(selectedCount) {
        if (selectedCount <= 0) {
            return [];
        }
        if (selectedCount === 1) {
            return [{ value: 'single_structure', label: 'Structure A' }];
        }
        if (selectedCount === 2) {
            return [{ value: 'none', label: 'A vs B' }];
        }
        if (selectedCount === 3) {
            return [
                { value: 'third_structure', label: 'A vs B, C outline' },
                { value: 'intersection_vs_c', label: '(A AND B) vs C' },
                { value: 'union_vs_c', label: '(A OR B) vs C' },
                { value: 'xor_vs_c', label: '(A XOR B) vs C' },
                { value: 'difference_vs_c', label: '(A - B) vs C' },
            ];
        }
        return [
            {
                value: 'all_outlines',
                label: 'A vs B, outlines of all other structures',
            },
        ];
    }

    setRelationshipOverlayOptions(selectedCount) {
        const overlaySelect = document.getElementById('relationshipOverlaySelect');
        const currentValue = overlaySelect ? overlaySelect.value : 'none';
        const options = this.getRelationshipOverlayOptions(selectedCount);

        overlaySelect.innerHTML = '';

        if (options.length === 0) {
            return currentValue;
        }

        options.forEach((option) => {
            const optionElement = document.createElement('option');
            optionElement.value = option.value;
            optionElement.textContent = option.label;
            overlaySelect.appendChild(optionElement);
        });

        const hasCurrentValue = options.some(
            (option) => option.value === currentValue
        );
        overlaySelect.value = hasCurrentValue ? currentValue : options[0].value;
        return overlaySelect.value;
    }

    getRequiredRelationshipOverlayCount(overlayMode) {
        if (overlayMode === 'single_structure') {
            return 1;
        }
        if (
            [
                'third_structure',
                'intersection_vs_c',
                'union_vs_c',
                'xor_vs_c',
                'difference_vs_c',
                'all_outlines',
            ].includes(overlayMode)
        ) {
            return 3;
        }
        return 2;
    }

    getRelationshipPlotRois(selectedRois, overlayMode) {
        if (overlayMode === 'single_structure') {
            return selectedRois.slice(0, 1);
        }
        if (
            [
                'third_structure',
                'intersection_vs_c',
                'union_vs_c',
                'xor_vs_c',
                'difference_vs_c',
            ].includes(overlayMode)
        ) {
            return selectedRois.slice(0, 3);
        }
        if (overlayMode === 'all_outlines') {
            return [...selectedRois];
        }
        return selectedRois.slice(0, 2);
    }

    updateRelationshipOverlayState() {
        const mode = this.getContourPlotMode();
        const overlaySelect = document.getElementById('relationshipOverlaySelect');
        const overlayHelp = document.getElementById('relationshipOverlayHelp');
        const selectedRois = this.getSelectedContourRois();
        const isRelationshipMode = mode === 'relationship';
        const activeOverlay = this.setRelationshipOverlayOptions(selectedRois.length);

        overlaySelect.disabled = !isRelationshipMode || selectedRois.length === 0;
        if (!isRelationshipMode) {
            overlayHelp.textContent = 'Overlay options are only used in relationship mode.';
            return;
        }

        if (selectedRois.length === 0) {
            overlayHelp.textContent = 'Select at least one structure to plot.';
            return;
        }

        const firstLabel = this.contourStructureLabels[String(selectedRois[0])] || 'Structure 1';
        if (selectedRois.length === 1) {
            overlayHelp.textContent = `A = ${firstLabel}.`;
            return;
        }

        const secondLabel = this.contourStructureLabels[String(selectedRois[1])] || 'Structure 2';
        if (selectedRois.length === 2) {
            overlayHelp.textContent = `A = ${firstLabel}; B = ${secondLabel}.`;
            return;
        }

        if (activeOverlay === 'all_outlines' || selectedRois.length > 3) {
            overlayHelp.textContent = `A = ${firstLabel}; B = ${secondLabel}; all additional selected structures are shown as outlines.`;
            return;
        }

        const thirdLabel = this.contourStructureLabels[String(selectedRois[2])] || 'Structure 3';
        overlayHelp.textContent = `A = ${firstLabel}; B = ${secondLabel}; C = ${thirdLabel}.`;
    }

    isInterpolatedSliceEnabled() {
        const toggle = document.getElementById('includeInterpolatedSlicesToggle');
        return !!(toggle && toggle.checked);
    }

    getSliceKey(sliceValue) {
        return Number(sliceValue).toFixed(4);
    }

    getDefaultSliceIndices() {
        if (this.isInterpolatedSliceEnabled()) {
            return [...this.allSliceIndices];
        }
        return [...this.allSliceIndicesOriginal];
    }

    getSlicesForStructure(roi, includeInterpolated) {
        if (!roi) {
            return [];
        }
        const key = String(parseInt(roi));
        const sourceMap = includeInterpolated
            ? this.structureSlices
            : this.structureSlicesOriginal;
        return sourceMap[key] || sourceMap[parseInt(roi)] || [];
    }

    normalizeSliceRelationships(rawRelationships) {
        const normalized = {};
        if (!rawRelationships || typeof rawRelationships !== 'object') {
            return normalized;
        }

        Object.entries(rawRelationships).forEach(([sliceKey, records]) => {
            const parsedSlice = Number.parseFloat(sliceKey);
            const canonicalKey = Number.isNaN(parsedSlice)
                ? String(sliceKey)
                : this.getSliceKey(parsedSlice);

            if (!normalized[canonicalKey]) {
                normalized[canonicalKey] = [];
            }

            if (Array.isArray(records)) {
                normalized[canonicalKey].push(...records);
            }
        });

        return normalized;
    }

    getSliceRelationshipSummaries(sliceValue, selectedRois) {
        if (!selectedRois || selectedRois.length < 2) {
            return [];
        }

        const roiA = Number.parseInt(selectedRois[0], 10);
        const roiB = Number.parseInt(selectedRois[1], 10);
        const targetPair = [roiA, roiB].sort((a, b) => a - b);
        const sliceKey = this.getSliceKey(sliceValue);
        const records = this.sliceRelationships[sliceKey] || [];

        return records.filter((record) => (
            Array.isArray(record.rois)
            && record.rois.length === 2
            && (() => {
                const pair = record.rois
                    .map((roi) => Number.parseInt(roi, 10))
                    .sort((a, b) => a - b);
                return pair[0] === targetPair[0] && pair[1] === targetPair[1];
            })()
        ));
    }

    getSliceRelationshipStatus(sliceValue, selectedRois, includeInterpolated) {
        if (!selectedRois || selectedRois.length < 2) {
            return '';
        }

        const roiA = Number.parseInt(selectedRois[0], 10);
        const roiB = Number.parseInt(selectedRois[1], 10);
        const sliceKey = this.getSliceKey(sliceValue);
        const labelA = this.contourStructureLabels[String(roiA)] || `ROI ${roiA}`;
        const labelB = this.contourStructureLabels[String(roiB)] || `ROI ${roiB}`;

        const hasA = this.getSlicesForStructure(roiA, includeInterpolated)
            .some((value) => this.getSliceKey(value) === sliceKey);
        const hasB = this.getSlicesForStructure(roiB, includeInterpolated)
            .some((value) => this.getSliceKey(value) === sliceKey);

        if (hasA && !hasB) {
            return `${labelA} only`;
        }
        if (!hasA && hasB) {
            return `${labelB} only`;
        }
        if (!hasA && !hasB) {
            return '';
        }

        const relationshipSummaries = this.getSliceRelationshipSummaries(
            sliceValue,
            [roiA, roiB]
        );

        const relationLabels = Array.from(new Set(
            relationshipSummaries
                .map((record) => String(record.relation_type || '').trim())
                .filter((label) => label.length > 0)
        ));

        const preferredRelationship = relationLabels.find(
            (label) => !label.toLowerCase().includes('unknown')
        ) || relationLabels[0] || 'has an Unknown relationship with';

        return `${labelA} ${preferredRelationship} ${labelB}`;
    }

    updateSliceDropdownOptions() {
        const dropdown = document.getElementById('sliceDropdown');
        const includeInterpolated = this.isInterpolatedSliceEnabled();
        const selectedRois = this.getSelectedContourRois();
        const originalSliceKeys = new Set(
            (this.allSliceIndicesOriginal || []).map((slice) => this.getSliceKey(slice))
        );

        dropdown.innerHTML = '';
        if (!this.sliceIndices || this.sliceIndices.length === 0) {
            const emptyOption = document.createElement('option');
            emptyOption.value = '0';
            emptyOption.textContent = 'No slices available';
            dropdown.appendChild(emptyOption);
            return;
        }

        this.sliceIndices.forEach((sliceValue, idx) => {
            const option = document.createElement('option');
            option.value = idx;

            const sliceKey = this.getSliceKey(sliceValue);
            const sliceLabel = `${sliceValue.toFixed(2)} cm`;
            const interpolationLabel =
                includeInterpolated && !originalSliceKeys.has(sliceKey)
                    ? ' [Interpolated]'
                    : '';
            const relationshipStatus = this.getSliceRelationshipStatus(
                sliceValue,
                selectedRois,
                includeInterpolated
            );
            const relationshipLabel = relationshipStatus
                ? ` | ${relationshipStatus}`
                : '';
            option.textContent = `${sliceLabel}${interpolationLabel}${relationshipLabel}`;
            dropdown.appendChild(option);
        });
    }

    updateSliceRangeForStructures() {
        const selectedRois = this.getSelectedContourRois();
        const includeInterpolated = this.isInterpolatedSliceEnabled();

        let filteredSlices = [];
        if (selectedRois.length > 0) {
            const slicePool = new Set();
            selectedRois.forEach((roi) => {
                this.getSlicesForStructure(roi, includeInterpolated).forEach((slice) => {
                    slicePool.add(slice);
                });
            });
            filteredSlices = Array.from(slicePool).sort((a, b) => a - b);
        } else {
            filteredSlices = this.getDefaultSliceIndices();
        }

        const slider = document.getElementById('sliceSlider');
        const oldSliceIndices = this.sliceIndices || [];
        const currentIndex = parseInt(slider.value);
        const currentSliceValue = oldSliceIndices[currentIndex];

        this.sliceIndices = filteredSlices;

        if (filteredSlices.length > 0) {
            slider.min = 0;
            slider.max = filteredSlices.length - 1;
            slider.step = 1;

            let closestIndex = Math.floor(filteredSlices.length / 2);
            if (typeof currentSliceValue === 'number') {
                let minDiff = Infinity;
                filteredSlices.forEach((slice, idx) => {
                    const diff = Math.abs(slice - currentSliceValue);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestIndex = idx;
                    }
                });
            }
            slider.value = closestIndex;
            this.updateSliceValue(closestIndex);
        } else {
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            document.getElementById('sliceValue').textContent = '0.0';
        }

        this.updateSliceDropdownOptions();
        document.getElementById('sliceDropdown').value = slider.value;
        this.updateSliderArrows();
    }

    updateSliceValue(sliderValue) {
        if (this.sliceIndices && this.sliceIndices.length > 0) {
            const index = parseInt(sliderValue);
            const sliceValue = this.sliceIndices[index];
            document.getElementById('sliceValue').textContent = sliceValue.toFixed(2);
            document.getElementById('sliceDropdown').value = String(index);
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
            this.plotContours({ suppressAlerts: true });
        }
    }

    getContourPlotMode() {
        const modeSelect = document.getElementById('plotModeSelect');
        return modeSelect ? modeSelect.value : 'contour';
    }

    getRelationshipOverlayMode() {
        const overlaySelect = document.getElementById('relationshipOverlaySelect');
        return overlaySelect ? overlaySelect.value : 'none';
    }

    shouldShowContourLegend() {
        const legendToggle = document.getElementById('showLegendToggle');
        return !!(legendToggle && legendToggle.checked);
    }

    shouldShowContourAxis() {
        const axisToggle = document.getElementById('showAxisToggle');
        return !!(axisToggle && axisToggle.checked);
    }

    getContourTolerance() {
        const toleranceToggle = document.getElementById('showToleranceToggle');
        const toleranceInput = document.getElementById('toleranceInput');

        if (!toleranceToggle || !toleranceToggle.checked) {
            return 0.0;
        }

        const parsed = Number.parseFloat(toleranceInput.value);
        if (!Number.isFinite(parsed) || parsed < 0) {
            return 0.0;
        }
        return parsed;
    }

    getContourPlotPayload() {
        const selectedRois = this.getSelectedContourRois();
        const sliderValue = document.getElementById('sliceSlider').value;
        const plotMode = this.getContourPlotMode();
        const overlayMode = this.getRelationshipOverlayMode();
        const roiList = plotMode === 'relationship'
            ? this.getRelationshipPlotRois(selectedRois, overlayMode)
            : selectedRois;
        const sliceIndex = this.sliceIndices[parseInt(sliderValue)];

        return {
            selectedRois,
            plotMode,
            overlayMode,
            roiList,
            payload: {
                session_id: this.sessionId,
                roi_list: roiList,
                slice_index: sliceIndex,
                include_interpolated_slices: this.isInterpolatedSliceEnabled(),
                plot_mode: plotMode,
                relationship_overlay: overlayMode,
                show_legend: this.shouldShowContourLegend(),
                add_axis: this.shouldShowContourAxis(),
                tolerance: this.getContourTolerance(),
            },
        };
    }

    getContourPlotCacheKey(payload) {
        return [
            payload.session_id || '',
            (payload.roi_list || []).join(','),
            Number(payload.slice_index).toFixed(4),
            payload.include_interpolated_slices ? '1' : '0',
            payload.plot_mode,
            payload.relationship_overlay,
            payload.show_legend ? '1' : '0',
            payload.add_axis ? '1' : '0',
            Number(payload.tolerance || 0).toFixed(4),
        ].join('|');
    }

    cacheContourPlotImage(cacheKey, imageUrl) {
        if (!cacheKey || !imageUrl) {
            return;
        }

        if (this.plotImageCache.has(cacheKey)) {
            this.plotImageCache.delete(cacheKey);
        }
        this.plotImageCache.set(cacheKey, imageUrl);

        while (this.plotImageCache.size > this.maxPlotCacheEntries) {
            const oldestEntry = this.plotImageCache.entries().next().value;
            if (!oldestEntry) {
                break;
            }
            const [oldestKey, oldestUrl] = oldestEntry;
            if (oldestUrl && oldestUrl.startsWith('blob:')) {
                URL.revokeObjectURL(oldestUrl);
            }
            this.plotImageCache.delete(oldestKey);
        }
    }

    showCachedContourPlot(cacheKey) {
        if (!cacheKey || !this.plotImageCache.has(cacheKey)) {
            return false;
        }

        const plotImg = document.getElementById('contourPlot');
        plotImg.src = this.plotImageCache.get(cacheKey);
        plotImg.style.display = 'block';
        this.lastRenderedPlotKey = cacheKey;
        return true;
    }

    clearContourPlotCache() {
        this.plotImageCache.forEach((imageUrl) => {
            if (imageUrl && imageUrl.startsWith('blob:')) {
                URL.revokeObjectURL(imageUrl);
            }
        });
        this.plotImageCache.clear();
        this.lastRenderedPlotKey = null;
    }

    schedulePlotContours(options = {}) {
        const { debounceMs = this.plotDebounceMs, suppressAlerts = false } = options;

        if (this.plotDebounceTimer) {
            clearTimeout(this.plotDebounceTimer);
            this.plotDebounceTimer = null;
        }

        if (debounceMs <= 0) {
            this.plotContours({ suppressAlerts });
            return;
        }

        this.plotDebounceTimer = setTimeout(() => {
            this.plotDebounceTimer = null;
            this.plotContours({ suppressAlerts });
        }, debounceMs);
    }

    initializePlotControls() {
        const viewport = document.getElementById('plotViewport');
        const plotImg = document.getElementById('contourPlot');

        // Zoom controls
        document.getElementById('zoomInBtn').addEventListener('click', () => {
            this.zoomPlot(1.2);
        });

        document.getElementById('zoomOutBtn').addEventListener('click', () => {
            this.zoomPlot(0.8);
        });

        document.getElementById('resetZoomBtn').addEventListener('click', () => {
            this.resetPlotTransform();
        });

        // Pan controls
        document.getElementById('panUpBtn').addEventListener('click', () => {
            this.panPlot(0, 50);
        });

        document.getElementById('panDownBtn').addEventListener('click', () => {
            this.panPlot(0, -50);
        });

        document.getElementById('panLeftBtn').addEventListener('click', () => {
            this.panPlot(50, 0);
        });

        document.getElementById('panRightBtn').addEventListener('click', () => {
            this.panPlot(-50, 0);
        });

        // Mouse wheel zoom
        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoomPlot(delta);
        });

        // Click and drag to pan
        viewport.addEventListener('mousedown', (e) => {
            if (plotImg.src) {
                this.isDragging = true;
                this.dragStartX = e.clientX - this.plotTranslateX;
                this.dragStartY = e.clientY - this.plotTranslateY;
                viewport.classList.add('grabbing');
            }
        });

        viewport.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                this.plotTranslateX = e.clientX - this.dragStartX;
                this.plotTranslateY = e.clientY - this.dragStartY;
                this.updatePlotTransform();
            }
        });

        viewport.addEventListener('mouseup', () => {
            this.isDragging = false;
            viewport.classList.remove('grabbing');
        });

        viewport.addEventListener('mouseleave', () => {
            this.isDragging = false;
            viewport.classList.remove('grabbing');
        });
    }

    zoomPlot(factor) {
        this.plotScale *= factor;
        this.plotScale = Math.max(0.1, Math.min(10, this.plotScale)); // Clamp between 0.1x and 10x
        this.updatePlotTransform();
    }

    panPlot(deltaX, deltaY) {
        this.plotTranslateX += deltaX;
        this.plotTranslateY += deltaY;
        this.updatePlotTransform();
    }

    updatePlotTransform() {
        const plotImg = document.getElementById('contourPlot');
        plotImg.style.transform = `translate(${this.plotTranslateX}px, ${this.plotTranslateY}px) scale(${this.plotScale})`;
    }

    resetPlotTransform() {
        this.plotScale = 1.0;
        this.plotTranslateX = 0;
        this.plotTranslateY = 0;
        this.updatePlotTransform();
    }

    async plotContours(options = {}) {
        const { suppressAlerts = false } = options;
        const {
            selectedRois,
            plotMode,
            overlayMode,
            payload,
        } = this.getContourPlotPayload();

        if (selectedRois.length === 0) {
            if (!suppressAlerts) {
                alert('Please select at least one structure to plot');
            }
            return;
        }

        if (plotMode === 'relationship') {
            const requiredCount = this.getRequiredRelationshipOverlayCount(
                overlayMode
            );
            if (selectedRois.length < requiredCount) {
                if (!suppressAlerts) {
                    const noun = requiredCount === 1 ? 'structure' : 'structures';
                    alert(
                        `This relationship view requires at least ${requiredCount} selected ${noun}`
                    );
                }
                return;
            }
        }

        const cacheKey = this.getContourPlotCacheKey(payload);
        const plotImg = document.getElementById('contourPlot');
        const loadingOverlay = document.getElementById('plotLoading');

        if (cacheKey === this.lastRenderedPlotKey && plotImg.src) {
            return;
        }

        if (this.showCachedContourPlot(cacheKey)) {
            loadingOverlay.classList.remove('active');
            return;
        }

        // Cancel any in-flight request
        if (this.plotAbortController) {
            this.plotAbortController.abort();
        }

        // Create new abort controller for this request
        this.plotAbortController = new AbortController();
        const signal = this.plotAbortController.signal;

        // Show loading indicator
        loadingOverlay.classList.add('active');

        try {
            const response = await fetch('/api/plot-contours', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload),
                signal: signal
            });

            if (!response.ok) {
                throw new Error('Failed to generate plot');
            }

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            this.cacheContourPlotImage(cacheKey, imageUrl);
            plotImg.src = imageUrl;
            plotImg.style.display = 'block';
            this.lastRenderedPlotKey = cacheKey;

            loadingOverlay.classList.remove('active');

        } catch (error) {
            loadingOverlay.classList.remove('active');

            if (error.name === 'AbortError') {
                console.log('Plot request cancelled');
                return;
            }

            console.error('Plot error:', error);
            alert('Failed to generate contour plot');
        } finally {
            this.plotAbortController = null;
        }
    }

    showStage(stageName) {
        if (stageName !== 'processing') {
            this.stopStatusPolling();
        }

        // Hide all stages
        document.querySelectorAll('.stage').forEach(stage => {
            stage.style.display = 'none';
        });

        // Show selected stage
        document.getElementById(`stage-${stageName}`).style.display = 'block';
    }

    resetApp() {
        this.stopStatusPolling();
        this.clearContourPlotCache();
        if (this.plotDebounceTimer) {
            clearTimeout(this.plotDebounceTimer);
            this.plotDebounceTimer = null;
        }
        if (this.plotAbortController) {
            this.plotAbortController.abort();
            this.plotAbortController = null;
        }
        // Reset application state for new analysis
        this.sessionId = null;
        this.selectedStructures.clear();
        this.summaryData = null;
        this.patientInfo = null;
        this.structureItems = [];
        this.structureItemsByRoi = new Map();
        this.diagramSelection.clear();
        this.diagramAppliedSelection.clear();
        this.diagramSelectionModalOpen = false;
        this.sortableListsInitialized = false;
        this.sortableInitScheduled = false;
        this.hiddenNodes.clear();
        this.hiddenLabels.clear();
        this.fixedNodes.clear();
        this.summaryHiddenRows.clear();
        this.summaryRowOrder = [];
        this.summaryDefaultOrder = [];
        this.summaryDragSortEnabled = false;
        this.summaryColumnOrder = this.getDefaultSummaryColumnOrder();
        this.summaryColumnSettings = this.getDefaultSummaryColumnSettings();
        this.currentSort = { column: 'roi', ascending: true };
        this.manualLayoutActive = false;
        this._dragFrozen = [];
        this._dismissContextMenu();
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        if (this.sortableRows) {
            this.sortableRows.available.destroy();
            this.sortableRows.selected.destroy();
            this.sortableRows = null;
        }
        if (this.sortableColumns) {
            this.sortableColumns.available.destroy();
            this.sortableColumns.selected.destroy();
            this.sortableColumns = null;
        }
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
        if (this.summarySortable) {
            this.summarySortable.destroy();
            this.summarySortable = null;
        }
        this.updateConnectionStatus(false);

        const structureSetInfo = document.getElementById('structureSetInfo');
        if (structureSetInfo) {
            structureSetInfo.innerHTML = '';
            structureSetInfo.style.display = 'none';
        }

        const badgeName = document.getElementById('structureSetBadgeName');
        if (badgeName) {
            badgeName.textContent = 'Structure Set';
        }
        const tooltip = document.getElementById('structureSetBadgeTooltip');
        if (tooltip) {
            tooltip.innerHTML = '';
            tooltip.classList.remove('visible');
        }
        this.updateDiagramLayoutButtons();
        const modal = document.getElementById('diagramStructureModal');
        if (modal) {
            modal.style.display = 'none';
        }

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
                if (targetTab === 'matrix' && window.app) {
                    window.app.ensureSortableListsInitialized();
                }
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
