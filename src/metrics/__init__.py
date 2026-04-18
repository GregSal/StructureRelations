"""Metrics subpackage for calculating spatial relationship metrics.

This subpackage provides quantitative metrics for analyzing spatial relationships
between radiotherapy structures. It follows a slice-oriented, region-aware
architecture that matches the existing relationship calculation approach.

Architecture:
- Slice-oriented: Per-slice calculation is PRIMARY, 3D summaries are derived
- Region-aware: Separate metrics for each region pair in multi-region structures
- Clinical focus: Uses actual contour boundaries for clinical relevance
- Configurable: JSON-based configuration for units, precision, enabled metrics

Metric Categories:
- Margins: Clearance distances inside containing structure (CONTAINS, SURROUNDS, SHELTERS)
- Distance: Gap between disjoint structures (DISJOINT, SHELTERS)
- Volume: Overlap ratios and Dice coefficients (OVERLAPS, PARTITION, CONTAINS, EQUAL)
- Surface: Boundary overlap for touching structures (BORDERS, CONFINES)
- Geometry: Centroids and geometric properties

Usage:
    from metrics import get_config, MetricCalculatorRegistry
    from metrics.data_structures import RelationshipMetrics

    # Load configuration
    config = get_config()

    # Get applicable calculators for a relationship
    calculators = MetricCalculatorRegistry.get_applicable_calculators(
        relationship_type=RelationshipType.CONTAINS,
        config=config
    )

    # Calculate metrics using orchestrator
    from metrics.orchestrator import MetricOrchestrator
    orchestrator = MetricOrchestrator(config)
    metrics = orchestrator.calculate_metrics(structure_a, structure_b, relationship)
"""

# Configuration
from metrics.config import MetricsConfig, get_config, reload_config

# Data structures
from metrics.data_structures import (
    MarginMetrics,
    DistanceMetrics,
    VolumeMetrics,
    SurfaceMetrics,
    GeometryMetrics,
    RelationshipMetrics,
)

# Base classes and registry
from metrics.base import (
    MetricCalculator,
    MetricCalculatorRegistry,
    register_calculator,
)

# Import calculator modules to trigger registration
# These imports ensure calculators are registered when metrics package is imported
try:
    from metrics import margins
    from metrics import distance
    from metrics import volume
    from metrics import surface
    from metrics import geometry
except ImportError as e:
    # Some calculators may not be implemented yet
    import logging
    logging.getLogger(__name__).debug(f'Could not import all calculator modules: {e}')

# Orchestrator (will be implemented in Phase 6)
try:
    from metrics.orchestrator import MetricOrchestrator
except ImportError:
    MetricOrchestrator = None


__all__ = [
    # Configuration
    'MetricsConfig',
    'get_config',
    'reload_config',
    # Data structures
    'MarginMetrics',
    'DistanceMetrics',
    'VolumeMetrics',
    'SurfaceMetrics',
    'GeometryMetrics',
    'RelationshipMetrics',
    # Base classes
    'MetricCalculator',
    'MetricCalculatorRegistry',
    'register_calculator',
    # Orchestrator
    'MetricOrchestrator',
]
