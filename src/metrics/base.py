"""Base classes and registry for metric calculators.

This module defines the abstract base class for all metric calculators
and provides a registry pattern for dynamic calculator discovery and instantiation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, Set
import logging

from relations import RelationshipType
from structures import StructureShape
from relationships import StructureRelationship
from metrics.config import MetricsConfig

logger = logging.getLogger(__name__)


class MetricCalculator(ABC):
    """Abstract base class for all metric calculators.

    Each metric calculator (orthogonal margins, minimum distance, etc.) inherits
    from this base class and implements the required methods.

    Calculators follow slice-oriented architecture:
    1. Identify regions in both structures (contour graph connected components)
    2. For each region pair, calculate per-slice metrics
    3. Aggregate per-slice results to per-region summaries
    4. Aggregate per-region results to overall 3D summary
    """

    def __init__(self, config: MetricsConfig):
        """Initialize calculator with configuration.

        Args:
            config: MetricsConfig instance with calculation settings
        """
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    @abstractmethod
    def get_name(self) -> str:
        """Get calculator name for registry and logging.

        Returns:
            Unique identifier for this calculator (e.g., 'orthogonal_margins')
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get calculator version for result tracking.

        Returns:
            Version string (e.g., '1.0.0')
        """
        pass

    @abstractmethod
    def is_applicable(self, relationship_type: RelationshipType) -> bool:
        """Check if this metric applies to the given relationship type.

        Args:
            relationship_type: The spatial relationship to check

        Returns:
            True if this metric is applicable, False otherwise

        Examples:
            - Orthogonal margins: CONTAINS, SURROUNDS, SHELTERS
            - Minimum distance: DISJOINT, SHELTERS (but NOT BORDERS, CONFINES, EQUAL)
            - Volume overlap: OVERLAPS, PARTITION, CONTAINS, EQUAL
            - Surface overlap: BORDERS, CONFINES
        """
        pass

    @abstractmethod
    def calculate(
        self,
        structure_a: StructureShape,
        structure_b: StructureShape,
        relationship: StructureRelationship
    ) -> Any:
        """Calculate metric for the given structure pair.

        This is the main calculation method. Implementations should:
        1. Check applicability (may return NaN if not applicable)
        2. Identify regions in both structures using contour_graph
        3. For each region pair:
           a. Iterate through slices where both regions exist
           b. Calculate per-slice metric using RegionSlice.select()
           c. Store per-slice results
        4. Aggregate per-slice to per-region summaries
        5. Aggregate per-region to overall 3D summary
        6. Return appropriate dataclass (MarginMetrics, DistanceMetrics, etc.)

        Args:
            structure_a: First structure
            structure_b: Second structure
            relationship: StructureRelationship with type and DE-27IM data

        Returns:
            Metric-specific dataclass (MarginMetrics, DistanceMetrics, VolumeMetrics,
            SurfaceMetrics, or GeometryMetrics) with results

        Raises:
            ValueError: If structures are invalid or incompatible
        """
        pass

    def get_non_applicable_value(self) -> Any:
        """Get value to use when metric is not applicable.

        Returns:
            math.nan, None, or 0 based on configuration
        """
        return self.config.get_non_applicable_value()

    def _identify_regions(self, structure: StructureShape) -> Dict[int, Set]:
        """Identify disconnected regions in structure using contour graph.

        Uses NetworkX connected_components on the structure's contour_graph
        to identify distinct 3D regions (archipelago structures).

        Args:
            structure: StructureShape to analyze

        Returns:
            Dictionary mapping region_id -> set of ContourIndex nodes in that region
        """
        import networkx as nx

        # Get connected components from contour graph
        # Each component is a set of ContourIndex values representing one 3D region
        components = nx.connected_components(structure.contour_graph)

        # Assign region IDs (starting from 0)
        regions = {i: component for i, component in enumerate(components)}

        self.logger.debug(
            f'Identified {len(regions)} region(s) in structure {structure.roi_number}'
        )

        return regions

    def _warn_non_applicable(self, relationship_type: RelationshipType):
        """Log warning when calculator is used on non-applicable relationship.

        Args:
            relationship_type: The relationship type
        """
        if self.config.warn_on_non_applicable:
            self.logger.warning(
                f'{self.get_name()} metric is not applicable to {relationship_type.name} '
                f'relationship. Returning {self.get_non_applicable_value()}'
            )


class MetricCalculatorRegistry:
    """Registry for metric calculators.

    Provides dynamic calculator discovery, registration, and instantiation.
    Calculators register themselves when their module is imported.
    """

    _calculators: Dict[str, Type[MetricCalculator]] = {}

    @classmethod
    def register(cls, calculator_class: Type[MetricCalculator]):
        """Register a calculator class.

        Args:
            calculator_class: MetricCalculator subclass to register

        Raises:
            ValueError: If calculator name is already registered
        """
        # Instantiate temporarily to get name
        temp_instance = calculator_class(MetricsConfig())
        name = temp_instance.get_name()

        if name in cls._calculators:
            raise ValueError(
                f'Calculator {name} is already registered by '
                f'{cls._calculators[name].__name__}'
            )

        cls._calculators[name] = calculator_class
        logger.info(f'Registered calculator: {name} ({calculator_class.__name__})')

    @classmethod
    def get_calculator(
        cls,
        name: str,
        config: Optional[MetricsConfig] = None
    ) -> MetricCalculator:
        """Get calculator instance by name.

        Args:
            name: Calculator name (e.g., 'orthogonal_margins')
            config: Optional config instance. If None, uses global config.

        Returns:
            Instantiated calculator

        Raises:
            KeyError: If calculator not found
        """
        if name not in cls._calculators:
            raise KeyError(
                f'Calculator {name} not found. Available: {list(cls._calculators.keys())}'
            )

        if config is None:
            from metrics.config import get_config
            config = get_config()

        return cls._calculators[name](config)

    @classmethod
    def get_all_calculators(cls, config: Optional[MetricsConfig] = None) -> Dict[str, MetricCalculator]:
        """Get all registered calculators.

        Args:
            config: Optional config instance. If None, uses global config.

        Returns:
            Dictionary mapping calculator name -> calculator instance
        """
        if config is None:
            from metrics.config import get_config
            config = get_config()

        return {
            name: calc_class(config)
            for name, calc_class in cls._calculators.items()
        }

    @classmethod
    def get_applicable_calculators(
        cls,
        relationship_type: RelationshipType,
        config: Optional[MetricsConfig] = None
    ) -> Dict[str, MetricCalculator]:
        """Get calculators applicable to a relationship type.

        Args:
            relationship_type: The spatial relationship
            config: Optional config instance

        Returns:
            Dictionary mapping calculator name -> calculator instance
            (only includes calculators where is_applicable returns True)
        """
        all_calculators = cls.get_all_calculators(config)

        applicable = {
            name: calc
            for name, calc in all_calculators.items()
            if calc.is_applicable(relationship_type)
        }

        logger.debug(
            f'Found {len(applicable)} applicable calculator(s) for {relationship_type.name}: '
            f'{list(applicable.keys())}'
        )

        return applicable

    @classmethod
    def list_calculator_names(cls) -> list[str]:
        """Get list of all registered calculator names.

        Returns:
            List of calculator names
        """
        return list(cls._calculators.keys())

    @classmethod
    def clear_registry(cls):
        """Clear all registered calculators.

        Primarily for testing purposes.
        """
        cls._calculators.clear()
        logger.info('Calculator registry cleared')


def register_calculator(calculator_class: Type[MetricCalculator]):
    """Decorator to register a calculator class.

    Usage:
        @register_calculator
        class OrthogonalMarginsCalculator(MetricCalculator):
            ...

    Args:
        calculator_class: MetricCalculator subclass to register

    Returns:
        The unmodified calculator class (decorator pattern)
    """
    MetricCalculatorRegistry.register(calculator_class)
    return calculator_class
