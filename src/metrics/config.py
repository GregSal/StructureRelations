"""Configuration management for metrics calculation.

This module loads and validates configuration settings from metrics_config.json
and provides access to configuration values throughout the metrics package.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Strongly-typed configuration for metrics calculation.

    Loads settings from metrics_config.json and provides validated access
    to configuration values.
    """

    # Units
    distance_unit: str = 'cm'
    volume_unit: str = 'cm³'
    area_unit: str = 'cm²'

    # Precision
    distance_precision: int = 2
    volume_precision: int = 2
    area_precision: int = 2
    ratio_precision: int = 3

    # Slice thickness
    default_slice_thickness_cm: float = 0.3
    use_dicom_slice_thickness: bool = True

    # Enabled metric categories
    margins_enabled: bool = True
    distance_enabled: bool = True
    volume_enabled: bool = True
    surface_enabled: bool = True
    geometry_enabled: bool = False

    # Margin settings
    calculate_orthogonal_margins: bool = True
    calculate_minimum_margin: bool = True
    calculate_maximum_margin: bool = False
    orthogonal_directions: List[str] = None
    use_anatomical_labels: bool = True
    anatomical_labels: Dict[str, str] = None

    # Distance settings
    calculate_minimum_distance: bool = True
    calculate_3d_components: bool = False

    # Volume settings
    calculate_overlap_ratio: bool = True
    calculate_dice: bool = True
    volume_ratio_basis: str = 'auto'

    # Surface settings
    calculate_surface_overlap_ratio: bool = True
    calculate_absolute_areas: bool = False
    surface_ratio_basis: str = 'larger'

    # Geometry settings
    calculate_centroids: bool = True
    calculate_centroid_distance: bool = False

    # Multi-region handling
    store_per_region_data: bool = True
    store_per_slice_data: bool = True
    aggregate_margins_method: str = 'minimum'
    aggregate_distance_method: str = 'minimum'
    aggregate_volume_method: str = 'sum'
    aggregate_surface_method: str = 'sum'

    # Non-applicable handling
    use_nan_for_non_applicable: bool = True
    use_none_for_non_applicable: bool = False
    use_zero_for_non_applicable: bool = False

    # Validation
    check_negative_margins: bool = True
    check_ratio_bounds: bool = True
    warn_on_non_applicable: bool = True

    # Performance
    cache_region_identification: bool = True
    parallel_slice_calculation: bool = False
    max_workers: int = 4

    # Raw config for reference
    _raw_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for list/dict fields."""
        if self.orthogonal_directions is None:
            self.orthogonal_directions = ['x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos']

        if self.anatomical_labels is None:
            self.anatomical_labels = {
                'x_neg': 'R',
                'x_pos': 'L',
                'y_neg': 'A',
                'y_pos': 'P',
                'z_neg': 'I',
                'z_pos': 'S'
            }

    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> 'MetricsConfig':
        """Load configuration from JSON file.

        Args:
            config_path: Path to config file. If None, uses default metrics_config.json
                        in same directory as this module.

        Returns:
            MetricsConfig instance with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If config contains invalid values
        """
        if config_path is None:
            # Default to metrics_config.json in same directory as this file
            config_path = Path(__file__).parent / 'metrics_config.json'

        logger.info(f'Loading metrics configuration from {config_path}')

        if not config_path.exists():
            raise FileNotFoundError(f'Metrics config file not found: {config_path}')

        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)

        # Extract values from nested structure
        config = cls(
            # Units
            distance_unit=raw_config.get('units', {}).get('distance', 'cm'),
            volume_unit=raw_config.get('units', {}).get('volume', 'cm³'),
            area_unit=raw_config.get('units', {}).get('area', 'cm²'),

            # Precision
            distance_precision=raw_config.get('precision', {}).get('distance', 2),
            volume_precision=raw_config.get('precision', {}).get('volume', 2),
            area_precision=raw_config.get('precision', {}).get('area', 2),
            ratio_precision=raw_config.get('precision', {}).get('ratio', 3),

            # Slice thickness
            default_slice_thickness_cm=raw_config.get('slice_thickness', {}).get('default_cm', 0.3),
            use_dicom_slice_thickness=raw_config.get('slice_thickness', {}).get('use_dicom_if_available', True),

            # Enabled metrics
            margins_enabled=raw_config.get('enabled_metrics', {}).get('margins', True),
            distance_enabled=raw_config.get('enabled_metrics', {}).get('distance', True),
            volume_enabled=raw_config.get('enabled_metrics', {}).get('volume', True),
            surface_enabled=raw_config.get('enabled_metrics', {}).get('surface', True),
            geometry_enabled=raw_config.get('enabled_metrics', {}).get('geometry', False),

            # Margin settings
            calculate_orthogonal_margins=raw_config.get('margin_settings', {}).get('calculate_orthogonal', True),
            calculate_minimum_margin=raw_config.get('margin_settings', {}).get('calculate_minimum', True),
            calculate_maximum_margin=raw_config.get('margin_settings', {}).get('calculate_maximum', False),
            orthogonal_directions=raw_config.get('margin_settings', {}).get('orthogonal_directions', None),
            use_anatomical_labels=raw_config.get('margin_settings', {}).get('anatomical_labels', {}).get('use_anatomical', True),
            anatomical_labels=cls._extract_anatomical_labels(raw_config),

            # Distance settings
            calculate_minimum_distance=raw_config.get('distance_settings', {}).get('calculate_minimum', True),
            calculate_3d_components=raw_config.get('distance_settings', {}).get('calculate_3d_components', False),

            # Volume settings
            calculate_overlap_ratio=raw_config.get('volume_settings', {}).get('calculate_overlap_ratio', True),
            calculate_dice=raw_config.get('volume_settings', {}).get('calculate_dice', True),
            volume_ratio_basis=raw_config.get('volume_settings', {}).get('ratio_basis', 'auto'),

            # Surface settings
            calculate_surface_overlap_ratio=raw_config.get('surface_settings', {}).get('calculate_overlap_ratio', True),
            calculate_absolute_areas=raw_config.get('surface_settings', {}).get('calculate_absolute_areas', False),
            surface_ratio_basis=raw_config.get('surface_settings', {}).get('ratio_basis', 'larger'),

            # Geometry settings
            calculate_centroids=raw_config.get('geometry_settings', {}).get('calculate_centroids', True),
            calculate_centroid_distance=raw_config.get('geometry_settings', {}).get('calculate_centroid_distance', False),

            # Multi-region handling
            store_per_region_data=raw_config.get('multi_region_handling', {}).get('store_per_region_data', True),
            store_per_slice_data=raw_config.get('multi_region_handling', {}).get('store_per_slice_data', True),
            aggregate_margins_method=raw_config.get('multi_region_handling', {}).get('aggregate_method', {}).get('margins', 'minimum'),
            aggregate_distance_method=raw_config.get('multi_region_handling', {}).get('aggregate_method', {}).get('distance', 'minimum'),
            aggregate_volume_method=raw_config.get('multi_region_handling', {}).get('aggregate_method', {}).get('volume', 'sum'),
            aggregate_surface_method=raw_config.get('multi_region_handling', {}).get('aggregate_method', {}).get('surface', 'sum'),

            # Non-applicable handling
            use_nan_for_non_applicable=raw_config.get('non_applicable_handling', {}).get('use_nan', True),
            use_none_for_non_applicable=raw_config.get('non_applicable_handling', {}).get('use_none', False),
            use_zero_for_non_applicable=raw_config.get('non_applicable_handling', {}).get('use_zero', False),

            # Validation
            check_negative_margins=raw_config.get('validation', {}).get('check_negative_margins', True),
            check_ratio_bounds=raw_config.get('validation', {}).get('check_ratio_bounds', True),
            warn_on_non_applicable=raw_config.get('validation', {}).get('warn_on_non_applicable', True),

            # Performance
            cache_region_identification=raw_config.get('performance', {}).get('cache_region_identification', True),
            parallel_slice_calculation=raw_config.get('performance', {}).get('parallel_slice_calculation', False),
            max_workers=raw_config.get('performance', {}).get('max_workers', 4),

            # Store raw config
            _raw_config=raw_config
        )

        # Validate configuration
        config._validate()

        logger.info('Metrics configuration loaded successfully')
        return config

    @staticmethod
    def _extract_anatomical_labels(raw_config: Dict) -> Dict[str, str]:
        """Extract anatomical labels from config."""
        margin_settings = raw_config.get('margin_settings', {})
        anatomical = margin_settings.get('anatomical_labels', {})

        return {
            'x_neg': anatomical.get('x_neg', 'R'),
            'x_pos': anatomical.get('x_pos', 'L'),
            'y_neg': anatomical.get('y_neg', 'A'),
            'y_pos': anatomical.get('y_pos', 'P'),
            'z_neg': anatomical.get('z_neg', 'I'),
            'z_pos': anatomical.get('z_pos', 'S')
        }

    def _validate(self):
        """Validate configuration values.

        Raises:
            ValueError: If configuration contains invalid values
        """
        # Validate precision values
        if self.distance_precision < 0:
            raise ValueError(f'distance_precision must be non-negative: {self.distance_precision}')
        if self.volume_precision < 0:
            raise ValueError(f'volume_precision must be non-negative: {self.volume_precision}')
        if self.ratio_precision < 0:
            raise ValueError(f'ratio_precision must be non-negative: {self.ratio_precision}')

        # Validate slice thickness
        if self.default_slice_thickness_cm <= 0:
            raise ValueError(f'default_slice_thickness_cm must be positive: {self.default_slice_thickness_cm}')

        # Validate ratio basis
        valid_volume_bases = ['auto', 'larger', 'smaller', 'average']
        if self.volume_ratio_basis not in valid_volume_bases:
            raise ValueError(f'volume_ratio_basis must be one of {valid_volume_bases}: {self.volume_ratio_basis}')

        valid_surface_bases = ['larger', 'smaller', 'average', 'exterior', 'interior']
        if self.surface_ratio_basis not in valid_surface_bases:
            raise ValueError(f'surface_ratio_basis must be one of {valid_surface_bases}: {self.surface_ratio_basis}')

        # Validate aggregate methods
        valid_aggregate = ['minimum', 'maximum', 'mean', 'sum']
        if self.aggregate_margins_method not in valid_aggregate:
            raise ValueError(f'aggregate_margins_method must be one of {valid_aggregate}: {self.aggregate_margins_method}')
        if self.aggregate_distance_method not in valid_aggregate:
            raise ValueError(f'aggregate_distance_method must be one of {valid_aggregate}: {self.aggregate_distance_method}')

        # Validate non-applicable handling (only one should be True)
        na_count = sum([self.use_nan_for_non_applicable, self.use_none_for_non_applicable, self.use_zero_for_non_applicable])
        if na_count != 1:
            raise ValueError(f'Exactly one non_applicable_handling option must be True, got {na_count}')

        # Validate performance settings
        if self.max_workers < 1:
            raise ValueError(f'max_workers must be at least 1: {self.max_workers}')

    def get_non_applicable_value(self) -> Any:
        """Get the value to use for non-applicable metrics.

        Returns:
            math.nan, None, or 0 depending on configuration
        """
        if self.use_nan_for_non_applicable:
            import math
            return math.nan
        elif self.use_none_for_non_applicable:
            return None
        else:  # use_zero_for_non_applicable
            return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        return self._raw_config if self._raw_config else {}


# Global configuration instance
_global_config: Optional[MetricsConfig] = None


def get_config() -> MetricsConfig:
    """Get global metrics configuration instance.

    Loads configuration on first call, returns cached instance on subsequent calls.

    Returns:
        MetricsConfig instance
    """
    global _global_config

    if _global_config is None:
        _global_config = MetricsConfig.load_from_file()

    return _global_config


def reload_config(config_path: Optional[Path] = None):
    """Reload configuration from file.

    Args:
        config_path: Optional custom path to config file
    """
    global _global_config
    _global_config = MetricsConfig.load_from_file(config_path)
    logger.info('Metrics configuration reloaded')
