'''Type stub file for relations module.

This file provides type hints for IDE autocomplete and type checking.
It contains both manually maintained type definitions and auto-generated
relationship constants.
'''

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import shapely
from contours import Contour
from region_slice import RegionSlice


# === Manual Type Definitions ===

@dataclass
class RelationshipType:
    '''Relationship type definition loaded from relationship_definitions.json.

    Attributes:
        relation_type: Unique identifier (e.g., 'CONTAINS', 'OVERLAPS')
        label: Human-readable label (e.g., 'Contains', 'Overlaps')
        symbol: Unicode symbol for relationship (e.g., '⊃', '∩')
        color: Hex color code for visualization (e.g., '#FF0000')
        description: Detailed description of relationship
        complementary_relation: Name of complementary relationship or empty string
        implied_relation: List of implied relationship names
        symmetric: Whether relationship is symmetric (A rel B ⇒ B rel A)
        transitive: Whether relationship is transitive (A rel B and B rel C ⇒ A rel C)
        reversed_arrow: Whether this is a complementary stub (True) or primary (False/None)
        pattern: DE-27IM pattern string (29 chars: T/F/* with tabs at pos 9, 19)
        mask: Binary mask as string (e.g., '0b111000111000111000111000111')
        value: Binary value as string (e.g., '0b111000000000111000000000000')
        examples: List of example scenarios
    '''
    relation_type: str
    label: str
    symbol: str
    color: str
    description: str
    complementary_relation: str
    implied_relation: List[str]
    symmetric: bool
    transitive: bool
    reversed_arrow: bool
    pattern: str
    mask: str
    value: str
    examples: List[str]

    @property
    def mask_decimal(self) -> int:
        ...

    @property
    def value_decimal(self) -> int:
        ...

    @property
    def is_symmetric(self) -> bool:
        '''Check if the relationship is symmetric.'''
        ...

    @property
    def is_transitive(self) -> bool:
        '''Check if the relationship is transitive.'''
        ...

    @property
    def complementary(self) -> Optional['RelationshipType']:
        '''Get the complementary relationship.'''
        ...

    @property
    def implied(self) -> List['RelationshipType']:
        '''Get list of implied relationships.'''
        ...

    def __bool__(self) -> bool:
        '''Return False for UNKNOWN, True for all others.'''
        ...


@dataclass
class RelationshipTest:
    '''The test binaries used to identify a relationship type.'''
    relation_type: RelationshipType
    mask: int
    value: int

    def test(self, relation: int) -> Optional[RelationshipType]:
        '''Apply the defined test to the supplied relation binary.'''
        ...


class DE9IM:
    '''The DE-9IM relationship string for two polygons.'''
    def __init__(
        self,
        poly_a: Optional['SinglePolygonType'] = None,
        poly_b: Optional['SinglePolygonType'] = None,
        relation_str: Optional[str] = None,
        tolerance: float = 0.0
    ) -> None: ...


class DE27IM:
    '''The DE-27IM relationship for two 3D structures.'''
    def __init__(
        self,
        region_a: Optional[RegionSlice] = None,
        region_b: Optional[RegionSlice] = None,
        relation_str: Optional[str] = None
    ) -> None: ...


# Type aliases
SinglePolygonType = Union[shapely.Polygon, shapely.MultiPolygon, Contour]
AllPolygonType = Union[RegionSlice, SinglePolygonType]


# === Module-level Registry and Lookup ===

# Dictionary of all relationship types keyed by relation_type name
RELATIONSHIP_TYPES: Dict[str, RelationshipType]

# List of all relationship tests for pattern matching
RELATIONSHIP_TESTS: List[RelationshipTest]

# Lookup function
def get_relationship_type(name: str) -> Optional[RelationshipType]:
    '''Get relationship type by name.

    Args:
        name: Relationship type name (e.g., 'CONTAINS', 'OVERLAPS')

    Returns:
        RelationshipType object or None if not found
    '''
    ...


# === Generated Constants (DO NOT EDIT MANUALLY) ===
# Auto-generated from relationship_definitions.json
# Run: python src/generate_stub_constants.py

BORDERS: RelationshipType  # Borders
CONFINES: RelationshipType  # Confines
CONTAINS: RelationshipType  # Contains
DISJOINT: RelationshipType  # Disjoint
EQUAL: RelationshipType  # Equal
OVERLAPS: RelationshipType  # Overlaps
PARTITIONED: RelationshipType  # Partitioned by
SHELTERS: RelationshipType  # Shelters
SURROUNDS: RelationshipType  # Surrounds
UNKNOWN: RelationshipType  # Unknown

# === End Generated Constants ===


# LOGICAL flag - separate from relationship types
LOGICAL: bool  # Flag for logical combination of relationships
