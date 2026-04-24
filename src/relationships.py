'''Data structures for representing relationships between structures.

This module defines the StructureRelationship class that encapsulates
all information about the spatial relationship between two structures,
including the DE27IM relationship, identity flag, logical flag, and
metrics placeholder.
'''
# %% Imports
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field

from relations import (
    DE27IM, RelationshipType, RELATIONSHIP_TYPES,
    RELATION_RANK, count_relations_by_rank,
)
from types_and_classes import ROI_Type, RegionIndex, RegionPairKey


# %% Class Definition
@dataclass
class StructureRelationship:
    '''Encapsulates all information about a relationship between two structures.

    This class stores the complete relationship data including the geometric
    relationship (DE27IM), flags indicating whether structures are identical
    or logically related, and a placeholder for relationship metrics.

    Attributes:
        de27im (Optional[DE27IM]): The DE27IM spatial relationship object.
            None for identical structures (same structure compared to itself)
            or when no geometric relationship has been calculated.
        is_identical (bool): True if this represents a structure compared to
            itself (diagonal of relationship matrix). False for all other
            relationships, even if two distinct structures are geometrically
            equal. Defaults to False.
        is_logical (bool): True if this relationship is derived from the
            relationship graph topology rather than being a direct geometric
            relationship. This flag is calculated by analyzing the graph
            structure in the finalize() method. Defaults to False.
        intermediate_structures (List[ROI_Type]): List of ROI numbers for
            structures that form the logical path between the two structures
            in this relationship. Only populated when is_logical is True.
            When the webapp's "Hide Logical Relations" option is enabled,
            this relationship will be hidden from display if all intermediate
            structures are currently shown, but will be displayed if any
            intermediate structure is hidden (since the relationship is no
            longer logically derivable from visible relationships). Will be
            populated by the calculate_logical_flags() implementation.
            Defaults to empty list.
        metrics (Optional[Any]): Placeholder for relationship metrics object.
            Will hold calculated metrics such as distances, volume ratios,
            overlap percentages, etc. The specific type and content will be
            determined as the metrics subpackage is developed. Defaults to None.

    Notes:
        - is_identical vs. geometric equality: Two distinct structures can be
          geometrically equal (same shape and position) but not identical
          (different ROI numbers). Only same-structure comparisons set
          is_identical=True.
        - is_logical will be calculated based on graph analysis to identify
          relationships that are implied by other relationships (e.g.,
          transitivity) rather than directly computed from geometry.
        - metrics field uses Any type for maximum flexibility as the metrics
          system evolves. It may eventually hold objects, dicts, dataclasses,
          or instances of abstract base metric classes.
    '''

    de27im: Optional[DE27IM] = None
    is_identical: bool = False
    is_logical: bool = False
    intermediate_structures: List[ROI_Type] = field(default_factory=list)
    metrics: Optional[Any] = None
    _override_type: Optional[RelationshipType] = None
    # Per-region relationships: maps (region_idx_a, region_idx_b) → DE27IM.
    # Populated by StructureSet.calculate_relationships() via
    # StructureShape.relate_regions().  None when not yet computed.
    per_region_relations: Optional[Dict[RegionPairKey, DE27IM]] = None

    @property
    def relationship_type(self) -> RelationshipType:
        '''Get the relationship type classification.

        Returns:
            RelationshipType: The classified relationship type. Returns
                EQUAL if is_identical is True, UNKNOWN if de27im is None,
                or the relationship type identified by the DE27IM object.
                If _override_type is set, returns that.
        '''
        # Check for override first
        if self._override_type is not None:
            return self._override_type

        # If this is a self-relationship, return EQUAL
        if self.is_identical:
            equals_type = RELATIONSHIP_TYPES.get('EQUAL')
            if equals_type is None:
                # Log error but return UNKNOWN to avoid crashing
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    'EQUAL relationship type not found in RELATIONSHIP_TYPES. '
                    'Available types: %s. '
                    'This indicates a problem with relationship_definitions.json loading.',
                    list(RELATIONSHIP_TYPES.keys())
                )
                # Return UNKNOWN as fallback
                return RELATIONSHIP_TYPES.get('UNKNOWN')
            return equals_type

        # If no DE27IM is available, return UNKNOWN
        if self.de27im is None:
            return RELATIONSHIP_TYPES.get('UNKNOWN')

        # Otherwise, identify the relationship from DE27IM
        return self.de27im.identify_relation()

    @property
    def region_relationship_types(
        self,
    ) -> Dict[RegionPairKey, RelationshipType]:
        '''Get the RelationshipType for each region pair.

        Returns:
            Dict mapping RegionPairKey to RelationshipType.  Empty dict when
            per_region_relations is None.
        '''
        if not self.per_region_relations:
            return {}
        return {
            key: de27im.identify_relation()
            for key, de27im in self.per_region_relations.items()
        }

    def has_multiple_regions(self) -> bool:
        '''Return True when there is more than one distinct region-pair key.

        This indicates that one or both structures have multiple 3D regions
        and the overall relationship may not capture all region-to-region
        distinctions.
        '''
        if not self.per_region_relations:
            return False
        return len(self.per_region_relations) > 1

    def display_label(
        self,
        show_multi: bool = False,
        drop_disjoint: bool = False,
        drop_overlaps: bool = False,
    ) -> str:
        '''Return a display label for this relationship.

        When show_multi is False (default) the consolidated relationship label
        is returned unchanged, matching pre-existing behaviour.

        When show_multi is True and multiple region pairs exist the label is
        enhanced according to the rules below:

        - All region pairs have the same relationship type → bare label
          (no brackets; identical to the single-region case).
        - Multiple types, consolidated type is not UNKNOWN → ``{label}``
          (curly brackets signal multiple underlying relationships).
        - Consolidated type is UNKNOWN → ``{label_1 & label_2 & ...}`` where
          the labels are sorted by RELATION_RANK priority.

        Args:
            show_multi: When True, apply multi-region labelling rules.
            drop_disjoint: Exclude DISJOINT pairs from label computation.
            drop_overlaps: Exclude OVERLAPS pairs from label computation.

        Returns:
            str: The display label for this relationship.
        '''
        base_type = self.relationship_type
        base_label = base_type.label if base_type else ''

        if not show_multi or not self.has_multiple_regions():
            return base_label

        per_types = self.region_relationship_types
        ranked = count_relations_by_rank(
            per_types,
            drop_disjoint=drop_disjoint,
            drop_overlaps=drop_overlaps,
        )

        if not ranked:
            return base_label

        # All pairs share the same type — no brackets needed.
        if len(ranked) == 1:
            return ranked[0][0].label

        # Multiple types present.
        if base_type and base_type.relation_type != 'UNKNOWN':
            return '{' + base_label + '}'

        # Consolidated result is UNKNOWN — enumerate the individual types.
        combined = ' & '.join(rel_type.label for rel_type, _ in ranked)
        return '{' + combined + '}'
