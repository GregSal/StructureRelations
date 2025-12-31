'''Data structures for representing relationships between structures.

This module defines the StructureRelationship class that encapsulates
all information about the spatial relationship between two structures,
including the DE27IM relationship, identity flag, logical flag, and
metrics placeholder.
'''
# %% Imports
from typing import Optional, Any
from dataclasses import dataclass

from relations import DE27IM, RelationshipType


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
    metrics: Optional[Any] = None

    @property
    def relationship_type(self) -> RelationshipType:
        '''Get the relationship type classification.

        Returns:
            RelationshipType: The classified relationship type. Returns
                RelationshipType.EQUALS if is_identical is True or de27im
                is None. Otherwise returns the relationship type identified
                by the DE27IM object.
        '''
        if self.is_identical or self.de27im is None:
            return RelationshipType.EQUALS
        return self.de27im.identify_relation()
