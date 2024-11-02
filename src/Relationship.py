# %% Relationship class
class Relationship():
    symmetric_relations = [
        RelationshipType.DISJOINT,
        RelationshipType.OVERLAPS,
        RelationshipType.BORDERS,
        RelationshipType.EQUALS,
        RelationshipType.UNKNOWN  # If unknown structure order is irrelevant.
        ]
    transitive_relations = [
        RelationshipType.EQUALS,
        RelationshipType.SHELTERS,
        RelationshipType.SURROUNDS,
        RelationshipType.CONTAINS,
        ]
    metric_match = {
        RelationshipType.DISJOINT: DistanceMetric,
        RelationshipType.BORDERS: OverlapSurfaceMetric,
        RelationshipType.BORDERS_INTERIOR: OverlapSurfaceMetric,
        RelationshipType.OVERLAPS: OverlapVolumeMetric,
        RelationshipType.PARTITION: OverlapVolumeMetric,
        RelationshipType.SHELTERS: MarginMetric,
        RelationshipType.SURROUNDS: MarginMetric,
        RelationshipType.CONTAINS: MarginMetric,
        RelationshipType.EQUALS: NoMetric,
        RelationshipType.UNKNOWN: NoMetric,
        }

    def __init__(self, structures: StructurePair,
                 slice_table: pd.DataFrame = pd.DataFrame(), **kwargs) -> None:
        self.is_logical = False
        self.show = True
        self.metric = {}
        # Sets the is_logical and metric attributes, if supplied.  Ignores any
        # other items in kwargs.
        self.set(**kwargs)
        # Order the structures with the largest first.
        self.structures = None
        self.set_structures(structures)
        # Determine the relationship type.  Either set it from a kwargs value
        # or determine it by comparing the structures.
        self.relationship_type = RelationshipType.UNKNOWN
        if 'relationship' in kwargs:
            self.relationship_type = RelationshipType[kwargs['relationship']]
        else:
            self.identify_relationship(slice_table)
            self.get_metric(slice_table=slice_table, **kwargs)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def get_metric(self, slice_table=pd.DataFrame(), **kwargs):
        # Select the appropriate metric for the identified relationship.
        metric_class = self.metric_match[self.relationship_type]
        self.metric = metric_class(self.structures, slice_table=slice_table,
                                   relation=self.relationship_type, **kwargs)

    @property
    def is_symmetric(self)-> bool:
        return self.relationship_type in self.symmetric_relations

    @property
    def is_transitive(self)-> bool:
        return self.relationship_type in self.transitive_relations

    def set_structures(self, structures: StructurePair,
                       slice_table: pd.DataFrame = pd.DataFrame()) -> None:
        # FIXME Stub method to be replaced with set_structures function.
        # Order the structures with the larger one first
        if slice_table.empty:
            self.structures = structures

    def identify_relationship(self, slice_table: pd.DataFrame) -> None:
        '''Get the 27 bit relationship integer for two structures,

            When written in binary, the 27 bit relationship contains 3 9 bit
            parts corresponding to DE-9IM relationships. The left-most 9 bits
            are the relationship between the second structure's contour and the
            first structure's convex hull.  The middle 9 bits are the
            relationship between the second structure's contour and the first
            structure's exterior. (The first structure's contour with any holes
            filled). The right-most 9 bits are the relationship between the
            second structure's contour and the first structure's actual contour.

            Note: The order of structures matters. For correct comparison, the
            first structure should always be the larger of the two structures.

            Args:
                slice_structures (pd.DataFrame): A table of structures, where the
                    values are the contours with type StructureSlice. The column
                    index contains the roi numbers for the structures.  The row
                    index contains the slice index distances.
        '''
        # Select the identified structure from the full table
        slice_structures = slice_table.loc[:, [self.structures[0],
                                                self.structures[1]]]
        # Remove Slices that have neither structure.
        slice_structures.dropna(how='all', inplace=True)
        boundary_slices = slice_structures.apply(find_boundary_slices)
        # For slices that have only one of the two structures, replace the nan
        # values with empty polygons for duck typing.
        slice_structures.fillna(StructureSlice([]), inplace=True)
        # Get the relationships between the two structures for all slices.
        relation_seq = slice_structures.agg(relate_structures, structures=self.structures,
                                            axis='columns')
        # Adjust the relationship metrics for the boundary slices of both structures.
        relation_seq = adjust_slice_boundary_relations(relation_seq,
                                                       self.structures,
                                                       boundary_slices)

        # Get the overall relationship for the two structures by merging the
        # relationships for the individual slices.
        relation_binary = merge_rel(relation_seq)
        self.relationship_type = identify_relation(relation_binary)
        return relation_binary
