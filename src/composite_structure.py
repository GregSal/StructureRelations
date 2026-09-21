'''CompositeStructure class and structure_boolean function.

Provides boolean algebra (UNION, INTERSECTION, DIFFERENCE) over StructureShape
objects, producing a new CompositeStructure with a synthetic negative ROI.
'''
# %% Imports
from typing import TYPE_CHECKING, List, Tuple
import re

import shapely

from structures import StructureShape
from contours import build_contour_table_from_polygons
from utilities import make_multi
from types_and_classes import ROI_Type

if TYPE_CHECKING:
    from structure_set import StructureSet


# %% Operator definitions
_OPERATOR_ALIASES = {
    'UNION': 'UNION', '|': 'UNION',
    'INTERSECTION': 'INTERSECTION', '&': 'INTERSECTION',
    'DIFFERENCE': 'DIFFERENCE', '-': 'DIFFERENCE',
}
_OPERATOR_FUNCTIONS = {
    'UNION': shapely.union,
    'INTERSECTION': shapely.intersection,
    'DIFFERENCE': shapely.difference,
}
# Matches any of the operator tokens, surrounded by whitespace.
_EXPRESSION_SPLIT_PATTERN = re.compile(
    r'\s+(UNION|INTERSECTION|DIFFERENCE|\||&|-)\s+', re.IGNORECASE
)


# %% CompositeStructure class
class CompositeStructure(StructureShape):
    '''A StructureShape derived from a boolean expression of other structures.

    Attributes:
        expression (str): The boolean expression, with operand references
            replaced by their structure names (or `[ROI]` for base
            structures) that produced this CompositeStructure.
    '''
    # Decremented for each new instance to produce unique negative ROIs.
    _roi_counter = 0

    def __init__(self, roi: ROI_Type, name: str, expression: str):
        super().__init__(roi=roi, name=name)
        self.expression = expression
        # Provenance: operand ROIs and operators that produced this composite.
        # Used by metric calculators to reuse existing composites.
        self.operand_rois: Tuple[ROI_Type, ...] = tuple()
        self.operators: Tuple[str, ...] = tuple()

    @classmethod
    def _next_roi(cls) -> ROI_Type:
        '''Return the next unique negative ROI number.'''
        cls._roi_counter -= 1
        return cls._roi_counter


# %% Expression parsing
def _parse_expression(expression: str) -> Tuple[List[str], List[str]]:
    '''Split a boolean expression into operand tokens and operators.

    Args:
        expression (str): A boolean expression of the form
            `A <operator> B <operator> C ...`, where `<operator>` is one of
            UNION/|, INTERSECTION/&, or DIFFERENCE/-.

    Returns:
        tuple: A tuple of (tokens, operators). `tokens` has one more entry
            than `operators`.
    '''
    parts = _EXPRESSION_SPLIT_PATTERN.split(expression)
    tokens = [part.strip() for part in parts[0::2]]
    operators = [_OPERATOR_ALIASES[part.upper()] for part in parts[1::2]]
    return tokens, operators


def _resolve_operand(structure_set: 'StructureSet',
                     token: str) -> StructureShape:
    '''Resolve an expression token to a StructureShape from a StructureSet.

    The token is first tried as an ROI number, then matched against
    structure names.

    Args:
        structure_set (StructureSet): The set of structures to search.
        token (str): The token to resolve, either an ROI number or name.

    Raises:
        ValueError: If no structure matches the token.

    Returns:
        StructureShape: The matching structure.
    '''
    try:
        roi = int(token)
    except ValueError:
        pass
    else:
        if roi in structure_set.structures:
            return structure_set.structures[roi]
    for structure in structure_set.structures.values():
        if structure.name == token:
            return structure
    raise ValueError(f'Could not resolve structure reference: {token!r}')


# %% Per-slice geometry
def _solid_polygon_for_slice(structure: StructureShape,
                             slice_index) -> shapely.MultiPolygon:
    '''Get the hole-subtracted MultiPolygon for a structure on one slice.

    Boundary and interpolated contours are excluded. Hole contours are
    unioned and subtracted from the union of the non-hole contours.

    Args:
        structure (StructureShape): The structure to extract geometry from.
        slice_index (SliceIndexType): The slice to extract geometry for.

    Returns:
        shapely.MultiPolygon: The resulting solid geometry, empty if the
            structure has no non-boundary, non-interpolated contours on the
            given slice.
    '''
    lookup = structure.contour_lookup
    if lookup.empty:
        return shapely.MultiPolygon()
    on_slice = ((lookup['SliceIndex'] == slice_index)
                & ~lookup['Boundary'] & ~lookup['Interpolated'])
    rows = lookup.loc[on_slice]
    if rows.empty:
        return shapely.MultiPolygon()

    is_hole = rows['HoleType'] != 'None'
    ext_polygons = [structure.contour_graph.nodes[label]['contour'].polygon
                    for label in rows.loc[~is_hole, 'Label']]
    hole_polygons = [structure.contour_graph.nodes[label]['contour'].polygon
                     for label in rows.loc[is_hole, 'Label']]

    ext_union = shapely.unary_union(ext_polygons) if ext_polygons \
        else shapely.Polygon()
    hole_union = shapely.unary_union(hole_polygons) if hole_polygons \
        else shapely.Polygon()
    solid = ext_union.difference(hole_union) if not hole_union.is_empty \
        else ext_union
    return make_multi(solid)


def _operand_display_name(operand: StructureShape) -> str:
    '''Return the name used to represent an operand in a composite name.

    CompositeStructure operands contribute their own descriptive name; base
    StructureShape operands contribute their ROI number in brackets.
    '''
    if isinstance(operand, CompositeStructure):
        return operand.name
    return f'[{operand.roi}]'


# %% Boolean operation
def structure_boolean(structure_set: 'StructureSet',
                      expression: str) -> CompositeStructure:
    '''Create a CompositeStructure from a boolean expression of structures.

    Args:
        structure_set (StructureSet): The set containing the operand
            structures, referenced in `expression` by ROI number or name.
        expression (str): A boolean expression of the form
            `A <operator> B <operator> C ...`, where `<operator>` is one of
            UNION/|, INTERSECTION/&, or DIFFERENCE/-.

    Returns:
        CompositeStructure: The new structure, already registered with
            `structure_set`.
    '''
    tokens, operators = _parse_expression(expression)
    operands = [_resolve_operand(structure_set, token) for token in tokens]
    display_names = [_operand_display_name(operand) for operand in operands]
    # Rebuild the display expression using the same operator text as written.
    raw_parts = _EXPRESSION_SPLIT_PATTERN.split(expression)
    display_parts = list(raw_parts)
    display_parts[0::2] = display_names
    structure_name = ' '.join(display_parts)

    new_roi = CompositeStructure._next_roi()

    all_slices = sorted({
        slice_index
        for operand in operands if not operand.contour_lookup.empty
        for slice_index in operand.contour_lookup['SliceIndex'].unique()
    })

    polygons = []
    for slice_index in all_slices:
        result = _solid_polygon_for_slice(operands[0], slice_index)
        for operator, operand in zip(operators, operands[1:]):
            other = _solid_polygon_for_slice(operand, slice_index)
            result = make_multi(_OPERATOR_FUNCTIONS[operator](result, other))
        if result.is_empty:
            continue
        polygons.append(make_multi(shapely.force_3d(result, slice_index)))

    contour_table, _ = build_contour_table_from_polygons(polygons, roi=new_roi)

    composite = CompositeStructure(roi=new_roi, name=structure_name,
                                   expression=structure_name)
    composite.operand_rois = tuple(operand.roi for operand in operands)
    composite.operators = tuple(operators)
    structure_set.slice_sequence = composite.add_contour_graph(
        contour_table, structure_set.slice_sequence
    )
    composite.finalize(structure_set.slice_sequence)
    structure_set.add_structure(composite)
    return composite
