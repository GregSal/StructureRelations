'''This script is designed to check the patterns and values in the relationship_definitions.json file.

It reads in the Json file and performs the following tests:
1. Binary Format Check
    - Ensure mask/value have '0b' prefix and exactly 27 bits.
    - Ensure mask/value contain only 0 and 1 characters.
2. Pattern Format Check
    - Ensure pattern contains only T, F, * and tabs.
    - Ensure pattern has exactly 29 characters (27 bits + 2 tabs).
    - Ensure tabs are at positions 9 and 19.
3. Pattern Check
    - Generate the pattern from the mask and value.
        1. Copy the mask as a string to a pattern test string.
        2. 0s in the pattern test string become '*'.
        3. locate the 1s in the value and replace the corresponding location in
            the pattern test string with 'T'.
        4. Replace any remaining 1s in the pattern test string with 'F'.
    - Compare the generated pattern to the pattern in the Json file.
4. Value Check
    - Ensure that the value only contains 1s where the mask contains 1s.
5. Decimal Check
    - Ensure that the decimal value matches the binary value for both the value
        and the mask.
6. Uniqueness Check
    - Ensure that all relationship definitions are unique across pattern, value,
        and mask, ignoring relationships that have an empty pattern.
7. Relation Type Check
    - Ensure that the relation_type is unique across all relationship definitions.
8. Symmetric Check
    - Ensure that for every symmetric relationship definition the
        complementary_relation is itself.
9. Complementary Check
    - Ensure that for every non-symmetric relationship definition except
        "UNKNOWN" a complementary_relation is given.
    - Ensure that the complementary_relation's complementary_relation is the
        original relationship definition.
10. Color Check
    - Ensure that the color is a valid hex code.
    - Ensure that the color is the same for complementary relations.
    - Ensure that the color is unique across all relationship definitions
        except its complementary relation.
11. Symbol Check
    - Ensure that the symbol is a single character.
    - Ensure that the symbol is the same for complementary relations.
12. Symbol Uniqueness Check
    - Ensure symbols are unique across non-complementary pairs.
13. Label Check
    - Ensure that the label is not empty.
    - Ensure that the label is unique across all relationship definitions.
    - Ensure that the using a case insensitive check, the relation_type is
        found in the label.
14. Arrow Check
    - Ensure that the reversed_arrow attribute is set for non-symmetric relations
        and not set for symmetric relations.
    - Ensure that the reversed_arrow attribute is true for relations that have
        an empty pattern and false for non-empty patterns.
15. Empty Pattern Consistency Check
    - Ensure reversed_arrow=True when pattern empty (except UNKNOWN).
16. Description Check
    - Ensure that the description is not empty.
    - Ensure that the description is unique across all relationship definitions.
    - Ensure that the descriptions of complementary relations are the same
        except for 'A' and 'B' being swapped where applicable.
17. Implied Relation Check
    - Ensure that implied_relation is only given for relations that are not
        transitive.
    - Ensure that the implied_relation is transitive.
18. Implied Circularity Check
    - Detect circular implied_relation chains using DFS.
19. Stub Completeness Check
    - Ensure empty-pattern relations have matching color/symbol with complementary.
20. Field Types Check
    - Validate symmetric/transitive are bool, decimals are int, examples is list.
21. Pattern/Mask/Value Consistency Check
    - Ensure all three fields present together or all empty.
22. Examples Check
    - Ensure that if examples is given, it give a list of strings describing
        relative image paths. (.png files)
    - Ensure that each example path exists.
23. Stub Sync Check
    - Verify relations.pyi stub file constants match JSON primary relationships.
'''

# %% Imports
import json
import sys
import re
import argparse
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# %% Constants
JSON_FILE = Path(__file__).parent / 'relationship_definitions.json'

REQUIRED_FIELDS = [
    'relation_type', 'label', 'symbol', 'color', 'description',
    'complementary_relation', 'symmetric', 'transitive'
]


# %% Helper Functions
def load_definitions() -> List[Dict]:
    '''Load and return relationship definitions from JSON file.

    Returns:
        List[Dict]: List of relationship definitions.

    Exits:
        If file cannot be read or JSON is invalid.
    '''
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            definitions = json.load(f)
        return definitions
    except FileNotFoundError:
        print(f'Error: File not found: {JSON_FILE}')
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in {JSON_FILE}: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Error loading {JSON_FILE}: {e}')
        sys.exit(1)


def build_lookup(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> Dict[str, Dict]:
    '''Build lookup dictionary mapping relation_type to definition.

    Only includes valid definitions (not in skip_indices).

    Args:
        definitions: List of all relationship definitions.
        skip_indices: Set of indices to skip (invalid definitions).

    Returns:
        Dict mapping relation_type to definition dict.
    '''
    lookup = {}
    for idx, defn in enumerate(definitions):
        if idx not in skip_indices:
            relation_type = defn.get('relation_type', '')
            if relation_type:
                lookup[relation_type] = defn
    return lookup


def binary_to_decimal(binary_str: str) -> int | None:
    '''Convert binary string to decimal integer.

    Args:
        binary_str: Binary string with '0b' prefix, or empty string.

    Returns:
        Integer value, or None if empty string.

    Raises:
        ValueError: If binary_str doesn't have '0b' prefix (when non-empty).
    '''
    if not binary_str:
        return None

    if not binary_str.startswith('0b'):
        raise ValueError(
            f"Binary string must start with '0b': {binary_str}"
        )

    return int(binary_str, 2)


def pattern_from_mask_value(mask: str, value: str) -> str:
    '''Generate pattern string from mask and value binary strings.

    Pattern generation rules:
    - Mask bit 0 → '*'
    - Mask bit 1 and value bit 1 → 'T'
    - Mask bit 1 and value bit 0 → 'F'

    Args:
        mask: Binary string with '0b' prefix, or empty string.
        value: Binary string with '0b' prefix, or empty string.

    Returns:
        Pattern string formatted as 3 groups of 9 chars separated by tabs,
        or empty string if inputs are empty.
    '''
    if not mask or not value:
        return ''

    # Strip '0b' prefix
    mask_bits = mask[2:] if mask.startswith('0b') else mask
    value_bits = value[2:] if value.startswith('0b') else value

    # Pad to 27 bits if necessary
    mask_bits = mask_bits.zfill(27)
    value_bits = value_bits.zfill(27)

    # Generate pattern
    pattern = []
    for m_bit, v_bit in zip(mask_bits, value_bits):
        if m_bit == '0':
            pattern.append('*')
        elif v_bit == '1':
            pattern.append('T')
        else:
            pattern.append('F')

    # Format as 3 groups of 9 separated by tabs
    pattern_str = ''.join(pattern)
    formatted = '\t'.join([
        pattern_str[0:9],
        pattern_str[9:18],
        pattern_str[18:27]
    ])

    return formatted


def swap_ab(text: str) -> str:
    '''Swap all occurrences of 'A' and 'B' in text, except in phrases "A and B" or "B and A".

    Args:
        text: String to swap A and B in.

    Returns:
        String with A and B swapped, preserving "A and B" and "B and A" phrases.
    '''
    import re

    # Protect "A and B" and "B and A" patterns
    temp = text.replace('A and B', '__a&b__')
    temp = temp.replace('B and A', '__b&a__')

    # Use word boundary regex to swap A and B
    # Match A or B as whole words (surrounded by non-word chars or start/end)
    temp = re.sub(r'\bA\b', '__TEMP__', temp)
    temp = re.sub(r'\bB\b', 'A', temp)
    temp = re.sub(r'\b__TEMP__\b', 'B', temp)

    # Restore protected patterns
    temp = temp.replace('__a&b__', 'A and B')
    temp = temp.replace('__b&a__', 'B and A')

    return temp


# %% Validation Functions
def check_structure(definitions: List[Dict]) -> Tuple[List[str], Set[int]]:
    '''Check that all definitions have required fields.

    Args:
        definitions: List of relationship definitions.

    Returns:
        Tuple of (error messages, set of invalid indices).
    '''
    errors = []
    skip_indices = set()

    for idx, defn in enumerate(definitions):
        relation_type = defn.get('relation_type', f'index {idx}')

        for field in REQUIRED_FIELDS:
            if field not in defn:
                errors.append(
                    f'Definition at index {idx} ({relation_type}): '
                    f'missing required field \'{field}\''
                )
                skip_indices.add(idx)

    return errors, skip_indices


def check_pattern(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that stored patterns match generated patterns.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        pattern = defn.get('pattern', '')
        mask = defn.get('mask', '')
        value = defn.get('value', '')

        # Skip empty patterns and UNKNOWN
        if not pattern or relation_type == 'UNKNOWN':
            continue

        generated_pattern = pattern_from_mask_value(mask, value)

        if pattern != generated_pattern:
            errors.append(
                f'{relation_type}: pattern mismatch\n'
                f'  Generated: {generated_pattern}\n'
                f'  Stored:    {pattern}'
            )

    return errors


def check_value(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that value bits are only set where mask bits are set.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        mask = defn.get('mask', '')
        value = defn.get('value', '')

        # Skip empty mask/value
        if not mask or not value:
            continue

        # Strip '0b' prefix
        mask_bits = mask[2:] if mask.startswith('0b') else mask
        value_bits = value[2:] if value.startswith('0b') else value

        # Pad to same length
        max_len = max(len(mask_bits), len(value_bits))
        mask_bits = mask_bits.zfill(max_len)
        value_bits = value_bits.zfill(max_len)

        # Check each bit position
        for pos, (m_bit, v_bit) in enumerate(zip(mask_bits, value_bits)):
            if v_bit == '1' and m_bit == '0':
                errors.append(
                    f'{relation_type}: value has bit set at position {pos} '
                    f'where mask has 0 (mask={mask}, value={value})'
                )

    return errors


def check_decimal(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that decimal values match binary values.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']

        # Check mask_decimal
        mask = defn.get('mask', '')
        mask_decimal = defn.get('mask_decimal', 0)

        if mask:
            try:
                expected_decimal = binary_to_decimal(mask)
                if mask_decimal != expected_decimal:
                    errors.append(
                        f'{relation_type}: mask_decimal {mask_decimal} does '
                        f'not match binary mask {mask} '
                        f'(expected {expected_decimal})'
                    )
            except ValueError as e:
                errors.append(
                    f'{relation_type}: invalid mask binary format: {e}'
                )

        # Check value_decimal
        value = defn.get('value', '')
        value_decimal = defn.get('value_decimal', 0)

        if value:
            try:
                expected_decimal = binary_to_decimal(value)
                if value_decimal != expected_decimal:
                    errors.append(
                        f'{relation_type}: value_decimal {value_decimal} '
                        f'does not match binary value {value} '
                        f'(expected {expected_decimal})'
                    )
            except ValueError as e:
                errors.append(
                    f'{relation_type}: invalid value binary format: {e}'
                )

    return errors


def check_uniqueness(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that patterns are unique and (mask, value) pairs are unique.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Build maps
    pattern_map = defaultdict(list)
    mask_value_map = defaultdict(list)

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        pattern = defn.get('pattern', '')
        mask = defn.get('mask', '')
        value = defn.get('value', '')

        # Check pattern uniqueness (non-empty only)
        if pattern:
            pattern_map[pattern].append(relation_type)

        # Check (mask, value) pair uniqueness (only if both non-empty)
        if mask and value:
            mask_value_map[(mask, value)].append(relation_type)

    # Check for duplicate patterns
    for pattern, relation_types in pattern_map.items():
        if len(relation_types) > 1:
            errors.append(
                f'Duplicate pattern \'{pattern}\' found in: '
                f'{", ".join(relation_types)}'
            )

    # Check for duplicate (mask, value) pairs
    for (mask, value), relation_types in mask_value_map.items():
        if len(relation_types) > 1:
            errors.append(
                f'Duplicate (mask, value) pair ({mask}, {value}) found in: '
                f'{", ".join(relation_types)}'
            )

    return errors


def check_relation_type(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that relation_type values are unique.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []
    seen = {}

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']

        if relation_type in seen:
            errors.append(
                f'Duplicate relation_type \'{relation_type}\' at indices '
                f'{seen[relation_type]} and {idx}'
            )
        else:
            seen[relation_type] = idx

    return errors


def check_symmetric(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that symmetric relations have self-referential complementary_relation.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        symmetric = defn['symmetric']
        complementary_relation = defn['complementary_relation']

        if symmetric and complementary_relation != relation_type:
            errors.append(
                f'{relation_type}: marked symmetric but '
                f'complementary_relation is \'{complementary_relation}\' '
                f'(should be \'{relation_type}\')'
            )

    return errors


def check_complementary(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check complementary relationship consistency.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        symmetric = defn['symmetric']
        complementary_relation = defn['complementary_relation']
        pattern = defn.get('pattern', '')

        # Non-symmetric relations (except UNKNOWN) must have complementary_relation
        if not symmetric and relation_type != 'UNKNOWN':
            if not complementary_relation:
                errors.append(
                    f'{relation_type}: non-symmetric relation must have '
                    f'complementary_relation'
                )
                continue

        # Check bidirectional matching
        if complementary_relation and complementary_relation in lookup:
            comp_defn = lookup[complementary_relation]
            comp_complementary = comp_defn.get('complementary_relation', '')

            if comp_complementary != relation_type:
                errors.append(
                    f'{relation_type}: complementary_relation='
                    f'\'{complementary_relation}\' but '
                    f'{complementary_relation}: complementary_relation='
                    f'\'{comp_complementary}\''
                )
        elif complementary_relation and complementary_relation != relation_type:
            # complementary_relation not found in lookup (unless self-referential)
            errors.append(
                f'{relation_type}: complementary_relation '
                f'\'{complementary_relation}\' not found in definitions'
            )

    return errors


def check_color(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check color validity and consistency.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []
    hex_pattern = re.compile(r'^#[0-9a-fA-F]{6}$')

    # Track colors for uniqueness check
    color_map = defaultdict(list)

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        color = defn['color']

        # Validate hex format
        if not hex_pattern.match(color):
            errors.append(
                f'{relation_type}: invalid color format \'{color}\' '
                f'(must be #RRGGBB)'
            )
            continue

        # Normalize to lowercase for comparison
        color_normalized = color.lower()

        # Check complementary pair matching
        complementary_relation = defn.get('complementary_relation', '')
        if (complementary_relation and
            complementary_relation in lookup and
            complementary_relation != relation_type):

            comp_color = lookup[complementary_relation]['color'].lower()
            if color_normalized != comp_color:
                errors.append(
                    f'{relation_type}: color {color} does not match '
                    f'complementary relation {complementary_relation} '
                    f'color {lookup[complementary_relation]["color"]}'
                )

        # Track for uniqueness (store original case)
        color_map[color_normalized].append((relation_type, color))

    # Check uniqueness (excluding complementary pairs)
    for color_norm, relations in color_map.items():
        if len(relations) > 2:
            # More than one complementary pair with same color
            relation_names = [r[0] for r in relations]
            errors.append(
                f'Color {relations[0][1]} used by more than one '
                f'complementary pair: {", ".join(relation_names)}'
            )

    return errors


def check_symbol(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check symbol validity and consistency.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        symbol = defn['symbol']

        # Check single character or empty
        if len(symbol) > 1:
            errors.append(
                f'{relation_type}: symbol must be single character or empty, '
                f'got \'{symbol}\''
            )

        # Check complementary pair matching
        complementary_relation = defn.get('complementary_relation', '')
        if (complementary_relation and
            complementary_relation in lookup and
            complementary_relation != relation_type):

            comp_symbol = lookup[complementary_relation]['symbol']
            if symbol != comp_symbol:
                errors.append(
                    f'{relation_type}: symbol \'{symbol}\' does not match '
                    f'complementary relation {complementary_relation} '
                    f'symbol \'{comp_symbol}\''
                )

    return errors


def check_label(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check label validity and uniqueness.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Track labels for uniqueness (case-insensitive)
    label_map = defaultdict(list)

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        label = defn['label']

        # Check non-empty
        if not label:
            errors.append(f'{relation_type}: label is empty')
            continue

        # Check relation_type in label (case-insensitive)
        if relation_type.lower() not in label.lower():
            errors.append(
                f'{relation_type}: relation_type not found in label '
                f'\'{label}\' (case-insensitive check)'
            )

        # Track for uniqueness
        label_map[label.lower()].append((relation_type, label))

    # Check uniqueness
    for label_lower, relations in label_map.items():
        if len(relations) > 1:
            relation_names = [f'{r[0]} (\'{r[1]}\')' for r in relations]
            errors.append(
                f'Duplicate label (case-insensitive): {", ".join(relation_names)}'
            )

    return errors


def check_arrow(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check reversed_arrow attribute consistency.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        symmetric = defn['symmetric']
        pattern = defn.get('pattern', '')
        has_arrow = 'reversed_arrow' in defn
        reversed_arrow = defn.get('reversed_arrow')

        # Symmetric relations should not have reversed_arrow
        if symmetric and has_arrow:
            errors.append(
                f'{relation_type}: symmetric relation should not have '
                f'reversed_arrow attribute'
            )

        # Non-symmetric relations should have reversed_arrow
        if not symmetric and not has_arrow:
            errors.append(
                f'{relation_type}: non-symmetric relation must have '
                f'reversed_arrow attribute'
            )

        # Check value based on pattern
        if has_arrow:
            if not pattern and reversed_arrow is not True:
                errors.append(
                    f'{relation_type}: has empty pattern but '
                    f'reversed_arrow={reversed_arrow} (should be True)'
                )
            elif pattern and reversed_arrow is not False:
                errors.append(
                    f'{relation_type}: has non-empty pattern but '
                    f'reversed_arrow={reversed_arrow} (should be False)'
                )

    return errors


def check_description(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check description validity and complementary pair consistency.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Track descriptions for uniqueness
    desc_map = defaultdict(list)

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        description = defn['description']

        # Check non-empty
        if not description:
            errors.append(f'{relation_type}: description is empty')
            continue

        # Track for uniqueness
        desc_map[description].append(relation_type)

        # Check complementary pair A/B swap
        complementary_relation = defn.get('complementary_relation', '')
        if (complementary_relation and
            complementary_relation in lookup and
            complementary_relation != relation_type):

            comp_description = lookup[complementary_relation]['description']
            expected_description = swap_ab(comp_description)

            if description != expected_description:
                errors.append(
                    f'{relation_type}: description does not match '
                    f'complementary relation {complementary_relation} '
                    f'with A/B swapped.\n'
                    f'  {relation_type}: "{description}"\n'
                    f'  {complementary_relation}: "{comp_description}"\n'
                    f'  Expected (swapped): "{expected_description}"'
                )

    # Check uniqueness
    for description, relation_types in desc_map.items():
        if len(relation_types) > 1:
            errors.append(
                f'Duplicate description in: {", ".join(relation_types)}'
            )

    return errors


def check_implied_relation(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check implied_relation validity.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        transitive = defn['transitive']
        implied_relation = defn.get('implied_relation', '')

        if implied_relation:
            # Should only be on non-transitive relations
            if transitive:
                errors.append(
                    f'{relation_type}: has implied_relation but is transitive '
                    f'(implied_relation should only be on non-transitive '
                    f'relations)'
                )

            # Check that implied relation exists
            if implied_relation not in lookup:
                errors.append(
                    f'{relation_type}: implied_relation '
                    f'\'{implied_relation}\' not found in definitions'
                )
            else:
                # Check that implied relation is transitive
                implied_defn = lookup[implied_relation]
                if not implied_defn.get('transitive', False):
                    errors.append(
                        f'{relation_type}: implied_relation '
                        f'\'{implied_relation}\' is not transitive '
                        f'(implied relations must be transitive)'
                    )

    return errors


def check_examples(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check examples paths validity.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Base path is parent of src/ directory
    base_path = Path(__file__).parent.parent

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        examples = defn.get('examples', [])

        if not examples:
            continue

        # Check that examples is a list
        if not isinstance(examples, list):
            errors.append(
                f'{relation_type}: examples must be a list, '
                f'got {type(examples).__name__}'
            )
            continue

        # Check each example
        for example in examples:
            # Check that it's a string
            if not isinstance(example, str):
                errors.append(
                    f'{relation_type}: example must be string, '
                    f'got {type(example).__name__}: {example}'
                )
                continue

            # Check that it ends with .png
            if not example.endswith('.png'):
                errors.append(
                    f'{relation_type}: example must be .png file, '
                    f'got {example}'
                )

            # Normalize path (backslashes to forward slashes)
            normalized_path = example.replace('\\', '/')

            # Construct absolute path
            abs_path = base_path / normalized_path

            # Check that file exists
            if not abs_path.exists():
                errors.append(
                    f'{relation_type}: example file not found: '
                    f'{normalized_path}'
                )

    return errors


def check_binary_format(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that mask/value have '0b' prefix and exactly 27 bits.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        mask = defn.get('mask', '')
        value = defn.get('value', '')

        # Skip empty mask/value (for UNKNOWN and complementary stubs)
        if not mask and not value:
            continue

        # Check mask format
        if mask:
            if not mask.startswith('0b'):
                errors.append(
                    f'{relation_type}: mask must start with \'0b\', '
                    f'got \'{mask}\''
                )
            else:
                # Check length (should be 27 bits + '0b' prefix = 29 chars)
                mask_bits = mask[2:]
                if len(mask_bits) != 27:
                    errors.append(
                        f'{relation_type}: mask must have exactly 27 bits, '
                        f'got {len(mask_bits)} bits'
                    )
                # Check only contains 0 and 1
                if not all(c in '01' for c in mask_bits):
                    errors.append(
                        f'{relation_type}: mask must contain only 0 and 1, '
                        f'got \'{mask}\''
                    )

        # Check value format
        if value:
            if not value.startswith('0b'):
                errors.append(
                    f'{relation_type}: value must start with \'0b\', '
                    f'got \'{value}\''
                )
            else:
                # Check length
                value_bits = value[2:]
                if len(value_bits) != 27:
                    errors.append(
                        f'{relation_type}: value must have exactly 27 bits, '
                        f'got {len(value_bits)} bits'
                    )
                # Check only contains 0 and 1
                if not all(c in '01' for c in value_bits):
                    errors.append(
                        f'{relation_type}: value must contain only 0 and 1, '
                        f'got \'{value}\''
                    )

    return errors


def check_pattern_format(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that pattern contains only T/F/* and has correct format.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        pattern = defn.get('pattern', '')

        # Skip empty patterns
        if not pattern:
            continue

        # Check total length (27 chars + 2 tabs = 29 chars)
        if len(pattern) != 29:
            errors.append(
                f'{relation_type}: pattern must be exactly 29 characters '
                f'(27 bits + 2 tabs), got {len(pattern)} characters'
            )
            continue

        # Check tab positions at indices 9 and 19
        if pattern[9] != '\t' or pattern[19] != '\t':
            errors.append(
                f'{relation_type}: pattern must have tabs at positions 9 and 19, '
                f'got characters \'{pattern[9]}\' and \'{pattern[19]}\''
            )
            continue

        # Check that all non-tab characters are T, F, or *
        chars = pattern.replace('\t', '')
        if not all(c in 'TF*' for c in chars):
            invalid_chars = set(c for c in chars if c not in 'TF*')
            errors.append(
                f'{relation_type}: pattern must contain only T, F, *, and tabs, '
                f'found invalid characters: {invalid_chars}'
            )

    return errors


def check_symbol_uniqueness(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that symbols are unique across non-complementary pairs.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Track symbols for uniqueness (excluding complementary pairs)
    symbol_map = defaultdict(list)

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        symbol = defn['symbol']

        # Track for uniqueness
        symbol_map[symbol].append(relation_type)

    # Check uniqueness (allow at most 2 relations per symbol - complementary pair)
    for symbol, relation_types in symbol_map.items():
        if len(relation_types) > 2:
            errors.append(
                f'Symbol \'{symbol}\' used by more than one complementary pair: '
                f'{", ".join(relation_types)}'
            )

    return errors


def check_empty_pattern_consistency(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check consistency of reversed_arrow for empty-pattern relationships.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        pattern = defn.get('pattern', '')
        has_arrow = 'reversed_arrow' in defn
        reversed_arrow = defn.get('reversed_arrow')

        # Empty pattern should have reversed_arrow=True (except UNKNOWN)
        if not pattern and has_arrow:
            if relation_type != 'UNKNOWN' and reversed_arrow is not True:
                errors.append(
                    f'{relation_type}: has empty pattern but reversed_arrow={reversed_arrow} '
                    f'(should be True for complementary stubs)'
                )
            elif relation_type == 'UNKNOWN' and reversed_arrow is not False:
                errors.append(
                    f'{relation_type}: is special case with empty pattern but '
                    f'reversed_arrow={reversed_arrow} (should be False)'
                )

    return errors


def check_implied_circularity(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check for circular implied_relation chains.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Build directed graph of implied relations
    implied_graph = {}
    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        implied_relation = defn.get('implied_relation', '')

        if implied_relation:
            implied_graph[relation_type] = implied_relation

    # Check for cycles using DFS
    def has_cycle(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)

        if node in implied_graph:
            neighbor = implied_graph[node]
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    visited = set()
    for node in implied_graph:
        if node not in visited:
            rec_stack = set()
            if has_cycle(node, visited, rec_stack):
                errors.append(
                    f'Circular implied_relation detected involving {node}'
                )
                break

    return errors


def check_stub_completeness(
    definitions: List[Dict],
    lookup: Dict[str, Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that empty-pattern relations match complementary's attributes.

    Args:
        definitions: List of relationship definitions.
        lookup: Dict mapping relation_type to definition.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        pattern = defn.get('pattern', '')
        reversed_arrow = defn.get('reversed_arrow')

        # Only check complementary stubs (empty pattern, reversed_arrow=True)
        if not pattern and reversed_arrow is True:
            complementary_relation = defn.get('complementary_relation', '')

            if complementary_relation and complementary_relation in lookup:
                comp_defn = lookup[complementary_relation]

                # Check color matches
                if defn['color'] != comp_defn['color']:
                    errors.append(
                        f'{relation_type}: stub color \'{defn["color"]}\' '
                        f'does not match complementary {complementary_relation} '
                        f'color \'{comp_defn["color"]}\''
                    )

                # Check symbol matches
                if defn['symbol'] != comp_defn['symbol']:
                    errors.append(
                        f'{relation_type}: stub symbol \'{defn["symbol"]}\' '
                        f'does not match complementary {complementary_relation} '
                        f'symbol \'{comp_defn["symbol"]}\''
                    )

    return errors


def check_field_types(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that field types are correct.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']

        # Check boolean fields
        for field in ['symmetric', 'transitive']:
            value = defn.get(field)
            if value is not None and not isinstance(value, bool):
                errors.append(
                    f'{relation_type}: {field} must be boolean, '
                    f'got {type(value).__name__}'
                )

        # Check reversed_arrow if present
        reversed_arrow = defn.get('reversed_arrow')
        if reversed_arrow is not None and not isinstance(reversed_arrow, bool):
            errors.append(
                f'{relation_type}: reversed_arrow must be boolean, '
                f'got {type(reversed_arrow).__name__}'
            )

        # Check integer fields
        for field in ['mask_decimal', 'value_decimal']:
            value = defn.get(field)
            if value is not None and not isinstance(value, int):
                errors.append(
                    f'{relation_type}: {field} must be integer, '
                    f'got {type(value).__name__}'
                )

        # Check examples is list if present
        examples = defn.get('examples')
        if examples is not None and not isinstance(examples, list):
            errors.append(
                f'{relation_type}: examples must be list, '
                f'got {type(examples).__name__}'
            )

    return errors


def check_pattern_mask_value_consistency(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that pattern, mask, value are all present or all empty.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']

        # UNKNOWN is a special case - allow pattern without mask/value
        if relation_type == 'UNKNOWN':
            continue

        pattern = defn.get('pattern', '')
        mask = defn.get('mask', '')
        value = defn.get('value', '')

        # Count which fields are present
        has_pattern = bool(pattern)
        has_mask = bool(mask)
        has_value = bool(value)

        # All three should be present together or all empty
        if has_pattern + has_mask + has_value not in [0, 3]:
            present = []
            if has_pattern:
                present.append('pattern')
            if has_mask:
                present.append('mask')
            if has_value:
                present.append('value')

            missing = []
            if not has_pattern:
                missing.append('pattern')
            if not has_mask:
                missing.append('mask')
            if not has_value:
                missing.append('value')

            errors.append(
                f'{relation_type}: inconsistent pattern/mask/value definition. '
                f'Present: {", ".join(present)}; Missing: {", ".join(missing)}'
            )

    return errors


def check_stub_sync(
    definitions: List[Dict],
    skip_indices: Set[int]
) -> List[str]:
    '''Check that relations.pyi stub file constants match JSON definitions.

    Args:
        definitions: List of relationship definitions.
        skip_indices: Set of indices to skip.

    Returns:
        List of error messages.
    '''
    errors = []

    # Path to stub file
    stub_file = Path(__file__).parent / 'relations.pyi'

    # Check stub file exists
    if not stub_file.exists():
        errors.append(
            f'Type stub file not found: {stub_file}\n'
            'Create src/relations.pyi with relationship constant declarations.\n'
            'Run: python src/generate_stub_constants.py'
        )
        return errors

    # Read stub file
    try:
        stub_content = stub_file.read_text(encoding='utf-8')
    except Exception as e:
        errors.append(f'Failed to read stub file {stub_file}: {e}')
        return errors

    # Extract constant declarations using regex
    # Pattern: name: RelationshipType (at start of line, allowing whitespace)
    pattern = r'^(\w+):\s*RelationshipType'
    stub_constants = set(re.findall(pattern, stub_content, re.MULTILINE))

    # Get primary relationships from JSON (reversed_arrow=False or missing)
    json_constants = set()
    for idx, defn in enumerate(definitions):
        if idx in skip_indices:
            continue

        relation_type = defn['relation_type']
        reversed_arrow = defn.get('reversed_arrow')

        # Only include primary relationships (not complementary stubs)
        if reversed_arrow is not True:
            json_constants.add(relation_type)

    # Check for missing constants in stub
    missing_in_stub = json_constants - stub_constants
    if missing_in_stub:
        errors.append(
            f'Constants missing from {stub_file}: {", ".join(sorted(missing_in_stub))}\n'
            'Run: python src/generate_stub_constants.py'
        )

    # Check for extra constants in stub
    extra_in_stub = stub_constants - json_constants
    if extra_in_stub:
        errors.append(
            f'Extra constants in {stub_file} not in JSON: {", ".join(sorted(extra_in_stub))}\n'
            'Update stub file to match relationship_definitions.json'
        )

    return errors


# %% Main Execution
def main():
    '''Run all validation checks and report results.'''
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Validate relationship_definitions.json schema and consistency'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed validation progress for each relationship'
    )
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # Load definitions
    definitions_data = load_definitions()
    definitions = definitions_data.get('Relationships', [])

    # Track all check results
    all_checks = []

    # Print validation header
    primary_count = sum(1 for d in definitions if d.get('reversed_arrow') is not True)
    complementary_count = sum(1 for d in definitions if d.get('reversed_arrow') is True)
    print(f'{"="*60}')
    print(f'Validating: {JSON_FILE}')
    print(f'Total relationships: {len(definitions)} ({primary_count} primary, {complementary_count} complementary)')
    print(f'{"="*60}\n')

    # Check 0: Structure Validation
    if args.verbose:
        print('Running Structure Validation...')
    struct_errors, skip_indices = check_structure(definitions)
    all_checks.append(('Structure Validation', struct_errors))

    if struct_errors:
        print('=== Structure Validation ===')
        for error in struct_errors:
            print(f'  {error}')

        # If all definitions failed, exit immediately
        if len(skip_indices) == len(definitions):
            print('\nAll definitions failed structure validation. Exiting.')
            sys.exit(1)
    else:
        print('✓ Structure Validation')

    # Build lookup dictionary (excluding invalid definitions)
    lookup = build_lookup(definitions, skip_indices)

    # Run format checks first (new ordering)
    if args.verbose:
        print('Running Binary Format Check...')
    all_checks.append(('Binary Format Check', check_binary_format(definitions, skip_indices)))

    if args.verbose:
        print('Running Pattern Format Check...')
    all_checks.append(('Pattern Format Check', check_pattern_format(definitions, skip_indices)))

    # Run existing checks (reordered to have format checks first)
    if args.verbose:
        print('Running Pattern Check...')
    all_checks.append(('Pattern Check', check_pattern(definitions, skip_indices)))

    if args.verbose:
        print('Running Value Check...')
    all_checks.append(('Value Check', check_value(definitions, skip_indices)))

    if args.verbose:
        print('Running Decimal Check...')
    all_checks.append(('Decimal Check', check_decimal(definitions, skip_indices)))

    if args.verbose:
        print('Running Uniqueness Check...')
    all_checks.append(('Uniqueness Check', check_uniqueness(definitions, skip_indices)))

    if args.verbose:
        print('Running Relation Type Check...')
    all_checks.append(('Relation Type Check', check_relation_type(definitions, skip_indices)))

    if args.verbose:
        print('Running Symmetric Check...')
    all_checks.append(('Symmetric Check', check_symmetric(definitions, skip_indices)))

    if args.verbose:
        print('Running Complementary Check...')
    all_checks.append(('Complementary Check', check_complementary(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Color Check...')
    all_checks.append(('Color Check', check_color(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Symbol Check...')
    all_checks.append(('Symbol Check', check_symbol(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Symbol Uniqueness Check...')
    all_checks.append(('Symbol Uniqueness Check', check_symbol_uniqueness(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Label Check...')
    all_checks.append(('Label Check', check_label(definitions, skip_indices)))

    if args.verbose:
        print('Running Arrow Check...')
    all_checks.append(('Arrow Check', check_arrow(definitions, skip_indices)))

    if args.verbose:
        print('Running Empty Pattern Consistency Check...')
    all_checks.append(('Empty Pattern Consistency Check', check_empty_pattern_consistency(definitions, skip_indices)))

    if args.verbose:
        print('Running Description Check...')
    all_checks.append(('Description Check', check_description(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Implied Relation Check...')
    all_checks.append(('Implied Relation Check', check_implied_relation(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Implied Circularity Check...')
    all_checks.append(('Implied Circularity Check', check_implied_circularity(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Stub Completeness Check...')
    all_checks.append(('Stub Completeness Check', check_stub_completeness(definitions, lookup, skip_indices)))

    if args.verbose:
        print('Running Field Types Check...')
    all_checks.append(('Field Types Check', check_field_types(definitions, skip_indices)))

    if args.verbose:
        print('Running Pattern/Mask/Value Consistency Check...')
    all_checks.append(('Pattern/Mask/Value Consistency Check', check_pattern_mask_value_consistency(definitions, skip_indices)))

    if args.verbose:
        print('Running Examples Check...')
    all_checks.append(('Examples Check', check_examples(definitions, skip_indices)))

    # Stub sync check at the end
    if args.verbose:
        print('Running Stub Sync Check...')
    all_checks.append(('Stub Sync Check', check_stub_sync(definitions, skip_indices)))

    # Print results for each check
    total_errors = 0
    failed_checks = 0

    if args.verbose:
        print()  # Add blank line before results

    for check_name, errors in all_checks[1:]:  # Skip structure (already printed)
        if errors:
            print(f'\n=== {check_name} ===')
            for error in errors:
                print(f'  {error}')
            total_errors += len(errors)
            failed_checks += 1
        else:
            print(f'✓ {check_name}')

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print summary
    total_checks = len(all_checks)
    passed_checks = total_checks - failed_checks

    print(f'\n{"="*60}')
    print(f'Validation complete: {passed_checks}/{total_checks} checks passed')
    print(f'Time elapsed: {elapsed_time:.2f}s')
    if failed_checks > 0:
        print(f'{failed_checks} checks failed ({total_errors} total errors)')
        sys.exit(1)
    else:
        print('All checks passed!')
        sys.exit(0)


if __name__ == '__main__':
    main()
