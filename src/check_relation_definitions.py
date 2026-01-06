'''This script is designed to check the patterns and values in the relationship_definitions.json file.

It reads in the Json file and preforms the following tests:
1. Pattern Check
    - Generate the pattern from the mask and value.
        1. Copy the mask as a string to a pattern test string.
        2. 0s in the pattern test string become '*'.
        2. locate the 1s in the value and replace the corresponding location in
            the pattern test string with 'T'.
        3. Replace any remaining 1s in the pattern test string with 'F'.
    - Compare the generated pattern to the pattern in the Json file.
2. Value Check
    - Ensure that the value only contains 1s where the mask contains 1s.
3. Decimal Check
    - Ensure that the decimal value matches the binary value for both the value
        and the mask.
4. Uniqueness Check
    - Ensure that all relationship definitions are unique across pattern, value,
        and mask, ignoring relationships that have an empty pattern.
5. Symmetric Check
    - Ensure that for every symmetric relationship definition the
        complementary_relation is itself.
5. Complimentary Check
    - Ensure that for every non-symmetric relationship definition except
        "UNKNOWN" a complementary_relation is given.
    - Ensure that the complementary_relation's complementary_relation is the
        original relationship definition.
6. Color Check
    - Ensure that the color is a valid hex code.
    - Ensure that the color is the same for complementary relations.
    - Ensure that the color is unique across all relationship definitions
        except its complementary relation.
7. Symbol Check
    - Ensure that the symbol is a single character.
    - Ensure that the symbol is the same for complementary relations.
    - Ensure that the symbol is unique across all relationship definitions
        except its complementary relation.
8. Label Check
    - Ensure that the label is not empty.
    - Ensure that the label is unique across all relationship definitions.
    - Ensure that the using a case insensitive check, the relation_type is
        found in the label.
9. Relation Type Check
    - Ensure that the relation_type is unique across all relationship definitions.
10. Arrow Check
    - Ensure that the reversed_arrow attribute is set for non-symmetric relations
        and not set for symmetric relations.
    - Ensure that the reversed_arrow attribute is true for relations that have
        an empty pattern and false for non-empty patterns.
11. Description Check
    - Ensure that the description is not empty.
    - Ensure that the description is unique across all relationship definitions.
    - Ensure that the descriptions of complementary relations are the same
        except for 'A' and 'B' being swapped where applicable.
12. Implied Relation Check
    - Ensure that implied_relation is only given for relations that are not
        transitive.
    - Ensure that the implied_relation is transitive.
13. Examples Check
    - Ensure that if examples is given, it give a list of strings describing
        relative image paths. (.png files)
    - Ensure that each example path exists.
'''

# %% Imports
import json
import sys
import re
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


# %% Main Execution
def main():
    '''Run all validation checks and report results.'''
    # Load definitions
    definitions = load_definitions()

    # Track all check results
    all_checks = []

    # Check 0: Structure Validation
    print('Running validation checks...\n')
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

    # Run all content checks
    all_checks.append(('Pattern Check', check_pattern(definitions, skip_indices)))
    all_checks.append(('Value Check', check_value(definitions, skip_indices)))
    all_checks.append(('Decimal Check', check_decimal(definitions, skip_indices)))
    all_checks.append(('Uniqueness Check', check_uniqueness(definitions, skip_indices)))
    all_checks.append(('Relation Type Check', check_relation_type(definitions, skip_indices)))
    all_checks.append(('Symmetric Check', check_symmetric(definitions, skip_indices)))
    all_checks.append(('Complementary Check', check_complementary(definitions, lookup, skip_indices)))
    all_checks.append(('Color Check', check_color(definitions, lookup, skip_indices)))
    all_checks.append(('Symbol Check', check_symbol(definitions, lookup, skip_indices)))
    all_checks.append(('Label Check', check_label(definitions, skip_indices)))
    all_checks.append(('Arrow Check', check_arrow(definitions, skip_indices)))
    all_checks.append(('Description Check', check_description(definitions, lookup, skip_indices)))
    all_checks.append(('Implied Relation Check', check_implied_relation(definitions, lookup, skip_indices)))
    all_checks.append(('Examples Check', check_examples(definitions, skip_indices)))

    # Print results for each check
    total_errors = 0
    failed_checks = 0

    for check_name, errors in all_checks[1:]:  # Skip structure (already printed)
        if errors:
            print(f'\n=== {check_name} ===')
            for error in errors:
                print(f'  {error}')
            total_errors += len(errors)
            failed_checks += 1
        else:
            print(f'✓ {check_name}')

    # Print summary
    total_checks = len(all_checks)
    passed_checks = total_checks - failed_checks

    print(f'\n{"="*60}')
    print(f'Validation complete: {passed_checks}/{total_checks} checks passed')
    if failed_checks > 0:
        print(f'{failed_checks} checks failed ({total_errors} total errors)')
        sys.exit(1)
    else:
        print('All checks passed!')
        sys.exit(0)


if __name__ == '__main__':
    main()
