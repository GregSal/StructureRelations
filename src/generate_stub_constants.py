#!/usr/bin/env python
'''Generate relationship constant declarations for relations.pyi stub file.

This script reads relationship_definitions.json and outputs constant declarations
that can be inserted into the "Generated Constants" section of relations.pyi.

Usage:
    python src/generate_stub_constants.py
    python src/generate_stub_constants.py > stub_constants.txt
'''

import json
from pathlib import Path
from typing import List, Dict


def load_definitions(filepath: Path) -> List[Dict]:
    '''Load relationship definitions from JSON file.

    Args:
        filepath: Path to relationship_definitions.json

    Returns:
        List of relationship definitions
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['Relationships']


def generate_constant_declarations(definitions: List[Dict]) -> str:
    '''Generate constant declarations for stub file.

    Args:
        definitions: List of relationship definitions

    Returns:
        String containing constant declarations
    '''
    lines = []

    # Only generate constants for primary relationships (reversed_arrow=False or null)
    primary_rels = [
        d for d in definitions
        if not d.get('reversed_arrow', False)
    ]

    # Sort alphabetically for consistent output
    primary_rels.sort(key=lambda d: d['relation_type'])

    for defn in primary_rels:
        relation_type = defn['relation_type']
        label = defn['label']
        lines.append(f'{relation_type}: RelationshipType  # {label}')

    return '\n'.join(lines)


def main():
    '''Main entry point.'''
    # Find relationship_definitions.json
    script_dir = Path(__file__).parent
    json_path = script_dir / 'relationship_definitions.json'

    if not json_path.exists():
        print(f'Error: {json_path} not found')
        return 1

    # Load definitions
    definitions = load_definitions(json_path)

    # Generate constant declarations
    constants = generate_constant_declarations(definitions)

    # Print to stdout
    print('# === Generated Constants (DO NOT EDIT MANUALLY) ===')
    print('# Auto-generated from relationship_definitions.json')
    print('# Run: python src/generate_stub_constants.py')
    print()
    print(constants)
    print()
    print('# === End Generated Constants ===')

    return 0


if __name__ == '__main__':
    exit(main())
