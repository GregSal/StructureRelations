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
