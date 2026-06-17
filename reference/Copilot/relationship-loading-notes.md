- In src/relations.py::_initialize_relationships, skip reversed_arrow definitions when building primary test_binaries; then generate reversed tests by transposing complementary primary tests to avoid duplicate matches.
- Symmetric relations in src/relationship_definitions.json should be transpose-invariant at mask/value level; test_DE27IM includes checks that transpose(mask)==mask and transpose(value)==value.

- In src/webapp/static/js/app.js, normalize slice_relationships keys to fixed 4-decimal strings (e.g., 3.25 -> 3.2500) before lookup; otherwise dropdown slice labels miss per-slice relationship summaries because getSliceKey uses toFixed(4).
