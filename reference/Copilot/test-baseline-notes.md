- Full pytest tests/ currently has known non-stage failures: src/test_diagram_modes.py missing DICOM file path, flaky Selenium workflow tests, and Shapely GEOS TopologyException in tests/test_3D_relations.py sphere interpolation cases.
- For DE27IM/relationship-focused validation, targeted suites in tests/test_DE27IM.py, tests/test_2D_relations.py, and tests/test_3D_relations.py are useful but still include the known sphere topology edge case.
- Selenium headless flake in tests/test_webapp_selenium.py: native `.click()` on matrix controls can raise ElementNotInteractableException when tab activation state is stale; using JS click plus waiting for `tab-<name>` active class is more reliable.
- Matrix relation text now appears as `is Equal to` in addition to `=` / `Equals`; symbol-toggle assertions should allow all observed variants.

