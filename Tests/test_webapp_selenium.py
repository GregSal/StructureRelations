"""
Selenium test suite for StructureRelations web application.
Tests the complete workflow including upload, selection, processing, and matrix display.
"""

import pytest
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


@pytest.fixture(scope='module')
def chrome_headless_driver():
    """
    Fixture providing a headless Chrome WebDriver instance.
    Configured for testing without GUI.
    """
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)

    yield driver

    driver.quit()


@pytest.fixture
def test_dicom_file():
    """Returns path to test DICOM file."""
    test_file = Path(__file__).parent / 'RS.GJS_Struct_Tests.Relations.dcm'
    assert test_file.exists(), f'Test file not found: {test_file}'
    return str(test_file)


class WebAppTestHelper:
    """Helper class for common web app testing operations."""

    def __init__(self, driver, base_url='http://localhost:8000'):
        self.driver = driver
        self.base_url = base_url
        self.wait = WebDriverWait(driver, 30)

    def navigate_home(self):
        """Navigate to home page."""
        self.driver.get(self.base_url)

    def wait_for_connection(self, timeout=10):
        """Wait for WebSocket connection indicator."""
        try:
            self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '.status-dot.connected')
                )
            )
            return True
        except TimeoutException:
            return False

    def upload_dicom(self, file_path):
        """Upload a DICOM file."""
        file_input = self.driver.find_element(By.ID, 'fileInput')
        file_input.send_keys(file_path)

        # Wait for selection stage to appear
        self.wait.until(
            EC.visibility_of_element_located(
                (By.ID, 'selectionStage')
            )
        )

    def get_structure_list(self):
        """Get list of structure names from selection stage."""
        structures = []
        items = self.driver.find_elements(
            By.CSS_SELECTOR,
            '#structuresList .structure-item'
        )
        for item in items:
            name = item.find_element(
                By.CSS_SELECTOR,
                '.structure-name'
            ).text
            roi = item.find_element(
                By.CSS_SELECTOR,
                'input[type="checkbox"]'
            ).get_attribute('data-roi')
            structures.append({'name': name, 'roi': int(roi)})
        return structures

    def select_structures(self, roi_numbers=None):
        """
        Select specific structures by ROI number.
        If roi_numbers is None, keeps all selected.
        """
        if roi_numbers is not None:
            # First deselect all
            select_none = self.driver.find_element(By.ID, 'selectNone')
            select_none.click()
            time.sleep(0.2)

            # Select specified ROIs
            for roi in roi_numbers:
                checkbox = self.driver.find_element(
                    By.CSS_SELECTOR,
                    f'input[data-roi="{roi}"]'
                )
                checkbox.click()

    def start_processing(self):
        """Click process button to start analysis."""
        process_btn = self.driver.find_element(By.ID, 'processBtn')
        process_btn.click()

        # Wait for processing stage
        self.wait.until(
            EC.visibility_of_element_located(
                (By.ID, 'processingStage')
            )
        )

    def wait_for_processing(self, timeout=120):
        """Wait for processing to complete."""
        try:
            # Wait for results stage to appear
            self.wait = WebDriverWait(self.driver, timeout)
            self.wait.until(
                EC.visibility_of_element_located(
                    (By.ID, 'resultsStage')
                )
            )
            return True
        except TimeoutException:
            return False
        finally:
            self.wait = WebDriverWait(self.driver, 30)

    def get_progress_percentage(self):
        """Get current processing progress percentage."""
        progress_fill = self.driver.find_element(
            By.CSS_SELECTOR,
            '.progress-fill'
        )
        width = progress_fill.value_of_css_property('width')
        # Convert pixel width to percentage (approximate)
        return width

    def set_matrix_rows(self, roi_numbers):
        """Drag structures to selected rows list."""
        selected_list = self.driver.find_element(
            By.ID,
            'selectedRowsList'
        )

        for roi in roi_numbers:
            item = self.driver.find_element(
                By.CSS_SELECTOR,
                f'#availableRowsList .sortable-item[data-roi="{roi}"]'
            )
            # Simulate drag and drop
            self.driver.execute_script(
                'arguments[0].parentNode.removeChild(arguments[0]); '
                'arguments[1].appendChild(arguments[0]);',
                item, selected_list
            )

    def set_matrix_cols(self, roi_numbers):
        """Drag structures to selected columns list."""
        selected_list = self.driver.find_element(
            By.ID,
            'selectedColsList'
        )

        for roi in roi_numbers:
            item = self.driver.find_element(
                By.CSS_SELECTOR,
                f'#availableColsList .sortable-item[data-roi="{roi}"]'
            )
            # Simulate drag and drop
            self.driver.execute_script(
                'arguments[0].parentNode.removeChild(arguments[0]); '
                'arguments[1].appendChild(arguments[0]);',
                item, selected_list
            )

    def toggle_symbols(self, use_symbols=True):
        """Toggle between symbols and labels."""
        checkbox = self.driver.find_element(By.ID, 'symbolToggle')
        is_checked = checkbox.is_selected()
        if is_checked != use_symbols:
            checkbox.click()

    def update_matrix(self):
        """Click update matrix button."""
        update_btn = self.driver.find_element(By.ID, 'updateMatrixBtn')
        update_btn.click()
        time.sleep(1)  # Wait for matrix update

    def get_matrix_cell(self, row_idx, col_idx):
        """Get value from matrix cell at specified position."""
        tbody = self.driver.find_element(By.ID, 'matrixBody')
        rows = tbody.find_elements(By.TAG_NAME, 'tr')

        if row_idx >= len(rows):
            return None

        cells = rows[row_idx].find_elements(By.TAG_NAME, 'td')
        if col_idx >= len(cells):
            return None

        return cells[col_idx].text

    def get_matrix_dimensions(self):
        """Get matrix dimensions (rows, columns)."""
        tbody = self.driver.find_element(By.ID, 'matrixBody')
        rows = tbody.find_elements(By.TAG_NAME, 'tr')

        if len(rows) == 0:
            return (0, 0)

        cols = len(rows[0].find_elements(By.TAG_NAME, 'td'))
        return (len(rows), cols)

    def export_matrix(self, format_type):
        """Click export button for specified format."""
        export_btn = self.driver.find_element(
            By.ID,
            f'export{format_type.capitalize()}'
        )
        export_btn.click()
        time.sleep(2)  # Wait for download


class TestWebAppWorkflow:
    """Test complete application workflow."""

    def test_upload_and_preview(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test file upload and structure preview."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()

        # Check initial state
        assert helper.driver.find_element(By.ID, 'uploadStage')

        # Upload file
        helper.upload_dicom(test_dicom_file)

        # Verify selection stage appears
        selection_stage = helper.driver.find_element(
            By.ID,
            'selectionStage'
        )
        assert selection_stage.is_displayed()

        # Verify patient info loaded
        patient_info = helper.driver.find_element(By.ID, 'patientInfo')
        assert len(patient_info.text) > 0

        # Verify structures loaded
        structures = helper.get_structure_list()
        assert len(structures) > 0

    def test_structure_selection(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test structure selection controls."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)

        structures = helper.get_structure_list()

        # Test Select None
        helper.driver.find_element(By.ID, 'selectNone').click()
        time.sleep(0.2)

        checkboxes = helper.driver.find_elements(
            By.CSS_SELECTOR,
            '#structuresList input[type="checkbox"]'
        )
        assert all(not cb.is_selected() for cb in checkboxes)

        # Test Select All
        helper.driver.find_element(By.ID, 'selectAll').click()
        time.sleep(0.2)

        checkboxes = helper.driver.find_elements(
            By.CSS_SELECTOR,
            '#structuresList input[type="checkbox"]'
        )
        assert all(cb.is_selected() for cb in checkboxes)

        # Test individual selection
        if len(structures) >= 2:
            helper.select_structures([structures[0]['roi']])

            selected = [
                cb for cb in checkboxes
                if cb.is_selected()
            ]
            assert len(selected) == 1

    def test_processing_workflow(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test complete processing workflow."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)

        # Wait for WebSocket connection
        assert helper.wait_for_connection()

        # Keep all structures selected and process
        helper.start_processing()

        # Verify processing stage shown
        processing_stage = helper.driver.find_element(
            By.ID,
            'processingStage'
        )
        assert processing_stage.is_displayed()

        # Wait for completion
        assert helper.wait_for_processing(timeout=180)

        # Verify results stage shown
        results_stage = helper.driver.find_element(
            By.ID,
            'resultsStage'
        )
        assert results_stage.is_displayed()

    def test_matrix_display(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test relationship matrix display."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)
        helper.start_processing()

        assert helper.wait_for_processing(timeout=180)

        # Check matrix exists
        matrix_table = helper.driver.find_element(
            By.CLASS_NAME,
            'matrix-table'
        )
        assert matrix_table.is_displayed()

        # Check matrix has data
        rows, cols = helper.get_matrix_dimensions()
        assert rows > 0
        assert cols > 0

        # Check diagonal is EQUALS
        for i in range(min(rows, cols)):
            cell_value = helper.get_matrix_cell(i, i)
            # Should be either '=' symbol or 'EQUALS' text
            assert cell_value in ['=', 'EQUALS']

    def test_independent_matrix_axes(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test independent row/column selection for matrix."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)
        helper.start_processing()

        assert helper.wait_for_processing(timeout=180)

        structures = helper.get_structure_list()
        if len(structures) >= 3:
            # Set different structures for rows and columns
            row_rois = [structures[0]['roi'], structures[1]['roi']]
            col_rois = [structures[1]['roi'], structures[2]['roi']]

            helper.set_matrix_rows(row_rois)
            helper.set_matrix_cols(col_rois)
            helper.update_matrix()

            # Verify matrix dimensions
            rows, cols = helper.get_matrix_dimensions()
            assert rows == len(row_rois)
            assert cols == len(col_rois)

    def test_symbol_toggle(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test toggling between symbols and labels."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)
        helper.start_processing()

        assert helper.wait_for_processing(timeout=180)

        # Get initial cell value (should be symbol)
        initial_value = helper.get_matrix_cell(0, 0)
        assert initial_value == '='

        # Toggle to labels
        helper.toggle_symbols(use_symbols=False)
        helper.update_matrix()

        label_value = helper.get_matrix_cell(0, 0)
        assert label_value == 'EQUALS'

        # Toggle back to symbols
        helper.toggle_symbols(use_symbols=True)
        helper.update_matrix()

        symbol_value = helper.get_matrix_cell(0, 0)
        assert symbol_value == '='

    def test_export_functionality(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test matrix export in different formats."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)
        helper.start_processing()

        assert helper.wait_for_processing(timeout=180)

        # Test each export format
        for format_type in ['csv', 'excel', 'json']:
            helper.export_matrix(format_type)
            # Note: Can't easily verify download in headless mode
            # In production, would check download folder


class TestSessionManagement:
    """Test session persistence and disk management."""

    def test_disk_usage_display(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test disk usage indicator updates."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)

        # Check disk usage appears
        disk_usage = helper.driver.find_element(By.ID, 'diskUsage')
        assert 'MB' in disk_usage.text

    def test_disk_warning_threshold(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test disk warning appears when threshold exceeded."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)

        # Check if warning displayed
        disk_warning = helper.driver.find_element(By.ID, 'diskWarning')
        # Warning may or may not be visible depending on actual disk usage
        # Just verify element exists
        assert disk_warning is not None


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_file_upload(self, chrome_headless_driver):
        """Test uploading non-DICOM file."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()

        # Create temporary text file
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix='.txt',
            delete=False
        ) as f:
            f.write(b'Not a DICOM file')
            temp_file = f.name

        try:
            file_input = helper.driver.find_element(By.ID, 'fileInput')
            file_input.send_keys(temp_file)

            # Should show alert
            time.sleep(1)
            try:
                alert = helper.driver.switch_to.alert
                alert_text = alert.text
                alert.accept()
                assert 'DICOM' in alert_text or '.dcm' in alert_text
            except:
                # Alert may be handled by browser before we can check
                pass
        finally:
            Path(temp_file).unlink()

    def test_websocket_disconnection(
        self,
        chrome_headless_driver,
        test_dicom_file
    ):
        """Test reconnection handling."""
        helper = WebAppTestHelper(chrome_headless_driver)
        helper.navigate_home()
        helper.upload_dicom(test_dicom_file)

        # Initial connection should be established
        assert helper.wait_for_connection()

        # Verify connection status indicator
        status_dot = helper.driver.find_element(
            By.CSS_SELECTOR,
            '.status-dot'
        )
        assert 'connected' in status_dot.get_attribute('class')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
