from os import stat
from pathlib import Path
from datetime import datetime, timedelta
import xlwings as xw
import PySimpleGUI as sg
import varian_query as vq

base_dir = Path(r'L:\temp\Plan Checking Temp')
file_name = 'StructuresCheck.xlsx'

DBSERVER = 'vardbpv1'
FILESERVER = 'varimgpv1'
SYSTEMDB = 'variansystem'
ENMDB = 'varianenm'
AURA = 'vaurapv1'
DWDB = 'variandw'
REPORTSDB = 'ReportServer'

SQL_PATH = Path.cwd() / 'SQL'

def get_patient_id():
    form_rows = [[sg.Text('Patient CR#:'), sg.InputText(key='CR#')],
                 [sg.Submit(), sg.Cancel()]]
    window = sg.Window('Select Patient', form_rows)
    button, values = window.read()
    window.close()
    patient_id = values['CR#']
    if any((button != 'Submit', patient_id == '')):
        patient_id = None
    return patient_id


def patient_info(connection):
    patient_id = get_patient_id()
    if not patient_id:
        print('Query Cancelled')
    else:
        patient_data = vq.query_dict(connection, SQL_PATH / 'patient_query.sql', patient_id=patient_id)
    return patient_data


def is_contoured(file_name):
    '''Determine if a structure has been contoured based on file size.
    '''
    file_size = stat(file_name).st_size
    if file_size > 90:
        return True
    else:
        return False


def structure_size(file_name):
    '''Determine if a structure has been contoured based on file size.
    '''
    return stat(file_name).st_size


def structure_modified(file_name):
    '''Determine if a structure has been contoured based on file size.
    '''
    mod_seconds = stat(file_name).st_mtime
    mod_time = datetime.min + timedelta(seconds=mod_seconds)
    return str(mod_time)


def make_path(file_name: str) -> str:
    '''Determine if a structure has been contoured based on file size.
    '''
    data_path = r'\\{}\va_data$\Filedata'.format(FILESERVER)
    full_file_name = file_name[2:].replace('IMAGEDIR1', data_path)
    full_file_name = full_file_name.replace('imagedir1', data_path)
    return full_file_name


def color_format(sheet, starting_cell='A1', column_name='RGB', size: int = None):
    start_range = sheet.range(starting_cell)
    end_range = start_range.end('right')
    header_range = xw.Range(start_range, end_range)
    headers = header_range.value
    column_index = headers.index('RGB')
    top_cell = start_range.offset(1,column_index)
    if size:
        num_rows = size
    else:
        end_cell = top_cell.end('down')
        num_rows = xw.Range(top_cell, end_cell).size
    for indx in range(num_rows):
        cell = top_cell.offset(indx,0)
        if cell.value:
            rgb = cell.value.strip()
            try:
                color_rgb = tuple(int(num) for num in rgb[1:-1].split(', '))
            except (ValueError, AttributeError):
                color_rgb = None
            else:
                cell.color = color_rgb


def get_structure_info(connection, patient_selection, sql_path):
    structures = vq.run_query(connection, sql_path / 'StructureCheck.sql', patient_selection)
    structures['File_Name'] = structures['File_Name'].map(make_path)
    structures['File_Size'] = structures['File_Name'].map(structure_size)
    structures['File_Modified'] = structures['File_Name'].map(structure_modified)
    structures['Contoured'] = structures['File_Name'].map(is_contoured)
    structures['RGB'] = structures['Color_Hex'].map(vq.hex2rgb)
    structures.set_index('Structure_Id', inplace=True)
    return structures


def save_info(starting_path: Path):
    if starting_path.is_dir():
        starting_dir = starting_path
        starting_file = ''
    else:
        starting_dir = starting_path.parent
        starting_file = str(starting_path)
    form_rows = [[sg.Text('Save Structure Check As:')],
                 [sg.InputText(key='save_file', default_text=starting_file),
                  sg.FileSaveAs(initial_folder=str(starting_dir),
                                file_types=(('Excel Files', '*.xlsx'),))],
                 [sg.Submit(), sg.Cancel()]]
    window = sg.Window('File Save', form_rows)
    button, values = window.read()
    window.close()
    save_file_name = Path(values['save_file'])
    if any((button != 'Submit', save_file_name == '')):
        save_file_path = None
    else:
        save_file_path = Path(save_file_name)
    return save_file_path


def save_structure_data(base_dir, file_name):
    save_file = base_dir / file_name
    save_file_path = save_info(save_file)
    if save_file_path:
        sheet_name = 'Structure Check'
        exel_app = xw.apps.active
        if not exel_app:
            exel_app = xw.App(visible=None, add_book=False)
        structure_query = exel_app.books.add()
        sheet = structure_query.sheets.add(sheet_name)
        sheet.range('A1').options(index=False).value = structures
        structure_query.save(str(save_file_path))
    else:
        sheet = None
    return sheet


def main():
    connection = vq.connect(DBSERVER)
    patient_data = patient_info(connection)
    patient_selection = patient_data[0] #TODO make patient Selector GUI
    structures = get_structure_info(connection, patient_selection)

    sheet = save_structure_data(base_dir, file_name)
    if sheet:
        color_format(sheet, starting_cell='A1', column_name='RGB', size=len(structures))
        sheet.book.save()

if __name__ == '__main__':
    main()
