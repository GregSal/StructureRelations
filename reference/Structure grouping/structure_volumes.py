# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:37:49 2021

@author: Greg
"""
#%% Imports
import xlwings as xw
import PySimpleGUI as sg
import pandas as pd

# FIXME OK can remain disabled when switching between Functions that require
# variables and those that do not.
# TODO Use selector for all functions including 1 variable ones
# TODO allow for inserting rows
# GUI tool for re-ordering structures

#%%  Functions
def get_vars(function):
    var_list = list()
    remainder = str(function)
    while remainder:
        split1 = remainder.partition('{')
        split2 = split1[2].partition('}')
        remainder = split2[2]
        var = split2[0]
        if var:
            var_list.append(var)
    return var_list


def build_reference(var_composite, selected_row, function_type,
                    structure_selection, indexes):
    def parse_var(var_composite, indexes):
        # Deal with @ references
        type_index = indexes['var_type'][0]
        var_parts = var_composite.split('@')
        if len(var_parts) == 2:
            var_name = var_parts[0]
            col_offset = type_index[var_parts[1]]
        else:
            var_name = var_composite
            col_offset = 0
        return col_offset, var_name

    def get_address(var, col_offset, selected_row, indexes):
        # Build Reference
        structure_index, structure_top = indexes['structure']
        #image_index, image_top = indexes['image']
        type_index = indexes['var_type'][0]
        if var in structure_index:
            row_offset = structure_index[var]
            reference = structure_top
        #elif var in image_index:
        #    row_offset = image_index[var]
        #    reference = image_top
        elif var in type_index:
            col_offset = type_index[var]
            row_offset = 0
            reference = selected_row
        else:
            raise ValueError(f'{var} is not a valid reference.')
        reference_range = reference.offset(row_offset, col_offset)
        var_address = reference_range.get_address(row_absolute=False,
                                                  column_absolute=False)
        return var_address

    def get_full_reference(var_name, col_offset, structure_selection, function_type, indexes):
        var_selection, join_method = structure_selection[var_name]
        join_str = join_method[function_type]
        if join_str:
            var_addresses = [get_address(single_var, col_offset, selected_row, indexes)
                             for single_var in var_selection]
            full_reference = join_str.join(var_addresses)
        else:
            full_reference = get_address(var_selection[0], col_offset, selected_row, indexes)
        return full_reference

    col_offset, var_name = parse_var(var_composite, indexes)
    # Deal with User Selected values
    if var_name in structure_selection:
        reference = get_full_reference(var_name, col_offset, structure_selection, function_type, indexes)
    else:
        reference = get_address(var_name, col_offset, selected_row, indexes)
    return reference


def build_function(function, function_type, selected_row,
                   structure_selection, indexes):
    var_list = get_vars(function)
    var_dict = dict()
    for var in var_list:
        var_address = build_reference(var.upper(), selected_row, function_type,
                                      structure_selection, indexes)
        var_dict[var] = var_address
    final_function = '=' + function.format(**var_dict)
    return final_function


def get_var_list(selected_function, function_data):
    variable_set = function_data['variables'].xs(selected_function,
                                                 level=0, axis=0)
    variable_set.reset_index(inplace=True)
    variable_set = variable_set.dropna(subset=['Variable'])
    variable_set.set_index(['Type', 'Variable'], inplace=True)
    variable_set = variable_set.unstack('Variable')
    variable_set = variable_set.droplevel(0, axis='columns')
    return variable_set


def insert_function(selected_function, structure_selection, selected_row,
                    function_data, indexes):
    type_index = indexes['var_type'][0]
    format_range = function_data['format']
    function_group_df = function_data['functions'].xs(selected_function,
                                                      level=0, axis=0)
    functions = function_group_df.to_dict(orient='dict')['Function']

    for function_type, function in functions.items():
        #variables_df = function_data['variables'].loc[(selected_function,function_type),:].set_index('Variable')

        function_type = function_type.upper()
        function_str = build_function(str(function), function_type,
                                      selected_row, structure_selection,
                                      indexes)
        target_range = selected_row.offset(0, type_index[function_type])
        format_range.copy(target_range)
        target_range.formula = function_str


#%%  Load Data Functions
# Structures
def get_structures_data(selected_sheet, cell='A2'):
    structure_range = selected_sheet.range(cell).expand()
    structure_df = structure_range.options(pd.DataFrame).value
    structure_index = {
        structure.upper(): offset
        for offset, structure in enumerate(list(structure_df.index))
        }
    structure_top = structure_range.resize(1,1).offset(1,0)
    structures_data = (structure_index, structure_top)

    type_index = {
        value.upper(): offset + 1
        for offset, value in enumerate(list(structure_df.columns))
        }
    type_index['STRUCTURES'] = 0
    var_type_data = (type_index, structure_top)

    return structures_data, var_type_data

# Image
def get_image_data(selected_sheet, cell='N2'):
    image_df = selected_sheet.range(cell).options(pd.Series,
                                                  expand='table',
                                                  header=False).value
    image_index = {
        value.upper(): offset
        for offset, value in enumerate(list(image_df.index))
        }
    image_top = selected_sheet.range(cell).offset(0,1)
    return image_index, image_top

def make_indexes(selected_sheet):
    structures_data, var_type_data = get_structures_data(selected_sheet)
    #image_data = get_image_data(selected_sheet)
    indexes = {'structure': structures_data,
               'var_type': var_type_data}
    return indexes

# Functions
def get_function_data(selected_book, sheet_name='Functions',
                      function_cell='A2', variable_cell='E2',
                      format_cell='M2'):
    selected_sheet = selected_book.sheets[sheet_name]

    function_range = selected_sheet.range(function_cell).expand()
    function_df = function_range.options(pd.DataFrame, index=2).value

    variable_range = selected_sheet.range(variable_cell).expand()
    variable_df = variable_range.options(pd.DataFrame, index=2).value

    format_range = selected_sheet.range(format_cell)
    function_data = {'functions': function_df,
                     'variables': variable_df,
                     'format': format_range}
    return function_data


#%% GUI Methods
def build_window(function_names, structure_list):
    function_selector = sg.Combo(function_names, enable_events=True,
                                    key='Function Selector')
    function_frame = sg.Frame('Function', key='Function',
                              layout = [[function_selector]])
    height = len(structure_list)
    width = max(len(var) for var in structure_list)
    selection_header = [sg.Text('Test Header', key='Variable Header',
                                justification='center', size=(width*2,1))]
    var_col1 = sg.Column([
            [sg.Text('Test Text1',key='var_label1',
                     justification='center', size=(width, 1))],
            [sg.Listbox(structure_list, key='var1', enable_events=True,
                        no_scrollbar=True, size=(width, height))]
             ], visible=True, key='var_col1')
    var_col2 = sg.Column([
            [sg.Text('Test Text1',key='var_label2',
                     justification='center', size=(width, 1))],
            [sg.Listbox(structure_list, key='var2', enable_events=True,
                        no_scrollbar=True, size=(width, height))]
             ], visible=True, key='var_col2')
    selector_list = [{'label': 'var_label1', 'selector':'var1', 'column':'var_col1'},
                     {'label': 'var_label2', 'selector':'var2', 'column':'var_col2'}]
    var_ref = {}
    structure_frame = sg.Frame('Variables', key='Variables',
                               layout=[selection_header, [var_col1, var_col2]],
                               metadata=(selector_list, var_ref)
                               )
    layout = [[sg.Text('Select a function')],
              [function_frame],
              [sg.HorizontalSeparator()],
              [structure_frame],
              [sg.OK(), sg.Cancel()] ]

    window = sg.Window('Structure Volume Functions', layout, finalize=True, resizable=True)
    window['Variables'].expand(expand_x = True, expand_y = True, expand_row = True)
    window['Variable Header'].expand(expand_x = True, expand_y = True, expand_row = True)
    return window


def build_var_frame(window, selected_function, function_data):
    variable_set = get_var_list(selected_function, function_data)
    var_list = list(variable_set.columns)
    selector_list, var_ref = window['Variables'].metadata
    var_ref.clear()

    header_text = f'Selected Variables for: {selected_function}'
    window['Variable Header'].update(header_text)
    window['Variable Header'].set_size((len(header_text), 1))
    window['Variable Header'].expand(expand_x = True)

    for var, selector_ref in zip(var_list, selector_list):
        window[selector_ref['label']].update(var)
        window[selector_ref['selector']].SetValue(var)

        join_method = variable_set[var].to_dict()
        for key, value in join_method.copy().items():
            join_method[key.upper()] = value
        window[selector_ref['selector']].metadata=join_method

        if any(join_method):
            select_mode = sg.LISTBOX_SELECT_MODE_EXTENDED
        else:
            select_mode = sg.LISTBOX_SELECT_MODE_BROWSE
        window[selector_ref['selector']].update(select_mode=select_mode)
        var_ref[selector_ref['selector']] = var
        window[selector_ref['column']].update(visible=True)

    #window['Variables'].expand(expand_x = True)

    window['Variables'].metadata = (selector_list, var_ref)
    window.refresh()
    return var_list


def found_vars(var_list, structure_selection):
    for var in var_list:
        yield var in structure_selection


def add_selection(window, event, values, var_ref, structure_selection):
    selection = values.get(event)
    join_method = window[event].metadata
    var_match = var_ref[event]
    structure_selection[var_match] = (selection, join_method)
    structure_selection[var_match.upper()] = (selection, join_method)
    return structure_selection


def add_functions(function_data, indexes, selected_book):
    function_names = list({ind[0] for ind in function_data['functions'].index})
    function_names.sort()
    structure_list = list(indexes['structure'][0].keys())
    var_list = list()
    structure_selection = dict()
    selected_function = None
    window = build_window(function_names, structure_list)
    _, var_ref = window['Variables'].metadata
    done = False
    while not done:
        event, values = window.Read(timeout=200)
        if event == sg.TIMEOUT_KEY:
            continue
        elif event is None:
            done = True
        elif event in 'Cancel':
            selected_function = None
            done = True
        elif event in 'OK':
            selected_row = selected_book.selection.resize(1,1)
            # TODO Keep row and set column to "A"
            if selected_function:
                insert_function(selected_function, structure_selection,
                                selected_row, function_data, indexes)
                selected_row.offset(1,0).select()
        elif event in 'Function Selector':
            selected_function = values['Function Selector']
            var_list = build_var_frame(window, selected_function, function_data)
            if var_list:
                window['OK'].update(disabled=True)
        elif event in var_ref:
            structure_selection = add_selection(window, event, values,
                                                var_ref, structure_selection)
            if all(found_vars(var_list, structure_selection)):
                window['OK'].update(disabled=False)
    window.Close()




#%% Worksheet lists
def get_sheets():
    book_list = xw.books
    books_and_sheets = {
        bk.name: [sht.name for sht in bk.sheets]
        for bk in book_list
        }
    return books_and_sheets



#%% Test Structure Selection
def test_selection():
    structure_selection = {
        'PTV': 'PTV 1 27Gy',
        'CTV': 'GTV 1 27Gy',
        'Left': 'Lung L',
        'Right': 'Lung R',
        'Both': 'Lung B',
        'PRV': 'PRV5 SpinalCanal',
        'Original': 'Spinal Canal',
        'IGTV': 'IGTV',
        'Phase GTVs': ['GTV 4D0', 'GTV 4D50'],
        'Parts': ['Lung L', 'Lung R'],
        'Total': 'Lung B'
        }
    selected_function = 'Boolean SUM'
    return structure_selection, selected_function


#%%  Main

def main():
    selected_book = xw.books.active
    selected_sheet = selected_book.sheets.active
    #selected_sheet = selected_book.sheets['Contouring']
    indexes = make_indexes(selected_sheet)
    function_data = get_function_data(selected_book)
    add_functions(function_data, indexes, selected_book)


if __name__ == '__main__':
    main()
#%% Test
#structure_selection, selected_function = test_selection()
