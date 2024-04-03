'''Structure Sorting.
'''
# %% Imports
from typing import Dict, List, Tuple, Generic

from pathlib import Path
import re
from math import log10, ceil
import pickle
from functools import partial

import numpy as np
import pandas as pd
import xlwings as xw


# %% Type classes
SortIndexes = Dict[str, List[str]]
GroupLabel = Generic()

# %% Sorting and Grouping Functions
def opt_ptrn(group_options: SortIndexes)->Dict[str, str]:
    '''Combine group options with OR ('|').

    Args:
        group_options (Dict[str, List[str]]): _description_

    Returns:
        Dict[str, str]: _description_
    '''
    group_patterns = {}
    for name, option_list in group_options.items():
        pattern = '|'.join(option_list)
        group_patterns[name] = pattern
    return group_patterns


def build_pattern(pattern_list: List[str], group_options: SortIndexes,
                  regex_flags: int)->re.Pattern:
    '''Build a regular expression pattern.

    To assemble the pattern, each section is added to a pattern list and the
    final pattern is built by combining the completed pattern and then
    *re*-compiling the resulting string.

    Args:
        pattern_list (List[str]): _description_
        group_options (Dict[str, str]): _description_
        regex_flags (int): _description_

    Returns:
        re.Pattern: _description_
    '''
    # Convert lists of group options into 'OR' patterns.
    grp_ptrn = opt_ptrn(group_options)
    # Combine all of the string groups into a complete pattern.
    pattern = ''.join(pattern_list)
    # Replace each pattern place-holder with its appropriate 'OR' pattern.
    pattern = pattern.format(**grp_ptrn)
    re_pattern = re.compile(pattern, flags=regex_flags)
    return re_pattern


def equivalent_items(data_column: pd.Series, eq_labels: list[GroupLabel],
                     new_label: GroupLabel, lower_case=False)->pd.Series:
    '''Combine multiple equivalent labels as a single label.

    Finds all instances of the items from eq_labels in data_column and replaces
    them with the new_label value.

    Args:
        data_column (pd.Series): Table column to be updated.
        eq_labels (List[GroupLabel]): List of item labels in the table column
            to be replaced.
        new_label (GroupLabel): Value to replace the identified labels with.
            Must be of the same type as the original labels.
        lower_case (bool, optional): If true, the contents of data_column and
            eq_labels will be converted to lowercase before comparison
            (not case sensitive) Ignored if `lower()` cannot be applied to the
            items.Defaults to False.

    Returns:
        pd.Series: A shallow copy of the  original data_column with items in
        eq_labels replaced with new_label.
    '''
    #convert to lowercase if required
    if lower_case:
        try:
            search_col = data_column.str.lower()
            test_labels = [lbl.lower() for lbl in eq_labels]
        except AttributeError:   # lower not applicable
            search_col = data_column.copy()
            test_labels = eq_labels
    else:
        search_col = data_column.copy()
        test_labels = eq_labels
    # find matching labels
    lbl_mask = search_col.isin(test_labels)
    new_column = data_column.where(~lbl_mask, new_label)
    return new_column


def merge_columns(df: pd.DataFrame, co1: str, co2: str)->pd.Series:
    '''Combine values from two columns.

    Values from the second column will only be used if the value in the
    first column is null.

    Args:
        df (pd.DataFrame): _description_
        co1 (str): _description_
        co2 (str): _description_
        new_col (str): _description_

    Returns:
        pd.Series: _description_
    '''
    new_col = df[co1]
    null_row = new_col.isnull()
    new_col.where(~null_row, df[co2], inplace=True)
    return new_col


def combine_columns(df: pd.DataFrame, columns, sep=' ')->pd.Series:
    '''Combine text from multiple columns with a separator.

    Args:
        df (pd.DataFrame): _description_
        columns (List[str]: _description_
        sep (str): _description_

    Returns:
        pd.Series: _description_
    '''
    row_dict = {}
    for index, row in df.iterrows():
        text_items = []
        for col in columns:
            text = row.at[col]
            if text:
                text_items.append(str(text))
        combined_text = sep.join(text_items)
        row_dict[index] = combined_text
    new_col = pd.Series(row_dict)
    return new_col


def to_num_str(num: float, conversion='{:2.0f}', na_value='')->str:
    '''Convert a number to a string representation.

    Args:
        num (float): Number to convert
        conversion (str, optional): Conversion format string.
            Defaults to '{:2.0f}'.
        na_value (str, optional): Value to return if num is not a number, or is
            NA. Defaults to ''.

    Returns:
        str: string representation of the supplied number.
    '''
    if isinstance(num, float):
        if np.isnan(num):
            return na_value
        return conversion.format(num).strip()
    return na_value


def to_mm(data: pd.DataFrame, size_column: str, unit_column: str)->pd.Series:
    '''Convert numbers with cm units to mm.

    Args:
        data (pd.DataFrame): _description_
        size_column (str): _description_
        unit_column (str): _description_

    Returns:
        pd.Series: _description_
    '''
    # Convert size to mm
    size_val = pd.to_numeric(data[size_column], errors='coerce')
    # Identify Expansion values in cm units
    # Find cm Units
    cm_unit = data[unit_column].str.lower().isin(['cm'])
    cm_unit.fillna(False, inplace=True)
    # Find Size < 1.0
    # Values in mm will always be greater than 1 because 1 mm ia approximately
    # the resolution limit for high resolution structures.
    small_num = size_val < 1.0
    small_num.fillna(False, inplace=True)
    # Either explicit cm units given OR size is less than 1.0.
    cm_unit = cm_unit | small_num
    # Convert cm to mm
    size_val.where(~cm_unit, size_val * 10, inplace=True)
    return size_val


def to_cgy(data: pd.DataFrame, dose_column: str, unit_column: str)->pd.DataFrame:
    '''Convert numbers with Gy units to cGy.

    Args:
        data (pd.DataFrame): _description_
        dose_column (str): _description_
        unit_column (str): _description_

    Returns:
        pd.DataFrame: _description_
    '''
    # Convert dose to cGy
    data[dose_column] = pd.to_numeric(data[dose_column], errors='coerce')
    # Identify Dose values in Gy units
    # Find Gy Units
    cgy_unit = data[unit_column].str.lower().isin(['cgy'],)
    cgy_unit.fillna(False, inplace=True)
    # Find '%' Units
    prcnt_unit = data[unit_column].isin(['%'],)
    prcnt_unit.fillna(False, inplace=True)
    # Find Size < 1.0
    # Values in Gy will never be greater than 100.
    large_num = data[dose_column] > 100
    large_num.fillna(False, inplace=True)
    # Do not do anything with '%' units
    cgy_unit = cgy_unit | large_num | prcnt_unit
    # Convert cm to mm
    data[dose_column].where(cgy_unit, data[dose_column] * 100, inplace=True)
    return data

# %% Sorting classes
class ColumnIndexer():
    '''Stores and applies information for sorting a data column.
    '''
    def __init__(self, *sort_parts: List[str], column_name: str,
                 case_sensitive=False, as_numeric=False,
                 sort_order='ascending', missing_first=False) -> None:
        self.column_name = column_name
        # convert sort instructions to pandas sort_values compatible parameters
        self.case_sensitive = case_sensitive
        if 'asc' in sort_order:
            self.sort_ascending = True
        else:
            self.sort_ascending = False
        if missing_first:
            self.na_pos = 'first'
        else:
            self.na_pos = 'last'

        # Configure the sorting key function
        if sort_parts:
            # Combine multiple lists to generate a sort order.
            sort_list = self.make_sort_list(*sort_parts)
            # check for duplicates in sort list
            unique_items = set(sort_list)
            if len(sort_list) != len(unique_items):
                msg = (f'All items in the sort list must be unique.\nGot '
                    f'{len(sort_list):d} list items and '
                    f'{len(unique_items):d} unique items')
                raise ValueError(msg)
            self.sort_list = sort_list
            if not self.case_sensitive:
                self.sort_dict = {name.lower(): index
                                  for index, name in enumerate(sort_list)}
            else:
                self.sort_dict = {name: index
                                  for index, name in enumerate(sort_list)}
            self.index_key = self.list_key
        elif as_numeric:
            self.index_key = partial(pd.to_numeric, errors='coerce')
        else:
            self.index_key = None

    def make_sort_list(self, *sort_lists: List[str])->List[str]:
        sort_list = []
        for srt_list in sort_lists:
            if not self.case_sensitive:
                part_list = [item.lower() for item in srt_list]
            else:
                part_list = list(srt_list)
            sort_list.extend(part_list)
        return sort_list

    def list_key(self, column: pd.Series)->pd.Series:
        def rank_item(text)->Dict[str, int]:
            if self.na_pos == 'first':
                not_found = -1
            else:
                not_found = len(self.sort_dict) + 1
            if not self.case_sensitive:
                test_text = str(text).lower()
            else:
                test_text = text
            score = self.sort_dict.get(test_text, not_found)
            return score
        column_rank = column.apply(rank_item)
        return column_rank

    def make_sort_lookup(self, column: pd.Series)->List[str]:
        sorted_column = column.sort_values(key=self.index_key,
                                           ascending=self.sort_ascending,
                                           na_position=self.na_pos)
        sorted_column.dropna(inplace=True)
        sorted_column.drop_duplicates(inplace=True)
        used_dict = {name: index
                     for index, name in enumerate(list(sorted_column))}
        index_lookup = pd.Series(used_dict)
        return index_lookup

    def apply(self, data: pd.DataFrame)->pd.Series:
        '''Generate a sort indexer for a particular dataset.

        The index will contain integers with increments of $10^priority_factor$
        indicating the sort order for the items in the column from the data
        labeled self.column_name.

        Args:
            data: (pd.DataFrame): Data table to be sorted. Must contain a
                column labeled self.column_name.
        Raises:
            ValueError: if a column labeled self.column_name does not exist.

        Returns
            pd.Series: A series with values integers reflecting the desired
            sort order for the data column. The sort values start at 1 if
            missing_first=False and at 1 if missing_first=True.
        '''
        if self.column_name not in list(data.columns):
            return pd.Series()
        # Identify the data column to sort
        column_name = self.column_name
        index_name = column_name + '_index'
        column = data[column_name]
        if all(column.isna()):
            return pd.Series()
        # Get the sort order for the actual column data.
        sorted_column = column.sort_values(key=self.index_key,
                                           ascending=self.sort_ascending,
                                           na_position=self.na_pos)
        # Build a new data-specific sort index.
        # This is required so that "identical" values will have the same
        # index number.
        sorted_column.dropna(inplace=True)
        sorted_column.drop_duplicates(inplace=True)
        used_dict = {name: index + 1
                     for index, name in enumerate(list(sorted_column))}
        indexed_lookup = pd.Series(used_dict)
        indexed_lookup.name = index_name
        # minimum enumerate value is 0 Add 1 so the minimum value becomes 1,
        # and 0 can be used for nan if desired. Maximum value becomes
        # len(used_dict).
        # re-apply the sort index to generate a numerical sort order.
        index_df = data.merge(indexed_lookup, how='left',
                              left_on=column_name, right_index=True)
        column_index = index_df[index_name]
        # assign numerical values for un-sorted items.
        if self.na_pos == 'first':
            na_value = 0
        else:
            na_value = len(used_dict) +1
        column_index = column_index.fillna(na_value)
        return column_index


class DataSorter():
    '''Generate and apply custom sorting for a data table.
    '''
    def __init__(self) -> None:
        self.max_rank = 1
        self.indexers = {}

    def add_indexer(self, *sort_parts: List[str], column_name: str,
                    rank: int = None, **kwargs) -> None:
        '''Insert a new index directive into the DataSorter.

        Args:
            column_name (str): _description_
            rank (int, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        '''
        if rank is None:
            rank = self.max_rank
        elif rank in self.indexers:
            raise ValueError(f'Rank of {rank} already in use.')
        new_index = ColumnIndexer(*sort_parts, column_name=column_name,
                                  **kwargs)
        self.indexers[rank] = new_index
        if rank >= self.max_rank:
            self.max_rank += 1

    def apply(self, data: pd.DataFrame):
        index_columns = [indexer.column_name
                         for indexer in self.indexers.values()]
        data = data[index_columns].copy()
        index_order = list(self.indexers.keys())
        index_order.sort()
        rank_factor = 1
        applied_indexes = []
        for rank in index_order:
            indexer = self.indexers[rank]
            column_name = indexer.column_name
            index_name = column_name + '_index'
            column_index = indexer.apply(data)
            if not column_index.empty:
                column_index.name = index_name
                rank_size = ceil(log10(column_index.dropna().max()))
                rank_factor = rank_factor + rank_size
                mul_factor = 10**(-rank_factor)
                column_index = column_index * mul_factor
                applied_indexes.append(column_index)
        if applied_indexes:
            composite_index = pd.concat(applied_indexes, axis='columns')
        else:
            composite_index = pd.DataFrame()
        return composite_index

    def combined_index(self, structure_groups: pd.DataFrame)->pd.Series:
        composite_index = self.apply(structure_groups)
        composite_index.dropna(how='all', inplace=True)
        composite_index['Combined'] = composite_index.sum(axis='columns')
        composite_index.sort_values('Combined', inplace=True)
        return composite_index.Combined


# %% Build the Structure Group patterns
def build_target_pattern()->Tuple[pd.Series, SortIndexes]:
    '''Create a regex pattern to distinguish between different Target structures.

    The pattern contains sections for each component of the Target volume name.
    The order of the *regular expression* sections matter partly because they
    define the order that the components appear in the name and partly because
    earlier sections grab components that might otherwise be confused with a
    component defined in a later section.

    The only required section of the *regular expression* pattern is the
    **Basic target type** e.g. GTV, CTV, PTV, ITV. All other sections are
    optional and may or may not appear in the name.

    Returns:
        Tuple[pd.Series, SortIndexes]: _description_
    '''
    target_pattern_list = []
    target_group_options = {}

    # Beginning of string
    target_pattern_list.append(r'^')
    # MOD (A prefix identifying *eval* or *opt* targets)
    target_group_options['mod_types'] = ['eval', 'opt', 'mod']
    target_pattern_list.append(
        r'(?P<Mod>'     # Start of *Mod* group
        r'{mod_types}'  # contains '|' separated items from mod_types.
        r')?'           # End of optional *Mod* group
        r'[ _]*'        # Optional space or '_'
        )
    # TargetType (The Basic target type. e.g. GTV, CTV, PTV)
    target_group_options['target_type'] = [
        'Nodes', 'Node', 'Iliac Vessels', 'Edema',
        'Cavity', 'Operative Bed', 'GTV', 'IGTV',
        'CTV', 'HTV', 'HRV', 'ITV', 'PTV'
        ]
    target_pattern_list.append(
        r'(?P<TargetType>'      # Start of required *TargetType* group
        r'{target_type}'  # contains '|' separated items from target_type.
        r')'              # End of *TargetType* group.
        )
    # TargetSuffix (Single letter target suffix. One of: n,p,m
    target_group_options['Suffix_options'] = ['n', 'p', 'm']
    target_pattern_list.append(
        r'(?:'                # Start of the optional non-capturing group.
        r'(?P<TargetSuffix>'  # Start of the *TargetSuffix* group.
        r'{Suffix_options}'   # contains '|' separated items from Suffix_options.
        r')'                  # End of the *TargetSuffix* group.
        r'(?![A-Za-z]{{2}})'  # NOT followed by 2 characters.
        r')?'                 # End of the optional non-capturing group.
        r'[ _]*'              # Optional space or '_' if  group is not present.
        )
    # TargetNumber designator
    target_pattern_list.append(
        r'(?:'                # Start of an optional non-capturing group.
        r'(?<![0-9.+-])'      # NOT preceded by another digit, '.', '+', or '-'.
        r'(?P<TargetNumber>'  # Start of the *TargetNumber* group.
        r'[1-9]'              # A single digit.
        r'|'                  # OR
        r'1[0-5]'             # A '1' and an additional digit between 0 and 4.
        r')'                  # End of the *TargetNumber* group.
        r'(?![0-9.Dcm])'      # NOT followed by a digit, '.', 'D', 'c', or 'm'.
        r')?'                 # End of the optional non-capturing group.
        r'[ _]*'              # Optional space or '_'
        )
    # Metabolic Tumor Volume (Targets based on specific imaging types.)
    target_group_options['metabolic_types'] = ['T1', 'T2', 'MRI', 'PET', 'CT']
    target_pattern_list.append(
        r'(?P<Metabolic>'     # Start of the optional *Metabolic* group
        r'{metabolic_types}'  # contains '|' separated items from metabolic_types.
        r')?'                 # End of the optional *Metabolic* group
        r'[ _]*'              # Optional space or '_'
        )
    # Motion (Targets related to 4D scanning.)
    target_group_options['motion_types'] = ['MIP', 'AIP', 'AVE']
    target_pattern_list.append(
        r'(?P<Motion>'     # Start of the optional *Motion* group
        r'4D[0-9][0-9]?'   # '4D' followed by one or two digits.
        r'|'               # OR
        r'{motion_types}'  # contains '|' separated items from motion_types.
        r')?'              # End of the optional *Motion* group
        r'[ _]*'           # Optional space or '_'
        )
    # Region (The name of a nodal group region.)
    target_group_options['regions'] = [
        'Pyloric', 'Subpyloric',
        'Gastric', 'Splenic', 'Celiac', 'Pancreatic',
        'Hepatoduod', 'Hepatogastr', 'Hepatic',
        'Para-Aortic', 'Obturator',
        'Int iliac', 'ext iliac', 'com iliac', 'Sacral',
        'Axilla 1', 'Axilla 2', 'Axilla 3',
        # Order is important here! 'Axilla III' and 'Axilla II' must come before
        # 'Axilla I' or 'Axilla I' will catch them.
        'Axilla III', 'Axilla II', 'Axilla I',
        'IMC', # 'SC'  Dropping SC because of Scalp
        ]
    target_pattern_list.append(
        r'(?P<Region>'  # Start of the optional *Region* group
        r'{regions}'    # contains '|' separated items from motion_types.
        r')?'           # End of the optional *Region* group
        r'[ _]*'        # Optional space or '_'
        )
    # Level (Dose level modifier indicating *nodal*, *primary*, etc.)
    target_group_options['Level_types'] = [
        'low', 'int', 'IR', 'PREOP', 'Cavity', 'RES', 'HR'
        ]
    target_pattern_list.append(
        r'(?:'                # Start of an optional non-capturing group.
        r'(?P<Level>'         # Start of the *Level* group
        r'{Level_types}'      # contains '|' separated items from Level_types.
        r')'                  # End of the *Level* group
        r'(?![D-Zd-z]{{2}})'  # NOT followed by any two characters except for a, b, or c.
        r')?'                 # End of the optional non-capturing group.
        r'[ _]*'              # Optional space or '_'
        )
    # Dose (Optional Dose Value modifier)
    target_pattern_list.append(
        # Pattern optionally ends with units that are not captured.
        # '-' is allowed as a delimiter because subtracting dose doesn't happen.
        # The double { and } brackets are used to 'escape' { and } when
        # applying str.format().
        r'(?:'                 # Start of an optional non-capturing group.
        r'(?:[- _]*)'          # Optional non-capturing blanks (space '_' or '-').
        r'(?P<TargetDose>'           # Start of the *TargetDose* group
        r'[0-9.]{{2,}}'        # Two or more digits (includes decimal)
        r')'                   # End of the *TargetDose* group
        r'(?:[ cGy]{{2,3}})?'  # Optional non-capturing units: cGy or Gy.
        # Optional non-capturing fractions '/' followed by number and optional '#'
        r'(?:/[0-9]+#?)?'
        r')?'                  # End of the optional non-capturing group
        r'[ _]*'               # Optional space or '_'
        )
    # Combined
    target_group_options['combined_types'] = [
        'Total', 'All']
    target_pattern_list.append(
        r'(?:'               # Start of an optional non-capturing group.
        r'(?:[- ]*)'         # non-capturing optional minus ('-') with spaces.
        r'(?P<Combined>'     # Start of the *Combined* group
        r'{combined_types}'  # contains '|' separated items from combined_types.
        r')'                 # End of the *Combined* group
        r')?'                # End of the optional non-capturing group
        r'[ _]*'             # Optional space or '_'
        )
    # TargetOrgan
    target_pattern_list.append(
        r'(?P<TargetOrgan>'      # Start of the optional *TargetOrgan* group
        r'[A-Za-z]{{3,}}'  # Name of an organ. (Must be longer than 3 letters.)
        r')?'              # End of the optional *TargetOrgan* group
        r'[ _]*'           # Optional space or '_'
        )
    # Expansions
    target_pattern_list.append(
        r'(?:'                  # Start of an optional non-capturing group.
        #  First option number followed by units
        r'(?P<ExpansionSize1>'  # Start of the *ExpansionSize1* group
        r'[0-9.]+'              # Number (one or more digits including decimal)
        r')'                    # End of the *ExpansionSize1* group
        r'(?P<ExpansionUnit1>'  # Start of the *ExpansionUnit1* group
        r'[cm]m'                # *mm* or *cm*
        r')'                    # End of the *ExpansionUnit1* group
        #  End of First option
        r'|'                    # OR
        #  Second option '+' or '-' followed by number with optional units
        r'(?P<ExpansionSign>'   # Start of the *ExpansionSign* group
        r'[+-])'                # Plus or minus sign
        r'(?P<ExpansionSize2>'  # Start of the *ExpansionSize2* group
        r'[0-9.]+'              # Number (one or more digits including decimal)
        r')'                    # End of the *ExpansionSize2* group
        r'(?P<ExpansionUnit2>'  # Start of the optional *ExpansionUnit2* group
        r'[cm]m'                # *mm* or *cm*
        r')?'                   # End of the optional *ExpansionUnit2* group
        #  End of Second option
        r')?'                   # End of the optional non-capturing group
        r'[ _]*'                # Optional space or '_'
        )
    # OrganSubtraction
    target_pattern_list.append(
        # Pattern starts with '-' that is not captured.
        r'(?:'                    # Start of an optional non-capturing group.
        r'(?P<SubtractionSign>'   # Start of the *SubtractionSign* group
        r'-'                      # Minus sign ('-')
        r')'                      # End of the *Subtraction* group.
        r' *'                     # Optional white space.
        r'(?P<OrganSubtraction>'  # Start of the *OrganSubtraction* group
        r'[A-Za-z]*'              # OAR label without spaces or numbers.
        r')'                      # End of the *OrganSubtraction* group.
        r')?'                     # End of the optional non-capturing group.
        r'[ _]*'                  # Optional space or '_'
        )
    # Laterality
    target_group_options['laterality_types'] = ['L', 'LT', 'R', 'RT']
    target_pattern_list.append(
        r'(?P<TargetLaterality>'     # Start of the optional *TargetLaterality* group
        r'{laterality_types}'  # contains '|' separated items from Level_types.
        r')?'                  # End of the optional *TargetLaterality* group
        r'[ _]*'               # Optional space or '_'
        )
    # SubGroup
    target_pattern_list.append(
        r'(?P<SubGroup>'  # Start of the optional *SubGroup* group
        r'[abc]'          # One of 'a', 'b', or 'c'
        r')?'             # End of the optional *SubGroup* group
        )
    # Remaining Text
    target_pattern_list.append(
        r'(?P<Remainder>'  # Start of the optional *Remainder* group
        r'.+'              # All non-captured text after the Target Type
        r')?'              # End of the optional *Remainder* group
        )
     # End of string
    target_pattern_list.append(r'$')

    # Build and apply the pattern
    target_pattern = build_pattern(target_pattern_list, target_group_options,
                                regex_flags=re.IGNORECASE)
    return target_pattern, target_group_options


def build_oar_pattern()->Tuple[pd.Series, SortIndexes]:
    '''Create a regex pattern to distinguish between different OAR structures.

    The pattern contains sections for each component of the OAR structure name.
    The order of the *regular expression* sections matter partly because they
    define the order that the components appear in the name and partly because
    earlier sections grab components that might otherwise be confused with a
    component defined in a later section.

    Only the *BaseOAR* section is required.  Some of the other parts can
    be combined, while some are mutually exclusive.

    Returns:
        Tuple[pd.Series, SortIndexes]: _description_
    '''
    oar_pattern_list = []
    oar_group_options = {}

    # Beginning of string
    # The pattern must begin at the beginning of the volume name to avoid
    # matching part of a name.
    oar_pattern_list.append(r'^')
    # Optional *opt* prefix
    oar_pattern_list.append(
        r'(?P<Opt>'  # Start of *Opt* group
        r'opt'       # contains 'opt'.
        r')?'        # End of optional *Opt* group
        r' *'        # Optional space after the *Prefix* text
        )
    # Optional *PRV* prefix
    # Contains literal text 'PRV' and optional numbers (including decimal).
    oar_pattern_list.append(
        r'(?P<PRV>'       # Start of optional *PRV* group
        r'(?P<PRV_Text>'  # Start of *PRV_Text* group
        r'PRV'            # Required literal text 'PRV'.
        r')'              # End of *PRV_Text* group
        r'(?: *)'         # Non-capturing group containing optional spaces
        r'(?P<PRV_Size>'  # Start of *PRV_Size* group
        r'[0-9.]*'        # Optional numbers and decimal.
        r')'              # End of *PRV_Size* group
        r'(?: *)'         # Non-capturing group containing optional spaces
        r'(?P<PRV_Unit>'  # Start of *PRV_Unit* group
        r'[cm]*'          # Optional cm or mm.
        r')'              # End of *PRV_Unit* group
        r')?'             # End of optional *PRV* group
        r' *'             # Optional space after the *PRV* text
        )
    # BaseOAR
    # - The root structure names available.
    # - This section is **NOT Optional**
    # - The order of the names in this list represents the sorting order for
    #   non-target structures.
    oar_group_options['base_oar'] = [
        'Body',
        'Normal Tissue',
        'Skin',
        'Bone',
        'Bones',
        'Skull',
        'Brain',
        'Temporal Lobes',
        'Hippocampi',
        'Hippo',
        'Pituitary',
        'Neural',
        'Neuro',
        'OpticNerve',
        'OpticChiasm',
        'Optic Chiasm',
        'Optics',
        'Globe',
        'Globes',
        'Eyes',
        'Lens',
        'Lenses',
        'Macula',
        'Lacrimal',
        'LacrimalGland',
        'BrainStem',
        'Brain Stem',
        'BR + op',
        'Optics+BrStm',
        'SpinalCord',
        'SpinalCanal',
        'Spinal Canal',
        'Spinal Cord',
        'Cord',
        'ThecalSac',
        'Cauda Equina',
        'Ear',
        'Ear Middle',
        'Ear Inner',
        'Cochlea',
        'Acoustic',
        'Oral Cavity',
        'OralCavity',
        'Oral',
        'Mucosa',
        'Mucosal',
        'Tongue',
        'Lip',
        'Lips',
        'Parotid',
        'Submandibular',
        'Mandible',
        'Musc_Sclmast',
        'Carotid',
        'Cricoid',
        'Pharynx',
        'PharynxConst',
        'PharynxConstrict',
        'Constrictors',
        'Esophagus',
        'Larynx',
        'Trachea',
        'Thymus',
        'Thyroid',
        'BronchialTree',
        'Bronchial Tree',
        'Humorous',
        'Humerus',
        'Radius',
        'Ulna',
        'Hand',
        'Digits',
        'Clavicle',
        'BrachialPlex',
        'Brachial_plexus',
        'BrachialPlexs',
        'BrachialPlexus',
        'Plexus',
        'brachiocephalic',
        'AxillaryNerves',
        'Prox Bronch Zone',
        'Prox Bronc  Zone',
        'Prox Bronch Tree',
        'ProxBronchZone',
        'Axillary vessels',
        'Breast',
        'Chest Wall',
        'Scapula',
        'subclavian Vein',
        'Pectoralis minor',
        'Pec major/minor',
        'Heart',
        'Aorta',
        'Lung',
        'Lungs',
        'PulmonaryA',
        'Pulmonary Artery',
        'PulmonaryArtery',
        'Pulmonary Arter',
        'SVC',
        'Great Vessels',
        'Celiac Artery',
        'Celiac trunk',
        'Celiac',
        'Ribs',
        'Stomach',
        'Spleen',
        'Splenic Hilum',
        'Duodenum',
        'Jejunum',
        'Liver',
        'Portal Vein',
        'Pancreas',
        'PeritonealCavity',
        'MesentericA S',
        'Sup Mesenteric A',
        'Sup Mesnt Artery',
        'Kidney',
        'Kidneys',
        'Renal hilum',
        'Large Bowel',
        'LargeBowel',
        'Colon',
        'Small Bowel',
        'SmallBowel',
        'Bowel',
        'Bowel Space',
        'BowelBag',
        'Bowel Bag',
        'Femur',
        'Tibia',
        'Fibula',
        'Pelvic Bones',
        'Hip',
        'Bone Marrow',
        'iliac crest',
        'Ilium',
        'Pubic Symphysis',
        'Sacrum',
        'Vessels',
        'Sacral plexus',
        'SacralPlexus',
        'Presacral space',
        'FemoralHead',
        'Femoral Head',
        'Femur_Heads',
        'Bladder',
        'Bladder wall',
        'Ureter',
        'Urethra',
        'groin',
        'Genitalia',
        'Genitals',
        'Ovary',
        'Uterus',
        'Cervix',
        'Vagina',
        'Vagina Empty',
        'Vagina Full',
        'Prostate',
        'Seminal Ves',
        'Penile  bulb',
        'Sigmoid',
        'Rectosigmoid',
        'Rectum',
        'MesoRectum',
        'Anorectosigmoid',
        'Ano-Rectum',
        'Anus',
        'Anal Canal'
        ]
    oar_pattern_list.append(
        r'(?P<BaseOAR>'  # Start of the *BaseOAR* group
        r'{base_oar}'    # contains '|' separated items from base_oar.
        r')'             # End of the *BaseOAR* group
        r'[_ ]*'         # Optional space or '_'
        )
    # Laterality. A suffix identifying laterality of a structure.
    oar_group_options['laterality_types'] = ['L', 'LT', 'R', 'RT', 'B']
    oar_pattern_list.append(
        r'(?P<OAR_Laterality>'     # Start of *OAR_Laterality* group
        r'{laterality_types}'  # contains '|' separated items from Level_types.
        r')?'                  # End of the optional *OAR_Laterality* group
        r'[ _]*'               # Optional space or '_'
        )
    # TargetCrop (Subtracting a target volume from an OAR.)
    oar_group_options['target_sub'] = ['GTV', 'IGTV', 'CTV', 'HTV', 'HRV',
                                       'ITV', 'PTV']
    oar_pattern_list.append(
        r'(?:'                 # Start of an optional non-capturing group.
        r'(?:- *)'             # non-capturing '-' followed by optional space.
        r'(?P<TargetCrop>'     # Start of the *TargetCrop* group
        r'{target_sub}'        # contains '|' separated items from target_sub.
        r')'                   # End of the *TargetCrop* group
        r'(?:[_ ]*)'           # non-capturing optional space or '_'
        r'(?P<TargetCropMod>'  # Start of the *TargetCropMod* group
        r'(?P<TargetModText>'  # Start of the *TargetModText* group
        r'[A-Za-z_]+'          # Text (including '_')
        r')?'                  # End of the optional *TargetModText* group
        r'(?: *)'              # non-capturing optional space
        r'(?P<TargetModSign>'  # Start of the *TargetModSign* group
        r'[+-] *'              # '+' or '-' followed by optional space
        r')?'                  # End of the optional *TargetModSign* group
        r'(?P<TargetModNum>'   # Start of the *TargetModNum* group
        r'[0-9.]+'             # Number (may be expansion, dose or occurrence)
        r')?'                  # End of the optional *TargetModNum* group
        r'(?P<TargetModRes>'   # Start of the *TargetModRes* group
        r'.*'                  # Remainder of the ID as additional modifier text.
        r')?'                  # End of the optional *TargetModRes* group
        r')?'                  # End of the optional *TargetCropMod* group
        r')?'                  # End of the optional group
        )
    # Subtracting the couch, bolus, or a support object from an OAR or body.
    oar_group_options['other_crop'] = ['Board', 'Bolus']
    oar_pattern_list.append(
        r'(?:'             # Start of an optional non-capturing group.
        r'(?:- *)'         # non-capturing '-' followed by optional white space.
        r'(?P<OtherCrop>'  # Start of the *OtherCrop* group
        r'{other_crop}'    # contains '|' separated items from other_crop.
        r')'               # End of the *OtherCrop* group
        r')?'              # End of the optional group
        r'[_ ]*'           # Optional space or '_'
        )
    # Volume Expansion (A '+' followed by a number.)
    oar_pattern_list.append(
        r'(?:'                 # Start of an optional non-capturing group.
        r'(?:[+] *)'           # non-capturing '+' followed by optional white space.
        r'(?P<ExpansionSize>'  # Start of the *ExpansionSize* group
        r'[0-9.]+'             # One or more digits, including decimal places.
        r')?'                  # End of the optional *ExpansionSize* group
        r'(?P<ExpansionUnit>'  # Start of the optional *ExpansionUnit* group
        r'[cm]{{2}}'           # 'cm' or 'mm' units
        r')?'                  # End of the optional *ExpansionUnit* group
        r'(?P<ExpansionText>'  # Start of the optional *ExpansionText* group
        r'[A-Za-z_]+'          # Text (including '_')
        r')?'                  # End of the optional *ExpansionText* group
        r')?'                  # End of the optional non-capturing group
        r'[_ ]*'               # Optional space or '_'
        )
    # Occurrence. (A suffix noting the occurrence number of the structure.)
    oar_pattern_list.append(
        r'(?P<Occurrence>'  # Start of *Occurrence* group
        r'[0-9]+'           # A number between 0 and 19
        r')?'               # End of optional *Occurrence* group
        )
    # End of string
    # The pattern should match the entire string.
    oar_pattern_list.append(r'$')
    oar_pattern = build_pattern(oar_pattern_list, oar_group_options,
                                regex_flags=re.IGNORECASE)
    return oar_pattern, oar_group_options


def build_oth_pattern()->Tuple[pd.Series, SortIndexes]:
    '''Create a regex pattern to identify other (non-target and non-OAR)
    structures.

    The pattern contains sections for each component of the Target volume name.
    The order of the *regular expression* sections matter partly because they
    define the order that the components appear in the name and partly because
    earlier sections grab components that might otherwise be confused with a
    component defined in a later section.

    Only the *BaseOAR* section is required.  Some of the other parts can
    be combined, while some are mutually exclusive.

    Returns:
        Tuple[pd.Series, SortIndexes]: _description_
    '''
    other_pattern_list = []
    other_group_options = {}
    #  Beginning of string
    other_pattern_list.append(r'^')
    # The pattern may contain anything at the beginning
    other_pattern_list.append(r'.*?')
    # Structures containing 'Avoid or 'Ring' designators.
    other_group_options['avoidance_types'] = ['Avoidance', 'Avoid', 'Ring']
    other_pattern_list.append(
        r'(?P<Avoidance>'      # Start of *Avoidance* group
        r'(?P<AvoidPrefix>'    # Start of *AvoidPrefix* group
        r'.*?'                 # Text before the Avoidance Type
        r')'                   # End of the *AvoidPrefix* group
        r'(?:[ _]*)'           # non-capturing group containing optional space or '_'
        r'(?P<AvoidanceType>'  # Start of *AvoidanceType* group
        r'{avoidance_types}'   # contains '|' separated items from avoidance_types.
        r')'                   # End of the *AvoidanceType* group
        r'(?:[ _]*)'           # non-capturing group containing optional space or '_'
        r'(?P<AvoidSuffix>'    # Start of *AvoidSuffix* group
        r'.*'                 # Text after the Avoidance Type
        r')'                   # End of the *AvoidSuffix* group
        r'(?:[ _]*)'           # non-capturing group containing optional space or '_'
        r')|'                  # End of the *Avoidance* group option
        )
    # Couch Structures
    other_group_options['couch_types'] = ['Surface', 'Interior',
                                        'RailLeft', 'RailRight']
    other_pattern_list.append(
        r'(?P<Couch>'        # Start of *Couch* group
        r'(?P<CouchLabel>'   # Start of *CouchLabel* group
        r'Couch'             # Starts with the word 'Couch'.
        r')'                 # End of the *CouchLabel* group
        r'(?:[ _]*)'         # non-capturing group containing optional space or '_'
        r'(?P<CouchType>'    # Start of *CouchType* group
        r'{couch_types}'     # contains '|' separated items from couch_types.
        r')?'                # End of the optional *CouchType* group
        r'(?P<CouchSuffix>'  # Start of *CouchSuffix* group
        r'.*'                # Text after the Couch Type
        r')'                 # End of the *CouchSuffix* group
        r')|'                # End of *Couch* group option
        )
    # Bolus Structures
    other_group_options['bolus_types'] = ['Bolus', 'Bol', 'Gauze', 'Barts', 'Pink']
    other_pattern_list.append(
        r'(?P<Bolus>'           # Start of *Bolus* group
        r'(?P<BolusText1>'      # Start of *BolusText1* group
        r'[A-Za-z]*'            # Text before the BolusSize
        r')'                    # End of the *BolusText1* group
        r'(?P<BolusSize>'       # Start of *BolusSize* group
        r'(?P<BolusSizeValue>'  # Start of *BolusSizeValue* group
        r'[0-9.]+'              # Required number (decimals allowed)
        r')'                    # End of the *BolusSizeValue* group
        r'(?:[ _]*)'            # non-capturing group containing optional space or '_'
        r'(?P<BolusSizeUnits>'  # Start of *BolusSizeUnits* group
        r'[cm]{{2}}'            # Required 'mm or cm'
        r')'                    # End of the *BolusSizeUnits* group
        r')?'                   # End of the optional *BolusSize* group
        r'(?P<BolusText2>'      # Start of *BolusText2* group
        r'.*?'                  # Remaining text before the Bolus Type (non-greedy to avoid trailing spaces)
        r')'                    # End of the *BolusText2* group
        r'(?:[ _]*)'            # non-capturing group containing optional space or '_'
        r'(?P<BolusType>'       # Start of *BolusType* group
        r'{bolus_types}'        # contains '|' separated items from bolus_types.
        r')'                    # End of the required *Bolus* group
        r'(?:[ _]*)'            # non-capturing group containing optional space or '_'
        r'(?P<BolusSuffix>'     # Start of *BolusSuffix* group
        r'.*'                   # Remaining text
        r')'                    # End of the *BolusSuffix* group
        r')|'                   # End of *Bolus* group option
        )
    # Foreign objects as Structures
    other_group_options['foreign_types'] = ['CIED', 'prosthesis', 'implant',
                                            'expander', 'screws', 'staples',
                                            'hardware', 'Metal', 'Dental',
                                            'anastomosis', 'stoma']
    other_pattern_list.append(
        r'(?P<Foreign>'        # Start of *Foreign* group
        r'(?P<ForeignPrefix>'  # Start of *ForeignPrefix* group
        r'.*?'                 # Text before the Foreign Type
        r')'                   # End of the *ForeignPrefix* group
        r'(?:[ _]*)'           # non-capturing group with optional space or '_'
        r'(?P<ForeignType>'    # Start of *ForeignType* group
        r'{foreign_types}'     # contains '|' separated items from foreign_types.
        r')'                   # End of the required *ForeignType* group
        r'(?:[ _]*)'           # non-capturing group with optional space or '_'
        r'(?P<ForeignSuffix>'  # Start of *ForeignSuffix* group
        r'.*'                  # Remaining text
        r')'                   # End of the *ForeignSuffix* group
        r')|'                  # End of *Foreign* group option
        )
    # Location reference objects as Structures
    other_group_options['marker_types'] = ['BBs', 'BB', 'fiducial', 'fiducials',
                                           'MARKER', 'OVOID',
                                           'Wires', 'Wire', 'Match', 'Baseline']
    other_pattern_list.append(
        r'(?P<Marker>'        # Start of *Marker* group
        r'(?P<MarkerPrefix>'  # Start of *MarkerPrefix* group
        r'.*?'                # Text before the Marker Type
        r')'                  # End of the *MarkerPrefix* group
        r'(?:[ _]*)'          # non-capturing group containing optional space or '_'
        r'(?P<MarkerType>'    # Start of *MarkerType* group
        r'{marker_types}'     # contains '|' separated items from marker_types.
        r's?'                 # optional s for plural.
        r')'                  # End of the required *MarkerType* group
        r'(?:[ _]*)'          # non-capturing group containing optional space or '_'
        r'(?P<MarkerSuffix>'  # Start of *MarkerSuffix* group
        r'.*'                 # Remaining text
        r')'                  # End of the *MarkerSuffix* group
        r')|'                 # End of *Marker* group option
        )
    # Structures for correcting density
    other_group_options['density_types'] = ['Air', 'Contrast']
    other_pattern_list.append(
        r'(?P<BulkDensity>'    # Start of *BulkDensity* group
        r'(?P<DensityPrefix>'  # Start of *DensityPrefix* group
        r'.*?'                 # Text before the BulkDensity Type
        r')'                   # End of the *DensityPrefix* group
        r'(?:[ _]*)'           # non-capturing group containing optional space or '_'
        r'(?P<DensityType>'    # Start of *DensityType* group
        r'{density_types}'     # contains '|' separated items from density_types.
        r')'                   # End of the required *DensityType* group
        r'(?:[ _]*)'           # non-capturing group containing optional space or '_'
        r'(?P<DensitySuffix>'  # Start of *DensitySuffix* group
        r'.*'                  # Remaining text
        r')'                   # End of the *DensitySuffix* group
        r')|'                  # End of *BulkDensity* group option
        )
    # Structures for field placement reference
    other_group_options['field_types'] = ['Field', 'Beam', 'edge', 'guide']
    other_pattern_list.append(
        r'(?P<FieldReference>'  # Start of *FieldReference* group
        r'(?P<FieldPrefix>'     # Start of *FieldPrefix* group
        r'.*?'                  # Text before the FieldType Type
        r')'                    # End of the *FieldPrefix* group
        r'(?:[ _]*)'            # non-capturing group containing optional space or '_'
        r'(?P<FieldType>'       # Start of *FieldType* group
        r'{field_types}'        # contains '|' separated items from field_types.
        r')'                    # End of the required *FieldType* group
        r'(?:[ _]*)'            # non-capturing group containing optional space or '_'
        r'(?P<FieldSuffix>'     # Start of *FieldSuffix* group
        r'.*'                   # Remaining text
        r')'                    # End of the *FieldSuffix* group
        r')|'                   # End of *FieldReference* group option
        )
    # Dose levels as structure
    other_group_options['dose_types'] = ['EVAL50', 'PREV', 'old',
                                        'Dose', 'D', 'BOB']
    other_pattern_list.append(
        r'(?P<Dose>'        # Start of optional *Dose* group
        r'(?P<DoseType>'   # Start of *DoseType* group
        r'{dose_types}'     # contains '|' separated items from dose_types.
        r')'               # End of *DoseType* group
        r'(?: *)'           # Non-capturing group containing optional spaces
        r'(?P<DoseValue>'  # Start of REQUIRED *DoseValue* group
        r'[0-9.]+'          # numbers and decimal.
        r')'                # End of *Dose_Value* group
        r'(?:[ []*)'        # Non-capturing group containing optional spaces or '['
        r'(?P<DoseUnits>'   # Start of *DoseUnits* group
        r'[%cGy]*'          # Optional %, cGy or Gy.
        r')'                # End of *DoseUnits* group
        r'(?:[ \]]*)'       # Non-capturing group containing optional spaces or ']'
        r'(?P<DoseSuffix>'  # Start of *DoseSuffix* group
        r'.*'                 # Remaining text
        r')'                  # End of the *DoseSuffix* group
        r')'               # End of *Dose* group
        )
    # Build and apply the pattern
    other_structure_pattern = build_pattern(other_pattern_list, other_group_options,
                                            regex_flags=re.IGNORECASE)
    return other_structure_pattern, other_group_options


# %% Functions for composite target groups
def build_target_groups(target_sections: pd.DataFrame, target_group_options: SortIndexes):
    def build_nodes_group(data: pd.DataFrame,
                        index_groups: SortIndexes
                        )->Tuple[pd.Series, SortIndexes]:
        '''Create Nodes column.

        Contains *Region* or target type 'Node', 'Nodes', 'Iliac Vessels'

        Args:
            data (pd.DataFrame): _description_
            target_group_options (Dict[str, List[str]]): _description_

        Returns:
            Tuple[pd.Series, SortIndexes]: _description_
        '''
        # Target Types that are nodes:
        node_type = ['Nodes', 'Node', 'Iliac Vessels']
        index_groups['node_type'] = node_type
        # Build the Nodes sort Group.
        node_group = data.Region.str.lower()
        # Blank Region.
        is_not_region = data.Region.isnull()
        # target  type is a node type.
        is_node_type = data.TargetType.isin(node_type)
        # Blank Region and Target Type is a node type.
        use_node_type = (is_not_region & is_node_type)
        # If Region is blank, use Node Target type if present.
        node_group.where(~use_node_type, data.TargetType.str.lower(),
                         inplace=True)
        # Remove node types from list of target types.
        new_target_types = [item for item in index_groups['target_type']
                            if item not in node_type]
        index_groups['target_type'] = new_target_types
        return node_group, index_groups

    def get_phase_order(data: pd.DataFrame,
                        sort_indexes: SortIndexes)->SortIndexes:
        '''Build the phase_target_order.

        Sort '4D' items by number and build sort index.

        Args:
            target_sections (_type_): _description_
            target_group_options (_type_): _description_
        '''
        # Get '4D' Motion items
        phase_mask = data.Motion.str.startswith('4D', na=False)
        if any(phase_mask):
            phase_targets = data.loc[phase_mask, 'Motion'].copy()
            # Extract number from 4D *Motion* items
            phase_targets = phase_targets.str.replace('4D', '')
            phase_targets = pd.to_numeric(phase_targets)
            # Sort the phase numbers
            phase_targets.sort_values(inplace=True)
            # Restore to original label
            phase_targets = '4D' + phase_targets.astype(str)
            phase_target_order = list(phase_targets.drop_duplicates())
        else:
            phase_target_order = []
        sort_indexes['phase_target'] = phase_target_order
        return sort_indexes

    def build_cgy_dose(data: pd.DataFrame)->pd.Series:
        '''Convert Target Dose to cGy.

        Dose values less than 100 are assumed tio be in Gy and are multiplied
        by 100 to convert them to cGy.

        Args:
            target_sections (pd.DataFrame): _description_

        Returns:
            pd.Series: _description_
        '''
        # Convert to numbers for correct sorting.
        cgy_dose = pd.to_numeric(data.TargetDose)
        # Convert all dose units to cGy for correct sorting.
        cgy_dose.where(cgy_dose > 100, cgy_dose * 100, inplace=True)
        return cgy_dose

    def build_expansions(target_sections: pd.DataFrame)->pd.DataFrame:
        ''' Build Target Expansion columns

        'ExpSize' is expansion or subtraction size in mm.
        'Expansions' is a text column made byt combining sign, size units (mm)
            and organ subtraction name.

        Args:
            target_sections (pd.DataFrame): _description_
            target_group_options (SortIndexes): _description_F

        Returns:
            pd.DataFrame: DataFrame with 4 columns:
                ['ExpSign', 'ExpSize', 'ExpUnit' 'Expansions']
        '''
        expansion_columns = []
        # Combine '1' and '2' columns for Sign, Size and Unit
        expansion_columns.append(merge_columns(target_sections, 'ExpansionSign',
                                            'SubtractionSign'))
        expansion_columns.append(merge_columns(target_sections, 'ExpansionSize1',
                                            'ExpansionSize2'))
        expansion_columns.append(merge_columns(target_sections, 'ExpansionUnit1',
                                            'ExpansionUnit2'))
        expansion = pd.concat(expansion_columns, axis='columns')
        expansion.columns = ['ExpSign', 'ExpSize', 'ExpUnit']
        # Convert expansion size to mm
        expansion.ExpSize = to_mm(expansion, 'ExpSize', 'ExpUnit')
        # Build the Expansions String columns
        # Join sign and size
        expansion['Expansions'] = (expansion.ExpSign.fillna('').str.strip() +
                                expansion.ExpSize.apply(to_num_str))
        # Add mm units where appropriate
        expansion.Expansions.where(expansion.ExpSize.isna(),
                                expansion.Expansions + 'mm', inplace=True)
        # Add Organ
        expansion.Expansions = (expansion['Expansions'] +
                                target_sections.OrganSubtraction.fillna(''))
        return expansion

    node_group, target_group_options = build_nodes_group(target_sections,
                                                         target_group_options)
    target_sections['Nodes'] = node_group
    target_group_options = get_phase_order(target_sections,
                                           target_group_options)
    target_sections['cGy_Dose'] = build_cgy_dose(target_sections)
    target_sections['Levels'] = merge_columns(target_sections,
                                              'Level', 'TargetSuffix')
    # target_sections['TargetNumber'] = merge_columns(target_sections,
    #                                                'TargetNumber', 'Combined')
    expansion = build_expansions(target_sections)
    target_sections = pd.concat([target_sections, expansion], axis='columns')
    target_sections.TargetLaterality = equivalent_items(
        target_sections.TargetLaterality,
        ['LT', 'L', 'Lt'], 'L')
    target_sections.TargetLaterality = equivalent_items(
        target_sections.TargetLaterality,
        ['RT', 'R', 'Rt'], 'R')
    return target_sections, target_group_options


def build_oar_groups(organ_sections: pd.DataFrame):
    organ_sections.PRV_Size = to_mm(organ_sections, 'PRV_Size', 'PRV_Unit')
    organ_sections.ExpansionSize = to_mm(organ_sections, 'ExpansionSize',
                                         'ExpansionUnit')
    organ_sections.OAR_Laterality = equivalent_items(organ_sections.OAR_Laterality,
                                                  ['LT', 'L', 'Lt'], 'L')
    organ_sections.OAR_Laterality = equivalent_items(organ_sections.OAR_Laterality,
                                                  ['RT', 'R', 'Rt'], 'R')
    return organ_sections


def build_oth_groups(other_sections: pd.DataFrame):
    other_sections['AvoidanceModifier'] = (other_sections.AvoidPrefix +
                                           other_sections.AvoidSuffix)
    other_sections['MarkerModifier'] = (other_sections.MarkerPrefix +
                                        other_sections.MarkerSuffix)
    other_sections['ForeignModifier'] = (other_sections.ForeignPrefix + ' ' +
                                         other_sections.ForeignSuffix)
    other_sections.DensitySuffix = other_sections.DensitySuffix.str.replace(r'/', '', regex=False)
    other_sections['DensityModifier'] = (other_sections.DensityPrefix +
                                         other_sections.DensitySuffix)
    other_sections['FieldModifier'] = (other_sections.FieldPrefix +
                                       other_sections.FieldSuffix)
    # Bol becomes Bolus
    bol_type = other_sections.BolusType.str.lower().isin(['bol'])
    other_sections.BolusType = other_sections.BolusType.where(~bol_type, 'Bolus')
    # Convert Bolus thickness to mm
    other_sections['BolusSize'] = to_mm(other_sections, 'BolusSizeValue',
                                        'BolusSizeUnits')
    # Merge all other bolus text
    other_sections['BolusModifier'] = (other_sections.BolusText1 + ' ' +
                                       other_sections.BolusText2 + ' ' +
                                       other_sections.BolusSuffix)
    #combine_columns(other_sections,
    #                ['BolusText1', 'BolusText2', 'BolusSuffix'],
    #                sep=' ')
    other_sections.DoseType = equivalent_items(other_sections.DoseType,
                                               ['D', 'Dose'], 'Dose')
    other_sections = to_cgy(other_sections, 'DoseValue', 'DoseUnits')
    return other_sections


# %% Build the sorting indexes
def build_target_sorter(target_group_options: SortIndexes,
                        structure_groups: pd.DataFrame)->pd.Series:
    # Build the Target Indexer
    target_sorter = DataSorter()
    # Nodes
    regions = target_group_options['regions']
    node_type = target_group_options['node_type']
    target_sorter.add_indexer(regions, node_type, column_name='Nodes',
                              missing_first=False)
    # Expansion
    target_sorter.add_indexer(column_name='ExpSign', as_numeric=False,
                              missing_first=True)
    target_sorter.add_indexer(column_name='OrganSubtraction', as_numeric=False,
                              missing_first=True)
    target_sorter.add_indexer(column_name='ExpSize', as_numeric=True,
                              missing_first=True)
    # Metabolic
    metabolic_types = target_group_options['metabolic_types']
    target_sorter.add_indexer(metabolic_types, column_name='Metabolic',
                              missing_first=False)
    # Motion
    phase_target_order = target_group_options['phase_target']
    phase_combined = target_group_options['motion_types']
    target_sorter.add_indexer(phase_target_order, phase_combined,
                              column_name='Motion', missing_first=False)
    # TargetOrgan
    target_sorter.add_indexer(column_name='TargetOrgan', as_numeric=False,
                              missing_first=True)
    # TargetNumber
    target_sorter.add_indexer(column_name='TargetNumber', as_numeric=True)
    # Combined
    combined_types = target_group_options['combined_types']
    target_sorter.add_indexer(combined_types, column_name='Combined',
                              missing_first=True)
    # cGy_Dose
    target_sorter.add_indexer(column_name='cGy_Dose', as_numeric=True)
    # TargetType
    target_types = target_group_options['target_type']
    target_sorter.add_indexer(target_types, column_name='TargetType')
    # Levels
    suffix_options = target_group_options['Suffix_options']
    level_options = target_group_options['Level_types']
    target_sorter.add_indexer(level_options, suffix_options,
                              column_name='Levels', missing_first=False)
    # Mod
    mod_types = target_group_options['mod_types']
    target_sorter.add_indexer(mod_types, suffix_options, column_name='Mod',
                              missing_first=True)
    # Laterality
    laterality_types = target_group_options['laterality_types']
    target_sorter.add_indexer(laterality_types, suffix_options,
                              column_name='TargetLaterality',
                              missing_first=True)
    # SubGroup
    target_sorter.add_indexer(column_name='SubGroup', as_numeric=False,
                              missing_first=False)

    target_index = target_sorter.combined_index(structure_groups)
    return target_index


def build_oar_sorter(group_options: SortIndexes,
                     structure_groups: pd.DataFrame)->pd.Series:
    oar_sorter = DataSorter()
    # BaseOAR
    base_oar = group_options['base_oar']
    oar_sorter.add_indexer(base_oar, column_name='BaseOAR')
    # Laterality
    laterality_types = group_options['laterality_types']
    oar_sorter.add_indexer(laterality_types, column_name='OAR_Laterality',
                           missing_first=True)
    # Occurrence
    oar_sorter.add_indexer(column_name='Occurrence', as_numeric=True,
                           missing_first=True)
    # PRV
    oar_sorter.add_indexer(column_name='PRV_Text', as_numeric=False,
                           missing_first=True)
    oar_sorter.add_indexer(column_name='PRV_Size', as_numeric=True,
                           missing_first=True)
    # Expansion
    oar_sorter.add_indexer(column_name='ExpansionSize', as_numeric=True,
                           missing_first=True)
    oar_sorter.add_indexer(column_name='ExpansionText', as_numeric=False,
                           missing_first=True)
    # OtherCrop
    other_crop = group_options['other_crop']
    oar_sorter.add_indexer(other_crop, column_name='OtherCrop',
                           missing_first=True)
    # Opt
    oar_sorter.add_indexer(column_name='Opt', as_numeric=False,
                           missing_first=True)
    # TargetCrop
    target_sub = group_options['target_sub']
    oar_sorter.add_indexer(target_sub, column_name='TargetCrop',
                           missing_first=True)
    # Assemble the composite index
    oar_index = oar_sorter.combined_index(structure_groups)
    return oar_index


def build_oth_sorter(group_options: SortIndexes, structure_groups: pd.DataFrame)->pd.Series:
    other_sorter = DataSorter()
    # Avoidance
    avoidance_types = group_options['avoidance_types']
    other_sorter.add_indexer(avoidance_types, column_name='AvoidanceType')
    other_sorter.add_indexer(column_name='AvoidanceModifier', as_numeric=False)
    # Couch
    couch_types = group_options['couch_types']
    other_sorter.add_indexer(couch_types, column_name='CouchType')
    # Marker
    marker_types = group_options['marker_types']
    other_sorter.add_indexer(marker_types, column_name='MarkerType')
    other_sorter.add_indexer(column_name='MarkerModifier', as_numeric=False)
    # Foreign
    foreign_types = group_options['foreign_types']
    other_sorter.add_indexer(foreign_types, column_name='ForeignType')
    other_sorter.add_indexer(column_name='ForeignModifier', as_numeric=False)
    # Field
    field_types = group_options['field_types']
    other_sorter.add_indexer(field_types, column_name='FieldType')
    other_sorter.add_indexer(column_name='FieldModifier', as_numeric=False)
    # Bolus
    bolus_types = group_options['bolus_types']
    other_sorter.add_indexer(bolus_types, column_name='BolusType')
    other_sorter.add_indexer(column_name='BolusSize', as_numeric=True)
    other_sorter.add_indexer(column_name='BolusModifier', as_numeric=False)
    # Dose
    dose_types = group_options['dose_types']
    other_sorter.add_indexer(dose_types, column_name='DoseType')
    other_sorter.add_indexer(column_name='DoseValue', as_numeric=True)
    other_sorter.add_indexer(column_name='DoseSuffix', as_numeric=False)
    # Assemble the composite index
    other_index = other_sorter.combined_index(structure_groups)
    return other_index

def build_sort_index(structure_column: pd.Series):
    # Identify the Target name parts for sorting.
    target_pattern, target_grp_opt = build_target_pattern()
    target_mask = structure_column.str.match(target_pattern)
    target_mask.fillna(False, inplace=True)
    target_struct = structure_column.loc[target_mask]
    target_sections = target_struct.str.extract(target_pattern, expand=True)
    target_sections, target_grp_opt = build_target_groups(target_sections,
                                                          target_grp_opt)
    target_index = build_target_sorter(target_grp_opt, target_sections)

    # Non-Target volumes are first selected as the structures that do not match
    # the *Target* *regular expression* pattern.
    remaining_struct = structure_column.loc[~target_mask]
    oar_pattern, oar_grp_opt = build_oar_pattern()
    oar_mask = remaining_struct.str.match(oar_pattern)
    oar_struct = remaining_struct.loc[oar_mask]
    oar_sections = oar_struct.str.extract(oar_pattern, expand=True)
    oar_sections = build_oar_groups(oar_sections)
    oar_index = build_oar_sorter(oar_grp_opt, oar_sections)
    oar_index = oar_index + 1

    # Other structures
    remaining_struct = remaining_struct.loc[~oar_mask]
    oth_pattern, oth_grp_opt = build_oth_pattern()
    oth_mask = remaining_struct.str.match(oth_pattern)
    oth_struct = remaining_struct.loc[oth_mask]
    oth_sections = oth_struct.str.extract(oth_pattern, expand=True)
    oth_sections = build_oth_groups(oth_sections)
    oth_index = build_oth_sorter(oth_grp_opt, oth_sections)
    oth_index = oth_index + 2

    sort_order = pd.concat([target_index, oar_index, oth_index], axis='rows')
    sort_order.name = 'SortOrder'
    sort_order = pd.concat([structure_column, sort_order], axis='columns')

    max_idx = max(sort_order.SortOrder)
    sort_order.SortOrder.fillna(max_idx + 1, inplace=True)
    sort_order = sort_order.SortOrder

    structure_sections = pd.concat(
        [target_sections, oar_sections, oth_sections, sort_order],
        axis='columns')
    selected_columns = ['SortOrder', 'TargetType', 'Nodes', 'BaseOAR',
                        'TargetLaterality', 'OAR_Laterality', 'cGy_Dose',
                        'SubGroup', 'Levels', 'Mod', 'Metabolic', 'Motion',
                        'TargetNumber', 'Combined',
                        'ExpSign', 'ExpSize', 'PRV_Size',
                        'TargetOrgan', 'TargetCrop',
                        'TargetModText', 'TargetModNum', 'OtherCrop',
                        'ExpansionSize', 'ExpansionText', 'Occurrence',
                        'BolusSize', 'BolusType', 'BolusModifier',
                        'ForeignType', 'ForeignModifier',
                        'MarkerType', 'MarkerModifier',
                        'DensityType', 'DensityModifier',
                        'AvoidanceType', 'AvoidanceModifier',
                        'DoseType', 'DoseValue', 'DoseUnits', 'DoseSuffix',
                        'CouchType',
                        'FieldType', 'FieldModifier'
                        ]
    structure_parts = structure_sections[selected_columns].copy()
    structure_parts.dropna(axis='columns', how='all', inplace=True)
    structure_parts.sort_values('SortOrder', inplace=True)
    return structure_parts


# %% Main
def main():
    # Relevant paths
    base_path = Path.cwd()
    base_path = base_path.resolve()
    data_path = base_path / 'Work In Progress' / 'Contours'
    test_data_file = data_path / 'analysis_structures.pkl'

    # Load Structures from pickle file
    with open(test_data_file, 'rb') as file:
        # load the Pickled DataFrame.
        analysis_structures = pickle.load(file)
    index_columns = ['StructureId', 'VolumeType']
    analysis_structures.drop_duplicates(subset=index_columns, inplace=True)
    analysis_structures = analysis_structures.set_index(index_columns,
                                                        drop=False)

    structure_sections = build_sort_index(analysis_structures.StructureId)

    xw.view(structure_sections)

if __name__ == '__main__':
    main()
