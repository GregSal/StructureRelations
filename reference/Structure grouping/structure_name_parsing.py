'''Structure Name Parsing

Parse a list of structure names based on TG-263.

It provides some simple functions for general validation of the standard, such
as name length.
It also provides regular expressions and functions for parsing structure names
that follow the nomenclature standard.

The parsing functions have two uses:
1. To verify that the structure name conforms to the TG283 standard.
2. To identify specific groups of structures for other analysis.
'''
# %% Imports
from typing import Dict, List, Union

from pathlib import Path
import re
import xml.etree.ElementTree as ET
from itertools import chain

import pandas as pd
import xlwings as xw

# Path
reference_path = Path.cwd()

# %% Utility Functions
def combine_columns(df: pd.DataFrame, columns: List[str], sep=' ')->pd.Series:
    '''Combine text from multiple columns with a separator.

    Args:
        df (pd.DataFrame): The table containing the columns to be merged.
            All of the columns should contain strings or NA values.
        columns (List[str]: The names of the columns to be merged.
        sep (str): A delimiter to place between the text from each column.

    Returns:
        pd.Series: A new text column containing the combined text from each
            column.
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


def to_cgy(text: str)->Dict[str, Union[float, int]]:
    '''Convert numbers with Gy units to cGy and identify fractions.

    Numbers without units or x are assumed to be total dose in cGy.
    Numbers with trailing 'Gy' are converted to cGy.
    Number preceded by x are fractions. dose values are assumed to be dose
    per fraction. Total dose is calculated by:
        $dose_per_fraction x fractions$
    The decimal point may also be represented by a 'p' e.g. 50p4Gy
    If the text does not match any of the valid formats, return the original
    text.

    Args:
        text (str): Dose as a string in one of the following forms:
            ####
            ##Gy
            ##.##Gy
            ##p##Gy
            ####x#
            ##Gyx#
            ##.##Gyx#
            ##p##Gyx#

    Returns:
        Tuple[float, int]: _description_
    '''
    if not text:
        dose_dict = {'TotalDose': text, 'Fractions': None}
        return pd.Series(dose_dict)
    # Convert 'p' to decimal point.
    try:
        text_cnv1 = text.replace('p', '.')
    except AttributeError:
        dose_dict = {'TotalDose': text, 'Fractions': None}
        return pd.Series(dose_dict)
    # Find fractions
    dose_parts = text_cnv1.split('x', 1)
    if len(dose_parts) > 1:
        try:
            fractions = int(dose_parts[1])
        except ValueError:
            dose_dict = {'TotalDose': text, 'Fractions': None}
            return pd.Series(dose_dict)
    else:
        fractions = None
    dose_str = dose_parts[0]
    # Convert Gy to cGy
    if dose_str.endswith('Gy'):
        try:
            dose = float(dose_str[:-2])  # Drop the Gy suffix
        except ValueError:
            dose_dict = {'TotalDose': text, 'Fractions': None}
            return pd.Series(dose_dict)
        dose = dose * 100   # Gy to cGy conversion
    else:
        try:
            dose = float(dose_str)
        except ValueError:
            dose_dict = {'TotalDose': text, 'Fractions': None}
            return pd.Series(dose_dict)
    # Convert dose per fraction to total dose
    if fractions:
        total_dose = dose * fractions
    else:
        total_dose = dose
    dose_dict = {'TotalDose': total_dose, 'Fractions': fractions}
    return pd.Series(dose_dict)


def extract_name_group(names: pd.DataFrame, re_pattern: re.Pattern,
                       match_column: str, idx: pd.Series = None)->pd.DataFrame:
    '''Extract portions of a structure name.

    The re_pattern is applied to the 'Remainder' column to extract named parts.
    The resulting DataFrame is merged with names and the Remainder columns is
    updated with new Remainders from the extraction.

    Args:
        nt_names (pd.DataFrame): A table with structure names. It must contain
            a column 'Remainder', which is used as the starting point for
            extracting name parts.
        re_pattern (re.Pattern): A regular expression with named groups.  It
            must contain one named group that will always be present if a
            successful match is made.  It must also contain a 'Remainder'
            named group that contains the unmatched part of the structure name.
        match_column (str): The name of the named group that is always present
            when a successful match is made.  Used to update the 'Remainder'
            column.
        idx (pd.Series, optional): A mask type index to a subset of names to
            apply the match to.  If not supplied, all names are matched.
            Default is None.

    Returns:
        pd.DataFrame: The supplied table with new columns containing the
            structure name parts.
    '''
    # Extract group parts based on regular expression.
    if idx is not None:
        extr_names = names.loc[idx, 'Remainder'].str.extract(re_pattern)
    else:
        extr_names = names.loc[:, 'Remainder'].str.extract(re_pattern)

    # Merge extracted group parts with structure names.
    names = names.merge(extr_names, how='left',
                            left_index=True, right_index=True,
                            suffixes=('', '_ex'))

    # Update Remainder text
    nt_idx = names[match_column].isna()
    # Where a match was not found, keep the original Remainder text, otherwise
    # update Remainder with resulting Remainder after the match.
    names.Remainder = names.Remainder.where(nt_idx, names.Remainder_ex)
    names.drop(columns=['Remainder_ex'], inplace=True)
    return names

# %% Test functions
def valid_length(text: str, max_length=16)->bool:
    '''Test for valid string length.

    String length must be less than or equal to max_length.

    Args:
        text (str): String to test for length.
        max_length (int, optional): Maximum allowable length. Defaults to 16.

    Returns:
        bool: True if text length is less than or equal to max_length.
    '''
    text_len = len(text)
    return text_len <= max_length


def no_spaces(text: str)->bool:
    '''Verify that text does not contain spaces.

    Args:
        text (str): String to test for spaces.

    Returns:
        bool: True if text does not contain spaces.
    '''
    has_space = ' ' in text
    return ~has_space


def no_dup(structures: List[str])->bool:
    '''Verify that a list of structure names does not contain duplicates.

    Duplicates tests will ignore case.

    Args:
        structures (List[str]): List of structure names to test for duplicates.

    Returns:
        bool: True if list of structure names does not contain case-insensitive
            duplicates.
    '''
    unique_structures = {struc.lower() for struc in structures}
    has_dup = len(unique_structures) < len(structures)
    return ~has_dup


def not_evaluated(text: str)->bool:
    '''Identify 'Not Evaluated' structure names.

    'Not Evaluated' structure names are prefixed with a 'z' or '_'.

    Args:
        text (str): Structure name to be checked.

    Returns:
        bool: True if structure name is 'Not Evaluated'.
    '''
    exclude_prefixes = ['Z', '_']
    exclude = text[0] in exclude_prefixes
    return exclude

# %% Regular Expression patterns
# Non-Evaluated Nomenclature
not_evaluated_pat = ''.join([
    r'(?P<NotEvalChar>',   # Start of named group NotEvalChar
    r'[zZ_]',              # Z or _ character
    r')',                  # End of named group NotEvalChar
    r'(?P<NotEvaluated>',  # Start of named group NotEvaluated
    r'.*',                 # Remainder of the text.
    r')',                  # End of named group NotEvaluated
    ])

# Non-Target Nomenclature
basic_sub_structure_pat = ''.join([
    r'(',               # Start of group
    r'[A-Z]',             # Starts with a capital letter.
    r'([A-Z]+|[a-z]+)?',  # Remaining text as all capitals or all lowercase.
    r'[si]?',             # Optional plural indicator 's' or 'i'
    r'[~]?',              # Optional partial indicator '~'
    r'[0-9]{0,2}',        # Optional trailing 1 or 2 digit number.
    r'_?'                 # Optional ending '_'
    r'\^?'                # Optional ending '^'
    r')'                # End of group
    ])

basic_structure_pat = ''.join([
    r'(?P<StructureName>',  # Start of named group StructureName
    r'(',                     # Start of group
    basic_sub_structure_pat,    # Sub-structure pattern definition
    r')+'                     # End of repeatable group
    r')'                    # End of group
    ])

# %% [markdown]
# ### Parsing Non-Target Structures
# The following patterns can be used to sub-divide OAR names.
# They are not part of the primary regular expression, but can be applied after
# the target structures have been identified.
# %% [markdown]
# #### Major Category
#
# |Prefix|Meaning|Example|
# |-|-|-|
# |A|artery|A_Aorta, A_Carotid|
# |V|vein|V_Portal, V_Pulmonary|
# |LN|lymph node|LN_Ax_L1, LN_IMN|
# |CN|cranial nerve|CN_IX_L, CN_XII_R|
# |Glnd|glandular structure|Glnd_Submand|
# |Bone|bone|Bone_Hyoid, Bone_Pelvic|
# |Musc|muscle|Musc_Masseter, Musc_Sclmast_L|
# |Spc|Space|Spc_Bowel, Spc_Retrophar_L|
# |VB|vertebral body||
# |Sinus|sinus|Sinus_Frontal, Sinus_Maxillary|
#

# %%
category_def = {
    'A': 'artery',
    'V': 'vein',
    'LN': 'lymph node',
    'CN': 'cranial nerve',
    'Glnd': 'glandular structure',
    'Bone': 'bone',
    'Musc': 'muscle',
    'Spc': 'Space',
    'VB': 'vertebral body',
    'Sinus': 'sinus'
    }


# %%
major_category_pat = re.compile(''.join([
    r'^'                        # Beginning of string.
    r'(?P<StructureCategory>',  # Start of named group StructureCategory
    r'(?:',                         # Start of non-captured Root options group
    r'A|V|LN|CN|Glnd|',             # Root options
    r'Bone|Musc|VB|Sinus',          # Root options continued
    r')'                          # End of group
    r'(?P<Pleural>',              # Start of Optional named group Pleural
    r'[si]',                        # Optional plural indicator 's' or 'i'
    r')?'                         # End of Optional Pleural group
    r')'                        # End of StructureCategory group
    r'(?:',                     # Start of non-captured optional group
    r'_'                          # '_' as delimiter
    r'(?P<Remainder>',            # Start of named group Remainder
    r'.*',                          # All remaining text
    r')'                          # End of Remainder group
    r')?'                       # End of optional group
    r'$'                        # End of string.
    ]))

# %% [markdown]
# #### Custom Qualifier Text
# - Custom qualifier text is delimited by '^' e.g. Lungs^Ex

# %%
custom_oar_qualifier_pat = re.compile(''.join([
    r'^'                    # Beginning of string.
    r'(?P<Remainder>',      # Start of named group Remainder
    r'[^^]*',                 # All text before a '^'
    r')'                    # End of Remainder group
    r'(?:',                 # Start of non-captured optional group
    r'(?:\^)'                 # '^' character (not captured)
    r'(?P<CustomStructure>',  # Start of named group CustomStructure
    r'.+'                       # Remainder of text
    r')'                      # End of CustomStructure group
    r')?'                   # End of optional group
    ]))

# %% [markdown]
# #### Spatial Categorizations
# |Suffix|Meaning|
# |-|-|
# |L|left|
# |R|Right|
# |A|Anterior|
# |P|Posterior|
# |I|Inferior|
# |S|Superior|
# |RUL|Right Upper lobe|
# |RLL|Right Lower lobe|
# |RML|Right middle lobe|
# |LUL|Left Upper lobe|
# |LLL|LeftLower lobe|
# |NAdj|non-adjacent|
# |Dist|distal|
# |Prox|proximal|
#
# %% [markdown]
# ##### Issues with Identifying Spatial Indicators
# Single character spatial indicators (*L*, *R*, *A*, *P*, *S*, and *I*)
# can be confused with cranial nerves and nodal levels that use roman numerals:
# - `CN_I` is the *first cranial nerve* not the *inferior cranial nerve*
# - `LN_Neck_IA_L` is the *Level IA (Submental) neck node Left*
#
# Vertebral body references use *L* and *S* to refer to the lumbar and sacral
# vertebrae:
# - `VB_L` refers to the *Lumbar Vertebrae*
# - `VB_S` refers to the *Sacral Vertebrae*
#
# It is also possible for spatial indicators to be combined:
# - `Nasalconcha_LI` is the *<u>Inferior</u> Nasal Concha <u>Left</u>*
#
# - Before searching for spatial indicators deal with special cases
#
# %% [markdown]
# ##### Vertebral Body References

# %%
vertebrae_level = {
    'C': 'Cervical',
    'T': 'Thoracic',
    'L': 'Lumbar',
    'S': 'Sacral'
    }


# %%
vb_ref_pat = re.compile(''.join([
    r'^'                      # Beginning of string.
    r'(?P<VertebraeLevel>',   # Start of named group VertebraeLevel
    r'[CTLS]'                   # One of Cervical Thoracic, Lumbar, Sacral
    r')'                      # End of VertebraeLevel group
    r'(?P<VertebraeNumber>',  # Start of named group VertebraeNumber
    r'[0-9]{0,2}'               # Optional 1 or 2 digit level number
    r')'                      # End of VertebraeNumber group
    r'(?:',                   # Start of non-captured optional group
    r'_'                        # '_' as delimiter
    r'(?P<Remainder>',          # Start of named group Remainder
    r'.*',                        # All remaining text
    r')'                        # End of Remainder group
    r')?'                     # End of optional group
    r'$'                      # End of string.
    ]))

# %% [markdown]
# ##### Cranial Nerve References

# %%
cn_ref_pat = re.compile(''.join([
    r'^'               # Beginning of string.
    r'(?P<NerveLevel>',  # Start of named group NerveLevel
    r'[IVX]+'              # Roman Numeral Characters
    r')'                 # End of NerveLevel group
    r'(?:',                   # Start of non-captured optional group
    r'_'                        # '_' as delimiter
    r'(?P<Remainder>',          # Start of named group Remainder
    r'.*',                        # All remaining text
    r')'                        # End of Remainder group
    r')?'                     # End of optional group
    r'$'                      # End of string.
    ]))

# %% [markdown]
# ##### Neck Node References

# %%
nn_ref_pat = re.compile(''.join([
    r'^'              # Beginning of string.
    r'(?P<NeckNode>',   # Start of named group NeckNode
    r'Neck'               # The text 'Neck'
    r')'                # End of NeckNode group
    r'_'                # '_' as delimiter
    r'(?P<NodeLevel>',  # Start of named group NodeLevel
    r'[IVX]+'             # Roman Numeral characters
    r'[AB]?'              # A or B sub level characters
    r')'                # End of NerveLevel group
    r'(?:',             # Start of non-captured optional group
    r'_'                  # '_' as delimiter
    r'(?P<Remainder>',    # Start of named group Remainder
    r'.*',                  # All remaining text
    r')'                  # End of Remainder group
    r')?'               # End of optional group
    r'$'              # End of string.
    ]))

# %% [markdown]
# ##### Find Spatial Indicators
# Once conflicting patterns have been identified, Spatial Indicators can be
# extracted.

# %%
spatial_def = {
    'L': 'left',
    'R': 'Right',
    'A': 'Anterior',
    'P': 'Posterior',
    'I': 'Inferior',
    'S': 'Superior',
    'RUL': 'Right Upper lobe',
    'RLL': 'Right Lower lobe',
    'RML': 'Right middle lobe',
    'LUL': 'Left Upper lobe',
    'LLL': 'LeftLower lobe',
    'NAdj': 'non-adjacent',
    'Dist': 'distal',
    'Prox': 'proximal'
    }


# %%
spatial_pat = re.compile(''.join([
    r'^'                     # Beginning of string.
    r'(?:',                  # Start of optional non-capturing group
    r'(?P<Remainder>',         # Start of named group Remainder
    r'.*?',                      # All unused text (non-greedy)
    r')'                       # End of Remainder group
    r'_'                       # '_' as delimiter
    r')?',                   # End of optional group
    r'(?P<SpatialIndicator>',  # Start of named group SpatialIndicator
    r'(?:'                       # Start of non-capture group
    r'L|R|A|P|I|S|',               # Basic directions
    r'NAdj|Dist|Prox|',            # Relative directions
    r'RUL|RLL|LUL|LLL',            # Lung quadrants
    r')+'                        # End of group with multiple Indicators
    r')'                       # End of SpatialIndicator group
    r'$'                       # End of string.
    ]))

# %% [markdown]
# #### Planning organ at risk volumes
# - Begins with 'PRV'
# - Ends with optional 1 or 2 digit expansion number (in mm)
#

# %%
prv_pat = re.compile(''.join([
    r'^'                # Beginning of string.
    r'(?P<Remainder>',  # Start of named group Remainder
    r'.*?',               # All unused text (non-greedy)
    r')'                # End of Remainder group
    r'_?'               # '_' as delimiter
    r'(?P<Prv>',        # Start of named group Prv
    r'PRV'                # PRV designator
    r'(?P<PrvSize>',      # Start of named group PrvSize
    r'[0-9]{1,2}'           # Optional expansion size as 1 or 2 digits
    r')?'                 # End of optional PrvSize group
    r')'                # End of Prv group
    r'$'                # End of string.
    ]))

# %% [markdown]
# #### Partial Structure Indicator
# - Partial structure indicated by '\~' suffix e.g. Brain\~, Lung\~_L|
#

# %%
partial_pat = re.compile(''.join([
    r'^'                # Beginning of string.
    r'(?P<Remainder>',  # Start of named group Remainder
    r'.*?',               # All unused text (non-greedy)
    r')'                # End of Remainder group
    r'(?P<Partial>',    # Start of named group Partial
    r'~'                  # Partial designator '~'
    r')'                # End of Partial group
    r'$'                # End of string.
    ]))

# %% [markdown]
# #### Base Structure Name
# %% [markdown]
# - An <u>underscore character ('_')</u> is used to separate categorizations
#   (e.g., Bowel_Bag).
#
# - <u>Camel case</u> (a compound word where each word starts with a capital
#   letter and there is no space between words such as CamelCase) is only used
#   when a structure name implies two concepts, but the concepts do not appear
#   as distinct categories in common usage (e.g., CaudaEquina instead of
#   Cauda_Equina) because there are not several examples of Cauda_xxxxx.
#
# - Compound structures are identified using the plural, i.e., <u>the name ends
# - with an *'s'* or an *'i'*</u> as appropriate on the root structure name
# - (e.g., Lungs, Kidneys, Hippocampi, LNs (for all lymph nodes), Ribs_L.)

# %%
base_structure_pat = re.compile(''.join([
    r'^'                       # Beginning of string.
    r'(?P<BaseStructure>',     # Start of named group BaseStructure
    r'(?:'                       # Start of non-captured group
    r'[A-Z]',                      # Starts with a capital letter.
    r'(?:[A-Z]+|[a-z]+)',          # Remaining text as all capitals or all lowercase.
    r'){1,2}'                    # End of group with optional repeat (CamelCase name)
    r'(?P<Pleural>',             # Start of optional named group Pleural
    r'[si]',                       # Optional plural indicator 's' or 'i'
    r')?'                        # End of optional Pleural group
    r')'                       # End of BaseStructure group
    r'(?:',                    # Start of non-captured optional group
    r'_'                         # '_' as delimiter
    r'(?P<StructureQualifier>',  # Start of named group StructureQualifier
    r'[A-Z]',                      # Capital letter to start Structure Qualifier
    r'(?:[A-Z]+|[a-z]+)',          # Remaining text as all capitals or all lowercase.
    r')'                         # End of StructureQualifier group
    r'|'                       # OR
    r'(?:',                      # Start of non-captured group
    r'_?'                          # optional '_' as delimiter
    r'(?P<StructureNumber>',       # Start of named group StructureNumber
    r'[A-Z]?',                       # Optional capital letter
    r'[0-9]+',                       # Numeric Structure Qualifier
    r')'                           # End of StructureNumber group
    r')'                         # End of non-captured group
    r')*'                      # End of optional multiple non-captured group
    r'(?P<Remainder>',         # Start of named group Remainder
    r'.*',                       # All remaining text
    r')'                       # End of Remainder group
    r'$'                       # End of string.
    ]))

# %% [markdown]
# ### Target Nomenclature
# %% [markdown]
# #### Target Type
#
# - The first set of characters must be one of the allowed target types:
#     - GTV
#     - CTV
#     - ITV
#     - IGTV (Internal Gross Target Volume—gross disease with margin for motion)
#     - ICTV (Internal Clinical Target Volume—clinical disease with margin for motion)
#     - PTV
#     - PTV! for low-dose PTV volumes that exclude overlapping high-dose volumes

# %%
target_type_def = {
    'GTV': 'Gross Target Volume',
    'CTV': 'Clinical Target Volume',
    'PTV': 'Planning Target Volume',
    'ITV': 'Internal Target Volume',
    'IGTV': 'Internal Gross Target Volume',
    'ICTV': 'Internal Clinical Target Volume',
    'PTV!': 'Partial Planning Target Volume'
    }


# %%
target_type_pat = ''.join([
    r'(',             # Start of required group
    r'GTV|CTV|PTV|',    # Target Type options
    r'ITV|IGTV|ICTV|',  # Internal target volume types
    r'PTV!'             # low-dose PTV volumes excluding high-dose volumes
    r')'              # End of the required group
    ])

# %% [markdown]
# #### Target Classifier
#
# - If used, the target classifier is placed after the target type with no spaces.
#     - Allowed target classifiers are listed below:
#     - n: nodal (e.g., PTVn)
#     - p: primary (e.g., GTVp)
#     - sb: surgical bed (e.g., CTVsb)
#     - par: parenchyma (e.g., GTVpar)
#     - v:venous thrombosis (e.g., CTVv)
#     - vas: vascular (e.g., CTVvas)

# %%
target_classifier_def = {
    'n': 'nodal',
    'p': 'primary',
    'sb': 'surgical bed',
    'par': 'parenchyma',
    'v': 'venous thrombosis',
    'vas': 'vascular'
    }


# %%
target_classifier_pat = ''.join([
    r'(',               # Start of optional group
    r'par|vas|sb|n|p|v',  # Target classifier options
    r')?'               # End of the optional group
    ])

# %% [markdown]
# #### Target Number
#
# - For multiple spatially distinct targets Arabic numerals are used after the
#   target type and classifier (e.g., PTV1, PTV2, GTVp1, GTVp2).

# %%
target_number_pat = ''.join([
    r'(',         # Start of optional group
    r'[0-9]{1,2}',  # Target number as 1 or 2 digits
    r')?'         # End of the optional group
    ])

# %% [markdown]
# #### Base Target
# - Base target include first three parts of target structure name:
#   1. Target Type
#   2. Target Classifier
#   3. Target Number
#
# - The base target is grouped separately because it can be used as a cropping
#   designator for OARs. e.g. `Brain-GTV`

# %%
base_target_pat = ''.join([
    r'(?P<BaseTarget>',      # Start of named group BaseTarget
    r'(?P<TargetType>',        # Start of named group TargetType
    target_type_pat,             # Target Type pattern definition
    r')',                      # End of the TargetType group
    r'(?P<TargetClassifier>',  # Start of optional named group TargetClassifier
    target_classifier_pat,       # Target Classifier pattern definition
    r')?',                     # End of the optional TargetClassifier group
    r'(?P<TargetNumber>',      # Start of optional named group TargetNumber
    target_number_pat,           # Target Number pattern definition
    r')?',                     # End of the optional TargetNumber group
    r')',                    # End of the BaseTarget group
    ])

# %% [markdown]
# - Imaging modality follows the type/classifier/enumerator with an underscore
#   and then the image modality type (CT, PT, MR, SP)
#
# - Image sequence order is indicated by a number immediately following the
#   image modality.
#
# - Multiple modalities can be included.  No additional underscore is used.

# %%
modality_def = {
    'CT': 'CT',
    'PT': 'Pet',
    'MR': 'MRI',
    'US': 'Ultrasound',
    'SP': 'Spect'
    }


# %%
modality_pat = ''.join([
    r'(',            # Start of optional group
    r'(?:_)',          # Underscore delimiter (Not captured)
    r'(?P<Modality>',  # Start of optional named group Modality
    r'('                 # Beginning of optional repeat group
    r'(CT|PT|MR|US|SP)',   # Modality designator group
    r'[0-9]{0,2}',         # Optional sequence number as 1 or 2 digits
    r'){1,2}'            # repeatable group for multiple modalities
    r')'               # End of named group Modality
    r')?'            # End of optional group
    ])

# %% [markdown]
# #### Structure Indicators
# - Structure indicators follow the type/classifier/enumerator/imaging with an underscore prefix
# - Structure indicators are values from the approved structure nomenclature list.
# - Examples: CTV_A_Aorta, CTV_A_Celiac, GTV_Preop, PTV_Boost, PTV_Eval, PTV_MR2_Prostate
#
#
# **Note:**
# - Relative dose indicators have a similar pattern to Structure Indicators, but
#   are limited to three text strings: 'High', 'Mid', or 'Low' (see next section).
# - None of the current valid Structure Indicators begin with this text.
# - Add a check to the pattern for these three text strings.

# %%
struct_ind_pat = ''.join([
    r'(',                      # Start of optional group
    r'(?:_)',                    # Underscore delimiter (Not captured)
    r'(?!Hig|Mid|Low)',          # Exclude text that begins with one of these patterns

    r'(?P<StructureIndicator>',  # Start of named group StructureIndicator
    r'(',                          # Start of group
    basic_sub_structure_pat,         # Sub-structure pattern definition
    r')+'                          # End of repeatable group
    r')',                        # End of named group StructureIndicator
    r')?'                      # End of optional group
    ])

# %% [markdown]
# - If the structure is cropped back from the external contour for the patient,
#   then the quantity of cropping by “-xx” millimeters is placed at the end of
#   the target string. The cropping length follows the dose indicator, with the
#   amount of cropping indicated by xx millimeters
#   (e.g., PTV_Eval_7000-08, PTV-03, CTVp2-05).

# %%
target_crop_pat = ''.join([
    r'(?P<ExternalCrop>',  # Start of optional named group ExternalCrop
    r'(?P<Sign>-)',          # Negative sign '-' as its own named group
    r'(?P<Size>[0-9]{2})',   # 2-digit Number
    r')?'                  # End of optional group
    ])

# %% [markdown]
# - If a custom qualifier string is used, the custom qualifier is placed at the
#   end after a ‘^’ character (e.g., PTV^Physician1, GTV_Liver^ICG).
# - Include *custom_qualifier_pat* from non-target structures
# %% [markdown]
# #### Dose Specifier
# %% [markdown]
# - Dose specifier is placed at the end of the target string prefixed with an underscore character.
#
# - Dose specifier can be one of:
#    - Relative Dose Level
#    - Numeric dose
#    - Dose per Fraction and number of Fractions
#
# %% [markdown]
# ##### Relative Dose
# - Relative dose is recommended
#     - High (e.g., PTV_High, CTV_High, GTV_High)
#     - Mid: (e.g., PTV_Mid, CTV_Mid, GTV_Mid)
#     - Low (e.g., PTV_Low, CTV_Low, GTV_Low) ◦
#
# - Mid+2-digit enumerator: allows specification of more than three relative
#   dose levels (e.g., PTV_Low, PTV_Mid01, PTV_Mid02, PTV_Mid03, PTV_High).
# - Lower numbers correspond to lower dose values.

# %%
rel_dose_pat = ''.join([
    r'(?P<RelativeDose>',     # Start of named group RelativeDose
    r'High|',                   # 'High' relative dose    OR
    r'Low|',                    # 'Low'  relative dose    OR
    r'Mid',                     # 'Mid'  relative dose   with
    r'(?P<RelativeDoseLevel>',  # Optional RelativeDoseLevel group
    r'[0-9]{2}',                  # 2-digit number
    r')?',                      # End of Optional RelativeDoseLevel group
    r')'                      # End of RelativeDose group
    ])

# %% [markdown]
# ##### Numeric Dose
# - Units of cGy are recommended for numeric dose values. (e.g., PTV_5040).
# - When specified in units of Gy, then ‘Gy’ should be appended to the numeric
#   value of the dose (e.g., PTV_50.4Gy).
# - For systems that do not allow use of a period, the ‘p’ character should be
#   substituted (e.g., PTV_50p4Gy)

# %%
numeric_dose_pat = ''.join([
    r'(?P<NumericDose>',  # Start of optional named group NumericDose
    r'[0-9]+',              # Number before decimal place
    r'[.p]?',               # '.' or 'p' as optional decimal place
    r'[0-9]*',              # Optional Number after decimal place
    r'[Gy]*',               # Optional units of Gy
    r')'                  # End of NumericDose group
    ])

# %% [markdown]
# ##### Dose per Fraction and number of Fractions
# - If the dose indicated must reflect the number of fractions used to reach the
#   total dose, then the numeric values of <u>dose per fraction</u> in cGy, or
#   in Gy with the <u>unit specifier</u>, '<u>x</u>' followed by the <u>number
#   of fractions</u> (e.g., PTV_Liver_2000x3 or PTV_Liver_20Gyx3).

# %%
dose_fraction_pat = ''.join([
    r'(?P<DoseFractionation>',  # Start of  named group DoseFractionation
    r'[0-9]+',                    # Number before decimal place
    r'[.p]?',                     # '.' or 'p' as optional decimal place
    r'[0-9]*',                    # Optional Number after decimal place
    r'[Gy]*',                     # Optional units of Gy
    r'x',                         # Fractions delimiter 'x'
    r'[0-9]+',                    # Number of fractions
    r')'                        # End of DoseFractionation group
    ])

# %% [markdown]
# - Dose Fractionation must be specified before Numeric Dose because otherwise Numeric Dose will catch part of Dose Fractionation

# %%
dose_specifier = ''.join([
   '(',                   # Beginning of optional Dose Specifier group
    r'(?:_)',               # Underscore delimiter (Not captured)
    r'(?P<DoseSpecifier>',  # Start of named group DoseSpecifier
    rel_dose_pat,             # Relative Dose pattern definition
    '|',                    # OR
    dose_fraction_pat,        # Dose Fractionation pattern definition
    '|',                    # OR
    numeric_dose_pat,         # Numeric Dose pattern definition
    ')'                     # End of Dose Specifier options
    ')?',                 # End of optional Dose Specifier group
   ])

# %% [markdown]
# #### Custom Qualifier
# - Custom Qualifier indicated by '^' e.g. Lungs^Ex

# %%
target_custom_qualifier_pat = ''.join([
    r'('                 # Start of optional group
    r'(?:\^)'              # '^' character (not captured)
    r'(?P<CustomTarget>',  # Start of optional named group Custom
    r'.+'                    # Remainder of text
    r')'                   # End of named group Custom
    r')?'                # End of optional group
    ])

# %% [markdown]
# #### Combine Target patterns
# **Order of Name Components:**  (Underline indicates required component)
#
# 1. <u>Base Target</u>  (consists of three parts)
#    1. <u>Target Type</u>
#    2. Target Classifier
#    3. Target Number
# 2. Modality
# 3. Structure Indicator
# 4. Dose Specifier
# 5. Target Cropping from External
# 6.  Custom Qualifier
#
# - Pattern must match the entire string
#

# %%
target_pattern = ''.join([
    r'(',                         # Start of all target group patterns
    base_target_pat,              # Base Target pattern definition
    modality_pat,                 # Target Modality pattern definition
    struct_ind_pat,               # Structure pattern definition
    dose_specifier,               # Dose Specifier pattern definition
    target_crop_pat,              # Target Cropping pattern definition
    target_custom_qualifier_pat,  # Custom Qualifier pattern definition
    r')',                         # End of all target group patterns
    ])

# %% [markdown]
# #### Cropped OARs
# - OARs can have target volumes subtracted from them to exclude tumour from
#   OAR dose calculations.
# - This is not mentioned explicitly in TG263, but is included in their structure
#   examples.
# - The pattern created here is inferred from the TG263 examples.
# - **OAR Component**  The allowable OAR components includes:
#     1. <u>Primary Structure</u>
#     2. Spatial Indicator
# - **Target Component**  The allowable target components include:
#      1. <u>Target Type</u>
#      2. Target Classifier
#      3. Target Number
#

# %%
cropped_oar_pat = ''.join([
    r'(?P<CroppedOAR>',   # Start of named group CroppedOAR
    r'(',                   # Start of group
    basic_sub_structure_pat,  # Sub-structure pattern definition
    r')+'                   # End of repeatable group
    r')'                  # End of named group CroppedOAR
    ])


# %%
target_crop_pat = ''.join([
    r'(?P<TargetCrop>',  # Start of named group TargetCrop
    target_type_pat,       # Target Type pattern definition
    r'(',                  # Start of optional group
    target_classifier_pat,   # Target Classifier pattern definition
    r')?',                 # End of optional group
    r'(',                  # Start of optional group
    target_number_pat,       # Target Number pattern definition
    r')?',                 # End of optional group
    r')'                 # End of the BaseTarget group
    ])


# %%
oar_crop_pat = ''.join([
    r'(?P<OARCrop>',  # Start of named group OARCrop
    cropped_oar_pat,    # Cropped OAR pattern definition
    r'-',               # negative sign
    target_crop_pat,    # TargetCrop pattern definition
    r')'              # End of the OARCrop group
    ])

# %% [markdown]
# ### All Patterns Combined
# - Target patterns must come before non-target patterns or non-target patterns
#   will capture target structures.

# %%
all_structure_pattern = ''.join([
    r'(',              # Start of primary group of patterns
    not_evaluated_pat,   # Not Evaluated pattern definition
    r'|',              # OR
    target_pattern,      # Target pattern definition
    r'|',              # OR
    oar_crop_pat,        # OAR with Target crop pattern definition
    r'|',              # OR
    basic_structure_pat,  # Non-Target pattern definition
    r')',              # End of primary group of patterns
    ])


# %%
structure_pat = re.compile(all_structure_pattern)

# %% [markdown]
# ## Apply the pattern matching to a list of structure names
# %% [markdown]
# ### Load the names from a text file

# %%
examples_file = reference_path / 'examples.txt'
examples = examples_file.read_text().splitlines()


# %%
matched_structure_list = []
for structure in examples:
    mtch = structure_pat.fullmatch(structure)
    if mtch:
        mtch_dict = mtch.groupdict()
        mtch_dict['Structure'] = structure
    else:
        mtch_dict = {'Structure': structure}
    matched_structure_list.append(mtch_dict)

matched_structures = pd.DataFrame(matched_structure_list)
matched_structures.drop_duplicates(inplace=True)
matched_structures.set_index('Structure', inplace=True)

# %% [markdown]
# ### Get Dose Values

# %%
dose_values = matched_structures.DoseSpecifier.apply(to_cgy)
matched_structures = matched_structures.join(dose_values)

# %% [markdown]
# ### Identify match failures

# %%
unmatched_idx = matched_structures.isna().all(axis='columns')
unmatched_idx.name = 'Unmatched'
matched_structures = matched_structures.join(unmatched_idx)

# %% [markdown]
# ### Parse Non-Target Structures

# %%
# Get Non-Target Structure Names.
nt_idx = ~matched_structures.StructureName.isna()
names = matched_structures.loc[nt_idx, ['StructureName', 'Unmatched']]
names['Remainder'] = names.StructureName

# Sequentially apply Non-Target Parsing Rules
names = extract_name_group(names, major_category_pat, 'StructureCategory')
names = extract_name_group(names, custom_oar_qualifier_pat, 'CustomStructure')

is_vb = names.StructureCategory == 'VB'
names = extract_name_group(names, vb_ref_pat, 'VertebraeLevel', is_vb)

is_cn = names.StructureCategory == 'CN'
names = extract_name_group(names, cn_ref_pat, 'NerveLevel', is_cn)

is_ln = names.StructureCategory == 'LN'
names = extract_name_group(names, nn_ref_pat, 'NeckNode', is_ln)

names = extract_name_group(names, spatial_pat, 'SpatialIndicator')
names = extract_name_group(names, prv_pat, 'Prv')
names = extract_name_group(names, partial_pat, 'Partial')
names = extract_name_group(names, base_structure_pat, 'BaseStructure')

# Merge Parsed Non-Target Structures with Parsed Target table
names.drop(columns=['StructureName', 'Unmatched'], inplace=True)
matched_structures = matched_structures.join(names)

# %% [markdown]
# ### Identify Non-Target Parsing Failures

# %%
matched_structures['Um2'] = matched_structures.Remainder.str.len() > 0
matched_structures.Um2 = matched_structures.Um2.fillna(False)
matched_structures.Unmatched.where(~matched_structures.Um2, True, inplace=True)
matched_structures.drop(columns=['Um2'], inplace=True)

# %% [markdown]
# ### Save the Parsing Test Examples

# %%
save_file = reference_path / 'Parsed Examples.xlsx'
xw.view(matched_structures)
wb = xw.books.active
wb.save(save_file)

# %% [markdown]
# ## Additional Notes and Issues
# %% [markdown]
# ### Possible Typos in the Examples
# %% [markdown]
# #### Spelling Mistakes?
# > **I think there is a spelling mistake in the following example structure names**
# > - `LN_lliac_Int_R` Should it be `LN_Iliac_Int_R`?
# > - `Colon_PTV09` Should it be `Colon_PRV09`?
# %% [markdown]
# #### *Wall* Designator Placement?
#
# **Two different orders are given for designating the _wall_ of an organ:**
# > - `Bladder_Wall` as *Bladder Wall*
# > - `Rectal_Wall` as *Rectal Wall*<br>
# > and<br>
# > - `Wall_Vagina` as *Wall of vagina*
# - The *Wall* designator is critical for evaluating dose, so order should be
# - standardized for this. Based on the discussion in the text, I suspect that
#   `Wall_Vagina` is a typographical error, and the *Wall* designator is intended
#   to follow the primary organ name.
#
# %% [markdown]
# ### Examples not Referenced in the Manuscript
# %% [markdown]
# #### Spatial Indicators
# **The Spatial Indicator (Middle) `M` is used in the examples, but is not mentioned in the document.**
#
# - `Musc_Constrict_M` is the middle pharyngeal constrictor.
# - Include `M` as *Middle* in the table of Spatial Indicators.
# - Consider adding `U` as *Upper* in the table of Spatial Indicators.
# - Using `L` as *Lower* would conflict with `L` as *Left*.
# - Should *Sup* and *Inf* be used instead of *Upper* and *Lower*?
# %% [markdown]
# #### Bolus
# **Bolus with units is given as an example, but bolus is not mentioned in the document.**
# - `Bolus_09mm`
# - Below is a regex pattern based on the the above example

# %%
bolus_pat = ''.join([
    r'^'                   # Beginning of string.
    r'(?P<Bolus>',         # Start of named group Bolus
    r'Bolus'                 # Bolus designator
    r'(?P<BolusThickness>',  # Start of optional named group BolusThickness
    r'[0-9]{1,2}'              # Expansion size as 1 or 2 digits
    r'[cm]m'                   # cm or mm units
    r')?'                    # End of optional BolusThickness group
    r')'                   # End of Bolus group
    r'$'                   # End of string.
    ])

# %% [markdown]
# ### Derived Structures
#
# Derived Structures are discussed in the beginning of the TG263 document
# (see below), but no guidance is explicitly given on nomenclatures for
# derivative structures except for targets cropped back from skin
# (e.g. `PTV-03`) and low-dose PTV volumes that exclude overlapping
# high-dose volumes (`PTV!`).
#
# > **4.4 Derived and Planning Structures**<br>
# > Derivative structures are formed from target or non-target structures,
# > typically using Boolean operations, e.g., intersection (x AND y), combination
# > (x OR y), subtraction (x AND NOT y), and margins (x+1.0). Five institutions
# > indicated that nomenclatures for derivative structures were used to define
# > conditions for evaluating the dose distribution (e.g., OAR contour excluding
# > PTV). Variations in several structures were common (e.g., body-ptv, PTV_EVAL,
# >  eval_PTV), but wide variation was noted for structures involving multiple
# >  concepts (e.g., NS_Brain-PTVs, optCTV-N2R5L_MRT1_ex-3600-v12).
#
# - Parsing instructions have been included for targets cropped from OARs, since
#   this form appears in multiple TG263 examples.
#
# - Derived *Normal Tissue* examples are discussed below.
#
# %% [markdown]
# #### *Normal Tissue* Structures
# **The following 'Normal Tissue' structure names are given in the examples, but not discussed in the text.**
#
# - `E-PTV_Ev05_xxxx` represents all tissue excluding the 5 mm expanded PTV.
#   Generated by subtracting the 5 mm expanded PTV receiving a dose of xxxx cGy
#   from the external contour.
#
# - `E-PTV_xxxx` represents all tissue excluding the PTV. Generated by
#   subtracting the PTV receiving a dose of xxxx cGy from the external contour.
#
# - Should this be a special case structure pattern?
# - Below is a regex pattern based on the the above examples

# %%
normal_t_pat = ''.join([
    r'^'                  # Beginning of string.
    r'(?P<NormalTissue>', # Start of named group NormalTissue
    r'E-'                   # The text 'E-'
    r'(?P<TargetCrop>',     # Start of named group TargetCrop
    r'PTV'                    # The text 'PTV'
    r'(?:'                    # Start of optional non-capturing group
    r'_Ev'                      # the text '_Ev'
    r'(?P<TargetExpansion>',    # Start of named group TargetExpansion
    r'[0-9]{2}'                   # Expansion size as 2 digits
    r')'                        # End of TargetExpansion group
    r')?'                     # End of optional group
    r'(?:'                    # Start of optional non-capturing group
    r'_'                        # '_' s delimiter
    r'(?P<TargetDose>',         # Start of named group TargetDose
    r'[0-9]{4}'                   # Target dose in cGy using 4 digits
    r')'                        # End of TargetDose group
    r')?'                     # End of optional non-capturing group
    r')'                    # End of TargetCrop group
    r')',                 # End of NormalTissue group
    r'$'                  # End of string.
    ])

# %% [markdown]
# ### Name Length Limitations
# - If it is not possible to follow the guidelines and remain within the
#   16-character limit, then preserve the relative ordering but remove underscore
#   characters, progressing from left to right as needed to meet the limit
#   (e.g PTVLiverR_2000x3.) <u>This last resort scenario undermines the
#   use of automated tools</u>.
#
# - It is possible, though not simple, to accommodate this in parsing and
#   checking functions.
# %% [markdown]
# ## Beyond the TG263 Standard
# %% [markdown]
# ### Dose Levels as Structure
# TG263 recommends the use of `IDL` as an Isodose Line e.g. IDL_5000 is the isodose line for 50 Gy.  However this does not appear to be in common use.  Eclipse has a default naming format used when generating isodose lines:
# - `Dose xxx[%]` or `Dose xxxx[cGy]`
#
# ### Dose From Previous Treatment
# - An additional qualifier `_PREV` for isodose lines taken from previous treatment.<br>
# OR<br>
# - `_<name>` (where `<name>` represents the name of a previous plan)
#
# %% [markdown]
# ### Target Motion Modifiers
# No mention is made of 4DCT related target designations.
# The following motion related designators may be useful.
# - `_MIP`
# - `_AVE`
# - `_##%` (Where `##` is the breathing phase the target was contoured on.)
# %% [markdown]
# ### Target Organ Subtraction
# There are times when modified targets are generated by subtracting a critical
# OAR from the initial target volume
# - `-<Organ>`  (Where `<Organ>` represents and valid OAR structure name)
# %% [markdown]
# ### Targets Combined
# - `_Total` as the combined structure for all targets at the same dose level.
# %% [markdown]
# ### Additional Target Type
# - HTV as High Risk target Volume
# %% [markdown]
# ### Target Expansion
# PTVs are sometimes expanded to evaluate the conformality of a plan.
# While not mentioned in the text, the supplied TG263 examples suggest the
# following format:
# - `_Ev##` (where `##` is the uniform expansion size in mm)
#
# %% [markdown]
# ### Target Subgroup
# For optimization purposes a target volume may be divided into subsections
# based on the proximity to hight dose targets.  The TG263 document implies that
# such structures should be prefixed with a `z` or an `_`.
#
# An alternative would be to append a letter suffix to the full target volume to
# designate that it is a subsection.
# - `~a`, `~b` `~c`
#
# The `~` delimiter would have the same meaning of *partial structure* that it
# has for OAR structures.
#
# %% [markdown]
# ### Additional Target Classifiers
# - `_Edema` as target volume based on CNS edema imaging.
# - `_Cavity` as target volume based on a surgical cavity.
# - `_PREOP` as target volume based on pre-operative imaging.
# - `_RES` as target volume for post-op residual disease
#
# %% [markdown]
# ### Additional Target Modifiers
# - `eval_` Target volume explicitly intended for DVH evaluation
# - `opt_` Target volume only intended for optimization. (TG263 recommends that
#   such structures should be prefixed with a `z` or an `_`, but this may be too
#   generic of an indicator.)
#
# %% [markdown]
# ### OAR Multiple Occurrence.
# A suffix noting the occurrence number of the structure.
# - `<Organ>_##` (Where `<Organ>` represents and valid OAR structure name
#   and `##` represents the occurrence number)
#
# %% [markdown]
# ### Bolus Types
# The term *Bolus* may be too generic.
#  Alternative bolus terms may also be needed.
# - `BartsBolus`
# - `WetGauze`
# - `PinkWax`
# %% [markdown]
# ### Foreign Location Reference Objects
# - `BB`
# - `Fiducial`
# - `MARKER`
# - `OVOID`
# - `Wire`
#
# %% [markdown]
# ### Structures for Field Placement Reference
# - `MatchPlane`
# - `Baseline`
# - `FieldEdge`
#
# %% [markdown]
# ### Structures for Correcting Density
# - `Air`
# - `Contrast`
# %% [markdown]
# ### Foreign Objects as Structures
# - `CIED`
# - `Prosthesis`
# - `Implant`
# - `Expander`
# - `Screws`
# - `Staples`
# - `Hardware`
# - `Metal`
# - `Dental`
# - `Anastomosis`
# - `Stoma`

