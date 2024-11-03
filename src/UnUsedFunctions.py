import pandas as pd


def build_slice_spacing_table(slice_table, shift_direction=-1)->pd.DataFrame:
    def slice_spacing(contour):
        # Index is the slice position of all slices in the image set
        # Columns are structure IDs
        # Values are the distance (INF) to the next contour
        inf = contour.dropna().index.min()
        sup = contour.dropna().index.max()
        contour_range = (contour.index <= sup) & (contour.index >= inf)
        slices = contour.loc[contour_range].dropna().index.to_series()
        gaps = slices.shift(shift_direction) - slices
        return gaps
    # Find distance between slices with contours
    def get_slices(structure: pd.Series):
        used_slices = structure.dropna().index.to_series()
        return used_slices

    contour_slices = slice_table.apply(get_slices)
    slice_spacing_data = contour_slices.apply(slice_spacing)
    return slice_spacing_data
