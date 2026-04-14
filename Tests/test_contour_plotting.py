import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from contour_plotting import plot_ab, plot_roi_slice


class _DummyRegionSlice:
    def __init__(self, polygon):
        self._polygon = polygon
        self.is_empty = polygon.is_empty

    def merge_regions(self):
        return self._polygon


class _DummyStructure:
    def __init__(self, name, polygon):
        self.name = name
        self._region_slice = _DummyRegionSlice(polygon)

    def get_slice(self, _slice_index):
        return self._region_slice


class _DummyStructureSet:
    def __init__(self, structures):
        self.structures = structures
        self.dicom_structure_file = None


def test_plot_ab_inverts_y_axis():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    fig, ax = plt.subplots()
    result_ax = plot_ab(poly_a, poly_b, add_axis=False, axes=ax)

    assert result_ax.yaxis_inverted()
    plt.close(fig)


def test_plot_roi_slice_inverts_y_axis_for_contour_mode():
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    structure_set = _DummyStructureSet({1: _DummyStructure('A', poly)})

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1],
        axes=ax,
        add_axis=False,
        plot_mode='contour',
    )

    assert result_ax.yaxis_inverted()
    plt.close(fig)


def test_plot_roi_slice_draws_axis_guides_when_enabled():
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    structure_set = _DummyStructureSet({1: _DummyStructure('A', poly)})

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1],
        axes=ax,
        add_axis=True,
        plot_mode='contour',
    )

    axis_lines = [line for line in result_ax.lines if line.get_linestyle() == '--']
    assert len(axis_lines) == 2
    plt.close(fig)


def test_plot_roi_slice_supports_single_structure_relationship_mode():
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    structure_set = _DummyStructureSet({1: _DummyStructure('A', poly)})

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1],
        axes=ax,
        add_axis=False,
        plot_mode='relationship',
        relationship_overlay='single_structure',
    )

    legend = result_ax.get_legend()
    assert result_ax.yaxis_inverted()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ['A']
    plt.close(fig)


def test_plot_roi_slice_supports_third_structure_outline_overlay():
    poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    poly_b = Polygon([(2, 1), (6, 1), (6, 5), (2, 5)])
    poly_c = Polygon([(1, 2), (5, 2), (5, 6), (1, 6)])
    structure_set = _DummyStructureSet(
        {
            1: _DummyStructure('A', poly_a),
            2: _DummyStructure('B', poly_b),
            3: _DummyStructure('C', poly_c),
        }
    )

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1, 2, 3],
        axes=ax,
        add_axis=False,
        plot_mode='relationship',
        relationship_overlay='third_structure',
    )

    labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert 'C' in labels
    plt.close(fig)


def test_plot_roi_slice_supports_intersection_vs_c_overlay():
    poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    poly_b = Polygon([(2, 0), (6, 0), (6, 4), (2, 4)])
    poly_c = Polygon([(1, 1), (3, 1), (3, 5), (1, 5)])
    structure_set = _DummyStructureSet(
        {
            1: _DummyStructure('A', poly_a),
            2: _DummyStructure('B', poly_b),
            3: _DummyStructure('C', poly_c),
        }
    )

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1, 2, 3],
        axes=ax,
        add_axis=False,
        plot_mode='relationship',
        relationship_overlay='intersection_vs_c',
    )

    labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert 'A AND B only' in labels
    assert 'C only' in labels
    plt.close(fig)


def test_plot_roi_slice_supports_all_outlines_overlay():
    poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    poly_b = Polygon([(2, 1), (6, 1), (6, 5), (2, 5)])
    poly_c = Polygon([(1, 2), (5, 2), (5, 6), (1, 6)])
    poly_d = Polygon([(3, -1), (7, -1), (7, 3), (3, 3)])
    structure_set = _DummyStructureSet(
        {
            1: _DummyStructure('A', poly_a),
            2: _DummyStructure('B', poly_b),
            3: _DummyStructure('C', poly_c),
            4: _DummyStructure('D', poly_d),
        }
    )

    fig, ax = plt.subplots()
    result_ax = plot_roi_slice(
        structure_set=structure_set,
        slice_index=0.0,
        roi_list=[1, 2, 3, 4],
        axes=ax,
        add_axis=False,
        plot_mode='relationship',
        relationship_overlay='all_outlines',
    )

    labels = [text.get_text() for text in result_ax.get_legend().get_texts()]
    assert 'C' in labels
    assert 'D' in labels
    plt.close(fig)
