"""Focused API tests for diagram edge metric actions."""

from types import SimpleNamespace
import math

import pytest
from fastapi.testclient import TestClient

from metrics import MetricCalculatorRegistry
from webapp.main import app
import webapp.main as web_main
from webapp.session_manager import SessionData, SessionManager


class FakeMetricStructureSet:
    def __init__(self, relation_type, stored_pair=(1, 2)):
        self.relationship = SimpleNamespace(
            relationship_type=SimpleNamespace(relation_type=relation_type),
            metrics=None,
        )
        self.stored_pair = stored_pair
        self.calculation_calls = []
        self.unit = 'cm'

    def get_relationship(self, roi_a, roi_b):
        if (roi_a, roi_b) == self.stored_pair:
            return self.relationship
        return None

    def calculate_metric(self, roi_a, roi_b, metric_name):
        self.calculation_calls.append((roi_a, roi_b, metric_name))
        if self.relationship.metrics is None:
            self.relationship.metrics = SimpleNamespace(
                margin=None,
                volume_ratio=None,
            )
        if metric_name == 'minimum_margins':
            result = SimpleNamespace(
                minimum_margin=1.234,
                orthogonal_margins={
                    'x_neg': 0.5, 'x_pos': 0.25,
                    'y_neg': 1.0, 'y_pos': 1.5,
                    'z_neg': math.nan, 'z_pos': 0.3,
                },
            )
            self.relationship.metrics.margin = result
        elif metric_name == 'overlapping_volume_ratio':
            if self.relationship.metrics.volume_ratio is None:
                self.relationship.metrics.volume_ratio = SimpleNamespace(
                    overlapping_ratio=None,
                    non_overlapping_ratio=None,
                )
            result = self.relationship.metrics.volume_ratio
            result.overlapping_ratio = 0.45678
        elif metric_name == 'non_overlapping_volume_ratio':
            if self.relationship.metrics.volume_ratio is None:
                self.relationship.metrics.volume_ratio = SimpleNamespace(
                    overlapping_ratio=None,
                    non_overlapping_ratio=None,
                )
            result = self.relationship.metrics.volume_ratio
            result.non_overlapping_ratio = 0.32123
        else:
            raise AssertionError(f'Unexpected calculator: {metric_name}')
        return result


def _make_client(monkeypatch, tmp_path, structure_set):
    manager = SessionManager(sessions_dir=tmp_path / 'sessions')
    manager.save_session(
        'edge-metric-session',
        SessionData(dicom_file_path='dummy.dcm', structure_set=structure_set),
    )
    monkeypatch.setattr(web_main, 'session_manager', manager)
    return TestClient(app), manager


def _request(from_node, to_node, metric_name=None, relation_type='CONTAINS'):
    payload = {
        'session_id': 'edge-metric-session',
        'from_node': from_node,
        'to_node': to_node,
        'relation_type': relation_type,
    }
    if metric_name is not None:
        payload['metric_name'] = metric_name
    return payload


def test_edge_metric_options_only_include_compatible_calculators(
    monkeypatch, tmp_path
):
    structure_set = FakeMetricStructureSet('CONTAINS')
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    response = client.post(
        '/api/diagram/edge-metric', json=_request(1, 2)
    )

    assert response.status_code == 200
    options = response.json()['metrics']
    names = {item['name'] for item in options}
    calculators = MetricCalculatorRegistry.get_all_calculators()
    expected = {
        name
        for name, spec in web_main._DIAGRAM_METRICS.items()
        if spec.calculator in calculators
        and calculators[spec.calculator].is_applicable(structure_set.relationship)
    }
    assert names == expected
    assert 'minimum_distance' not in names
    paths = {item['name']: item['menu_path'] for item in options}
    assert paths['orthogonal_margins'] == ['Margins', 'Orthogonal']
    assert paths['minimum_margin'] == ['Margins', 'Minimum']
    assert paths['overlapping_volume_ratio'] == ['Volume Ratio', 'Overlapping']


def test_margin_views_share_one_calculation(monkeypatch, tmp_path):
    structure_set = FakeMetricStructureSet('CONTAINS')
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    orthogonal = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, 'orthogonal_margins'),
    )
    minimum = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, 'minimum_margin'),
    )

    assert orthogonal.status_code == minimum.status_code == 200
    payload = orthogonal.json()
    assert payload['value'] is None
    assert payload['unit'] == 'cm'
    assert [item['value'] for item in payload['values']] == [
        '0.50', '0.25', '1.00', '1.50', 'N/A', '0.30',
    ]
    assert [item['direction'] for item in payload['values']] == [
        'x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos',
    ]
    config = web_main.get_metrics_config()
    if config.use_anatomical_labels:
        assert [item['label'] for item in payload['values']] == [
            config.anatomical_labels[direction]
            for direction in config.orthogonal_directions
        ]
    assert minimum.json()['value'] == '1.23'
    assert structure_set.calculation_calls == [(1, 2, 'minimum_margins')]


def test_edge_metric_is_calculated_once_and_persisted(monkeypatch, tmp_path):
    structure_set = FakeMetricStructureSet('CONTAINS')
    client, manager = _make_client(monkeypatch, tmp_path, structure_set)
    payload = _request(1, 2, 'minimum_margin')

    first = client.post('/api/diagram/edge-metric', json=payload)
    second = client.post('/api/diagram/edge-metric', json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()['value'] == '1.23'
    assert first.json()['unit'] == 'cm'
    assert second.json()['value'] == '1.23'
    assert structure_set.calculation_calls == [(1, 2, 'minimum_margins')]

    persisted = SessionManager(sessions_dir=manager.sessions_dir).load_session(
        'edge-metric-session'
    )
    assert persisted.structure_set.relationship.metrics.margin.minimum_margin == 1.234


def test_symmetric_edge_calculates_using_stored_relationship_direction(
    monkeypatch, tmp_path
):
    structure_set = FakeMetricStructureSet('OVERLAPS')
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    response = client.post(
        '/api/diagram/edge-metric',
        json=_request(
            2,
            1,
            'overlapping_volume_ratio',
            relation_type='OVERLAPS',
        ),
    )

    assert response.status_code == 200
    assert response.json()['value'] == '45.7'
    assert response.json()['unit'] == '%'
    assert structure_set.calculation_calls == [
        (1, 2, 'overlapping_volume_ratio')
    ]


def test_volume_ratio_calculators_reuse_their_own_shared_result_field(
    monkeypatch, tmp_path
):
    structure_set = FakeMetricStructureSet('CONTAINS')
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    overlap = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, 'overlapping_volume_ratio'),
    )
    non_overlap = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, 'non_overlapping_volume_ratio'),
    )
    overlap_again = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, 'overlapping_volume_ratio'),
    )

    assert overlap.json()['value'] == '45.7'
    assert non_overlap.json()['value'] == '32.1'
    assert overlap_again.json()['value'] == '45.7'
    assert structure_set.calculation_calls == [
        (1, 2, 'overlapping_volume_ratio'),
        (1, 2, 'non_overlapping_volume_ratio'),
    ]


def test_directional_relationship_does_not_fall_back_to_reverse_pair(
    monkeypatch, tmp_path
):
    structure_set = FakeMetricStructureSet(
        'CONTAINS',
        stored_pair=(2, 1),
    )
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    response = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, relation_type='CONTAINS'),
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ('metric_name', 'expected_status'),
    [('minimum_distance', 400), ('not_a_metric', 404)],
)
def test_edge_metric_rejects_incompatible_or_unknown_calculators(
    monkeypatch, tmp_path, metric_name, expected_status
):
    structure_set = FakeMetricStructureSet('CONTAINS')
    client, _ = _make_client(monkeypatch, tmp_path, structure_set)

    response = client.post(
        '/api/diagram/edge-metric',
        json=_request(1, 2, metric_name),
    )

    assert response.status_code == expected_status
    assert structure_set.calculation_calls == []
