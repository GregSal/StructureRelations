"""Automated vis-network layout tuning for StructureRelations webapp.

This script runs a full headless browser workflow:
1. Upload DICOM
2. Process structures
3. Render diagram
4. Iteratively tune layout rules
5. Score each candidate using deterministic layout metrics
6. Save screenshots and a JSON report

Optionally, it can write the best discovered layout rules back to
``diagram_settings.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import subprocess
import time
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.error import URLError
from urllib.request import Request, urlopen

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


LOGGER = logging.getLogger('webapp.layout_tuner')


@dataclass
class CandidateResult:
    """Represents one layout candidate evaluation."""

    name: str
    rules: Dict[str, Any]
    metrics: Dict[str, float]
    screenshot_path: str = ''


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def _load_layout_rules(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['diagram_options']['diagram_layout']['layout_rules']


def _save_layout_rules(config_path: Path, layout_rules: Dict[str, Any]) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['diagram_options']['diagram_layout']['layout_rules'] = layout_rules
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def _chrome_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(4)
    driver.command_executor._client_config.timeout = 300
    return driver


def _wait_for_stage(driver: webdriver.Chrome, stage_id: str, timeout: int = 240) -> None:
    WebDriverWait(driver, timeout).until(
        EC.visibility_of_element_located((By.ID, stage_id))
    )


def _upload_and_process(
    driver: webdriver.Chrome,
    base_url: str,
    dicom_path: Path,
    timeout: int,
) -> None:
    api_base_url = base_url.split('?', 1)[0].rstrip('/')
    LOGGER.info('Opening %s', base_url)
    driver.get(base_url)

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, 'fileInput'))
    )
    driver.find_element(By.ID, 'fileInput').send_keys(str(dicom_path.resolve()))

    # Either manual selection stage or auto-processing mode.
    try:
        _wait_for_stage(driver, 'stage-selection', timeout=20)
        LOGGER.info('Stage selection visible; starting processing')
        driver.find_element(By.ID, 'processBtn').click()
    except TimeoutException:
        LOGGER.info('Selection stage not shown; expecting auto-processing path')

    session_id = WebDriverWait(driver, 30).until(
        lambda d: d.execute_script(
            'return window.app && window.app.sessionId ? window.app.sessionId : null;'
        )
    )
    job_status_url = f'{api_base_url}/api/jobs/{session_id}/status'
    LOGGER.info('Tracking backend job for session=%s url=%s', session_id, job_status_url)
    deadline = time.time() + timeout
    poll_count = 0
    last_status = None
    while time.time() < deadline:
        poll_count += 1
        try:
            with urlopen(job_status_url, timeout=15) as response:
                payload = json.load(response)
        except (URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError, Exception):
            LOGGER.debug('Transient status poll failure for %s', job_status_url, exc_info=True)
            time.sleep(0.5)
            continue

        status = str(payload.get('status', '')).lower()
        if status != last_status:
            LOGGER.info(
                'Job poll %s for session=%s -> status=%s stage=%s progress=%s message=%s',
                poll_count,
                session_id,
                status,
                payload.get('stage', ''),
                payload.get('progress', ''),
                payload.get('message', ''),
            )
            last_status = status
        if status == 'completed':
            break
        if status in {'failed', 'cancelled'}:
            raise RuntimeError(f'Job ended early with status={status}: {payload!r}')
        time.sleep(0.5)
    else:
        raise TimeoutError(
            f'Processing did not complete within {timeout}s; '
            f'last_status={last_status!r}; session_id={session_id}; '
            f'job_status_url={job_status_url}'
        )

    session_data_path = (
        Path(__file__).resolve().parents[2] / 'webapp_sessions' / f'{session_id}.pkl'
    )
    if not session_data_path.exists():
        raise FileNotFoundError(f'Session data not found: {session_data_path}')

    with open(session_data_path, 'rb') as f:
        session_data = pickle.load(f)

    hidden_rois = {int(roi) for roi in getattr(session_data, 'hidden_rois', []) or []}
    structure_set = getattr(session_data, 'structure_set', None)
    if structure_set is None:
        raise RuntimeError(f'Session has no structure set: {session_id}')

    summary_df = structure_set.summary()
    visible_rois = [
        int(roi)
        for roi in summary_df['ROI'].tolist()
        if int(roi) not in hidden_rois
    ]
    if not visible_rois:
        visible_rois = [int(roi) for roi in summary_df['ROI'].tolist()]

    LOGGER.info(
        'Requesting diagram payload for %s visible rois (session=%s)',
        len(visible_rois),
        session_id,
    )
    diagram_request = {
        'session_id': session_id,
        'row_rois': visible_rois,
        'col_rois': visible_rois,
        'show_disjoint': False,
        'show_unknown': False,
        'logical_relations_mode': 'limited',
    }
    req = Request(
        f'{api_base_url}/api/diagram',
        data=json.dumps(diagram_request).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(req, timeout=60) as response:
        diagram_data = json.load(response)
    LOGGER.info(
        'Diagram payload received: %s node(s), %s edge(s)',
        len(diagram_data.get('nodes', [])),
        len(diagram_data.get('edges', [])),
    )

    LOGGER.info('Rendering diagram in browser')
    driver.execute_script(
        '''const data = arguments[0];
const snapshot = window.app && window.app._captureDiagramPositionSnapshot
  ? window.app._captureDiagramPositionSnapshot()
  : null;
window.app.renderDiagram(data, snapshot);''',
        diagram_data,
    )
    LOGGER.info('Browser render call returned')

    WebDriverWait(driver, 60).until(
        lambda d: d.execute_script(
            'return Boolean(window.app && window.app.network && '
            'window.app.latestDiagramData);'
        )
    )
    LOGGER.info('Browser diagram state is ready')


def _wait_for_server(base_url: str, timeout_s: int = 45) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{base_url.rstrip('/')}/api/config/symbols"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if 200 <= int(response.status) < 500:
                    return
        except URLError:
            time.sleep(0.5)
            continue
        except OSError:
            time.sleep(0.5)
            continue
    raise TimeoutError(f'Server did not become ready within {timeout_s}s: {base_url}')


def _start_server(
    python_executable: str,
    project_root: Path,
    host: str,
    port: int,
    module: str,
) -> subprocess.Popen[str]:
    command = [
        python_executable,
        '-m',
        'uvicorn',
        module,
        '--host',
        host,
        '--port',
        str(port),
    ]
    LOGGER.info('Starting server: %s', ' '.join(command))
    process = subprocess.Popen(
        command,
        cwd=str(project_root / 'src'),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process


def _stop_server(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _js_eval_metrics_script() -> str:
    # Score components are chosen to prioritize crossing minimization first,
    # then improve node spacing and edge length consistency.
    return r'''
const layoutRules = arguments[0];
if (!window.app || !window.app.latestDiagramData) {
  return {ok: false, error: 'App or diagram data is unavailable'};
}
const data = window.app.latestDiagramData;
const positions = window.app.computeDeterministicLayout(
  data.nodes,
  data.edges,
  layoutRules
);
const crossingParticipants = window.app.detectEdgeCrossings(
  positions,
  data.edges,
  true
);
const crossings = crossingParticipants.size;

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

const candidateEdges = data.edges.filter(e => e.layout_candidate !== false);
const edgeLengths = [];
for (const edge of candidateEdges) {
  const p1 = positions[String(edge.from_node)];
  const p2 = positions[String(edge.to_node)];
  if (!p1 || !p2) continue;
  edgeLengths.push(dist(p1, p2));
}

const meanEdge = edgeLengths.length
  ? edgeLengths.reduce((a, b) => a + b, 0) / edgeLengths.length
  : 0;
const edgeStd = edgeLengths.length
  ? Math.sqrt(
      edgeLengths.reduce((sum, value) => {
        const delta = value - meanEdge;
        return sum + (delta * delta);
      }, 0) / edgeLengths.length
    )
  : 0;

const nodeIds = data.nodes.map(n => String(n.id));
let minNodeDistance = Number.POSITIVE_INFINITY;
for (let i = 0; i < nodeIds.length; i++) {
  for (let j = i + 1; j < nodeIds.length; j++) {
    const p1 = positions[nodeIds[i]];
    const p2 = positions[nodeIds[j]];
    if (!p1 || !p2) continue;
    const d = dist(p1, p2);
    if (d < minNodeDistance) minNodeDistance = d;
  }
}
if (!Number.isFinite(minNodeDistance)) minNodeDistance = 0;

const spacingPenalty = minNodeDistance > 0 ? (100 / minNodeDistance) : 1000;
const score = (crossings * 1000) + (edgeStd * 2.0) + (spacingPenalty * 5.0);

window.app.diagramOptions.layout.layout_rules = layoutRules;
window.app.renderDiagram(data);

return {
  ok: true,
  crossings,
  edge_std: edgeStd,
  edge_mean: meanEdge,
  min_node_distance: minNodeDistance,
  spacing_penalty: spacingPenalty,
  score,
  node_count: data.nodes.length,
  edge_count: data.edges.length,
};
'''


def _apply_and_measure(
    driver: webdriver.Chrome,
    rules: Dict[str, Any],
) -> Dict[str, float]:
    payload = driver.execute_script(_js_eval_metrics_script(), rules)
    if not payload.get('ok', False):
        raise RuntimeError(str(payload.get('error', 'Unknown browser evaluation error')))
    return {
        'score': float(payload['score']),
        'crossings': float(payload['crossings']),
        'edge_std': float(payload['edge_std']),
        'edge_mean': float(payload['edge_mean']),
        'min_node_distance': float(payload['min_node_distance']),
        'spacing_penalty': float(payload['spacing_penalty']),
        'node_count': float(payload['node_count']),
        'edge_count': float(payload['edge_count']),
    }


def _capture_diagram_screenshot(driver: webdriver.Chrome, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    element = driver.find_element(By.ID, 'networkDiagram')
    element.screenshot(str(output_path))


def _candidate_variants(
    base_rules: Dict[str, Any],
    full_search: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    variants: List[Tuple[str, Dict[str, Any]]] = []
    variants.append(('baseline', copy.deepcopy(base_rules)))

    def add_variant(name: str, path: List[str], deltas: List[float]) -> None:
        node = None
        for delta in deltas:
            candidate = copy.deepcopy(base_rules)
            node = candidate
            for key in path[:-1]:
                node = node[key]
            leaf = path[-1]
            original = float(node[leaf])
            updated = max(0.0, original + delta)
            if leaf in {'curvature_threshold'}:
                updated = max(0.0, min(1.0, updated))
            if leaf in {
                'link_distance',
                'node_padding',
                'gap',
                'base_roundness',
                'crossing_roundness',
            }:
                node[leaf] = round(updated, 4)
            else:
                node[leaf] = int(round(updated))
            suffix = f"{delta:+g}".replace('.', 'p').replace('+', 'plus').replace('-', 'minus')
            variants.append((f'{name}_{suffix}', candidate))

    # Fast default profile: focus on the parameters that most often move score.
    add_variant('link_distance', ['link_distance'], [-15, 15])
    add_variant('node_padding', ['node_padding'], [-3, 3])
    add_variant('flow_gap', ['flow', 'gap'], [-12, 12])
    add_variant('opt_link_distance', ['opt_grouping', 'link_distance'], [-15, -8, 8])

    if full_search:
        add_variant('link_distance_ext', ['link_distance'], [-25, 25])
        add_variant('node_padding_ext', ['node_padding'], [-6, 6])
        add_variant('flow_gap_ext', ['flow', 'gap'], [-25, 25])
        add_variant('side_gap', ['side_bias', 'gap'], [-35, -20, 20, 35])
        add_variant('opt_link_distance_ext', ['opt_grouping', 'link_distance'], [15])
        add_variant('target_padding', ['target_grouping', 'padding'], [-10, -6, 6, 10])
        add_variant(
            'curvature_threshold',
            ['edge_routing', 'curvature_threshold'],
            [-0.2, -0.1, 0.1, 0.2],
        )
        add_variant(
            'crossing_roundness',
            ['edge_routing', 'crossing_roundness'],
            [-0.12, -0.06, 0.06, 0.12],
        )

    return variants


def _run_search(
    driver: webdriver.Chrome,
    base_rules: Dict[str, Any],
    screenshots_dir: Path,
    full_search: bool,
    screenshot_mode: str,
) -> List[CandidateResult]:
    results: List[CandidateResult] = []
    for name, rules in _candidate_variants(base_rules, full_search=full_search):
        metrics = _apply_and_measure(driver, rules)
        shot = ''
        if screenshot_mode == 'all':
            shot_path = screenshots_dir / f'{name}.png'
            _capture_diagram_screenshot(driver, shot_path)
            shot = str(shot_path)
        LOGGER.info(
            'Candidate %-24s score=%8.2f crossings=%3.0f edge_std=%7.2f min_node_dist=%7.2f',
            name,
            metrics['score'],
            metrics['crossings'],
            metrics['edge_std'],
            metrics['min_node_distance'],
        )
        results.append(
            CandidateResult(
                name=name,
                rules=rules,
                metrics=metrics,
                screenshot_path=shot,
            )
        )
    return results


def _capture_key_screenshots(
    driver: webdriver.Chrome,
    baseline: CandidateResult,
    best: CandidateResult,
    screenshots_dir: Path,
) -> None:
    seen: set[str] = set()
    for result in (baseline, best):
        if result.name in seen:
            continue
        _apply_and_measure(driver, result.rules)
        shot_path = screenshots_dir / f'{result.name}.png'
        _capture_diagram_screenshot(driver, shot_path)
        result.screenshot_path = str(shot_path)
        seen.add(result.name)


def _select_best(results: List[CandidateResult]) -> CandidateResult:
    return min(results, key=lambda item: item.metrics['score'])


def _write_report(
    output_path: Path,
    baseline: CandidateResult,
    best: CandidateResult,
    all_results: List[CandidateResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_epoch_s': int(time.time()),
        'baseline': {
            'name': baseline.name,
            'metrics': baseline.metrics,
            'rules': baseline.rules,
            'screenshot_path': baseline.screenshot_path,
        },
        'best': {
            'name': best.name,
            'metrics': best.metrics,
            'rules': best.rules,
            'screenshot_path': best.screenshot_path,
        },
        'candidates': [
            {
                'name': result.name,
                'metrics': result.metrics,
                'rules': result.rules,
                'screenshot_path': result.screenshot_path,
            }
            for result in sorted(all_results, key=lambda item: item.metrics['score'])
        ],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Tune vis-network layout rules automatically using headless Selenium.'
    )
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--dicom-path', required=True)
    parser.add_argument(
        '--diagram-config-path',
        default=str(Path(__file__).parent / 'config' / 'diagram_settings.json'),
    )
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).parent / 'layout_tuner_output'),
    )
    parser.add_argument('--timeout-seconds', type=int, default=300)
    parser.add_argument('--server-start-timeout-seconds', type=int, default=45)
    parser.add_argument('--apply-best', action='store_true')
    parser.add_argument(
        '--filter-path',
        default=str(Path(__file__).resolve().parents[2] / 'tests' / 'Pros_Equal_target_rules.json'),
    )
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--server-host', default='127.0.0.1')
    parser.add_argument('--server-port', type=int, default=8000)
    parser.add_argument('--server-module', default='webapp.main:app')
    parser.add_argument(
        '--python-executable',
        default=os.environ.get('SR_LAYOUT_TUNER_PYTHON', os.sys.executable),
    )
    parser.add_argument('--no-headless', action='store_true')
    parser.add_argument(
        '--full-search',
        action='store_true',
        help='Evaluate the full candidate set (slower but broader sweep).',
    )
    parser.add_argument(
        '--screenshot-mode',
        choices=['none', 'key', 'all'],
        default='key',
        help='Screenshot policy: none, key candidates (baseline+best), or all.',
    )
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.verbose)

    dicom_path = Path(args.dicom_path)
    if not dicom_path.exists() or not dicom_path.is_file():
        raise FileNotFoundError(f'DICOM path does not exist: {dicom_path}')

    config_path = Path(args.diagram_config_path)
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f'Diagram config path does not exist: {config_path}')

    output_dir = Path(args.output_dir)
    screenshots_dir = output_dir / 'screenshots'
    report_path = output_dir / 'layout_tuning_report.json'

    original_rules = _load_layout_rules(config_path)
    filter_path = Path(args.filter_path)
    if not filter_path.exists() or not filter_path.is_file():
        fallback_filter_path = Path(__file__).resolve().parents[2] / 'tests' / 'Pros_Equal_target_rules.json'
        if fallback_filter_path.exists() and fallback_filter_path.is_file():
            filter_path = fallback_filter_path
        else:
            raise FileNotFoundError(f'Filter path does not exist: {filter_path}')

    driver: Optional[webdriver.Chrome] = None
    server_process: Optional[subprocess.Popen[str]] = None
    try:
        base_url = (
            f'{args.base_url.rstrip("/")}/?filter_path={quote(str(filter_path.resolve()))}'
        )
        if args.start_server:
            project_root = Path(__file__).resolve().parents[2]
            server_process = _start_server(
                python_executable=str(args.python_executable),
                project_root=project_root,
                host=str(args.server_host),
                port=int(args.server_port),
                module=str(args.server_module),
            )
            _wait_for_server(
                base_url=args.base_url,
                timeout_s=int(args.server_start_timeout_seconds),
            )

        driver = _chrome_driver(headless=not args.no_headless)
        _upload_and_process(
            driver=driver,
            base_url=base_url,
            dicom_path=dicom_path,
            timeout=args.timeout_seconds,
        )

        results = _run_search(
            driver=driver,
            base_rules=original_rules,
            screenshots_dir=screenshots_dir,
            full_search=bool(args.full_search),
            screenshot_mode=str(args.screenshot_mode),
        )
        baseline = next(result for result in results if result.name == 'baseline')
        best = _select_best(results)

        if args.screenshot_mode == 'key':
            _capture_key_screenshots(
                driver=driver,
                baseline=baseline,
                best=best,
                screenshots_dir=screenshots_dir,
            )

        _write_report(
            output_path=report_path,
            baseline=baseline,
            best=best,
            all_results=results,
        )

        LOGGER.info(
            'Best candidate: %s (score=%.2f crossings=%.0f)',
            best.name,
            best.metrics['score'],
            best.metrics['crossings'],
        )
        LOGGER.info('Report written to %s', report_path)

        if args.apply_best:
            _save_layout_rules(config_path, best.rules)
            LOGGER.info('Applied best rules to %s', config_path)

    finally:
        if driver is not None:
            driver.quit()
        if server_process is not None:
            _stop_server(server_process)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
