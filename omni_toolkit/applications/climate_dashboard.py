"""
Interactive climate regime dashboard using Plotly.

Generates a self-contained HTML page with:
1. Geodesic distance timeline with event markers and threshold bands
2. 2D manifold embedding (PGA projection of covariance trajectory)
3. Animated correlation heatmap
4. V+/V- decomposition gauge for each transition
5. Regime state timeline

Usage:
    from omni_toolkit.applications.climate_dashboard import build_dashboard

    # From detection results
    build_dashboard(results, output="dashboard.html")

    # Full pipeline from scratch
    build_dashboard_from_data(data, output="dashboard.html")

    # Static version (no Plotly dependency, pure D3.js)
    build_static_dashboard(results, output="dashboard.html")
"""
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .climate_analysis import (
    ClimateRegimeDetector, ClimateDataLoader, ClimateData,
    VDecomposition, DetectionResults, ENSO_EVENTS, VOLCANIC_EVENTS,
    PDO_SHIFTS, AMO_SHIFTS,
)
from .spd_ml import SPDLayer, _matrix_log, _symmetrize


# =====================================================================
# Data preparation for visualization
# =====================================================================

def prepare_dashboard_data(results: DetectionResults,
                           vdecomp: Optional[VDecomposition] = None) -> Dict:
    """Prepare all data needed for the dashboard.

    Returns a JSON-serializable dict with all visualization data.
    """
    # Timeline data
    dates = results.center_dates[1:]  # align with distances
    timeline = {
        "dates": [f"{y}-{m:02d}" for y, m in dates],
        "geodesic": results.geodesic_distances.tolist(),
        "euclidean": results.euclidean_distances.tolist(),
        "mean_geo": float(np.mean(results.geodesic_distances)),
        "std_geo": float(np.std(results.geodesic_distances)),
    }

    # Transition markers
    transitions = []
    for tr in results.transitions:
        transitions.append({
            "date": f"{tr.date[0]}-{tr.date[1]:02d}",
            "distance": tr.geodesic_distance,
            "sigma": tr.sigma_above,
            "category": tr.category,
            "event": tr.matched_event or "Unknown",
        })

    # Known event markers
    events = []
    start_yr = results.center_dates[0][0]
    end_yr = results.center_dates[-1][0]
    for ey, em, etype, estr in ENSO_EVENTS:
        if start_yr <= ey <= end_yr:
            events.append({
                "date": f"{ey}-{em:02d}",
                "type": "enso",
                "label": f"{'El Niño' if etype == 'nino' else 'La Niña'} ({estr})",
            })
    for vy, vm, vname in VOLCANIC_EVENTS:
        if start_yr <= vy <= end_yr:
            events.append({
                "date": f"{vy}-{vm:02d}",
                "type": "volcanic",
                "label": vname,
            })
    for py, pm, ptype in PDO_SHIFTS:
        if start_yr <= py <= end_yr:
            events.append({
                "date": f"{py}-{pm:02d}",
                "type": "pdo",
                "label": f"PDO → {ptype}",
            })

    # 2D manifold embedding via log-Euclidean PCA
    covs = results.covariances
    d = covs.shape[1]
    log_covs = np.array([_matrix_log(c) for c in covs])
    # Flatten to [N, d*d]
    flat = log_covs.reshape(len(covs), -1)
    flat_centered = flat - flat.mean(axis=0)
    # SVD for top 2 components
    U, S, Vt = np.linalg.svd(flat_centered, full_matrices=False)
    coords_2d = U[:, :2] * S[:2]
    # Also 3D
    coords_3d = U[:, :3] * S[:3] if len(S) >= 3 else np.column_stack([coords_2d, np.zeros(len(covs))])

    embedding = {
        "x": coords_2d[:, 0].tolist(),
        "y": coords_2d[:, 1].tolist(),
        "z": coords_3d[:, 2].tolist() if coords_3d.shape[1] > 2 else [0.0] * len(covs),
        "dates": [f"{y}-{m:02d}" for y, m in results.center_dates],
        "variance_explained": (S[:3] ** 2 / (S ** 2).sum() * 100).tolist()[:3],
    }

    # Correlation matrix snapshots (every 12 months for animation)
    corr_snapshots = []
    for i in range(0, len(covs), 12):
        C = covs[i]
        diag = np.sqrt(np.diag(C))
        diag = np.where(diag > 0, diag, 1e-10)
        corr = C / np.outer(diag, diag)
        np.fill_diagonal(corr, 1.0)
        corr_snapshots.append({
            "date": f"{results.center_dates[i][0]}-{results.center_dates[i][1]:02d}",
            "matrix": corr.tolist(),
        })

    # V+/V- decomposition
    v_decomp = []
    if vdecomp is not None:
        for dec in vdecomp.decompose_all_transitions():
            v_decomp.append({
                "date": f"{dec['date'][0]}-{dec['date'][1]:02d}",
                "category": dec["category"],
                "event": dec.get("matched_event", "Unknown"),
                "norm_plus": dec["norm_plus"],
                "norm_minus": dec["norm_minus"],
                "volume_fraction": dec["volume_fraction"],
                "ratio": dec["ratio"],
            })

    return {
        "timeline": timeline,
        "transitions": transitions,
        "events": events,
        "embedding": embedding,
        "correlations": corr_snapshots,
        "v_decomposition": v_decomp,
        "index_names": results.index_names,
        "validation": {
            "enso_recall": results.enso_recall,
            "enso_precision": results.enso_precision,
            "mean_lead_time": results.mean_lead_time,
        },
        "config": {
            "window": results.window,
            "threshold": results.threshold,
            "n_indices": len(results.index_names),
        },
    }


# =====================================================================
# Static HTML dashboard (no server, pure JS)
# =====================================================================

def build_static_dashboard(results: DetectionResults,
                           output: str = "climate_dashboard.html",
                           title: str = "Riemannian Climate Regime Monitor"):
    """Build a self-contained HTML dashboard with embedded D3.js/Plotly.

    The output is a single HTML file with all data embedded as JSON.
    No server needed — just open in a browser.
    """
    vd = VDecomposition(results)

    # Prepare data
    data = prepare_dashboard_data(results, vd)
    data_json = json.dumps(data, indent=None)

    html = _DASHBOARD_TEMPLATE.replace("/*__DATA__*/", data_json)
    html = html.replace("__TITLE__", title)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html)

    return output


def build_dashboard_from_data(data: ClimateData,
                              output: str = "climate_dashboard.html",
                              **detector_kwargs) -> str:
    """Full pipeline: data → detection → validation → dashboard."""
    detector = ClimateRegimeDetector(**detector_kwargs)
    results = detector.detect(data)
    detector.validate(results)
    return build_static_dashboard(results, output)


# =====================================================================
# HTML template with embedded Plotly.js
# =====================================================================

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922;
    --purple: #bc8cff; --pink: #f778ba;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
}
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.8em; margin-bottom: 4px; color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 20px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-full { grid-column: 1 / -1; }
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
}
.card h2 { font-size: 1.1em; color: var(--muted); margin-bottom: 12px;
    text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
.stats { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 20px; }
.stat { text-align: center; }
.stat .value { font-size: 2em; font-weight: 700; color: var(--accent); }
.stat .label { font-size: 0.8em; color: var(--muted); }
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600;
}
.badge-enso { background: #1f3a5f; color: var(--accent); }
.badge-volcanic { background: #3d1f1f; color: var(--red); }
.badge-pdo { background: #2d3a1f; color: var(--green); }
.badge-amo { background: #3d2d1f; color: var(--orange); }
.badge-unknown { background: #2d2d2d; color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; }
.plotly-chart { width: 100%; height: 350px; }
.plotly-chart-tall { width: 100%; height: 450px; }
#correlationSlider { width: 100%; margin-top: 8px; }
.slider-label { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.8em; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
    <h1>Riemannian Climate Regime Monitor</h1>
    <div class="subtitle">
        Geodesic distance on SPD(<span id="dimLabel"></span>) covariance manifold
        &mdash; <span id="dateRange"></span>
    </div>

    <div class="stats" id="statsBar"></div>

    <div class="grid">
        <div class="card grid-full">
            <h2>Geodesic Distance Timeline</h2>
            <div id="timelinePlot" class="plotly-chart-tall"></div>
        </div>

        <div class="card">
            <h2>Manifold Embedding (PGA)</h2>
            <div id="embeddingPlot" class="plotly-chart"></div>
        </div>

        <div class="card">
            <h2>Correlation Structure</h2>
            <div id="corrPlot" class="plotly-chart"></div>
            <input type="range" id="correlationSlider" min="0" max="0" value="0">
            <div class="slider-label"><span id="corrDate"></span><span id="corrIdx"></span></div>
        </div>

        <div class="card">
            <h2>V+ / V&minus; Decomposition</h2>
            <div id="vdecompPlot" class="plotly-chart"></div>
        </div>

        <div class="card">
            <h2>Detected Transitions</h2>
            <div style="max-height: 350px; overflow-y: auto;">
                <table id="transitionsTable">
                    <thead><tr><th>Date</th><th>Distance</th><th>&sigma;</th><th>Type</th><th>Event</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="card" style="margin-top: 16px; text-align: center; color: var(--muted); font-size: 0.8em;">
        Powered by <strong>omni_toolkit</strong> &mdash; Riemannian geometry on GL<sup>+</sup>(d)/SO(d)
        &mdash; <a href="https://omnisciences.io" style="color: var(--accent);">omnisciences.io</a>
    </div>
</div>

<script>
const DATA = /*__DATA__*/null;

// ── Stats bar ──
function initStats() {
    const s = document.getElementById('statsBar');
    const v = DATA.validation;
    const c = DATA.config;
    const stats = [
        { value: c.n_indices + 'D', label: 'Indices' },
        { value: DATA.transitions.length, label: 'Transitions' },
        { value: v.enso_recall !== null ? (v.enso_recall * 100).toFixed(1) + '%' : '—', label: 'ENSO Recall' },
        { value: v.enso_precision !== null ? (v.enso_precision * 100).toFixed(1) + '%' : '—', label: 'Precision' },
        { value: v.mean_lead_time !== null ? v.mean_lead_time.toFixed(1) + ' mo' : '—', label: 'Lead Time' },
    ];
    s.innerHTML = stats.map(st =>
        `<div class="stat"><div class="value">${st.value}</div><div class="label">${st.label}</div></div>`
    ).join('');
    document.getElementById('dimLabel').textContent = c.n_indices;
    const dates = DATA.timeline.dates;
    document.getElementById('dateRange').textContent = dates[0] + ' to ' + dates[dates.length - 1];
}

// ── Timeline ──
function plotTimeline() {
    const t = DATA.timeline;
    const threshold = t.mean_geo + DATA.config.threshold * t.std_geo;
    const traces = [
        {
            x: t.dates, y: t.geodesic, type: 'scatter', mode: 'lines',
            name: 'Geodesic', line: { color: '#58a6ff', width: 1.5 },
        },
        {
            x: t.dates, y: t.euclidean, type: 'scatter', mode: 'lines',
            name: 'Euclidean', line: { color: '#8b949e', width: 1, dash: 'dot' },
            visible: 'legendonly',
        },
        {
            x: t.dates, y: Array(t.dates.length).fill(t.mean_geo),
            type: 'scatter', mode: 'lines', name: 'Mean',
            line: { color: '#3fb950', width: 1, dash: 'dash' },
        },
        {
            x: t.dates, y: Array(t.dates.length).fill(threshold),
            type: 'scatter', mode: 'lines', name: 'Threshold',
            line: { color: '#f85149', width: 1, dash: 'dash' },
        },
    ];

    // Event markers
    const colors = { enso: '#58a6ff', volcanic: '#f85149', pdo: '#3fb950', amo: '#d29922' };
    for (const ev of DATA.events) {
        traces.push({
            x: [ev.date, ev.date], y: [0, threshold * 1.5],
            type: 'scatter', mode: 'lines', showlegend: false,
            line: { color: colors[ev.type] || '#8b949e', width: 0.5, dash: 'dot' },
            hovertext: ev.label, hoverinfo: 'text',
        });
    }

    // Transition markers
    const trDates = DATA.transitions.map(tr => tr.date);
    const trDists = DATA.transitions.map(tr => tr.distance);
    const trTexts = DATA.transitions.map(tr =>
        `${tr.date}<br>${tr.sigma.toFixed(1)}σ<br>${tr.event}`);
    const trColors = DATA.transitions.map(tr => colors[tr.category] || '#8b949e');
    traces.push({
        x: trDates, y: trDists, type: 'scatter', mode: 'markers',
        name: 'Transitions', marker: { color: trColors, size: 8, symbol: 'diamond' },
        text: trTexts, hoverinfo: 'text',
    });

    const layout = {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#e6edf3', size: 11 },
        xaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
        yaxis: { title: 'Distance', gridcolor: '#21262d', linecolor: '#30363d' },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 10, r: 20 },
        hovermode: 'x unified',
    };
    Plotly.newPlot('timelinePlot', traces, layout, { responsive: true });
}

// ── Embedding ──
function plotEmbedding() {
    const e = DATA.embedding;
    const N = e.x.length;
    const colors = Array.from({length: N}, (_, i) => i);

    const trace = {
        x: e.x, y: e.y, type: 'scatter', mode: 'lines+markers',
        marker: { color: colors, colorscale: 'Viridis', size: 3, opacity: 0.7 },
        line: { color: '#30363d', width: 0.5 },
        text: e.dates, hoverinfo: 'text',
    };

    // Mark transitions
    const trIdx = DATA.transitions.map(tr => {
        const dateStr = tr.date;
        return e.dates.indexOf(dateStr);
    }).filter(i => i >= 0);

    const trMarker = {
        x: trIdx.map(i => e.x[i]),
        y: trIdx.map(i => e.y[i]),
        type: 'scatter', mode: 'markers',
        marker: { color: '#f85149', size: 10, symbol: 'star' },
        name: 'Transitions',
        text: trIdx.map(i => e.dates[i]), hoverinfo: 'text',
    };

    const ve = e.variance_explained;
    const layout = {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#e6edf3', size: 11 },
        xaxis: { title: `PG1 (${ve[0].toFixed(1)}%)`, gridcolor: '#21262d' },
        yaxis: { title: `PG2 (${ve[1].toFixed(1)}%)`, gridcolor: '#21262d' },
        showlegend: false, margin: { t: 10, r: 10 },
    };
    Plotly.newPlot('embeddingPlot', [trace, trMarker], layout, { responsive: true });
}

// ── Correlation heatmap ──
function plotCorrelation() {
    const slider = document.getElementById('correlationSlider');
    const corrs = DATA.correlations;
    slider.max = corrs.length - 1;
    const names = DATA.index_names;

    function update(idx) {
        const snap = corrs[idx];
        const trace = {
            z: snap.matrix, type: 'heatmap',
            x: names, y: names,
            colorscale: [[0, '#0d47a1'], [0.5, '#0d1117'], [1, '#b71c1c']],
            zmin: -1, zmax: 1,
            hovertemplate: '%{x}-%{y}: %{z:.2f}<extra></extra>',
        };
        const layout = {
            paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
            font: { color: '#e6edf3', size: 10 },
            margin: { t: 10, r: 10, l: 60, b: 60 },
            xaxis: { tickangle: -45 }, yaxis: { autorange: 'reversed' },
        };
        Plotly.react('corrPlot', [trace], layout, { responsive: true });
        document.getElementById('corrDate').textContent = snap.date;
        document.getElementById('corrIdx').textContent = (parseInt(idx) + 1) + '/' + corrs.length;
    }

    slider.addEventListener('input', () => update(slider.value));
    update(0);
}

// ── V+/V- ──
function plotVDecomp() {
    const vd = DATA.v_decomposition;
    if (!vd || vd.length === 0) {
        document.getElementById('vdecompPlot').innerHTML =
            '<p style="color:#8b949e;text-align:center;padding:40px;">No transitions to decompose</p>';
        return;
    }

    const catColors = { enso: '#58a6ff', volcanic: '#f85149', pdo: '#3fb950', amo: '#d29922', unknown: '#8b949e' };

    const trace = {
        x: vd.map(d => d.norm_plus),
        y: vd.map(d => d.norm_minus),
        type: 'scatter', mode: 'markers',
        marker: {
            color: vd.map(d => catColors[d.category] || '#8b949e'),
            size: 10,
        },
        text: vd.map(d => `${d.date}<br>${d.category}<br>${d.event}<br>V-/V+=${d.ratio.toFixed(2)}`),
        hoverinfo: 'text',
    };

    // Diagonal line (equal V+/V-)
    const maxVal = Math.max(...vd.map(d => Math.max(d.norm_plus, d.norm_minus))) * 1.1;
    const diag = {
        x: [0, maxVal], y: [0, maxVal],
        type: 'scatter', mode: 'lines',
        line: { color: '#30363d', dash: 'dash', width: 1 },
        showlegend: false,
    };

    const layout = {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#e6edf3', size: 11 },
        xaxis: { title: '|V+| (volume)', gridcolor: '#21262d' },
        yaxis: { title: '|V−| (shape)', gridcolor: '#21262d' },
        showlegend: false, margin: { t: 10, r: 10 },
        annotations: [
            { x: maxVal * 0.8, y: maxVal * 0.3, text: 'Volume-<br>dominated', showarrow: false, font: { color: '#8b949e', size: 10 } },
            { x: maxVal * 0.2, y: maxVal * 0.8, text: 'Shape-<br>dominated', showarrow: false, font: { color: '#8b949e', size: 10 } },
        ],
    };
    Plotly.newPlot('vdecompPlot', [diag, trace], layout, { responsive: true });
}

// ── Transitions table ──
function fillTable() {
    const tbody = document.querySelector('#transitionsTable tbody');
    tbody.innerHTML = DATA.transitions.map(tr => {
        const badgeClass = 'badge badge-' + tr.category;
        return `<tr>
            <td>${tr.date}</td>
            <td>${tr.distance.toFixed(4)}</td>
            <td>${tr.sigma.toFixed(1)}</td>
            <td><span class="${badgeClass}">${tr.category}</span></td>
            <td>${tr.event}</td>
        </tr>`;
    }).join('');
}

// ── Init ──
initStats();
plotTimeline();
plotEmbedding();
plotCorrelation();
plotVDecomp();
fillTable();
</script>
</body>
</html>"""


# =====================================================================
# Streaming dashboard (live updates via file watch)
# =====================================================================

def build_live_dashboard(monitor_state_path: str,
                         output: str = "live_dashboard.html") -> str:
    """Build a dashboard that auto-refreshes from a monitor state file.

    The dashboard polls the state file every 60 seconds and updates
    the visualizations. Suitable for deployment on omnisciences.io/climate/.
    """
    html = _LIVE_DASHBOARD_TEMPLATE.replace("__STATE_PATH__", monitor_state_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html)
    return output


_LIVE_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Climate Regime Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
}
.container { max-width: 1000px; margin: 0 auto; padding: 20px; }
h1 { color: var(--accent); font-size: 1.5em; }
.status-bar {
    display: flex; gap: 20px; align-items: center;
    padding: 16px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin: 16px 0;
}
.status-indicator {
    width: 12px; height: 12px; border-radius: 50%;
    animation: pulse 2s infinite;
}
.status-stable { background: var(--green); }
.status-transition { background: var(--red); }
.status-warmup { background: var(--muted); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.metric { text-align: center; }
.metric .val { font-size: 1.5em; font-weight: 700; color: var(--accent); }
.metric .lbl { font-size: 0.75em; color: var(--muted); }
#distPlot { width: 100%; height: 300px; margin-top: 16px; }
.footer { text-align: center; color: var(--muted); font-size: 0.8em; margin-top: 20px; }
</style>
</head>
<body>
<div class="container">
    <h1>🌊 Live Climate Regime Monitor</h1>

    <div class="status-bar">
        <div class="status-indicator" id="statusDot"></div>
        <div>
            <strong id="regimeLabel">Loading...</strong>
            <div style="font-size:0.8em;color:var(--muted)" id="lastUpdate"></div>
        </div>
        <div style="flex:1"></div>
        <div class="metric"><div class="val" id="distVal">—</div><div class="lbl">Step Distance</div></div>
        <div class="metric"><div class="val" id="sigmaVal">—</div><div class="lbl">σ Above Mean</div></div>
        <div class="metric"><div class="val" id="sinceVal">—</div><div class="lbl">Mo Since Transition</div></div>
    </div>

    <div id="distPlot"></div>

    <div class="footer">
        Auto-refreshes every 60s &mdash;
        <a href="https://omnisciences.io" style="color:var(--accent)">omnisciences.io</a>
    </div>
</div>

<script>
const STATE_PATH = "__STATE_PATH__";

async function fetchState() {
    try {
        const resp = await fetch(STATE_PATH + '?t=' + Date.now());
        return await resp.json();
    } catch(e) {
        console.warn('Failed to fetch state:', e);
        return null;
    }
}

function updateUI(state) {
    if (!state || !state.streaming) return;
    const s = state.streaming;

    // Status
    const dot = document.getElementById('statusDot');
    const regime = s.regime_label || 'unknown';
    dot.className = 'status-indicator status-' + (regime === 'transition' ? 'transition' : regime === 'warmup' ? 'warmup' : 'stable');
    document.getElementById('regimeLabel').textContent = regime.toUpperCase();

    // Metrics
    const dists = s.distance_history || [];
    const latest = dists.length > 0 ? dists[dists.length - 1] : 0;
    const std = Math.sqrt(Math.max(s.running_var_dist || 0, 0));
    const sigma = std > 0 ? ((latest - (s.running_mean_dist || 0)) / std) : 0;
    document.getElementById('distVal').textContent = latest.toFixed(4);
    document.getElementById('sigmaVal').textContent = sigma.toFixed(1) + 'σ';

    const since = s.last_transition_idx > 0 ? s.n_updates - s.last_transition_idx : s.n_updates;
    document.getElementById('sinceVal').textContent = since;

    const dates = (s.date_history || []).map(d => d[0] + '-' + String(d[1]).padStart(2, '0'));
    document.getElementById('lastUpdate').textContent = dates.length > 0 ? 'Last: ' + dates[dates.length - 1] : '';

    // Plot
    const threshold = (s.running_mean_dist || 0) + 2.0 * std;
    const traces = [
        { x: dates, y: dists, type: 'scatter', mode: 'lines', name: 'Distance', line: { color: '#58a6ff' } },
        { x: dates, y: Array(dates.length).fill(s.running_mean_dist || 0), type: 'scatter', mode: 'lines', name: 'Mean', line: { color: '#3fb950', dash: 'dash' } },
        { x: dates, y: Array(dates.length).fill(threshold), type: 'scatter', mode: 'lines', name: 'Threshold', line: { color: '#f85149', dash: 'dash' } },
    ];
    const layout = {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#e6edf3' },
        xaxis: { gridcolor: '#21262d' }, yaxis: { gridcolor: '#21262d', title: 'Geodesic Distance' },
        legend: { orientation: 'h', y: -0.2 }, margin: { t: 10, r: 10 },
    };
    Plotly.react('distPlot', traces, layout, { responsive: true });
}

async function refresh() {
    const state = await fetchState();
    updateUI(state);
}

refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>"""


# =====================================================================
# Convenience: run full pipeline and open dashboard
# =====================================================================

def run_demo(d: int = 12, output: str = "climate_dashboard.html",
             verbose: bool = True) -> str:
    """Generate synthetic data, run detection, build dashboard, return path."""
    data = ClimateDataLoader.generate_synthetic(T=816, d=d)

    if verbose:
        print(f"Generating {d}D climate dashboard...")

    detector = ClimateRegimeDetector(
        window=72, step=1, threshold=2.0, shrinkage=0.1
    )
    results = detector.detect(data)
    detector.validate(results)

    path = build_static_dashboard(results, output)

    if verbose:
        print(f"Dashboard saved to: {path}")
        print(f"  {len(results.transitions)} transitions detected")
        print(f"  ENSO recall: {results.enso_recall:.1%}")
        print(f"  Open in browser: file://{Path(path).resolve()}")

    return path
