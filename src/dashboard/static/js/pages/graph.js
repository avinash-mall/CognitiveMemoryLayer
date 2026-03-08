/**
 * Knowledge Graph Page
 * Interactive graph visualization using vis-network (loaded from CDN)
 * and the REST API endpoints for graph data.
 */

import { getGraphStats, getGraphOverview, getGraphExplore, getGraphSearch, getTenants } from '../api.js';
import { formatNumber, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-graph');

let visNetwork = null;
let visNodes = null;
let visEdges = null;
let currentTenants = [];
let lastGraphParams = null;
let visLoaded = false;
let visLoadPromise = null;

const TYPE_COLORS = {
    person: '#6c8cff', location: '#34d399', organization: '#fbbf24',
    event: '#f87171', concept: '#a78bfa', thing: '#f472b6',
    center: '#6c8cff', unknown: '#94a3b8',
};

function colorForType(t) {
    return TYPE_COLORS[(t || '').toLowerCase()] || TYPE_COLORS.unknown;
}

function getDepth(pageEl) {
    const raw = parseInt(pageEl?.querySelector('#graph-depth')?.value || '2', 10);
    return Math.min(5, Math.max(1, isNaN(raw) ? 2 : raw));
}

function errToMessage(err) {
    if (err == null) return 'Unknown error';
    if (typeof err === 'string') return err;
    if (err.message) return err.message;
    if (err.code && err.message) return `${err.code}: ${err.message}`;
    if (err.code) return String(err.code);
    try {
        const s = JSON.stringify(err);
        if (s !== '{}') return s;
    } catch (_) {}
    return String(err);
}

function ensureVisLoaded() {
    if (visLoaded && window.vis) return Promise.resolve();
    if (visLoadPromise) return visLoadPromise;
    visLoadPromise = new Promise((resolve, reject) => {
        if (window.vis) { visLoaded = true; resolve(); return; }
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/dist/vis-network.min.css';
        document.head.appendChild(link);
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js';
        script.onload = () => { visLoaded = true; resolve(); };
        script.onerror = () => reject(new Error('Failed to load vis-network library'));
        document.head.appendChild(script);
    });
    return visLoadPromise;
}

const VIS_OPTIONS = {
    nodes: {
        font: {
            color: '#fff',
            size: 13,
            face: 'Inter, system-ui, sans-serif',
            background: 'rgba(0,0,0,0.5)',
            strokeWidth: 2,
            strokeColor: 'rgba(0,0,0,0.8)',
        },
        borderWidth: 2,
        shadow: true,
        shape: 'dot',
        size: 20,
    },
    edges: {
        font: {
            size: 10,
            color: 'rgba(255,255,255,0.9)',
            strokeWidth: 0,
            background: 'rgba(0,0,0,0.5)',
            align: 'middle',
        },
        color: { color: 'rgba(108, 140, 255, 0.6)', highlight: '#6c8cff' },
        smooth: { type: 'cubicBezier', roundness: 0.4 },
        arrows: { to: { enabled: true, scaleFactor: 0.7 } },
        width: 1.5,
    },
    physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
            springLength: 160,
            springConstant: 0.03,
            gravitationalConstant: -80,
            damping: 0.4,
        },
        stabilization: { iterations: 200, fit: true },
    },
    interaction: {
        hover: true,
        tooltipDelay: 150,
    },
};

function apiDataToVis(data) {
    const nodes = (data.nodes || []).map(n => {
        const c = colorForType(n.entity_type);
        return {
            id: n.id || n.entity,
            label: n.entity,
            title: `<b>${escapeHtml(n.entity)}</b><br/><span style="color:#94a3b8">${escapeHtml(n.entity_type || 'unknown')}</span>`,
            color: { background: c, border: c, highlight: { background: c, border: '#fff' } },
            shape: 'dot',
            size: n.entity_type === 'center' ? 28 : 20,
            _entity: n.entity,
            _entity_type: n.entity_type || 'unknown',
            _properties: n.properties || {},
        };
    });
    const edges = (data.edges || []).map((e, i) => ({
        id: `e-${i}`,
        from: e.source,
        to: e.target,
        label: e.predicate || '',
        title: `<b>${escapeHtml(e.predicate || 'RELATED')}</b>${e.confidence != null ? ` (${Number(e.confidence).toFixed(2)})` : ''}`,
        _predicate: e.predicate,
        _confidence: e.confidence,
    }));
    return { nodes, edges };
}

function createNetwork(vizDiv, nodesArr, edgesArr, pageEl) {
    visNodes = new vis.DataSet(nodesArr);
    visEdges = new vis.DataSet(edgesArr);

    visNetwork = new vis.Network(vizDiv, { nodes: visNodes, edges: visEdges }, VIS_OPTIONS);

    const detailPanel = pageEl.querySelector('#graph-detail-panel');
    const detailContent = pageEl.querySelector('#graph-detail-content');
    if (!detailPanel || !detailContent) return;

    visNetwork.on('click', (params) => {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = visNodes.get(nodeId);
            if (!node) return;
            const rest = Object.entries(node._properties || {}).filter(([k]) => !['entity', 'entity_type', 'tenant_id', 'scope_id'].includes(k));
            detailContent.innerHTML = `
                <div class="detail-grid">
                    <div class="component-detail"><span class="component-detail-label">Entity</span><span class="component-detail-value">${escapeHtml(String(node._entity))}</span></div>
                    <div class="component-detail"><span class="component-detail-label">Type</span><span class="component-detail-value"><span class="badge badge-type">${escapeHtml(String(node._entity_type))}</span></span></div>
                    ${rest.map(([k, v]) => `<div class="component-detail"><span class="component-detail-label">${escapeHtml(k)}</span><span class="component-detail-value">${escapeHtml(String(v))}</span></div>`).join('')}
                </div>`;
            detailPanel.classList.remove('hidden');
        } else if (params.edges.length > 0) {
            const edgeId = params.edges[0];
            const edge = visEdges.get(edgeId);
            if (!edge) return;
            detailContent.innerHTML = `
                <div class="detail-grid">
                    <div class="component-detail"><span class="component-detail-label">Source</span><span class="component-detail-value">${escapeHtml(String(edge.from))}</span></div>
                    <div class="component-detail"><span class="component-detail-label">Predicate</span><span class="component-detail-value">${escapeHtml(String(edge._predicate || 'RELATED'))}</span></div>
                    <div class="component-detail"><span class="component-detail-label">Target</span><span class="component-detail-value">${escapeHtml(String(edge.to))}</span></div>
                    <div class="component-detail"><span class="component-detail-label">Confidence</span><span class="component-detail-value">${escapeHtml(String(edge._confidence ?? '-'))}</span></div>
                </div>`;
            detailPanel.classList.remove('hidden');
        }
    });
}

async function loadOverview(el, tenantId, scopeId) {
    const graphContainer = el.querySelector('#graph-container');
    graphContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading graph...</div>';
    try {
        const data = await getGraphOverview(tenantId, scopeId);
        if (!data.nodes || data.nodes.length === 0) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">No graph data for this tenant. Add memories to build the knowledge graph.</div>`;
            return;
        }
        await ensureVisLoaded();
        const { nodes, edges } = apiDataToVis(data);
        const vizDiv = document.createElement('div');
        vizDiv.id = 'graph-viz';
        vizDiv.style.cssText = 'width:100%;height:100%;min-height:400px;';
        graphContainer.innerHTML = '';
        graphContainer.appendChild(vizDiv);
        createNetwork(vizDiv, nodes, edges, el);
        lastGraphParams = { type: 'overview', tenant_id: tenantId, scope_id: scopeId || tenantId };
    } catch (err) {
        graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Failed to load graph: ${escapeHtml(errToMessage(err))}</div>`;
    }
}

async function loadExplore(el, tenantId, entity, scopeId) {
    const graphContainer = el.querySelector('#graph-container');
    const depth = getDepth(el);
    const scope = scopeId || tenantId;
    graphContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading graph...</div>';
    try {
        const data = await getGraphExplore(tenantId, entity, scope, depth);
        if (!data.nodes || data.nodes.length === 0) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">Entity not found or no connections.</div>`;
            return;
        }
        await ensureVisLoaded();
        const { nodes, edges } = apiDataToVis(data);
        const vizDiv = document.createElement('div');
        vizDiv.id = 'graph-viz';
        vizDiv.style.cssText = 'width:100%;height:100%;min-height:400px;';
        graphContainer.innerHTML = '';
        graphContainer.appendChild(vizDiv);
        createNetwork(vizDiv, nodes, edges, el);
        lastGraphParams = { type: 'explore', tenant_id: tenantId, scope_id: scope, entity };
    } catch (err) {
        graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Failed to load graph: ${escapeHtml(errToMessage(err))}</div>`;
    }
}

export async function renderGraph({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading graph data...</div>`;

    try {
        const [stats, tenantsData] = await Promise.all([
            getGraphStats(),
            getTenants(),
        ]);
        currentTenants = tenantsData.tenants || [];
        el.innerHTML = buildPage(stats, tenantId);
        attachListeners(el, tenantId);

        const graphTenants = stats.tenants_with_graph || [];
        const graphContainer = el.querySelector('#graph-container');
        if (!graphContainer) return;

        if (graphTenants.length === 0) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">No graph data yet. Select a tenant and add memories to build the knowledge graph.</div>`;
            return;
        }

        const tenantSelect = el.querySelector('#graph-tenant');
        const selectedTenant = tenantSelect?.value || graphTenants[0];
        if (!selectedTenant) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">Select a tenant to explore the graph.</div>`;
            return;
        }

        await loadOverview(el, selectedTenant, selectedTenant);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load graph: ${escapeHtml(errToMessage(err))}</p></div>`;
    }
}

function buildPage(stats, tenantId) {
    const graphTenants = stats.tenants_with_graph || [];
    const tenantIds = [...new Set([
        ...currentTenants.map(t => t.tenant_id),
        ...graphTenants,
    ])];
    const preferredTenant = tenantId && graphTenants.includes(tenantId)
        ? tenantId
        : graphTenants[0] || tenantIds[0] || '';
    const tenantOptions = tenantIds.map(tid =>
        `<option value="${escapeHtml(tid)}" ${tid === preferredTenant ? 'selected' : ''}>${escapeHtml(tid)}</option>`
    ).join('');

    const hasGraphData = graphTenants.length > 0;
    const emptyStateMsg = hasGraphData
        ? 'Select a tenant and type an entity name to explore connections.'
        : 'No graph data yet. Select a tenant and add memories to build the knowledge graph.';

    return `
        <p class="page-desc">Interactive knowledge graph powered by Neo4j. Select a tenant and search for entities to explore semantic relationships between memory concepts.</p>
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Total Nodes</div><div class="kpi-value">${formatNumber(stats.total_nodes)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Edges</div><div class="kpi-value">${formatNumber(stats.total_edges)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Entity Types</div><div class="kpi-value">${formatNumber(Object.keys(stats.entity_types || {}).length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Tenants with Graph</div><div class="kpi-value">${formatNumber((stats.tenants_with_graph || []).length)}</div></div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Explore Graph</div>
            <div class="graph-controls">
                <select id="graph-tenant" class="select-sm">
                    <option value="">Select Tenant</option>
                    ${tenantOptions}
                </select>
                <input type="text" id="graph-search" class="input-sm" placeholder="Search entity name..." style="min-width:200px">
                <input type="range" id="graph-depth" min="1" max="5" value="2" style="width:100px">
                <span id="graph-depth-label" style="font-size:0.85rem;color:var(--text-secondary)">Depth: 2</span>
                <button id="graph-explore-btn" class="btn btn-primary btn-sm">Explore</button>
            </div>

            <div id="graph-search-results" class="graph-search-results hidden"></div>

            <div id="graph-container" class="graph-container">
                <div class="empty-state graph-empty-state" style="padding:60px">${emptyStateMsg}</div>
            </div>
        </div>

        <div id="graph-detail-panel" class="card hidden" style="margin-top:16px;">
            <div class="card-title">Selected Node Details</div>
            <div id="graph-detail-content"></div>
        </div>

        ${Object.keys(stats.entity_types || {}).length > 0 ? `
        <div class="card" style="margin-top:16px;">
            <div class="card-title">Entity Types Distribution</div>
            ${Object.entries(stats.entity_types).sort((a, b) => b[1] - a[1]).map(([type, count]) => `
                <div class="component-detail">
                    <span class="component-detail-label"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${colorForType(type)};margin-right:8px"></span>${escapeHtml(type)}</span>
                    <span class="component-detail-value">${formatNumber(count)}</span>
                </div>
            `).join('')}
        </div>
        ` : ''}
    `;
}

function attachListeners(el, initialTenantId) {
    const depthSlider = el.querySelector('#graph-depth');
    const depthLabel = el.querySelector('#graph-depth-label');
    depthSlider?.addEventListener('input', () => {
        depthLabel.textContent = `Depth: ${depthSlider.value}`;
    });
    depthSlider?.addEventListener('change', () => {
        if (!lastGraphParams) return;
        if (lastGraphParams.type === 'overview') {
            loadOverview(el, lastGraphParams.tenant_id, lastGraphParams.scope_id);
        } else {
            loadExplore(el, lastGraphParams.tenant_id, lastGraphParams.entity, lastGraphParams.scope_id);
        }
    });

    el.querySelector('#graph-tenant')?.addEventListener('change', () => {
        const tid = el.querySelector('#graph-tenant')?.value;
        if (!tid) return;
        loadOverview(el, tid, tid);
    });

    let searchTimeout;
    const searchInput = el.querySelector('#graph-search');
    const searchResults = el.querySelector('#graph-search-results');

    searchInput?.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const q = searchInput.value.trim();
        if (q.length < 2) { searchResults?.classList.add('hidden'); return; }
        searchTimeout = setTimeout(async () => {
            const tid = el.querySelector('#graph-tenant')?.value;
            try {
                const data = await getGraphSearch(q, tid, 10);
                if (data.results.length > 0) {
                    searchResults.innerHTML = data.results.map(r =>
                        `<div class="search-result-item" data-entity="${escapeHtml(r.entity)}" data-tenant="${escapeHtml(r.tenant_id)}" data-scope="${escapeHtml(r.scope_id)}">
                            <strong>${escapeHtml(r.entity)}</strong>
                            <span class="badge badge-type" style="margin-left:8px">${escapeHtml(r.entity_type)}</span>
                            <span style="color:var(--text-muted);font-size:0.8rem;margin-left:8px">${escapeHtml(r.tenant_id)}</span>
                        </div>`
                    ).join('');
                    searchResults.classList.remove('hidden');
                    searchResults.querySelectorAll('.search-result-item').forEach(item => {
                        item.addEventListener('click', () => {
                            searchInput.value = item.dataset.entity;
                            el.querySelector('#graph-tenant').value = item.dataset.tenant;
                            searchResults.classList.add('hidden');
                            loadExplore(el, item.dataset.tenant, item.dataset.entity, item.dataset.scope);
                        });
                    });
                } else {
                    searchResults.innerHTML = '<div class="search-result-item" style="color:var(--text-muted)">No entities found</div>';
                    searchResults.classList.remove('hidden');
                }
            } catch { searchResults?.classList.add('hidden'); }
        }, 300);
    });

    el.querySelector('#graph-explore-btn')?.addEventListener('click', () => {
        const tid = el.querySelector('#graph-tenant')?.value;
        const entity = searchInput?.value?.trim();
        if (!tid || !entity) return;
        searchResults?.classList.add('hidden');
        loadExplore(el, tid, entity, null);
    });
}
