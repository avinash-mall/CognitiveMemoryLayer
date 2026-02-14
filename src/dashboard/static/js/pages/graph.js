/**
 * Knowledge Graph Page
 * Interactive graph visualization using vis-network + Neo4j data.
 */

import { getGraphStats, getGraphOverview, getGraphExplore, getGraphSearch, getTenants } from '../api.js';
import { formatNumber, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-graph');

let network = null;
let currentTenants = [];

const TYPE_COLORS = {
    person: '#6c8cff', location: '#34d399', organization: '#fbbf24',
    event: '#f87171', concept: '#a78bfa', thing: '#f472b6',
    center: '#6c8cff', unknown: '#94a3b8',
};

function colorForType(t) {
    return TYPE_COLORS[(t || '').toLowerCase()] || TYPE_COLORS.unknown;
}

export async function renderGraph({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading graph data...</div>`;

    try {
        const [stats, tenantsData] = await Promise.all([getGraphStats(), getTenants()]);
        currentTenants = tenantsData.tenants || [];
        el.innerHTML = buildPage(stats, tenantId);
        attachListeners(el, tenantId);

        const graphTenants = stats.tenants_with_graph || [];
        const firstTenantId = graphTenants[0];
        if (firstTenantId) {
            const graphContainer = el.querySelector('#graph-container');
            const tenantSelect = el.querySelector('#graph-tenant');
            if (tenantSelect && Array.from(tenantSelect.options).some(o => o.value === firstTenantId)) {
                tenantSelect.value = firstTenantId;
                graphContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading graph...</div>';
                try {
                    const data = await getGraphOverview(firstTenantId);
                    if (data?.nodes?.length > 0) {
                        graphContainer.innerHTML = '';
                        renderNetwork(graphContainer, data, el);
                    } else {
                        graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">Select a tenant and type an entity name (e.g. a person, place, or concept) to explore connections. Tip: Results appear as you type.</div>`;
                    }
                } catch {
                    graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">Select a tenant and type an entity name (e.g. a person, place, or concept) to explore connections. Tip: Results appear as you type.</div>`;
                }
            }
        }
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load graph: ${err.message}</p></div>`;
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
        ? 'Select a tenant and type an entity name (e.g. a person, place, or concept) to explore connections. Tip: Results appear as you type.'
        : 'No graph data yet. Select a tenant and add memories to build the knowledge graph.';

    return `
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

    // Search entity
    let searchTimeout;
    const searchInput = el.querySelector('#graph-search');
    const searchResults = el.querySelector('#graph-search-results');

    searchInput?.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const q = searchInput.value.trim();
        if (q.length < 2) { searchResults.classList.add('hidden'); return; }
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
                            doExplore(el, item.dataset.tenant, item.dataset.entity, item.dataset.scope);
                        });
                    });
                } else {
                    searchResults.innerHTML = '<div class="search-result-item" style="color:var(--text-muted)">No entities found</div>';
                    searchResults.classList.remove('hidden');
                }
            } catch { searchResults.classList.add('hidden'); }
        }, 300);
    });

    el.querySelector('#graph-explore-btn')?.addEventListener('click', () => {
        const tid = el.querySelector('#graph-tenant')?.value;
        const entity = searchInput?.value?.trim();
        if (!tid || !entity) return;
        searchResults.classList.add('hidden');
        doExplore(el, tid, entity, null);
    });
}

async function doExplore(el, tenantId, entity, scopeId) {
    const graphContainer = el.querySelector('#graph-container');
    const depth = parseInt(el.querySelector('#graph-depth')?.value || '2', 10);
    graphContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading graph...</div>';

    try {
        const data = await getGraphExplore(tenantId, entity, scopeId, depth);
        if (!data.nodes || data.nodes.length === 0) {
            graphContainer.innerHTML = '<div class="empty-state" style="padding:60px">No graph data found for this entity</div>';
            return;
        }
        graphContainer.innerHTML = '';
        renderNetwork(graphContainer, data, el);
    } catch (err) {
        graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Error: ${err.message}</div>`;
    }
}

function renderNetwork(containerEl, data, pageEl) {
    const isCenter = (e) => e.entity === data.center_entity;
    const nodes = new vis.DataSet(data.nodes.map(n => ({
        id: n.entity,
        label: n.entity,
        color: {
            background: colorForType(n.entity_type),
            border: colorForType(n.entity_type),
            highlight: { background: '#fff', border: colorForType(n.entity_type) },
        },
        font: {
            color: '#fff',
            size: isCenter(n) ? 14 : 12,
            face: 'Inter, sans-serif',
            background: 'rgba(0,0,0,0.5)',
            strokeWidth: 2,
            strokeColor: 'rgba(0,0,0,0.8)',
        },
        shape: isCenter(n) ? 'diamond' : 'dot',
        size: isCenter(n) ? 28 : 20,
        borderWidth: 2,
        shadow: true,
        title: `${n.entity} (${n.entity_type})`,
        _data: n,
    })));

    const edges = new vis.DataSet(data.edges.map((e, i) => ({
        id: i,
        from: e.source,
        to: e.target,
        label: e.predicate.replace(/_/g, ' ').toLowerCase(),
        arrows: 'to',
        font: {
            size: 11,
            color: 'var(--text-secondary)',
            strokeWidth: 0,
            background: 'rgba(0,0,0,0.4)',
            align: 'middle',
        },
        color: { color: 'rgba(108, 140, 255, 0.6)', highlight: '#6c8cff' },
        width: 2,
        smooth: { type: 'cubicBezier', roundness: 0.5 },
        _data: e,
    })));

    const options = {
        layout: { improvedLayout: true },
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                springLength: 120,
                springConstant: 0.05,
            },
            stabilization: { iterations: 200 },
        },
        interaction: { hover: true, tooltipDelay: 200 },
    };

    if (network) { network.destroy(); }
    network = new vis.Network(containerEl, { nodes, edges }, options);

    network.on('click', (params) => {
        const detailPanel = pageEl.querySelector('#graph-detail-panel');
        const detailContent = pageEl.querySelector('#graph-detail-content');
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            if (node && node._data) {
                const n = node._data;
                const props = Object.entries(n.properties || {}).filter(([k]) => !['entity', 'entity_type'].includes(k));
                detailContent.innerHTML = `
                    <div class="detail-grid">
                        <div class="component-detail"><span class="component-detail-label">Entity</span><span class="component-detail-value">${escapeHtml(n.entity)}</span></div>
                        <div class="component-detail"><span class="component-detail-label">Type</span><span class="component-detail-value"><span class="badge badge-type">${escapeHtml(n.entity_type)}</span></span></div>
                        ${props.map(([k, v]) => `<div class="component-detail"><span class="component-detail-label">${escapeHtml(k)}</span><span class="component-detail-value">${escapeHtml(String(v))}</span></div>`).join('')}
                    </div>`;
                detailPanel.classList.remove('hidden');
            }
        } else if (params.edges.length > 0) {
            const edgeId = params.edges[0];
            const edge = edges.get(edgeId);
            if (edge && edge._data) {
                const e = edge._data;
                detailContent.innerHTML = `
                    <div class="detail-grid">
                        <div class="component-detail"><span class="component-detail-label">Source</span><span class="component-detail-value">${escapeHtml(e.source)}</span></div>
                        <div class="component-detail"><span class="component-detail-label">Predicate</span><span class="component-detail-value">${escapeHtml(e.predicate)}</span></div>
                        <div class="component-detail"><span class="component-detail-label">Target</span><span class="component-detail-value">${escapeHtml(e.target)}</span></div>
                        <div class="component-detail"><span class="component-detail-label">Confidence</span><span class="component-detail-value">${e.confidence}</span></div>
                    </div>`;
                detailPanel.classList.remove('hidden');
            }
        } else {
            detailPanel.classList.add('hidden');
        }
    });
}
