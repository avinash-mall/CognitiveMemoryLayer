/**
 * Knowledge Graph Page
 * Interactive graph visualization using neovis.js (connects directly to Neo4j).
 */

import NeoVis from 'neovis.js/dist/neovis.js';
import { getGraphStats, getGraphNeo4jConfig, getGraphSearch, getTenants } from '../api.js';
import { formatNumber, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-graph');

let neoViz = null;
let currentTenants = [];
let lastGraphParams = null;

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

const EXPLORE_CYPHER = `
MATCH (start:Entity {tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity})
MATCH path = (start)-[*1..$depth]-(neighbor:Entity)
WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
WITH path LIMIT 500
UNWIND relationships(path) AS r
WITH DISTINCT r, startNode(r) AS sn, endNode(r) AS en
RETURN sn, r, en
`;

const OVERVIEW_CYPHER = `
MATCH (n:Entity {tenant_id: $tenant_id, scope_id: $scope_id})
WITH n ORDER BY COUNT { (n)--() } DESC LIMIT 1
MATCH path = (n)-[*1..$depth]-(m:Entity)
WHERE m.tenant_id = $tenant_id AND m.scope_id = $scope_id
WITH path LIMIT 500
UNWIND relationships(path) AS r
WITH DISTINCT r, startNode(r) AS sn, endNode(r) AS en
RETURN sn, r, en
`;

function buildNeovisConfig(neo4jConfig, pageEl) {
    const vizContainerId = 'graph-viz';
    return {
        containerId: vizContainerId,
        neo4j: {
            serverUrl: neo4jConfig.server_url,
            serverUser: neo4jConfig.server_user,
            serverPassword: neo4jConfig.server_password,
        },
        labels: {
            Entity: {
                label: 'entity',
                group: 'entity_type',
                value: 'entity',
                [NeoVis.NEOVIS_ADVANCED_CONFIG]: {
                    function: {
                        title: (node) => {
                            const props = node.properties || {};
                            const ent = props.entity || node.id;
                            const typ = props.entity_type || 'unknown';
                            const html = `<div style="padding:8px;max-width:240px"><strong>${escapeHtml(String(ent))}</strong><br/><span style="color:#94a3b8">${escapeHtml(String(typ))}</span></div>`;
                            const div = document.createElement('div');
                            div.innerHTML = html;
                            return div;
                        },
                    },
                },
            },
        },
        visConfig: {
            groups: Object.fromEntries(
                Object.entries(TYPE_COLORS).map(([k, v]) => [k, { color: { background: v, border: v } }])
            ),
            nodes: {
                font: {
                    color: '#fff',
                    size: 12,
                    face: 'Inter, system-ui, sans-serif',
                    background: 'rgba(0,0,0,0.5)',
                    strokeWidth: 2,
                    strokeColor: 'rgba(0,0,0,0.8)',
                },
                borderWidth: 2,
                shadow: true,
            },
            edges: {
                font: {
                    size: 10,
                    color: 'rgba(255,255,255,0.9)',
                    strokeWidth: 0,
                    background: 'rgba(0,0,0,0.5)',
                    align: 'middle',
                },
                color: { color: 'rgba(108, 140, 255, 0.7)', highlight: '#6c8cff' },
                smooth: { type: 'cubicBezier', roundness: 0.5 },
                arrows: { to: { enabled: true } },
            },
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    springLength: 120,
                    springConstant: 0.05,
                },
                stabilization: { iterations: 200 },
            },
        },
    };
}

function attachClickHandlers(neoVizInstance, pageEl) {
    const detailPanel = pageEl.querySelector('#graph-detail-panel');
    const detailContent = pageEl.querySelector('#graph-detail-content');
    if (!detailPanel || !detailContent) return;

    neoVizInstance.registerOnEvent('clickNode', (event) => {
        const nodeId = event.node;
        const nodes = neoVizInstance.nodes;
        if (!nodes) return;
        const node = nodes.get(nodeId);
        if (!node) return;
        const props = node.properties || node;
        const entity = props.entity ?? props.id ?? String(nodeId);
        const entityType = props.entity_type ?? 'unknown';
        const rest = Object.entries(props).filter(([k]) => !['entity', 'entity_type', 'tenant_id', 'scope_id'].includes(k));
        detailContent.innerHTML = `
            <div class="detail-grid">
                <div class="component-detail"><span class="component-detail-label">Entity</span><span class="component-detail-value">${escapeHtml(String(entity))}</span></div>
                <div class="component-detail"><span class="component-detail-label">Type</span><span class="component-detail-value"><span class="badge badge-type">${escapeHtml(String(entityType))}</span></span></div>
                ${rest.map(([k, v]) => `<div class="component-detail"><span class="component-detail-label">${escapeHtml(k)}</span><span class="component-detail-value">${escapeHtml(String(v))}</span></div>`).join('')}
            </div>`;
        detailPanel.classList.remove('hidden');
    });

    neoVizInstance.registerOnEvent('clickEdge', (event) => {
        const edgeId = event.edge;
        const edges = neoVizInstance.edges;
        if (!edges) return;
        const edge = edges.get(edgeId);
        if (!edge) return;
        const from = edge.from ?? edge.fromId;
        const to = edge.to ?? edge.toId;
        const label = edge.label ?? edge.type ?? 'RELATED';
        const conf = edge.confidence ?? edge.properties?.confidence ?? '-';
        detailContent.innerHTML = `
            <div class="detail-grid">
                <div class="component-detail"><span class="component-detail-label">Source</span><span class="component-detail-value">${escapeHtml(String(from))}</span></div>
                <div class="component-detail"><span class="component-detail-label">Predicate</span><span class="component-detail-value">${escapeHtml(String(label))}</span></div>
                <div class="component-detail"><span class="component-detail-label">Target</span><span class="component-detail-value">${escapeHtml(String(to))}</span></div>
                <div class="component-detail"><span class="component-detail-label">Confidence</span><span class="component-detail-value">${escapeHtml(String(conf))}</span></div>
            </div>`;
        detailPanel.classList.remove('hidden');
    });
}

export async function renderGraph({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading graph data...</div>`;

    try {
        const [stats, tenantsData, neo4jConfig] = await Promise.all([
            getGraphStats(),
            getTenants(),
            getGraphNeo4jConfig(),
        ]);
        currentTenants = tenantsData.tenants || [];
        el.innerHTML = buildPage(stats, tenantId);
        attachListeners(el, tenantId);

        const graphTenants = stats.tenants_with_graph || [];
        const firstTenantId = graphTenants[0];
        const graphContainer = el.querySelector('#graph-container');

        if (!graphContainer) return;

        const hasGraphData = graphTenants.length > 0;
        if (!hasGraphData) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">No graph data yet. Select a tenant and add memories to build the knowledge graph.</div>`;
            return;
        }

        const tenantSelect = el.querySelector('#graph-tenant');
        const selectedTenant = tenantSelect?.value || firstTenantId;
        if (!selectedTenant) {
            graphContainer.innerHTML = `<div class="empty-state graph-empty-state" style="padding:60px">Select a tenant to explore the graph.</div>`;
            return;
        }

        const vizDiv = document.createElement('div');
        vizDiv.id = 'graph-viz';
        vizDiv.style.cssText = 'width:100%;height:500px;min-height:400px;';
        graphContainer.innerHTML = '';
        graphContainer.appendChild(vizDiv);

        const config = buildNeovisConfig(neo4jConfig, el);
        neoViz = new NeoVis(config);
        attachClickHandlers(neoViz, el);
        neoViz.registerOnEvent('error', (event) => {
            const err = event?.error ?? event;
            const msg = errToMessage(err);
            graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Neo4j error: ${escapeHtml(msg)}. Check that Neo4j is running at bolt://localhost:7687 and reachable from your browser.</div>`;
        });

        const scopeId = selectedTenant;
        const depth = getDepth(el);
        try {
            neoViz.renderWithCypher(OVERVIEW_CYPHER, {
                tenant_id: selectedTenant,
                scope_id: scopeId,
                depth,
            });
            lastGraphParams = { type: 'overview', tenant_id: selectedTenant, scope_id: scopeId };
        } catch (e) {
            graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Failed to load graph: ${escapeHtml(errToMessage(e))}</div>`;
        }
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
        if (!lastGraphParams || !neoViz) return;
        const graphContainer = el.querySelector('#graph-container');
        if (!graphContainer?.querySelector('#graph-viz')) return;
        const depth = getDepth(el);
        try {
            neoViz.clearNetwork();
            if (lastGraphParams.type === 'overview') {
                neoViz.renderWithCypher(OVERVIEW_CYPHER, {
                    tenant_id: lastGraphParams.tenant_id,
                    scope_id: lastGraphParams.scope_id,
                    depth,
                });
            } else {
                neoViz.renderWithCypher(EXPLORE_CYPHER, {
                    tenant_id: lastGraphParams.tenant_id,
                    scope_id: lastGraphParams.scope_id,
                    entity: lastGraphParams.entity,
                    depth,
                });
            }
        } catch (e) {
            graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Error: ${escapeHtml(String(e?.message || e))}</div>`;
        }
    });

    el.querySelector('#graph-tenant')?.addEventListener('change', () => {
        const tid = el.querySelector('#graph-tenant')?.value;
        if (!tid || !neoViz) return;
        const graphContainer = el.querySelector('#graph-container');
        if (!graphContainer?.querySelector('#graph-viz')) return;
        const depth = getDepth(el);
        try {
            neoViz.clearNetwork();
            neoViz.renderWithCypher(OVERVIEW_CYPHER, { tenant_id: tid, scope_id: tid, depth });
            lastGraphParams = { type: 'overview', tenant_id: tid, scope_id: tid };
        } catch (e) {
            graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Error: ${escapeHtml(String(e?.message || e))}</div>`;
        }
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
                            doExplore(el, item.dataset.tenant, item.dataset.entity, item.dataset.scope);
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
        doExplore(el, tid, entity, null);
    });
}

async function doExplore(el, tenantId, entity, scopeId) {
    const graphContainer = el.querySelector('#graph-container');
    const depth = getDepth(el);
    const scope = scopeId || tenantId;

    if (!neoViz || !graphContainer?.querySelector('#graph-viz')) {
        graphContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div> Loading graph...</div>';
        return;
    }

    try {
        neoViz.clearNetwork();
        neoViz.renderWithCypher(EXPLORE_CYPHER, {
            tenant_id: tenantId,
            scope_id: scope,
            entity,
            depth,
        });
        lastGraphParams = { type: 'explore', tenant_id: tenantId, scope_id: scope, entity };
    } catch (err) {
        graphContainer.innerHTML = `<div class="empty-state" style="padding:40px">Error: ${escapeHtml(errToMessage(err))}</div>`;
    }
}
