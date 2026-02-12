/**
 * Overview Dashboard Page
 * Displays KPI cards, charts, system health, recent events,
 * reconsolidation status, and request sparkline.
 */

import { getOverview, getTimeline, getComponents, getEvents, getLabile, getRequestStats } from '../api.js';
import { formatNumber, formatFloat, formatMB, formatDate, formatLatency } from '../utils/formatters.js';
import { createDoughnutChart, createBarChart, createLineChart } from '../utils/charts.js';

const container = () => document.getElementById('page-overview');

export async function renderOverview({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading dashboard...</div>`;

    try {
        // Fetch all data in parallel
        const [overview, timeline, components, events, labile, reqStats] = await Promise.all([
            getOverview(tenantId),
            getTimeline(30, tenantId),
            getComponents(),
            getEvents({ page: 1, perPage: 10, tenantId }),
            getLabile(tenantId).catch(() => ({ total_db_labile: 0, total_redis_scopes: 0, total_redis_sessions: 0, total_redis_memories: 0 })),
            getRequestStats(24).catch(() => ({ points: [], total_last_24h: 0 })),
        ]);

        el.innerHTML = buildOverviewHTML(overview, timeline, components, events, labile, reqStats);

        // Render charts after DOM is ready
        requestAnimationFrame(() => {
            renderTypeChart(overview);
            renderStatusChart(overview);
            renderTimelineChart(timeline);
            renderFactsChart(overview);
            renderRequestSparkline(reqStats);
        });
    } catch (err) {
        el.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">&#9888;</div>
            <p>Failed to load dashboard: ${err.message}</p>
        </div>`;
    }
}

function buildOverviewHTML(overview, timeline, components, events, labile, reqStats) {
    return `
        <!-- KPI Cards -->
        <div class="kpi-grid">
            ${kpiCard('Total Memories', formatNumber(overview.total_memories), 'Across all types')}
            ${kpiCard('Active', formatNumber(overview.active_memories), `${formatNumber(overview.silent_memories)} silent`)}
            ${kpiCard('Avg Confidence', formatFloat(overview.avg_confidence, 3), scoreBar(overview.avg_confidence))}
            ${kpiCard('Avg Importance', formatFloat(overview.avg_importance, 3), scoreBar(overview.avg_importance))}
            ${kpiCard('Storage', formatMB(overview.estimated_size_mb), `${formatNumber(overview.total_events)} events`)}
            ${kpiCard('Semantic Facts', formatNumber(overview.total_semantic_facts), `${formatNumber(overview.current_semantic_facts)} current`)}
            ${kpiCard('Labile', formatNumber(overview.labile_memories), `${formatNumber(labile.total_redis_scopes)} scopes, ${formatNumber(labile.total_redis_memories)} in Redis`)}
            ${kpiCard('Requests (24h)', formatNumber(reqStats.total_last_24h), '<canvas id="chart-req-sparkline" height="30" style="max-width:120px"></canvas>')}
        </div>

        <!-- Reconsolidation Quick Status -->
        ${labile.total_db_labile > 0 || labile.total_redis_scopes > 0 ? `
        <div class="card" style="margin-bottom:16px;border-left:3px solid var(--warning);">
            <div class="card-title">Reconsolidation Queue</div>
            <div style="display:flex;gap:32px;flex-wrap:wrap;">
                <div class="component-detail"><span class="component-detail-label">DB Labile</span><span class="component-detail-value">${formatNumber(labile.total_db_labile)}</span></div>
                <div class="component-detail"><span class="component-detail-label">Redis Scopes</span><span class="component-detail-value">${formatNumber(labile.total_redis_scopes)}</span></div>
                <div class="component-detail"><span class="component-detail-label">Redis Sessions</span><span class="component-detail-value">${formatNumber(labile.total_redis_sessions)}</span></div>
                <div class="component-detail"><span class="component-detail-label">Redis Memories</span><span class="component-detail-value">${formatNumber(labile.total_redis_memories)}</span></div>
            </div>
        </div>
        ` : ''}

        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-card">
                <div class="card-title">Memory Types Distribution</div>
                <canvas id="chart-types"></canvas>
            </div>
            <div class="chart-card">
                <div class="card-title">Memory Status Breakdown</div>
                <canvas id="chart-status"></canvas>
            </div>
            <div class="chart-card">
                <div class="card-title">Activity Timeline (Last 30 Days)</div>
                <canvas id="chart-timeline"></canvas>
            </div>
            <div class="chart-card">
                <div class="card-title">Semantic Facts by Category</div>
                <canvas id="chart-facts"></canvas>
            </div>
        </div>

        <!-- System Health & Recent Events -->
        <div class="chart-grid">
            <div class="card">
                <div class="card-title">System Health</div>
                ${buildComponentsSummary(components)}
            </div>
            <div class="card">
                <div class="card-title">Recent Events</div>
                ${buildRecentEvents(events)}
            </div>
        </div>

        <!-- Event Operations & Temporal Info -->
        <div class="chart-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Events by Type</div>
                ${buildEventBreakdown(overview.events_by_type)}
            </div>
            <div class="card">
                <div class="card-title">Events by Operation</div>
                ${buildEventBreakdown(overview.events_by_operation)}
            </div>
        </div>

        <div class="chart-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Temporal Range</div>
                <div class="component-detail">
                    <span class="component-detail-label">Oldest Memory</span>
                    <span class="component-detail-value">${formatDate(overview.oldest_memory)}</span>
                </div>
                <div class="component-detail">
                    <span class="component-detail-label">Newest Memory</span>
                    <span class="component-detail-value">${formatDate(overview.newest_memory)}</span>
                </div>
                <div class="component-detail">
                    <span class="component-detail-label">Avg Fact Confidence</span>
                    <span class="component-detail-value">${formatFloat(overview.avg_fact_confidence, 3)}</span>
                </div>
                <div class="component-detail">
                    <span class="component-detail-label">Avg Evidence Count</span>
                    <span class="component-detail-value">${formatFloat(overview.avg_evidence_count, 1)}</span>
                </div>
            </div>
        </div>
    `;
}

function kpiCard(label, value, sub) {
    return `
        <div class="kpi-card">
            <div class="kpi-label">${label}</div>
            <div class="kpi-value">${value}</div>
            <div class="kpi-sub">${sub}</div>
        </div>
    `;
}

function scoreBar(value) {
    const pct = Math.min(100, Math.max(0, (value || 0) * 100));
    let cls = 'gauge-fill-accent';
    if (pct >= 70) cls = 'gauge-fill-success';
    else if (pct >= 40) cls = 'gauge-fill-accent';
    else cls = 'gauge-fill-warning';
    return `<div class="gauge-bar"><div class="gauge-fill ${cls}" style="width:${pct}%"></div></div>`;
}

function buildComponentsSummary(data) {
    if (!data?.components?.length) return '<div class="empty-state">No component data</div>';
    return data.components.map(c => `
        <div class="component-detail">
            <span class="component-detail-label">${c.name}</span>
            <span>
                <span class="${c.status === 'ok' ? 'badge badge-ok' : c.status === 'error' ? 'badge badge-error' : 'badge badge-unknown'}">${c.status}</span>
                ${c.latency_ms !== null ? `<span style="margin-left:8px;color:var(--text-muted);font-size:0.8rem">${formatLatency(c.latency_ms)}</span>` : ''}
            </span>
        </div>
    `).join('');
}

function buildRecentEvents(data) {
    if (!data?.items?.length) return '<div class="empty-state">No recent events</div>';
    return `<div style="max-height:280px;overflow-y:auto;">
        ${data.items.map(e => `
            <div class="component-detail">
                <span class="component-detail-label" style="font-size:0.8rem;">
                    <span class="badge badge-type">${e.event_type}</span>
                    ${e.operation ? `<span style="margin-left:4px;color:var(--text-muted)">${e.operation}</span>` : ''}
                </span>
                <span class="component-detail-value" style="font-size:0.78rem;">${formatDate(e.created_at)}</span>
            </div>
        `).join('')}
    </div>`;
}

function buildEventBreakdown(map) {
    if (!map || !Object.keys(map).length) return '<div class="empty-state">No data</div>';
    return Object.entries(map).sort((a, b) => b[1] - a[1]).map(([k, v]) => `
        <div class="component-detail">
            <span class="component-detail-label">${k}</span>
            <span class="component-detail-value">${formatNumber(v)}</span>
        </div>
    `).join('');
}

// ---- Chart Renderers ----

function renderTypeChart(overview) {
    const canvas = document.getElementById('chart-types');
    if (!canvas) return;
    const entries = Object.entries(overview.by_type || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return;
    createDoughnutChart(canvas, entries.map(e => e[0]), entries.map(e => e[1]));
}

function renderStatusChart(overview) {
    const canvas = document.getElementById('chart-status');
    if (!canvas) return;
    const entries = Object.entries(overview.by_status || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return;
    createBarChart(canvas, entries.map(e => e[0]), entries.map(e => e[1]));
}

function renderTimelineChart(timeline) {
    const canvas = document.getElementById('chart-timeline');
    if (!canvas) return;
    const points = timeline?.points || [];
    if (!points.length) return;
    const labels = points.map(p => {
        const d = new Date(p.date);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });
    createLineChart(canvas, labels, points.map(p => p.count));
}

function renderFactsChart(overview) {
    const canvas = document.getElementById('chart-facts');
    if (!canvas) return;
    const entries = Object.entries(overview.facts_by_category || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
        canvas.parentElement.innerHTML += '<div class="empty-state" style="padding:20px;">No facts data yet</div>';
        canvas.style.display = 'none';
        return;
    }
    createDoughnutChart(canvas, entries.map(e => e[0]), entries.map(e => e[1]));
}

function renderRequestSparkline(reqStats) {
    const canvas = document.getElementById('chart-req-sparkline');
    if (!canvas || !reqStats.points.length) return;
    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: reqStats.points.map(() => ''),
            datasets: [{
                data: reqStats.points.map(p => p.count),
                borderColor: '#6c8cff',
                borderWidth: 1.5,
                fill: false,
                pointRadius: 0,
                tension: 0.3,
            }],
        },
        options: {
            responsive: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: { x: { display: false }, y: { display: false } },
            animation: false,
        },
    });
}
