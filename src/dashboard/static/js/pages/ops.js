/**
 * Operations Metrics Page
 * Runtime status, DB pool, Redis cache, retrieval step latency, and filterable Prometheus samples.
 */

import { getOpsMetrics } from '../api.js';
import { escapeHtml, formatDate, formatFloat, formatNumber, prettyJson } from '../utils/formatters.js';

const container = () => document.getElementById('page-ops');

let allMetrics = [];

export async function renderOps() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading operations metrics...</div>`;

    try {
        const data = await getOpsMetrics();
        allMetrics = data.metrics || [];
        el.innerHTML = buildPage(data);
        attachListeners(el);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load operations metrics: ${escapeHtml(err.message)}</p></div>`;
    }
}

function formatUptime(seconds) {
    if (seconds == null) return '—';
    const s = Math.floor(seconds);
    const hrs = Math.floor(s / 3600);
    const mins = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
}

function renderMetricsRows(metrics, search, hideZeros) {
    const q = (search || '').trim().toLowerCase();
    const filtered = metrics.filter(m => {
        if (hideZeros && Number(m.value) === 0) return false;
        if (!q) return true;
        return (
            m.name.toLowerCase().includes(q) ||
            JSON.stringify(m.labels || {}).toLowerCase().includes(q)
        );
    });
    if (!filtered.length) {
        return `<tr><td colspan="3" style="text-align:center;color:var(--text-muted);padding:16px;">No matching metric samples.</td></tr>`;
    }
    return filtered.map(metric => `
        <tr>
            <td class="mono" style="font-size:0.78rem;">${escapeHtml(metric.name)}</td>
            <td class="mono" style="font-size:0.76rem;white-space:pre-wrap;">${escapeHtml(prettyJson(metric.labels || {}))}</td>
            <td class="mono">${typeof metric.value === 'number' ? formatFloat(metric.value, 4) : escapeHtml(String(metric.value))}</td>
        </tr>
    `).join('');
}

function attachListeners(el) {
    const searchInput = el.querySelector('#ops-metric-search');
    const hideZeroCb = el.querySelector('#ops-hide-zeros');
    const tbody = el.querySelector('#ops-metrics-tbody');
    const countSpan = el.querySelector('#ops-metrics-count');
    if (!tbody) return;

    const refresh = () => {
        const search = searchInput?.value || '';
        const hideZeros = Boolean(hideZeroCb?.checked);
        tbody.innerHTML = renderMetricsRows(allMetrics, search, hideZeros);
        if (countSpan) {
            const shown = tbody.querySelectorAll('tr').length;
            countSpan.textContent = `${shown} / ${allMetrics.length}`;
        }
    };

    searchInput?.addEventListener('input', refresh);
    hideZeroCb?.addEventListener('change', refresh);
}

function buildPage(data) {
    const h = data.highlights || {};
    const runtime = h.runtime || {};
    const stepLatency = h.retrieval_step_avg_ms || {};
    const stepEntries = Object.entries(stepLatency);
    const poolSummary = h.db_pool_size != null
        ? `${formatNumber(h.db_pool_checked_out || 0)} / ${formatNumber(h.db_pool_size)}`
        : formatNumber(h.db_pool_checked_out || 0);
    const redisHitPct = h.redis_hit_rate != null ? `${formatFloat(h.redis_hit_rate * 100, 1)}%` : '—';

    return `
        <p class="page-desc">
            Live runtime health, CML v3.0 engine status, Postgres connection pool, Redis cache efficiency,
            per-source retrieval latency, and Prometheus metric samples.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Reads / Writes</div>
                <div class="kpi-value">${formatNumber(h.memory_reads_total || 0)} / ${formatNumber(h.memory_writes_total || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Timeouts / Step Failures</div>
                <div class="kpi-value">${formatNumber(h.retrieval_timeouts_total || 0)} / ${formatNumber(h.retrieval_step_failures_total || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">DB Pool (Out / Size)</div>
                <div class="kpi-value">${poolSummary}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Redis Cache (Hit / Mem)</div>
                <div class="kpi-value" style="font-size:1.05rem;">${redisHitPct} • ${h.redis_used_memory_mb != null ? `${formatFloat(h.redis_used_memory_mb, 1)} MB` : '—'}</div>
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">CML v3.0 Runtime Status</div>
                <div class="detail-field">
                    <span class="detail-field-label">Process Uptime</span>
                    <div class="detail-field-value">${formatUptime(h.uptime_seconds)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">v3.0 Whole-Brain Subsystems</span>
                    <div class="detail-field-value">
                        <span class="badge badge-ok">${formatNumber(runtime.v3_subsystems_enabled ?? 0)} / ${formatNumber(runtime.v3_subsystems_total ?? 10)} enabled</span>
                    </div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">LLM Mode</span>
                    <div class="detail-field-value">
                        <span class="badge ${runtime.llm_enabled ? 'badge-ok' : 'badge-warning'}">${runtime.llm_enabled ? 'Unified LLM' : 'Heuristic Only'}</span>
                        ${runtime.llm_model ? `<span class="mono" style="margin-left:8px;font-size:0.8rem;">${escapeHtml(runtime.llm_provider || '')}/${escapeHtml(runtime.llm_model)}</span>` : ''}
                    </div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Embedding Backend</span>
                    <div class="detail-field-value mono" style="font-size:0.82rem;">${escapeHtml(runtime.embedding_provider || '—')} / ${escapeHtml(runtime.embedding_model || '—')}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Generated</span>
                    <div class="detail-field-value">${formatDate(data.generated_at)}</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Infrastructure & Corpus Snapshot</div>
                <div class="detail-field">
                    <span class="detail-field-label">Postgres Pool</span>
                    <div class="detail-field-value">
                        size ${formatNumber(h.db_pool_size ?? 0)} • checked in ${formatNumber(h.db_pool_checked_in ?? 0)} • out ${formatNumber(h.db_pool_checked_out ?? 0)} • overflow ${formatNumber(h.db_pool_overflow ?? 0)}
                    </div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Redis Keys / Hit Rate</span>
                    <div class="detail-field-value">${formatNumber(h.redis_keys ?? 0)} keys • ${redisHitPct} hit rate (${formatNumber(h.redis_keyspace_hits ?? 0)} hits / ${formatNumber(h.redis_keyspace_misses ?? 0)} misses)</div>
                </div>
                ${runtime.memory_records_total != null ? `
                    <div class="detail-field">
                        <span class="detail-field-label">Stored Corpus</span>
                        <div class="detail-field-value">
                            ${formatNumber(runtime.memory_records_total)} memories • ${formatNumber(runtime.semantic_facts_total || 0)} facts • ${formatNumber(runtime.hyperedge_engrams_total || 0)} hyperedges • ${formatNumber(runtime.dmn_profiles_total || 0)} DMN profiles • ${formatNumber(runtime.procedural_skills_total || 0)} skills
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>

        ${stepEntries.length ? `
            <div class="card" style="margin-top:16px;">
                <div class="card-title">Retrieval Step Average Latency (ms)</div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Retrieval Source</th><th>Avg Latency (ms)</th></tr></thead>
                        <tbody>
                            ${stepEntries.map(([src, ms]) => `
                                <tr>
                                    <td class="mono">${escapeHtml(src)}</td>
                                    <td class="mono">${formatFloat(ms, 2)} ms</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        ` : ''}

        <div class="card" style="margin-top:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px;">
                <div class="card-title" style="margin-bottom:0;">Prometheus Metric Samples <span id="ops-metrics-count" style="font-size:0.78rem;color:var(--text-muted);font-weight:400;margin-left:6px;">${allMetrics.length} / ${allMetrics.length}</span></div>
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <input type="text" id="ops-metric-search" class="input-sm" placeholder="Filter metric name or label..." style="min-width:220px;">
                    <label style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:var(--text-secondary);cursor:pointer;">
                        <input type="checkbox" id="ops-hide-zeros"> Hide zero values
                    </label>
                </div>
            </div>
            ${allMetrics.length ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Name</th><th>Labels</th><th>Value</th></tr></thead>
                        <tbody id="ops-metrics-tbody">
                            ${renderMetricsRows(allMetrics, '', false)}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;">No metric samples exposed yet. Run a read or write to populate counters.</div>'}
        </div>
    `;
}
