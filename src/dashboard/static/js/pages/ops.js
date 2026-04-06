/**
 * Operations Metrics Page
 * Prometheus/runtime metric highlights and raw samples.
 */

import { getOpsMetrics } from '../api.js';
import { escapeHtml, formatDate, formatFloat, formatNumber, prettyJson } from '../utils/formatters.js';

const container = () => document.getElementById('page-ops');

export async function renderOps() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading operations metrics...</div>`;

    try {
        const data = await getOpsMetrics();
        el.innerHTML = buildPage(data);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load operations metrics: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildPage(data) {
    const highlights = data.highlights || {};
    const metrics = data.metrics || [];

    return `
        <p class="page-desc">
            Operational view of Prometheus and runtime internals already instrumented in the service:
            retrieval timeouts, step failures, DB pool usage, request counters, Redis usage, and fact-hit signals.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Retrieval Timeouts</div><div class="kpi-value">${formatNumber(highlights.retrieval_timeouts_total || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Step Failures</div><div class="kpi-value">${formatNumber(highlights.retrieval_step_failures_total || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">DB Pool Checked Out</div><div class="kpi-value">${formatNumber(highlights.db_pool_checked_out || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Fact Hit Rate</div><div class="kpi-value">${highlights.fact_hit_rate != null ? `${formatFloat(highlights.fact_hit_rate * 100, 1)}%` : '—'}</div></div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Runtime Highlights</div>
                <div class="json-tree">${escapeHtml(prettyJson(highlights))}</div>
            </div>
            <div class="card">
                <div class="card-title">Generated</div>
                <div class="detail-field">
                    <span class="detail-field-label">Timestamp</span>
                    <div class="detail-field-value">${formatDate(data.generated_at)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Metric Samples</span>
                    <div class="detail-field-value">${formatNumber(metrics.length)}</div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Raw Metric Samples</div>
            ${metrics.length ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Name</th><th>Labels</th><th>Value</th></tr></thead>
                        <tbody>
                            ${metrics.map(metric => `
                                <tr>
                                    <td class="mono" style="font-size:0.78rem;">${escapeHtml(metric.name)}</td>
                                    <td class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson(metric.labels || {}))}</td>
                                    <td>${typeof metric.value === 'number' ? formatFloat(metric.value, 4) : escapeHtml(String(metric.value))}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;">No metric samples exposed.</div>'}
        </div>
    `;
}
