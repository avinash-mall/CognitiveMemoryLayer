/**
 * API Usage & Rate Limits Page
 * Shows current rate-limit usage per key and hourly request charts.
 */

import { getRateLimits, getRequestStats } from '../api.js';
import { formatNumber, escapeHtml } from '../utils/formatters.js';
import { createLineChart } from '../utils/charts.js';

const container = () => document.getElementById('page-apiusage');

export async function renderApiUsage() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading API usage data...</div>`;

    try {
        const [rl, stats] = await Promise.all([getRateLimits(), getRequestStats(24)]);
        el.innerHTML = buildPage(rl, stats);
        requestAnimationFrame(() => renderRequestChart(stats));
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load API usage: ${err.message}</p></div>`;
    }
}

function utilizationBar(pct) {
    let cls = 'gauge-fill-success';
    if (pct >= 80) cls = 'gauge-fill-danger';
    else if (pct >= 50) cls = 'gauge-fill-warning';
    return `<div class="gauge-bar" style="width:120px;display:inline-flex;vertical-align:middle;margin-right:8px"><div class="gauge-fill ${cls}" style="width:${Math.min(100, pct)}%"></div></div><span style="font-size:0.82rem">${pct.toFixed(1)}%</span>`;
}

function buildPage(rl, stats) {
    const avgUtil = rl.entries.length ? (rl.entries.reduce((s, e) => s + e.utilization_pct, 0) / rl.entries.length) : 0;

    return `
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Active Rate-Limit Keys</div><div class="kpi-value">${formatNumber(rl.entries.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Avg Utilization</div><div class="kpi-value">${avgUtil.toFixed(1)}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Configured RPM</div><div class="kpi-value">${formatNumber(rl.configured_rpm)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Requests (24h)</div><div class="kpi-value">${formatNumber(stats.total_last_24h)}</div></div>
        </div>

        <div class="chart-grid" style="margin-top:16px;">
            <div class="chart-card" style="grid-column: 1 / -1;">
                <div class="card-title">Hourly Request Volume (Last 24h)</div>
                <canvas id="chart-requests"></canvas>
            </div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Current Rate Limit Buckets</div>
            <div class="table-responsive">
                <table class="data-table">
                    <thead>
                        <tr><th>Type</th><th>Identifier</th><th>Current</th><th>Limit</th><th>Utilization</th><th>TTL</th></tr>
                    </thead>
                    <tbody>
                        ${rl.entries.map(e => `
                            <tr>
                                <td><span class="badge badge-type">${escapeHtml(e.key_type)}</span></td>
                                <td style="font-family:monospace;font-size:0.85rem">${escapeHtml(e.identifier)}</td>
                                <td>${formatNumber(e.current_count)}</td>
                                <td>${formatNumber(e.limit)}</td>
                                <td>${utilizationBar(e.utilization_pct)}</td>
                                <td>${e.ttl_seconds}s</td>
                            </tr>
                        `).join('')}
                        ${rl.entries.length === 0 ? '<tr><td colspan="6" class="empty-state">No active rate limit buckets</td></tr>' : ''}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

function renderRequestChart(stats) {
    const canvas = document.getElementById('chart-requests');
    if (!canvas || !stats.points.length) return;
    const labels = stats.points.map(p => {
        const d = new Date(p.hour);
        return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    });
    createLineChart(canvas, labels, stats.points.map(p => p.count));
}
