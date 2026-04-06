/**
 * Quality Dashboard Page
 * Data quality and graph hygiene overview.
 */

import { getQualityOverview } from '../api.js';
import { escapeHtml, formatDate, formatNumber } from '../utils/formatters.js';

const container = () => document.getElementById('page-quality');

export async function renderQuality({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading quality overview...</div>`;

    try {
        const data = await getQualityOverview(tenantId);
        el.innerHTML = buildPage(data);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load quality overview: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildPage(data) {
    const issues = data.issues || [];
    const hotspots = data.invalidation_hotspots || [];
    const graph = data.graph_hygiene || {};

    return `
        <p class="page-desc">
            Inspect low-confidence memories, weakly-supported facts, broken lineage, invalidation hotspots,
            and graph hygiene signals that often explain confusing memory behavior.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Issues Tracked</div><div class="kpi-value">${formatNumber(issues.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Labile Sessions</div><div class="kpi-value">${formatNumber(data.labile_sessions || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Duplicate Entity Groups</div><div class="kpi-value">${formatNumber(graph.duplicate_entity_groups || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Stale Graph Nodes</div><div class="kpi-value">${formatNumber(graph.stale_nodes || 0)}</div></div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Issue Breakdown</div>
            <div class="stack-list">
                ${issues.map(issue => `
                    <div class="stack-item">
                        <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                            <strong>${escapeHtml(issue.label)}</strong>
                            <span class="badge badge-warning">${formatNumber(issue.count)}</span>
                        </div>
                        <div style="margin-top:8px;color:var(--text-muted);">${escapeHtml(issue.description || '')}</div>
                        ${issue.sample_ids?.length ? `<div class="mono" style="margin-top:8px;font-size:0.78rem;">samples: ${escapeHtml(issue.sample_ids.join(', '))}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Invalidation Hotspots</div>
                ${hotspots.length ? `
                    <div class="table-responsive">
                        <table class="data-table">
                            <thead><tr><th>Fact Key</th><th>Invalidations</th></tr></thead>
                            <tbody>
                                ${hotspots.map(item => `
                                    <tr>
                                        <td class="mono" style="font-size:0.78rem;">${escapeHtml(item.key)}</td>
                                        <td>${formatNumber(item.count)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : '<div class="empty-state" style="padding:12px;">No invalidation hotspots detected.</div>'}
            </div>

            <div class="card">
                <div class="card-title">Graph Hygiene</div>
                <div class="detail-field">
                    <span class="detail-field-label">Duplicate Entity Groups</span>
                    <div class="detail-field-value">${formatNumber(graph.duplicate_entity_groups || 0)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Stale Nodes</span>
                    <div class="detail-field-value">${formatNumber(graph.stale_nodes || 0)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Generated</span>
                    <div class="detail-field-value">${formatDate(data.generated_at)}</div>
                </div>
            </div>
        </div>
    `;
}
