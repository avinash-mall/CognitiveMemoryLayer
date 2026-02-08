/**
 * Memory Detail Page
 * Full detail view for a single memory record.
 */

import { getMemoryDetail } from '../api.js';
import { navigateTo } from '../app.js';
import {
    formatDate, formatFloat, formatNumber, statusBadgeClass,
    prettyJson, escapeHtml,
} from '../utils/formatters.js';

const container = () => document.getElementById('page-detail');

export async function renderDetail({ memoryId } = {}) {
    const el = container();

    if (!memoryId) {
        el.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">&#128269;</div>
            <p>No memory selected. Go to <a href="#memories">Memory Explorer</a> and click a row.</p>
        </div>`;
        return;
    }

    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading memory detail...</div>`;

    try {
        const mem = await getMemoryDetail(memoryId);
        el.innerHTML = buildDetail(mem);
        attachListeners();
    } catch (err) {
        el.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">&#9888;</div>
            <p>Failed to load memory: ${escapeHtml(err.message)}</p>
        </div>`;
    }
}

function buildDetail(mem) {
    const statusCls = statusBadgeClass(mem.status);
    const confPct = Math.round((mem.confidence || 0) * 100);
    const impPct = Math.round((mem.importance || 0) * 100);

    return `
        <a class="back-link" id="detail-back" href="#memories">&#8592; Back to Memory Explorer</a>

        <!-- Header -->
        <div class="detail-header">
            <h2 style="font-size:1.1rem;">Memory Detail</h2>
            <span class="badge badge-type">${mem.type}</span>
            <span class="${statusCls}">${mem.status}</span>
            ${mem.labile ? '<span class="badge badge-labile">labile</span>' : ''}
            ${mem.version > 1 ? `<span class="badge badge-unknown">v${mem.version}</span>` : ''}
        </div>
        <div class="detail-id">ID: ${mem.id}</div>

        <!-- Content -->
        <h3 class="section-title">Content</h3>
        <div class="card" style="margin-bottom:16px;">
            <div class="detail-field">
                <span class="detail-field-label">Text</span>
                <div class="detail-field-value" style="white-space:pre-wrap;">${escapeHtml(mem.text)}</div>
            </div>
            ${mem.key ? `
            <div class="detail-field">
                <span class="detail-field-label">Key</span>
                <div class="detail-field-value mono">${escapeHtml(mem.key)}</div>
            </div>` : ''}
            ${mem.namespace ? `
            <div class="detail-field">
                <span class="detail-field-label">Namespace</span>
                <div class="detail-field-value mono">${escapeHtml(mem.namespace)}</div>
            </div>` : ''}
            ${mem.context_tags?.length ? `
            <div class="detail-field">
                <span class="detail-field-label">Context Tags</span>
                <div class="tag-list">${mem.context_tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>
            </div>` : ''}
            ${mem.source_session_id ? `
            <div class="detail-field">
                <span class="detail-field-label">Session ID</span>
                <div class="detail-field-value mono">${escapeHtml(mem.source_session_id)}</div>
            </div>` : ''}
        </div>

        <!-- Metrics -->
        <h3 class="section-title">Metrics</h3>
        <div class="detail-grid">
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Confidence</span>
                    <div class="detail-field-value">${formatFloat(mem.confidence, 4)}</div>
                    <div class="gauge-bar"><div class="gauge-fill ${confPct >= 70 ? 'gauge-fill-success' : confPct >= 40 ? 'gauge-fill-accent' : 'gauge-fill-warning'}" style="width:${confPct}%"></div></div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Importance</span>
                    <div class="detail-field-value">${formatFloat(mem.importance, 4)}</div>
                    <div class="gauge-bar"><div class="gauge-fill ${impPct >= 70 ? 'gauge-fill-success' : impPct >= 40 ? 'gauge-fill-accent' : 'gauge-fill-warning'}" style="width:${impPct}%"></div></div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Access Count</span>
                    <div class="detail-field-value">${formatNumber(mem.access_count)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Decay Rate</span>
                    <div class="detail-field-value">${formatFloat(mem.decay_rate, 4)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Last Accessed</span>
                    <div class="detail-field-value">${formatDate(mem.last_accessed_at)}</div>
                </div>
            </div>

            <!-- Temporal -->
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Timestamp</span>
                    <div class="detail-field-value">${formatDate(mem.timestamp)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Written At</span>
                    <div class="detail-field-value">${formatDate(mem.written_at)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Valid From</span>
                    <div class="detail-field-value">${formatDate(mem.valid_from)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Valid To</span>
                    <div class="detail-field-value">${formatDate(mem.valid_to)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Tenant</span>
                    <div class="detail-field-value mono">${escapeHtml(mem.tenant_id)}</div>
                </div>
                ${mem.agent_id ? `
                <div class="detail-field">
                    <span class="detail-field-label">Agent ID</span>
                    <div class="detail-field-value mono">${escapeHtml(mem.agent_id)}</div>
                </div>` : ''}
            </div>
        </div>

        <!-- Provenance -->
        <h3 class="section-title">Provenance</h3>
        <div class="detail-grid">
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Version</span>
                    <div class="detail-field-value">${mem.version}</div>
                </div>
                ${mem.supersedes_id ? `
                <div class="detail-field">
                    <span class="detail-field-label">Supersedes</span>
                    <div class="detail-field-value mono"><a href="#detail/${mem.supersedes_id}">${mem.supersedes_id}</a></div>
                </div>` : ''}
                ${mem.content_hash ? `
                <div class="detail-field">
                    <span class="detail-field-label">Content Hash</span>
                    <div class="detail-field-value mono">${escapeHtml(mem.content_hash)}</div>
                </div>` : ''}
            </div>
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Provenance Data</span>
                    <div class="json-tree">${escapeHtml(prettyJson(mem.provenance))}</div>
                </div>
            </div>
        </div>

        <!-- Entities & Relations -->
        ${(mem.entities && (Array.isArray(mem.entities) ? mem.entities.length : true)) || (mem.relations && (Array.isArray(mem.relations) ? mem.relations.length : true)) ? `
        <h3 class="section-title">Entities & Relations</h3>
        <div class="detail-grid">
            ${mem.entities && (Array.isArray(mem.entities) ? mem.entities.length : true) ? `
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Entities</span>
                    <div class="json-tree">${escapeHtml(prettyJson(mem.entities))}</div>
                </div>
            </div>` : ''}
            ${mem.relations && (Array.isArray(mem.relations) ? mem.relations.length : true) ? `
            <div class="card">
                <div class="detail-field">
                    <span class="detail-field-label">Relations</span>
                    <div class="json-tree">${escapeHtml(prettyJson(mem.relations))}</div>
                </div>
            </div>` : ''}
        </div>` : ''}

        <!-- Metadata -->
        ${mem.metadata && Object.keys(mem.metadata).length ? `
        <h3 class="section-title">Metadata</h3>
        <div class="card" style="margin-bottom:16px;">
            <div class="json-tree">${escapeHtml(prettyJson(mem.metadata))}</div>
        </div>` : ''}

        <!-- Related Events -->
        ${mem.related_events?.length ? `
        <h3 class="section-title">Related Events (${mem.related_events.length})</h3>
        <div class="table-wrapper" style="margin-bottom:24px;">
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Type</th>
                        <th>Operation</th>
                        <th>Created</th>
                    </tr>
                </thead>
                <tbody>
                    ${mem.related_events.map(e => `
                    <tr>
                        <td><code class="mono" style="font-size:0.8rem;">${e.id ? e.id.slice(0, 8) + '...' : '-'}</code></td>
                        <td><span class="badge badge-type">${e.event_type || '-'}</span></td>
                        <td>${e.operation || '-'}</td>
                        <td>${formatDate(e.created_at)}</td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>` : ''}
    `;
}

function attachListeners() {
    document.getElementById('detail-back')?.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo('memories');
    });
}
