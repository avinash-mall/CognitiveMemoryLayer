/**
 * Memory Detail Page
 * Full lifecycle view for a single memory record.
 */

import { getMemoryLineage } from '../api.js';
import { navigateTo } from '../app.js';
import {
    escapeHtml,
    formatDate,
    formatFloat,
    formatNumber,
    prettyJson,
    statusBadgeClass,
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

    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading memory lifecycle...</div>`;

    try {
        const lineage = await getMemoryLineage(memoryId);
        el.innerHTML = buildPage(lineage);
        attachListeners();
    } catch (err) {
        el.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">&#9888;</div>
            <p>Failed to load memory lifecycle: ${escapeHtml(err.message)}</p>
        </div>`;
    }
}

function buildPage(lineage) {
    const mem = lineage.memory;
    const statusCls = statusBadgeClass(mem.status);
    const confPct = Math.round((mem.confidence || 0) * 100);
    const impPct = Math.round((mem.importance || 0) * 100);
    const flags = lineage.lifecycle_flags || [];

    return `
        <a class="back-link" id="detail-back" href="#memories">&#8592; Back to Memory Explorer</a>

        <div class="detail-header">
            <h2 style="font-size:1.1rem;">Memory Lifecycle</h2>
            <span class="badge badge-type">${escapeHtml(mem.type)}</span>
            <span class="${statusCls}">${escapeHtml(mem.status)}</span>
            ${mem.labile ? '<span class="badge badge-labile">labile</span>' : ''}
            ${mem.version > 1 ? `<span class="badge badge-unknown">v${mem.version}</span>` : ''}
        </div>
        <div class="detail-id">ID: ${escapeHtml(mem.id)}</div>

        ${flags.length ? `
            <div class="card" style="margin:14px 0 18px;">
                <div class="card-title">Lifecycle Flags</div>
                <div class="tag-list">${flags.map(flag => `<span class="tag">${escapeHtml(flag)}</span>`).join('')}</div>
            </div>
        ` : ''}

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
                </div>
            ` : ''}
            ${mem.namespace ? `
                <div class="detail-field">
                    <span class="detail-field-label">Namespace</span>
                    <div class="detail-field-value mono">${escapeHtml(mem.namespace)}</div>
                </div>
            ` : ''}
            ${mem.context_tags?.length ? `
                <div class="detail-field">
                    <span class="detail-field-label">Context Tags</span>
                    <div class="tag-list">${mem.context_tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div>
                </div>
            ` : ''}
            ${mem.source_session_id ? `
                <div class="detail-field">
                    <span class="detail-field-label">Session ID</span>
                    <div class="detail-field-value mono">${escapeHtml(mem.source_session_id)}</div>
                </div>
            ` : ''}
        </div>

        <div class="detail-grid">
            <div class="card">
                <div class="card-title">Scores</div>
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

            <div class="card">
                <div class="card-title">Temporal</div>
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
                    </div>
                ` : ''}
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Version Lineage</div>
                ${buildMemoryList(lineage.ancestors, 'Ancestors')}
                ${buildMemoryList(lineage.descendants, 'Descendants')}
                ${buildMemoryList(lineage.same_key_versions, 'Same Key Versions')}
            </div>
            <div class="card">
                <div class="card-title">Provenance</div>
                <div class="json-tree">${escapeHtml(prettyJson(mem.provenance))}</div>
                ${mem.content_hash ? `
                    <div class="detail-field" style="margin-top:12px;">
                        <span class="detail-field-label">Content Hash</span>
                        <div class="detail-field-value mono">${escapeHtml(mem.content_hash)}</div>
                    </div>
                ` : ''}
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Downstream Facts</div>
                ${buildFactList(lineage.evidence_facts)}
            </div>
            <div class="card">
                <div class="card-title">Related Graph Entities</div>
                ${buildEntityList(lineage.related_entities)}
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            ${(mem.entities && (Array.isArray(mem.entities) ? mem.entities.length : true)) ? `
                <div class="card">
                    <div class="card-title">Entities</div>
                    <div class="json-tree">${escapeHtml(prettyJson(mem.entities))}</div>
                </div>
            ` : ''}
            ${(mem.relations && (Array.isArray(mem.relations) ? mem.relations.length : true)) ? `
                <div class="card">
                    <div class="card-title">Relations</div>
                    <div class="json-tree">${escapeHtml(prettyJson(mem.relations))}</div>
                </div>
            ` : ''}
        </div>

        ${mem.metadata && Object.keys(mem.metadata).length ? `
            <div class="card" style="margin-top:16px;">
                <div class="card-title">Metadata</div>
                <div class="json-tree">${escapeHtml(prettyJson(mem.metadata))}</div>
            </div>
        ` : ''}

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Related Events</div>
                ${buildEventList(mem.related_events)}
            </div>
            <div class="card">
                <div class="card-title">Related Jobs</div>
                ${buildJobList(lineage.related_jobs)}
            </div>
        </div>
    `;
}

function buildMemoryList(items = [], title) {
    if (!items.length) {
        return `<div class="detail-field">
            <span class="detail-field-label">${escapeHtml(title)}</span>
            <div class="detail-field-value" style="color:var(--text-muted)">None</div>
        </div>`;
    }

    return `
        <div class="detail-field">
            <span class="detail-field-label">${escapeHtml(title)}</span>
            <div class="stack-list">
                ${items.map(item => `
                    <a href="#detail/${item.id}" class="stack-item">
                        <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                            <span>
                                <span class="badge badge-type">${escapeHtml(item.type)}</span>
                                <span class="${statusBadgeClass(item.status)}" style="margin-left:4px;">${escapeHtml(item.status)}</span>
                                <span class="mono" style="margin-left:6px;font-size:0.8rem">v${item.version}</span>
                            </span>
                            <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.timestamp)}</span>
                        </div>
                        <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.text || '').slice(0, 180)}</div>
                    </a>
                `).join('')}
            </div>
        </div>
    `;
}

function buildFactList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No downstream facts linked to this memory.</div>';
    }

    return `
        <div class="stack-list">
            ${items.map(item => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span class="mono" style="font-size:0.8rem">${escapeHtml(item.key)}</span>
                        <span class="badge ${item.is_current ? 'badge-ok' : 'badge-status-archived'}">${item.is_current ? 'current' : 'superseded'}</span>
                    </div>
                    <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.value)}</div>
                    <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);">
                        ${escapeHtml(item.category)} • confidence ${formatFloat(item.confidence, 3)} • evidence ${formatNumber(item.evidence_count)}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function buildEntityList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No related graph entities found.</div>';
    }

    return `
        <div class="stack-list">
            ${items.map(item => `
                <a class="stack-item" href="#graph">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span>${escapeHtml(item.entity)}</span>
                        <span class="badge badge-type">${escapeHtml(item.entity_type || 'entity')}</span>
                    </div>
                    <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);">
                        tenant ${escapeHtml(item.tenant_id)} • scope ${escapeHtml(item.scope_id)}
                    </div>
                </a>
            `).join('')}
        </div>
    `;
}

function buildEventList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No related events.</div>';
    }

    return `
        <div class="stack-list">
            ${items.map(item => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span>
                            <span class="badge badge-type">${escapeHtml(item.event_type || 'event')}</span>
                            ${item.operation ? `<span style="margin-left:6px;color:var(--text-muted)">${escapeHtml(item.operation)}</span>` : ''}
                        </span>
                        <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.created_at)}</span>
                    </div>
                    <div style="margin-top:8px;" class="mono">${escapeHtml(item.id || '')}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function buildJobList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No related jobs found in recent history.</div>';
    }

    return `
        <div class="stack-list">
            ${items.map(item => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span>
                            <span class="badge badge-type">${escapeHtml(item.job_type)}</span>
                            <span class="badge ${item.status === 'completed' ? 'badge-ok' : item.status === 'failed' ? 'badge-error' : 'badge-warning'}" style="margin-left:6px;">${escapeHtml(item.status)}</span>
                        </span>
                        <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.started_at)}</span>
                    </div>
                    ${item.result ? `
                        <div class="json-tree" style="margin-top:8px;max-height:140px;overflow:auto;">${escapeHtml(prettyJson(item.result))}</div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

function attachListeners() {
    document.getElementById('detail-back')?.addEventListener('click', (event) => {
        event.preventDefault();
        navigateTo('memories');
    });
}
