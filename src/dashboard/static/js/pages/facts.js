/**
 * Facts Explorer Page
 * Browse, filter, and invalidate semantic facts.
 */

import {
    exportFacts,
    getFactDetail,
    getFactEvidence,
    getFacts,
    getTenants,
    invalidateFact,
} from '../api.js';
import { navigateTo, showToast } from '../app.js';
import { formatNumber, formatFloat, formatDate, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-facts');

let tenants = [];
let currentPage = 1;
let selectedFactId = null;
const PER_PAGE = 50;

export async function renderFacts({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading facts...</div>`;

    try {
        const tenantsData = await getTenants().catch(() => ({ tenants: [] }));
        tenants = tenantsData.tenants || [];
    } catch { tenants = []; }

    currentPage = 1;
    el.innerHTML = buildPage(tenantId);
    attachListeners(el, tenantId);
    await loadFacts(el);
}

function buildPage(tenantId) {
    const tenantOptions = tenants.map(t =>
        `<option value="${escapeHtml(t.tenant_id)}" ${t.tenant_id === tenantId ? 'selected' : ''}>${escapeHtml(t.tenant_id)}</option>`
    ).join('');

    return `
        <p class="page-desc">Semantic facts extracted from episodic memories during consolidation. Facts track beliefs with confidence, evidence counts, and version lineage.</p>

        <div class="filter-bar">
            <select id="facts-tenant" class="select-sm" title="Filter by tenant">
                <option value="">All Tenants</option>
                ${tenantOptions}
            </select>
            <input type="text" id="facts-category" class="input-sm" placeholder="Category filter..." style="min-width:150px">
            <div class="toggle-wrapper">
                <div id="facts-current-toggle" class="toggle-switch active" title="Show only current facts"></div>
                <span id="facts-current-label" style="font-size:0.85rem;color:var(--text-secondary);cursor:pointer;">Current only</span>
            </div>
            <button id="facts-search-btn" class="btn btn-primary btn-sm">Apply</button>
            <button id="facts-export" class="btn btn-ghost btn-sm" title="Export facts as JSON">&#8681; Export</button>
        </div>

        <div id="facts-summary" class="kpi-grid" style="margin-bottom:16px;"></div>

        <div id="fact-detail-panel" class="hidden" style="margin-bottom:16px;"></div>

        <div id="facts-table-container">
            <div class="loading-overlay"><div class="spinner"></div> Loading...</div>
        </div>

        <div id="facts-pagination" class="pagination hidden"></div>
    `;
}

async function loadFacts(el) {
    const tableContainer = el.querySelector('#facts-table-container');
    const summaryDiv = el.querySelector('#facts-summary');
    const paginationDiv = el.querySelector('#facts-pagination');
    const detailPanel = el.querySelector('#fact-detail-panel');
    tableContainer.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading...</div>`;

    const tenantId = el.querySelector('#facts-tenant')?.value || undefined;
    const category = el.querySelector('#facts-category')?.value?.trim() || undefined;
    const currentToggle = el.querySelector('#facts-current-toggle');
    const currentOnly = currentToggle?.classList.contains('active') ?? true;
    const offset = (currentPage - 1) * PER_PAGE;

    try {
        const data = await getFacts({ tenantId, category, currentOnly, limit: PER_PAGE, offset });
        const items = data.items || [];
        const total = data.total || 0;
        const totalPages = Math.ceil(total / PER_PAGE);

        summaryDiv.innerHTML = `
            <div class="kpi-card"><div class="kpi-label">Total Facts</div><div class="kpi-value">${formatNumber(total)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Showing</div><div class="kpi-value">${formatNumber(items.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Page</div><div class="kpi-value">${currentPage} / ${Math.max(1, totalPages)}</div></div>
        `;

        if (items.length === 0) {
            tableContainer.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#128218;</div><p>No semantic facts found matching the current filters.</p><p style="color:var(--text-muted);margin-top:6px;">Run consolidation to extract facts from episodic memories.</p></div>`;
            paginationDiv.classList.add('hidden');
            return;
        }

        tableContainer.innerHTML = `
            <div class="table-responsive">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Key</th>
                            <th>Value</th>
                            <th>Category</th>
                            <th>Confidence</th>
                            <th>Evidence</th>
                            <th>Version</th>
                            <th>Status</th>
                            <th>Updated</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${items.map(f => `
                            <tr>
                                <td><span class="mono" style="font-size:0.82rem">${escapeHtml(f.key)}</span></td>
                                <td class="text-preview" style="max-width:280px">${escapeHtml(f.value)}</td>
                                <td><span class="badge badge-type">${escapeHtml(f.category)}</span></td>
                                <td>
                                    <div style="display:flex;align-items:center;gap:6px;">
                                        <div class="gauge-bar" style="width:60px;display:inline-flex">
                                            <div class="gauge-fill ${f.confidence >= 0.7 ? 'gauge-fill-success' : f.confidence >= 0.4 ? 'gauge-fill-accent' : 'gauge-fill-warning'}" style="width:${Math.min(100, f.confidence * 100)}%"></div>
                                        </div>
                                        <span style="font-size:0.82rem">${formatFloat(f.confidence, 3)}</span>
                                    </div>
                                </td>
                                <td>${formatNumber(f.evidence_count)}</td>
                                <td>v${f.version}</td>
                                <td>${f.is_current ? '<span class="badge badge-ok">current</span>' : '<span class="badge badge-status-archived">superseded</span>'}</td>
                                <td style="font-size:0.82rem">${formatDate(f.updated_at)}</td>
                                <td>
                                    <button class="btn btn-ghost btn-xs fact-view-btn" data-fact-id="${escapeHtml(f.id)}">Inspect</button>
                                    ${f.is_current ? `<button class="btn btn-ghost btn-xs fact-invalidate-btn" data-fact-id="${escapeHtml(f.id)}" data-fact-key="${escapeHtml(f.key)}">Invalidate</button>` : '<span style="color:var(--text-muted);font-size:0.78rem">—</span>'}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        if (totalPages > 1) {
            paginationDiv.classList.remove('hidden');
            paginationDiv.innerHTML = `
                <span class="pagination-info">${formatNumber(total)} facts total</span>
                <div class="pagination-controls">
                    <button class="btn btn-ghost btn-sm" id="facts-prev" ${currentPage <= 1 ? 'disabled' : ''}>Previous</button>
                    <span style="font-size:0.85rem;color:var(--text-secondary)">Page ${currentPage} of ${totalPages}</span>
                    <button class="btn btn-ghost btn-sm" id="facts-next" ${currentPage >= totalPages ? 'disabled' : ''}>Next</button>
                </div>
            `;
            el.querySelector('#facts-prev')?.addEventListener('click', () => { if (currentPage > 1) { currentPage--; loadFacts(el); } });
            el.querySelector('#facts-next')?.addEventListener('click', () => { if (currentPage < totalPages) { currentPage++; loadFacts(el); } });
        } else {
            paginationDiv.classList.add('hidden');
        }

        tableContainer.querySelectorAll('.fact-view-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                selectedFactId = btn.dataset.factId;
                detailPanel.classList.remove('hidden');
                detailPanel.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading fact detail...</div>`;
                try {
                    const [detail, evidence] = await Promise.all([
                        getFactDetail(selectedFactId),
                        getFactEvidence(selectedFactId),
                    ]);
                    detailPanel.innerHTML = buildFactDetail(detail, evidence);
                    attachFactDetailListeners(detailPanel);
                } catch (err) {
                    detailPanel.innerHTML = `<div class="card"><div class="empty-state" style="color:var(--danger)">Failed to load fact detail: ${escapeHtml(err.message)}</div></div>`;
                }
            });
        });

        tableContainer.querySelectorAll('.fact-invalidate-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const factId = btn.dataset.factId;
                const factKey = btn.dataset.factKey;
                if (!confirm(`Invalidate fact "${factKey}"? This marks it as no longer current.`)) return;
                btn.disabled = true;
                btn.textContent = '...';
                try {
                    await invalidateFact(factId);
                    showToast(`Fact "${factKey}" invalidated`);
                    await loadFacts(el);
                    if (selectedFactId === factId) {
                        detailPanel.classList.add('hidden');
                        detailPanel.innerHTML = '';
                    }
                } catch (err) {
                    showToast(`Failed: ${err.message}`, 'error');
                    btn.disabled = false;
                    btn.textContent = 'Invalidate';
                }
            });
        });
    } catch (err) {
        tableContainer.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load facts: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildFactDetail(detail, evidence) {
    return `
        <div class="card">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px;">
                <div class="card-title" style="margin:0;">Fact Inspection</div>
                <button class="btn btn-ghost btn-xs" id="fact-detail-close">Close</button>
            </div>
            <div class="detail-grid">
                <div>
                    <div class="detail-field">
                        <span class="detail-field-label">Key</span>
                        <div class="detail-field-value mono">${escapeHtml(detail.key)}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Subject / Predicate</span>
                        <div class="detail-field-value">${escapeHtml(detail.subject)} / ${escapeHtml(detail.predicate)}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Value</span>
                        <div class="detail-field-value">${escapeHtml(String(detail.value))}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Confidence</span>
                        <div class="detail-field-value">${formatFloat(detail.confidence, 3)}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Evidence Count</span>
                        <div class="detail-field-value">${formatNumber(detail.evidence_count)}</div>
                    </div>
                </div>
                <div>
                    <div class="detail-field">
                        <span class="detail-field-label">Category</span>
                        <div class="detail-field-value"><span class="badge badge-type">${escapeHtml(detail.category)}</span></div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Status</span>
                        <div class="detail-field-value">${detail.is_current ? '<span class="badge badge-ok">current</span>' : '<span class="badge badge-status-archived">superseded</span>'}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Version</span>
                        <div class="detail-field-value">v${detail.version}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Created / Updated</span>
                        <div class="detail-field-value">${formatDate(detail.created_at)} / ${formatDate(detail.updated_at)}</div>
                    </div>
                    <div class="detail-field">
                        <span class="detail-field-label">Context Tags</span>
                        <div class="tag-list">${(detail.context_tags || []).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('') || '<span style="color:var(--text-muted)">None</span>'}</div>
                    </div>
                </div>
            </div>

            <div class="detail-grid" style="margin-top:16px;">
                <div class="card" style="margin:0;">
                    <div class="card-title">Supersession Lineage</div>
                    <div class="json-tree">${escapeHtml(JSON.stringify(detail.lineage || [], null, 2))}</div>
                    ${(detail.superseded_by || []).length ? `
                        <div class="card-title" style="margin-top:12px;">Superseded By</div>
                        <div class="json-tree">${escapeHtml(JSON.stringify(detail.superseded_by || [], null, 2))}</div>
                    ` : ''}
                </div>
                <div class="card" style="margin:0;">
                    <div class="card-title">Evidence Memories</div>
                    ${buildEvidenceList(evidence)}
                </div>
            </div>
        </div>
    `;
}

function buildEvidenceList(evidence) {
    if (!evidence?.evidence?.length && !evidence?.missing_evidence_ids?.length) {
        return '<div class="empty-state" style="padding:12px;">No evidence linked.</div>';
    }

    const linked = (evidence.evidence || []).map(item => `
        <div class="stack-item">
            <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                <span>
                    <span class="badge badge-type">${escapeHtml(item.type)}</span>
                    <span class="badge badge-ok" style="margin-left:6px;">${escapeHtml(item.status)}</span>
                </span>
                <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.timestamp)}</span>
            </div>
            <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.text)}</div>
            <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);display:flex;gap:12px;flex-wrap:wrap;">
                <span>confidence ${formatFloat(item.confidence, 3)}</span>
                <a href="#detail/${item.id}" class="fact-evidence-link" data-memory-id="${escapeHtml(item.id)}">open memory</a>
            </div>
        </div>
    `).join('');

    const missing = (evidence.missing_evidence_ids || []).length
        ? `<div class="stack-item"><strong>Missing evidence ids</strong><div class="mono" style="margin-top:8px;">${escapeHtml(evidence.missing_evidence_ids.join(', '))}</div></div>`
        : '';

    return `<div class="stack-list">${linked}${missing}</div>`;
}

function attachFactDetailListeners(panel) {
    panel.querySelector('#fact-detail-close')?.addEventListener('click', () => {
        panel.classList.add('hidden');
        panel.innerHTML = '';
        selectedFactId = null;
    });

    panel.querySelectorAll('.fact-evidence-link').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            navigateTo('detail', { memoryId: link.dataset.memoryId });
        });
    });
}

function attachListeners(el, initialTenantId) {
    let currentOnly = true;
    const currentToggle = el.querySelector('#facts-current-toggle');
    const currentLabel = el.querySelector('#facts-current-label');

    const toggleCurrent = () => {
        currentOnly = !currentOnly;
        currentToggle?.classList.toggle('active', currentOnly);
        if (currentLabel) currentLabel.textContent = currentOnly ? 'Current only' : 'All versions';
    };
    currentToggle?.addEventListener('click', toggleCurrent);
    currentLabel?.addEventListener('click', toggleCurrent);

    el.querySelector('#facts-export')?.addEventListener('click', () => {
        const tenantId = el.querySelector('#facts-tenant')?.value || undefined;
        exportFacts(tenantId);
    });

    el.querySelector('#facts-search-btn')?.addEventListener('click', () => {
        currentPage = 1;
        loadFacts(el);
    });

    el.querySelector('#facts-category')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { currentPage = 1; loadFacts(el); }
    });
}
