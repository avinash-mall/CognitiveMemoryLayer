/**
 * Facts Explorer Page
 * Browse, filter, and invalidate semantic facts.
 */

import { getFacts, invalidateFact, getTenants } from '../api.js';
import { showToast } from '../app.js';
import { formatNumber, formatFloat, formatDate, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-facts');

let tenants = [];
let currentPage = 1;
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
        </div>

        <div id="facts-summary" class="kpi-grid" style="margin-bottom:16px;"></div>

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

    el.querySelector('#facts-search-btn')?.addEventListener('click', () => {
        currentPage = 1;
        loadFacts(el);
    });

    el.querySelector('#facts-category')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { currentPage = 1; loadFacts(el); }
    });
}
