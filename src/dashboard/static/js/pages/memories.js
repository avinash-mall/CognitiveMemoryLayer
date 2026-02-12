/**
 * Memory Explorer Page
 * Filterable, sortable, paginated table of memory records with bulk actions and export.
 */

import { getMemories, bulkAction, exportMemories } from '../api.js';
import { navigateTo, showToast } from '../app.js';
import { formatDate, formatFloat, formatNumber, truncate, shortUuid, statusBadgeClass, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-memories');

// State
let state = {
    page: 1,
    perPage: 25,
    type: '',
    status: '',
    search: '',
    sortBy: 'timestamp',
    order: 'desc',
    tenantId: '',
    data: null,
    selectedIds: new Set(),
};

const MEMORY_TYPES = [
    'episodic_event', 'semantic_fact', 'procedure', 'constraint', 'hypothesis',
    'preference', 'task_state', 'conversation', 'message', 'tool_result',
    'reasoning_step', 'scratch', 'knowledge', 'observation', 'plan',
];

const MEMORY_STATUSES = ['active', 'silent', 'compressed', 'archived', 'deleted'];

export async function renderMemories({ tenantId } = {}) {
    state.tenantId = tenantId || '';
    state.selectedIds.clear();

    // Check for session filter from Sessions page
    const sessionFilter = sessionStorage.getItem('cml_filter_session_id');
    if (sessionFilter) {
        sessionStorage.removeItem('cml_filter_session_id');
    }

    const el = container();
    el.innerHTML = buildShell();
    attachListeners();
    await loadData(sessionFilter);
}

function buildShell() {
    return `
        <div class="filter-bar">
            <input type="text" id="mem-search" class="input-sm" placeholder="Search memory text..." value="${escapeHtml(state.search)}">
            <select id="mem-type" class="select-sm">
                <option value="">All Types</option>
                ${MEMORY_TYPES.map(t => `<option value="${t}" ${state.type === t ? 'selected' : ''}>${t}</option>`).join('')}
            </select>
            <select id="mem-status" class="select-sm">
                <option value="">All Statuses</option>
                ${MEMORY_STATUSES.map(s => `<option value="${s}" ${state.status === s ? 'selected' : ''}>${s}</option>`).join('')}
            </select>
            <select id="mem-sort" class="select-sm">
                <option value="timestamp" ${state.sortBy === 'timestamp' ? 'selected' : ''}>Sort: Timestamp</option>
                <option value="confidence" ${state.sortBy === 'confidence' ? 'selected' : ''}>Sort: Confidence</option>
                <option value="importance" ${state.sortBy === 'importance' ? 'selected' : ''}>Sort: Importance</option>
                <option value="access_count" ${state.sortBy === 'access_count' ? 'selected' : ''}>Sort: Access Count</option>
            </select>
            <select id="mem-order" class="select-sm">
                <option value="desc" ${state.order === 'desc' ? 'selected' : ''}>Descending</option>
                <option value="asc" ${state.order === 'asc' ? 'selected' : ''}>Ascending</option>
            </select>
            <button id="mem-apply" class="btn btn-primary btn-sm">Apply</button>
            <button id="mem-export" class="btn btn-ghost btn-sm" title="Export memories as JSON">Export</button>
        </div>
        <div id="mem-bulk-bar" class="bulk-action-bar hidden">
            <span id="mem-selected-count">0 selected</span>
            <button class="btn btn-ghost btn-xs" id="bulk-archive">Archive</button>
            <button class="btn btn-ghost btn-xs" id="bulk-silence">Silence</button>
            <button class="btn btn-danger btn-xs" id="bulk-delete">Delete</button>
            <button class="btn btn-ghost btn-xs" id="bulk-clear">Clear</button>
        </div>
        <div id="mem-table-area">
            <div class="loading-overlay"><div class="spinner"></div> Loading...</div>
        </div>
    `;
}

function attachListeners() {
    const el = container();

    el.querySelector('#mem-apply')?.addEventListener('click', () => {
        state.search = el.querySelector('#mem-search')?.value || '';
        state.type = el.querySelector('#mem-type')?.value || '';
        state.status = el.querySelector('#mem-status')?.value || '';
        state.sortBy = el.querySelector('#mem-sort')?.value || 'timestamp';
        state.order = el.querySelector('#mem-order')?.value || 'desc';
        state.page = 1;
        loadData();
    });

    el.querySelector('#mem-search')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') el.querySelector('#mem-apply')?.click();
    });

    el.querySelector('#mem-export')?.addEventListener('click', () => {
        exportMemories(state.tenantId);
    });

    // Bulk actions
    el.querySelector('#bulk-archive')?.addEventListener('click', () => doBulkAction('archive'));
    el.querySelector('#bulk-silence')?.addEventListener('click', () => doBulkAction('silence'));
    el.querySelector('#bulk-delete')?.addEventListener('click', () => doBulkAction('delete'));
    el.querySelector('#bulk-clear')?.addEventListener('click', () => {
        state.selectedIds.clear();
        updateBulkBar();
        document.querySelectorAll('#page-memories .mem-checkbox').forEach(cb => { cb.checked = false; });
    });
}

async function doBulkAction(action) {
    if (state.selectedIds.size === 0) return;
    const ids = [...state.selectedIds];
    try {
        const result = await bulkAction(ids, action);
        showToast(`${action}: ${result.affected} memories affected`);
        state.selectedIds.clear();
        updateBulkBar();
        loadData();
    } catch (err) {
        showToast(`Bulk ${action} failed: ${err.message}`, 'error');
    }
}

function updateBulkBar() {
    const bar = document.getElementById('mem-bulk-bar');
    const count = document.getElementById('mem-selected-count');
    if (!bar) return;
    if (state.selectedIds.size > 0) {
        bar.classList.remove('hidden');
        count.textContent = `${state.selectedIds.size} selected`;
    } else {
        bar.classList.add('hidden');
    }
}

async function loadData(sourceSessionId) {
    const area = document.getElementById('mem-table-area');
    if (!area) return;
    area.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading...</div>`;

    try {
        const data = await getMemories({
            page: state.page,
            perPage: state.perPage,
            type: state.type,
            status: state.status,
            search: state.search,
            tenantId: state.tenantId,
            sortBy: state.sortBy,
            order: state.order,
            sourceSessionId: sourceSessionId || undefined,
        });
        state.data = data;
        area.innerHTML = buildTable(data) + buildPagination(data);
        attachTableListeners();
    } catch (err) {
        area.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>${err.message}</p></div>`;
    }
}

function buildTable(data) {
    if (!data.items?.length) {
        return `<div class="empty-state"><div class="empty-state-icon">&#128466;</div><p>No memories found</p></div>`;
    }
    const rows = data.items.map(m => `
        <tr class="clickable-row" data-id="${m.id}">
            <td><input type="checkbox" class="mem-checkbox" data-id="${m.id}" ${state.selectedIds.has(String(m.id)) ? 'checked' : ''}></td>
            <td><code class="mono" style="font-size:0.8rem;">${shortUuid(String(m.id))}</code></td>
            <td><div class="text-preview">${escapeHtml(truncate(m.text, 100))}</div></td>
            <td><span class="badge badge-type">${m.type}</span></td>
            <td><span class="${statusBadgeClass(m.status)}">${m.status}</span>${m.labile ? ' <span class="badge badge-labile">labile</span>' : ''}</td>
            <td>${formatFloat(m.confidence, 3)}</td>
            <td>${formatFloat(m.importance, 3)}</td>
            <td>${formatNumber(m.access_count)}</td>
            <td style="white-space:nowrap;">${formatDate(m.timestamp)}</td>
        </tr>
    `).join('');

    return `
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th style="width:30px"><input type="checkbox" id="mem-select-all"></th>
                        <th>ID</th>
                        <th>Text</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Confidence</th>
                        <th>Importance</th>
                        <th>Accesses</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function buildPagination(data) {
    const { page, total_pages, total, per_page } = data;
    const start = Math.min((page - 1) * per_page + 1, total);
    const end = Math.min(page * per_page, total);

    return `
        <div class="pagination">
            <div class="pagination-info">
                Showing ${formatNumber(start)}-${formatNumber(end)} of ${formatNumber(total)} memories
            </div>
            <div class="pagination-controls">
                <select id="mem-per-page" class="select-sm">
                    ${[25, 50, 100].map(n => `<option value="${n}" ${state.perPage === n ? 'selected' : ''}>${n} / page</option>`).join('')}
                </select>
                <button class="btn btn-ghost btn-sm" id="mem-prev" ${page <= 1 ? 'disabled' : ''}>&#8592; Prev</button>
                <span style="color:var(--text-secondary);font-size:0.85rem;">Page ${page} of ${total_pages}</span>
                <button class="btn btn-ghost btn-sm" id="mem-next" ${page >= total_pages ? 'disabled' : ''}>Next &#8594;</button>
            </div>
        </div>
    `;
}

function attachTableListeners() {
    // Row clicks (but not on checkbox)
    document.querySelectorAll('#page-memories .clickable-row').forEach(row => {
        row.addEventListener('click', (e) => {
            if (e.target.type === 'checkbox') return;
            const id = row.dataset.id;
            if (id) navigateTo('detail', { memoryId: id });
        });
    });

    // Checkboxes
    document.querySelectorAll('#page-memories .mem-checkbox').forEach(cb => {
        cb.addEventListener('change', (e) => {
            e.stopPropagation();
            if (cb.checked) { state.selectedIds.add(cb.dataset.id); }
            else { state.selectedIds.delete(cb.dataset.id); }
            updateBulkBar();
        });
    });

    // Select all
    document.getElementById('mem-select-all')?.addEventListener('change', (e) => {
        const checked = e.target.checked;
        document.querySelectorAll('#page-memories .mem-checkbox').forEach(cb => {
            cb.checked = checked;
            if (checked) { state.selectedIds.add(cb.dataset.id); }
            else { state.selectedIds.delete(cb.dataset.id); }
        });
        updateBulkBar();
    });

    // Pagination
    document.getElementById('mem-prev')?.addEventListener('click', () => {
        if (state.page > 1) { state.page--; loadData(); }
    });
    document.getElementById('mem-next')?.addEventListener('click', () => {
        if (state.data && state.page < state.data.total_pages) { state.page++; loadData(); }
    });
    document.getElementById('mem-per-page')?.addEventListener('change', (e) => {
        state.perPage = parseInt(e.target.value, 10);
        state.page = 1;
        loadData();
    });
}
