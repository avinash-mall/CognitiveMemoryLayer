/**
 * Memory Explorer Page
 * Filterable, sortable, paginated table of memory records.
 */

import { getMemories } from '../api.js';
import { navigateTo } from '../app.js';
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
};

const MEMORY_TYPES = [
    'episodic_event', 'semantic_fact', 'procedure', 'constraint', 'hypothesis',
    'preference', 'task_state', 'conversation', 'message', 'tool_result',
    'reasoning_step', 'scratch', 'knowledge', 'observation', 'plan',
];

const MEMORY_STATUSES = ['active', 'silent', 'compressed', 'archived', 'deleted'];

export async function renderMemories({ tenantId } = {}) {
    state.tenantId = tenantId || '';
    const el = container();
    el.innerHTML = buildShell();
    attachListeners();
    await loadData();
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

    // Allow pressing Enter in search field
    el.querySelector('#mem-search')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            el.querySelector('#mem-apply')?.click();
        }
    });
}

async function loadData() {
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
    // Row clicks
    document.querySelectorAll('#page-memories .clickable-row').forEach(row => {
        row.addEventListener('click', () => {
            const id = row.dataset.id;
            if (id) navigateTo('detail', { memoryId: id });
        });
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
