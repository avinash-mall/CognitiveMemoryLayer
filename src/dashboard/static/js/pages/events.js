/**
 * Event Log Page
 * Paginated event log with filtering, expandable rows, and auto-refresh.
 */

import { getEvents } from '../api.js';
import { formatDate, shortUuid, prettyJson, escapeHtml, formatNumber } from '../utils/formatters.js';

const container = () => document.getElementById('page-events');

// State
let state = {
    page: 1,
    perPage: 25,
    eventType: '',
    operation: '',
    tenantId: '',
    data: null,
    autoRefresh: false,
    refreshTimer: null,
};

export async function renderEvents({ tenantId } = {}) {
    state.tenantId = tenantId || '';
    stopAutoRefresh();
    const el = container();
    el.innerHTML = buildShell();
    attachListeners();
    await loadData();
}

function buildShell() {
    return `
        <div class="filter-bar">
            <input type="text" id="evt-type-filter" class="input-sm" placeholder="Event type filter..." value="${escapeHtml(state.eventType)}">
            <input type="text" id="evt-op-filter" class="input-sm" placeholder="Operation filter..." value="${escapeHtml(state.operation)}">
            <button id="evt-apply" class="btn btn-primary btn-sm">Apply</button>
            <div style="margin-left:auto;" class="toggle-wrapper">
                <div id="evt-auto-toggle" class="toggle-switch ${state.autoRefresh ? 'active' : ''}" title="Auto-refresh every 5s"></div>
                <label style="cursor:pointer;font-size:0.82rem;" id="evt-auto-label">Auto-refresh</label>
            </div>
        </div>
        <div id="evt-table-area">
            <div class="loading-overlay"><div class="spinner"></div> Loading events...</div>
        </div>
    `;
}

function attachListeners() {
    const el = container();

    el.querySelector('#evt-apply')?.addEventListener('click', () => {
        state.eventType = el.querySelector('#evt-type-filter')?.value || '';
        state.operation = el.querySelector('#evt-op-filter')?.value || '';
        state.page = 1;
        loadData();
    });

    el.querySelector('#evt-type-filter')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') el.querySelector('#evt-apply')?.click();
    });
    el.querySelector('#evt-op-filter')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') el.querySelector('#evt-apply')?.click();
    });

    // Auto-refresh toggle
    const toggle = el.querySelector('#evt-auto-toggle');
    const label = el.querySelector('#evt-auto-label');
    const toggleFn = () => {
        state.autoRefresh = !state.autoRefresh;
        toggle?.classList.toggle('active', state.autoRefresh);
        if (state.autoRefresh) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    };
    toggle?.addEventListener('click', toggleFn);
    label?.addEventListener('click', toggleFn);
}

function startAutoRefresh() {
    stopAutoRefresh();
    state.refreshTimer = setInterval(() => loadData(), 5000);
}

function stopAutoRefresh() {
    if (state.refreshTimer) {
        clearInterval(state.refreshTimer);
        state.refreshTimer = null;
    }
}

async function loadData() {
    const area = document.getElementById('evt-table-area');
    if (!area) { stopAutoRefresh(); return; }

    // Don't show spinner on auto-refresh
    if (!state.autoRefresh) {
        area.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading events...</div>`;
    }

    try {
        const data = await getEvents({
            page: state.page,
            perPage: state.perPage,
            eventType: state.eventType,
            operation: state.operation,
            tenantId: state.tenantId,
        });
        state.data = data;
        area.innerHTML = buildTable(data) + buildPagination(data);
        attachTableListeners();
    } catch (err) {
        area.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>${escapeHtml(err.message)}</p></div>`;
        stopAutoRefresh();
    }
}

function buildTable(data) {
    if (!data.items?.length) {
        return `<div class="empty-state"><div class="empty-state-icon">&#128220;</div><p>No events found</p></div>`;
    }

    const rows = data.items.map((e, i) => {
        const payloadStr = prettyJson(e.payload);
        const payloadPreview = payloadStr.length > 80 ? payloadStr.slice(0, 80) + '...' : payloadStr;
        return `
            <tr class="clickable-row evt-row" data-idx="${i}">
                <td style="white-space:nowrap;">${formatDate(e.created_at)}</td>
                <td><span class="badge badge-type">${e.event_type}</span></td>
                <td>${e.operation || '—'}</td>
                <td><code class="mono" style="font-size:0.78rem;">${escapeHtml(e.tenant_id)}</code></td>
                <td><code class="mono" style="font-size:0.78rem;">${shortUuid(String(e.id))}</code></td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${e.memory_ids?.length ? e.memory_ids.length + ' memories' : '—'}</td>
            </tr>
            <tr class="evt-payload-row" id="evt-payload-${i}" style="display:none;">
                <td colspan="6">
                    <div class="json-tree" style="margin:0;">${escapeHtml(payloadStr)}</div>
                    ${e.memory_ids?.length ? `<div style="margin-top:8px;font-size:0.82rem;color:var(--text-secondary);">
                        Memory IDs: ${e.memory_ids.map(id => `<code class="mono">${String(id).slice(0, 8)}...</code>`).join(', ')}
                    </div>` : ''}
                </td>
            </tr>
        `;
    }).join('');

    return `
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Event Type</th>
                        <th>Operation</th>
                        <th>Tenant</th>
                        <th>Event ID</th>
                        <th>Related</th>
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
                Showing ${formatNumber(start)}-${formatNumber(end)} of ${formatNumber(total)} events
                ${state.autoRefresh ? '<span style="color:var(--success);margin-left:8px;">&#9679; Live</span>' : ''}
            </div>
            <div class="pagination-controls">
                <select id="evt-per-page" class="select-sm">
                    ${[25, 50, 100].map(n => `<option value="${n}" ${state.perPage === n ? 'selected' : ''}>${n} / page</option>`).join('')}
                </select>
                <button class="btn btn-ghost btn-sm" id="evt-prev" ${page <= 1 ? 'disabled' : ''}>&#8592; Prev</button>
                <span style="color:var(--text-secondary);font-size:0.85rem;">Page ${page} of ${total_pages}</span>
                <button class="btn btn-ghost btn-sm" id="evt-next" ${page >= total_pages ? 'disabled' : ''}>Next &#8594;</button>
            </div>
        </div>
    `;
}

function attachTableListeners() {
    // Expandable rows
    document.querySelectorAll('#page-events .evt-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = row.dataset.idx;
            const payloadRow = document.getElementById(`evt-payload-${idx}`);
            if (payloadRow) {
                const isVisible = payloadRow.style.display !== 'none';
                payloadRow.style.display = isVisible ? 'none' : 'table-row';
            }
        });
    });

    // Pagination
    document.getElementById('evt-prev')?.addEventListener('click', () => {
        if (state.page > 1) { state.page--; loadData(); }
    });
    document.getElementById('evt-next')?.addEventListener('click', () => {
        if (state.data && state.page < state.data.total_pages) { state.page++; loadData(); }
    });
    document.getElementById('evt-per-page')?.addEventListener('change', (e) => {
        state.perPage = parseInt(e.target.value, 10);
        state.page = 1;
        loadData();
    });
}
