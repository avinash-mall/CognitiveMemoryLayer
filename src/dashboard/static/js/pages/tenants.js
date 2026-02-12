/**
 * Tenants Page
 * Lists all tenants with memory/fact/event counts, last activity, and quick links.
 */

import { getTenants } from '../api.js';
import { setTenantAndNavigate } from '../app.js';
import { formatNumber, formatDate, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-tenants');

let sortKey = 'tenant_id';
let sortAsc = true;
let tenantsData = [];

export async function renderTenants() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading tenants...</div>`;

    try {
        const data = await getTenants();
        tenantsData = data.tenants || [];
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load tenants: ${err.message}</p></div>`;
        return;
    }

    renderTable(el);
}

function renderTable(el) {
    const sorted = [...tenantsData].sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        if (typeof va === 'string') { va = va.toLowerCase(); vb = (vb || '').toLowerCase(); }
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
    });

    const totalMemories = tenantsData.reduce((s, t) => s + t.memory_count, 0);
    const totalFacts = tenantsData.reduce((s, t) => s + t.fact_count, 0);
    const totalEvents = tenantsData.reduce((s, t) => s + t.event_count, 0);
    const mostActive = tenantsData.length ? tenantsData.reduce((a, b) => a.memory_count > b.memory_count ? a : b) : null;

    el.innerHTML = `
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Total Tenants</div><div class="kpi-value">${formatNumber(tenantsData.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Memories</div><div class="kpi-value">${formatNumber(totalMemories)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Facts</div><div class="kpi-value">${formatNumber(totalFacts)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Most Active</div><div class="kpi-value" style="font-size:1rem">${mostActive ? escapeHtml(mostActive.tenant_id) : '-'}</div><div class="kpi-sub">${mostActive ? formatNumber(mostActive.memory_count) + ' memories' : ''}</div></div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">All Tenants</div>
            <div class="table-responsive">
                <table class="data-table" id="tenants-table">
                    <thead>
                        <tr>
                            ${thCol('tenant_id', 'Tenant ID')}
                            ${thCol('memory_count', 'Memories')}
                            ${thCol('active_memory_count', 'Active')}
                            ${thCol('fact_count', 'Facts')}
                            ${thCol('event_count', 'Events')}
                            ${thCol('last_memory_at', 'Last Memory')}
                            ${thCol('last_event_at', 'Last Event')}
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sorted.map(t => `
                            <tr>
                                <td class="tenant-id-cell"><strong>${escapeHtml(t.tenant_id)}</strong></td>
                                <td>${formatNumber(t.memory_count)}</td>
                                <td>${formatNumber(t.active_memory_count)}</td>
                                <td>${formatNumber(t.fact_count)}</td>
                                <td>${formatNumber(t.event_count)}</td>
                                <td style="font-size:0.82rem">${formatDate(t.last_memory_at)}</td>
                                <td style="font-size:0.82rem">${formatDate(t.last_event_at)}</td>
                                <td>
                                    <div class="action-btns">
                                        <button class="btn btn-ghost btn-xs quick-link" data-tenant="${escapeHtml(t.tenant_id)}" data-page="overview">Overview</button>
                                        <button class="btn btn-ghost btn-xs quick-link" data-tenant="${escapeHtml(t.tenant_id)}" data-page="memories">Memories</button>
                                        <button class="btn btn-ghost btn-xs quick-link" data-tenant="${escapeHtml(t.tenant_id)}" data-page="events">Events</button>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                        ${sorted.length === 0 ? '<tr><td colspan="8" class="empty-state">No tenants found</td></tr>' : ''}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // Attach sort listeners
    el.querySelectorAll('.sortable-th').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sort;
            if (sortKey === key) { sortAsc = !sortAsc; } else { sortKey = key; sortAsc = true; }
            renderTable(el);
        });
    });

    // Quick links
    el.querySelectorAll('.quick-link').forEach(btn => {
        btn.addEventListener('click', () => {
            setTenantAndNavigate(btn.dataset.tenant, btn.dataset.page);
        });
    });
}

function thCol(key, label) {
    const arrow = sortKey === key ? (sortAsc ? ' &#9650;' : ' &#9660;') : '';
    return `<th class="sortable-th" data-sort="${key}" style="cursor:pointer">${label}${arrow}</th>`;
}
