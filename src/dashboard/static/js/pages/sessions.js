/**
 * Sessions Page
 * Shows active sessions from Redis and memory counts per source_session_id from DB.
 */

import { getSessions } from '../api.js';
import { navigateTo } from '../app.js';
import { formatNumber, formatDate, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-sessions');

export async function renderSessions({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading sessions...</div>`;

    try {
        const data = await getSessions(tenantId);
        el.innerHTML = buildSessions(data);
        attachListeners(el);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load sessions: ${err.message}</p></div>`;
    }
}

function ttlBadge(ttl) {
    if (ttl <= 0) return '<span class="badge badge-error">expired</span>';
    if (ttl < 3600) return `<span class="badge badge-warning">${formatDuration(ttl)}</span>`;
    return `<span class="badge badge-ok">${formatDuration(ttl)}</span>`;
}

function formatDuration(seconds) {
    if (seconds <= 0) return 'expired';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

function buildSessions(data) {
    const activeSessions = data.sessions.filter(s => s.ttl_seconds > 0);
    const dbOnlySessions = data.sessions.filter(s => s.ttl_seconds <= 0 && s.memory_count > 0);

    return `
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Active Sessions</div><div class="kpi-value">${formatNumber(data.total_active)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Sessions</div><div class="kpi-value">${formatNumber(data.sessions.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Memories with Sessions</div><div class="kpi-value">${formatNumber(data.total_memories_with_session)}</div></div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Active Sessions (Redis)</div>
            <div class="table-responsive">
                <table class="data-table">
                    <thead>
                        <tr><th>Session ID</th><th>Tenant</th><th>Created</th><th>Expires</th><th>TTL</th><th>Memories</th></tr>
                    </thead>
                    <tbody>
                        ${activeSessions.map(s => `
                            <tr>
                                <td><a href="#" class="session-link" data-sid="${escapeHtml(s.session_id)}" style="color:var(--accent)">${escapeHtml(s.session_id.substring(0, 12))}...</a></td>
                                <td>${escapeHtml(s.tenant_id || '-')}</td>
                                <td style="font-size:0.82rem">${formatDate(s.created_at)}</td>
                                <td style="font-size:0.82rem">${formatDate(s.expires_at)}</td>
                                <td>${ttlBadge(s.ttl_seconds)}</td>
                                <td>${formatNumber(s.memory_count)}</td>
                            </tr>
                        `).join('')}
                        ${activeSessions.length === 0 ? '<tr><td colspan="6" class="empty-state">No active sessions</td></tr>' : ''}
                    </tbody>
                </table>
            </div>
        </div>

        ${dbOnlySessions.length > 0 ? `
        <div class="card" style="margin-top:16px;">
            <div class="card-title">Historical Sessions (DB only - expired from Redis)</div>
            <div class="table-responsive">
                <table class="data-table">
                    <thead><tr><th>Session ID</th><th>Memories</th><th>Actions</th></tr></thead>
                    <tbody>
                        ${dbOnlySessions.map(s => `
                            <tr>
                                <td>${escapeHtml(s.session_id.substring(0, 20))}...</td>
                                <td>${formatNumber(s.memory_count)}</td>
                                <td><button class="btn btn-ghost btn-xs session-link" data-sid="${escapeHtml(s.session_id)}">View Memories</button></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
        ` : ''}
    `;
}

function attachListeners(el) {
    el.querySelectorAll('.session-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Navigate to memories page filtered by source_session_id
            window.location.hash = `#memories`;
            // Store the session ID for the memories page to pick up
            sessionStorage.setItem('cml_filter_session_id', link.dataset.sid);
            navigateTo('memories');
        });
    });
}
