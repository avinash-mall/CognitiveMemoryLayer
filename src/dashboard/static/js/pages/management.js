/**
 * Management Page
 * Consolidation and forgetting trigger panels with results display,
 * job history, and reconsolidation status.
 */

import { triggerConsolidate, triggerForget, getTenants, resetDatabase, getJobs, getLabile } from '../api.js';
import { showToast } from '../app.js';
import { prettyJson, escapeHtml, formatNumber, formatDate, formatFloat } from '../utils/formatters.js';

const container = () => document.getElementById('page-management');

let tenants = [];

export async function renderManagement({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading...</div>`;

    try {
        const [tenantsData, jobsData, labileData] = await Promise.all([
            getTenants(),
            getJobs(null, null, 20).catch(() => ({ items: [], total: 0 })),
            getLabile(tenantId).catch(() => ({ tenants: [], total_db_labile: 0, total_redis_scopes: 0, total_redis_sessions: 0, total_redis_memories: 0 })),
        ]);
        tenants = tenantsData.tenants || [];
        el.innerHTML = buildManagement(jobsData, labileData);
        attachListeners();
    } catch {
        tenants = [];
        el.innerHTML = buildManagement({ items: [], total: 0 }, { tenants: [], total_db_labile: 0, total_redis_scopes: 0, total_redis_sessions: 0, total_redis_memories: 0 });
        attachListeners();
    }
}

function buildManagement(jobsData, labileData) {
    const tenantOptions = tenants.map(t =>
        `<option value="${t.tenant_id}">${escapeHtml(t.tenant_id)} (${formatNumber(t.memory_count)} memories)</option>`
    ).join('');

    return `
        <p style="color:var(--text-secondary);margin-bottom:20px;">
            Trigger maintenance operations for memory consolidation and active forgetting.
            These operations run server-side and may take a few seconds to complete.
        </p>

        <!-- Reconsolidation Status -->
        <div class="card" style="margin-bottom:20px;">
            <div class="card-title">Reconsolidation Status</div>
            <div class="kpi-grid" style="margin-bottom:12px;">
                <div class="kpi-card"><div class="kpi-label">DB Labile Memories</div><div class="kpi-value">${formatNumber(labileData.total_db_labile)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Redis Scopes</div><div class="kpi-value">${formatNumber(labileData.total_redis_scopes)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Redis Sessions</div><div class="kpi-value">${formatNumber(labileData.total_redis_sessions)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Redis Labile Memories</div><div class="kpi-value">${formatNumber(labileData.total_redis_memories)}</div></div>
            </div>
            ${labileData.tenants && labileData.tenants.length > 0 ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Tenant</th><th>DB Labile</th><th>Scopes</th><th>Sessions</th><th>Memories (Redis)</th></tr></thead>
                        <tbody>
                            ${labileData.tenants.map(t => `
                                <tr>
                                    <td>${escapeHtml(t.tenant_id)}</td>
                                    <td>${formatNumber(t.db_labile_count)}</td>
                                    <td>${formatNumber(t.redis_scope_count)}</td>
                                    <td>${formatNumber(t.redis_session_count)}</td>
                                    <td>${formatNumber(t.redis_memory_count)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px">No labile memories currently</div>'}
        </div>

        <div class="management-grid">
            <!-- Consolidation Panel -->
            <div class="management-panel">
                <h3>Memory Consolidation</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Consolidation migrates episodic memories into semantic facts by sampling recent episodes,
                    clustering them semantically, extracting gists, and migrating to the neocortical store.
                </p>
                <div class="management-form">
                    <div class="form-row">
                        <label for="cons-tenant">Tenant</label>
                        <select id="cons-tenant" class="select-sm">
                            ${tenantOptions || '<option value="">No tenants found</option>'}
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="cons-user">User ID</label>
                        <input type="text" id="cons-user" class="input-sm" placeholder="Optional (defaults to tenant)">
                    </div>
                    <button id="cons-trigger" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>
                        Run Consolidation
                    </button>
                </div>
                <div id="cons-result" class="result-box hidden"></div>
            </div>

            <!-- Forgetting Panel -->
            <div class="management-panel">
                <h3>Active Forgetting</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Active forgetting scores memories for relevance and applies decay, silencing,
                    compression, deletion, and duplicate resolution. Use dry-run to preview actions.
                </p>
                <div class="management-form">
                    <div class="form-row">
                        <label for="fgt-tenant">Tenant</label>
                        <select id="fgt-tenant" class="select-sm">
                            ${tenantOptions || '<option value="">No tenants found</option>'}
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="fgt-user">User ID</label>
                        <input type="text" id="fgt-user" class="input-sm" placeholder="Optional (defaults to tenant)">
                    </div>
                    <div class="form-row">
                        <label for="fgt-max">Max Memories</label>
                        <input type="number" id="fgt-max" class="input-sm" value="5000" min="1" max="50000">
                    </div>
                    <div class="form-row">
                        <label>Dry Run</label>
                        <div class="toggle-wrapper">
                            <div id="fgt-dry-toggle" class="toggle-switch active" title="Preview without applying changes"></div>
                            <span id="fgt-dry-label" style="font-size:0.85rem;color:var(--text-secondary);cursor:pointer;">Yes (preview only)</span>
                        </div>
                    </div>
                    <button id="fgt-trigger" class="btn btn-warning" ${!tenants.length ? 'disabled' : ''}>
                        Run Forgetting
                    </button>
                </div>
                    <div id="fgt-result" class="result-box hidden"></div>
            </div>

            <!-- Database Reset Panel -->
            <div class="management-panel">
                <h3>Database</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Drop all tables and re-run migrations. Uses current .env (e.g. EMBEDDING__DIMENSIONS).
                    All memory data, events, and facts will be permanently deleted.
                </p>
                <div class="management-form">
                    <div class="form-row">
                        <label for="db-reset-confirm">Type DELETE to confirm</label>
                        <input type="text" id="db-reset-confirm" class="input-sm" placeholder="DELETE" autocomplete="off">
                    </div>
                    <button id="db-reset-trigger" class="btn btn-danger" disabled>
                        Delete and recreate database
                    </button>
                </div>
                <div id="db-reset-result" class="result-box hidden"></div>
            </div>
        </div>

        <!-- Job History -->
        <div class="card" style="margin-top:24px;">
            <div class="card-title">Job History (Last ${jobsData.items.length} of ${jobsData.total})</div>
            ${jobsData.items.length > 0 ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr><th>Type</th><th>Tenant</th><th>Status</th><th>Dry Run</th><th>Duration</th><th>Started</th><th>Details</th></tr>
                        </thead>
                        <tbody>
                            ${jobsData.items.map(j => `
                                <tr class="expandable-row" data-job-id="${j.id}">
                                    <td><span class="badge badge-type">${escapeHtml(j.job_type)}</span></td>
                                    <td>${escapeHtml(j.tenant_id)}</td>
                                    <td><span class="badge badge-${j.status === 'completed' ? 'ok' : j.status === 'failed' ? 'error' : 'warning'}">${j.status}</span></td>
                                    <td>${j.dry_run ? 'Yes' : 'No'}</td>
                                    <td>${j.duration_seconds != null ? formatFloat(j.duration_seconds, 1) + 's' : '-'}</td>
                                    <td style="font-size:0.82rem">${formatDate(j.started_at)}</td>
                                    <td><button class="btn btn-ghost btn-xs job-expand-btn" data-job='${escapeHtml(JSON.stringify(j.result || j.error || {}))}'>View</button></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:16px">No jobs recorded yet. Run a consolidation or forgetting operation above.</div>'}
        </div>

        <!-- Info Section -->
        <div class="card" style="margin-top:24px;">
            <div class="card-title">Process Details</div>
            <div class="detail-grid">
                <div>
                    <h4 style="font-size:0.95rem;margin-bottom:8px;">Consolidation Pipeline</h4>
                    <ol style="color:var(--text-secondary);font-size:0.88rem;padding-left:20px;line-height:2;">
                        <li><strong>Sample</strong> recent episodic memories</li>
                        <li><strong>Cluster</strong> semantically related episodes</li>
                        <li><strong>Summarize</strong> clusters into gist facts</li>
                        <li><strong>Align</strong> gists to fact schemas</li>
                        <li><strong>Migrate</strong> to neocortical store</li>
                    </ol>
                </div>
                <div>
                    <h4 style="font-size:0.95rem;margin-bottom:8px;">Forgetting Actions</h4>
                    <ol style="color:var(--text-secondary);font-size:0.88rem;padding-left:20px;line-height:2;">
                        <li><strong>Score</strong> all memories by relevance/recency</li>
                        <li><strong>Decay</strong> low-scoring memories</li>
                        <li><strong>Silence</strong> rarely accessed memories</li>
                        <li><strong>Compress</strong> verbose memories</li>
                        <li><strong>Delete</strong> expired/irrelevant records</li>
                        <li><strong>Deduplicate</strong> near-identical entries</li>
                    </ol>
                </div>
            </div>
        </div>
    `;
}

function attachListeners() {
    const el = container();

    // Dry-run toggle
    let dryRun = true;
    const dryToggle = el.querySelector('#fgt-dry-toggle');
    const dryLabel = el.querySelector('#fgt-dry-label');
    const toggleDry = () => {
        dryRun = !dryRun;
        dryToggle?.classList.toggle('active', dryRun);
        if (dryLabel) dryLabel.textContent = dryRun ? 'Yes (preview only)' : 'No (apply changes!)';
    };
    dryToggle?.addEventListener('click', toggleDry);
    dryLabel?.addEventListener('click', toggleDry);

    // Database reset
    const dbConfirmInput = el.querySelector('#db-reset-confirm');
    const dbResetBtn = el.querySelector('#db-reset-trigger');
    const dbResetResult = el.querySelector('#db-reset-result');
    const updateDbResetButton = () => {
        if (dbResetBtn) dbResetBtn.disabled = (dbConfirmInput?.value || '') !== 'DELETE';
    };
    dbConfirmInput?.addEventListener('input', updateDbResetButton);
    dbConfirmInput?.addEventListener('paste', updateDbResetButton);

    dbResetBtn?.addEventListener('click', async () => {
        if ((dbConfirmInput?.value || '') !== 'DELETE') return;
        dbResetBtn.disabled = true;
        dbResetBtn.textContent = 'Running...';
        if (dbResetResult) { dbResetResult.classList.add('hidden'); dbResetResult.textContent = ''; }
        try {
            const result = await resetDatabase();
            if (dbResetResult) { dbResetResult.textContent = prettyJson(result); dbResetResult.classList.remove('hidden'); }
            showToast('Database reset and recreated');
            dbConfirmInput.value = '';
            updateDbResetButton();
        } catch (err) {
            if (dbResetResult) { dbResetResult.textContent = `Error: ${err.message}`; dbResetResult.classList.remove('hidden'); }
            showToast('Database reset failed: ' + err.message, 'error');
        } finally {
            dbResetBtn.disabled = (dbConfirmInput?.value || '') !== 'DELETE';
            dbResetBtn.textContent = 'Delete and recreate database';
        }
    });

    // Consolidation trigger
    el.querySelector('#cons-trigger')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#cons-tenant')?.value;
        const user = el.querySelector('#cons-user')?.value || '';
        if (!tenant) return;
        const btn = el.querySelector('#cons-trigger');
        const resultBox = el.querySelector('#cons-result');
        btn.disabled = true; btn.textContent = 'Running...';
        resultBox?.classList.add('hidden');
        try {
            const result = await triggerConsolidate(tenant, user);
            resultBox.textContent = prettyJson(result);
            resultBox?.classList.remove('hidden');
            showToast('Consolidation completed successfully');
        } catch (err) {
            resultBox.textContent = `Error: ${err.message}`;
            resultBox?.classList.remove('hidden');
            showToast('Consolidation failed: ' + err.message, 'error');
        } finally { btn.disabled = false; btn.textContent = 'Run Consolidation'; }
    });

    // Forgetting trigger
    el.querySelector('#fgt-trigger')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#fgt-tenant')?.value;
        const user = el.querySelector('#fgt-user')?.value || '';
        const maxMemories = parseInt(el.querySelector('#fgt-max')?.value || '5000', 10);
        if (!tenant) return;
        const btn = el.querySelector('#fgt-trigger');
        const resultBox = el.querySelector('#fgt-result');
        btn.disabled = true; btn.textContent = 'Running...';
        resultBox?.classList.add('hidden');
        try {
            const result = await triggerForget(tenant, user, dryRun, maxMemories);
            resultBox.textContent = prettyJson(result);
            resultBox?.classList.remove('hidden');
            showToast(`Forgetting completed${dryRun ? ' (dry run)' : ''}`);
        } catch (err) {
            resultBox.textContent = `Error: ${err.message}`;
            resultBox?.classList.remove('hidden');
            showToast('Forgetting failed: ' + err.message, 'error');
        } finally { btn.disabled = false; btn.textContent = 'Run Forgetting'; }
    });

    // Job detail expand
    el.querySelectorAll('.job-expand-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const row = btn.closest('tr');
            const existing = row.nextElementSibling;
            if (existing && existing.classList.contains('job-detail-row')) {
                existing.remove();
                return;
            }
            const detailRow = document.createElement('tr');
            detailRow.className = 'job-detail-row';
            detailRow.innerHTML = `<td colspan="7"><pre class="result-box" style="margin:0">${prettyJson(JSON.parse(btn.dataset.job))}</pre></td>`;
            row.after(detailRow);
        });
    });
}
