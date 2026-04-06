/**
 * Management Page
 * Operations workbench for consolidation, forgetting, write simulation, and reconsolidation.
 */

import {
    getJobs,
    getLabile,
    getReconsolidationSessions,
    getTenants,
    previewForgetting,
    resetDatabase,
    simulateWrite,
    triggerConsolidate,
    triggerForget,
    triggerReconsolidate,
} from '../api.js';
import { navigateTo, showToast } from '../app.js';
import { escapeHtml, formatDate, formatFloat, formatNumber, prettyJson } from '../utils/formatters.js';

const container = () => document.getElementById('page-management');

let tenants = [];

export async function renderManagement({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading workbench...</div>`;

    try {
        const [tenantsData, jobsData, labileData, sessionsData] = await Promise.all([
            getTenants(),
            getJobs(null, null, 25).catch(() => ({ items: [], total: 0 })),
            getLabile(tenantId).catch(() => ({ tenants: [], total_db_labile: 0, total_redis_scopes: 0, total_redis_sessions: 0, total_redis_memories: 0 })),
            getReconsolidationSessions(tenantId).catch(() => ({ items: [], total: 0 })),
        ]);
        tenants = tenantsData.tenants || [];
        el.innerHTML = buildPage(jobsData, labileData, sessionsData, tenantId);
        attachListeners(el);
    } catch (err) {
        tenants = [];
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load management workbench: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildPage(jobsData, labileData, sessionsData, tenantId) {
    const tenantOptions = tenants.map(tenant => (
        `<option value="${tenant.tenant_id}" ${tenantId === tenant.tenant_id ? 'selected' : ''}>${escapeHtml(tenant.tenant_id)} (${formatNumber(tenant.memory_count)} memories)</option>`
    )).join('');

    return `
        <p class="page-desc">
            Operate the memory lifecycle end to end: inspect the write path, preview forgetting decisions,
            trigger maintenance jobs, and inspect active labile sessions before reconsolidation clears them.
        </p>

        <div class="kpi-grid" style="margin-bottom:16px;">
            <div class="kpi-card"><div class="kpi-label">DB Labile Memories</div><div class="kpi-value">${formatNumber(labileData.total_db_labile || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Redis Scopes</div><div class="kpi-value">${formatNumber(labileData.total_redis_scopes || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Redis Sessions</div><div class="kpi-value">${formatNumber(labileData.total_redis_sessions || 0)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Inspectable Sessions</div><div class="kpi-value">${formatNumber(sessionsData.total || 0)}</div></div>
        </div>

        <div class="card" style="margin-bottom:16px;">
            <div class="card-title">Active Labile Sessions</div>
            ${buildSessionsTable(sessionsData.items || [])}
        </div>

        <div class="management-grid">
            <div class="management-panel">
                <h3>Write Path Inspector</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Dry-run the write path to inspect chunking, gate decisions, redaction, chosen memory types,
                    extracted constraints/facts, and extractor comparisons before anything is stored.
                </p>
                <div class="management-form">
                    <div class="form-row">
                        <label for="write-tenant">Tenant</label>
                        <select id="write-tenant" class="select-sm">
                            ${tenantOptions || '<option value="">No tenants found</option>'}
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="write-session">Session ID</label>
                        <input type="text" id="write-session" class="input-sm" placeholder="Optional source session">
                    </div>
                    <div class="form-row">
                        <label for="write-tags">Context Tags</label>
                        <input type="text" id="write-tags" class="input-sm" placeholder="preferences, work">
                    </div>
                    <div class="form-row">
                        <label for="write-memory-type">Memory Type Override</label>
                        <input type="text" id="write-memory-type" class="input-sm" placeholder="Optional, e.g. constraint">
                    </div>
                    <div class="form-row">
                        <label for="write-content">Content</label>
                        <textarea id="write-content" class="input-sm" rows="6" style="resize:vertical;min-height:140px" placeholder="Paste the user or assistant content you want to inspect."></textarea>
                    </div>
                    <div class="form-row">
                        <label>Compare Extractors</label>
                        <div class="toggle-wrapper">
                            <div id="write-compare-toggle" class="toggle-switch" title="Run local/unified comparison when available"></div>
                            <span id="write-compare-label" style="font-size:0.85rem;color:var(--text-secondary);cursor:pointer;">Off</span>
                        </div>
                    </div>
                    <button id="write-simulate-btn" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>Run Dry-Run Simulation</button>
                </div>
                <div id="write-sim-result" class="hidden" style="margin-top:16px;"></div>
            </div>

            <div class="management-panel">
                <h3>Memory Consolidation</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Trigger episodic-to-semantic consolidation. The tracked job payload now persists migration artifacts
                    such as facts created, facts updated, and episodes marked.
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
                    <button id="cons-trigger" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>Run Consolidation</button>
                </div>
                <div id="cons-result" class="result-box hidden"></div>
            </div>

            <div class="management-panel">
                <h3>Forgetting Preview & Run</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Preview score breakdowns, protected memories, and duplicate candidates before running forgetting.
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
                        <input type="number" id="fgt-max" class="input-sm" value="200" min="1" max="5000">
                    </div>
                    <div class="form-row">
                        <label>Dry Run</label>
                        <div class="toggle-wrapper">
                            <div id="fgt-dry-toggle" class="toggle-switch active" title="Preview without applying changes"></div>
                            <span id="fgt-dry-label" style="font-size:0.85rem;color:var(--text-secondary);cursor:pointer;">Yes</span>
                        </div>
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;">
                        <button id="fgt-preview-btn" class="btn btn-ghost" ${!tenants.length ? 'disabled' : ''}>Preview Scores</button>
                        <button id="fgt-trigger" class="btn btn-warning" ${!tenants.length ? 'disabled' : ''}>Run Forgetting</button>
                    </div>
                </div>
                <div id="fgt-result" class="result-box hidden"></div>
                <div id="fgt-preview-result" class="hidden" style="margin-top:16px;"></div>
            </div>

            <div class="management-panel">
                <h3>Reconsolidation & Database</h3>
                <p style="color:var(--text-secondary);font-size:0.88rem;margin-bottom:16px;">
                    Release all labile sessions for a tenant, or rebuild the database from scratch when you need a clean environment.
                </p>
                <div class="management-form">
                    <div class="form-row">
                        <label for="rec-tenant">Tenant</label>
                        <select id="rec-tenant" class="select-sm">
                            ${tenantOptions || '<option value="">No tenants found</option>'}
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="rec-user">User ID</label>
                        <input type="text" id="rec-user" class="input-sm" placeholder="Optional (defaults to tenant)">
                    </div>
                    <button id="rec-trigger" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>Release Labile State</button>
                </div>
                <div id="rec-result" class="result-box hidden"></div>
                <div class="management-form" style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);">
                    <div class="form-row">
                        <label for="db-reset-confirm">Type DELETE to confirm</label>
                        <input type="text" id="db-reset-confirm" class="input-sm" placeholder="DELETE">
                    </div>
                    <button id="db-reset-trigger" class="btn btn-danger" disabled>Delete and recreate database</button>
                </div>
                <div id="db-reset-result" class="result-box hidden"></div>
            </div>
        </div>

        <div class="card" style="margin-top:24px;">
            <div class="card-title">Job History</div>
            ${buildJobHistory(jobsData.items || [])}
        </div>
    `;
}

function buildSessionMemoriesDetail(item) {
    const memories = item.memories || [];
    const retrieved = item.retrieved_texts || [];
    return `
        <div style="padding:12px 16px 12px 24px;background:var(--surface-2,#f9f9f9);border-top:1px solid var(--border);">
            ${retrieved.length ? `
                <div style="margin-bottom:10px;">
                    <strong style="font-size:0.82rem;">Retrieved Texts (${retrieved.length})</strong>
                    <ul style="margin:4px 0 0 16px;padding:0;font-size:0.80rem;color:var(--text-secondary);">
                        ${retrieved.slice(0, 5).map(t => `<li>${escapeHtml(t)}</li>`).join('')}
                        ${retrieved.length > 5 ? `<li style="color:var(--text-muted)">…${retrieved.length - 5} more</li>` : ''}
                    </ul>
                </div>
            ` : ''}
            ${memories.length ? `
                <strong style="font-size:0.82rem;">Labile Memories (${memories.length})</strong>
                <div class="table-responsive" style="margin-top:6px;">
                    <table class="data-table" style="font-size:0.78rem;">
                        <thead><tr><th>Memory ID</th><th>Relevance</th><th>Orig. Confidence</th><th>Context</th><th>Retrieved At</th></tr></thead>
                        <tbody>
                            ${memories.map(mem => `
                                <tr>
                                    <td>
                                        <a href="#detail/${escapeHtml(mem.memory_id)}" class="session-memory-link" data-memory-id="${escapeHtml(mem.memory_id)}" style="font-family:monospace;">${escapeHtml(mem.memory_id)}</a>
                                    </td>
                                    <td>${formatFloat(mem.relevance_score ?? 0, 3)}</td>
                                    <td>${formatFloat(mem.original_confidence ?? 0, 3)}</td>
                                    <td class="text-preview" style="max-width:240px;">${escapeHtml(mem.context || '')}</td>
                                    <td>${formatDate(mem.retrieved_at)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div style="font-size:0.82rem;color:var(--text-muted);margin-top:4px;">No memory entries in this session.</div>'}
        </div>
    `;
}

function buildSessionsTable(items) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No active labile sessions.</div>';
    }
    return `
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr><th></th><th>Session</th><th>Tenant</th><th>Scope</th><th>Query</th><th>Memories</th><th>Created</th><th>Expires</th></tr>
                </thead>
                <tbody>
                    ${items.map((item, idx) => `
                        <tr class="session-summary-row" data-session-idx="${idx}" style="cursor:pointer;" title="Click to expand labile memory details">
                            <td style="width:24px;text-align:center;">
                                <span class="session-expand-toggle" data-idx="${idx}" style="font-size:0.9rem;user-select:none;">&#9654;</span>
                            </td>
                            <td class="mono" style="font-size:0.78rem;">${escapeHtml(item.session_id)}</td>
                            <td>${escapeHtml(item.tenant_id)}</td>
                            <td class="mono" style="font-size:0.78rem;">${escapeHtml(item.scope_id)}</td>
                            <td class="text-preview" style="max-width:240px">${escapeHtml(item.query || '')}</td>
                            <td>${formatNumber((item.memories || []).length)}</td>
                            <td style="font-size:0.82rem">${formatDate(item.created_at)}</td>
                            <td style="font-size:0.82rem">${formatDate(item.expires_at)}</td>
                        </tr>
                        <tr class="session-detail-row hidden" data-session-idx="${idx}">
                            <td colspan="8" style="padding:0;">${buildSessionMemoriesDetail(item)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function buildJobHistory(items) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No tracked jobs yet.</div>';
    }
    return `
        <div class="stack-list">
            ${items.map(item => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                        <span>
                            <span class="badge badge-type">${escapeHtml(item.job_type)}</span>
                            <span class="badge ${item.status === 'completed' ? 'badge-ok' : item.status === 'failed' ? 'badge-error' : 'badge-warning'}" style="margin-left:6px;">${escapeHtml(item.status)}</span>
                            ${item.dry_run ? '<span class="badge badge-warning" style="margin-left:6px;">dry-run</span>' : ''}
                        </span>
                        <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.started_at)}</span>
                    </div>
                    <div style="margin-top:8px;color:var(--text-muted);font-size:0.78rem;">
                        tenant ${escapeHtml(item.tenant_id)} • duration ${item.duration_seconds != null ? `${formatFloat(item.duration_seconds, 1)}s` : '—'}
                    </div>
                    ${(item.result || item.error) ? `<div class="json-tree" style="margin-top:8px;max-height:160px;overflow:auto;">${escapeHtml(prettyJson(item.result || { error: item.error }))}</div>` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

function attachListeners(el) {
    let dryRun = true;
    let compareExtractors = false;

    const dryToggle = el.querySelector('#fgt-dry-toggle');
    const dryLabel = el.querySelector('#fgt-dry-label');
    const toggleDry = () => {
        dryRun = !dryRun;
        dryToggle?.classList.toggle('active', dryRun);
        if (dryLabel) dryLabel.textContent = dryRun ? 'Yes' : 'No';
    };
    dryToggle?.addEventListener('click', toggleDry);
    dryLabel?.addEventListener('click', toggleDry);

    const compareToggle = el.querySelector('#write-compare-toggle');
    const compareLabel = el.querySelector('#write-compare-label');
    const toggleCompare = () => {
        compareExtractors = !compareExtractors;
        compareToggle?.classList.toggle('active', compareExtractors);
        if (compareLabel) compareLabel.textContent = compareExtractors ? 'On' : 'Off';
    };
    compareToggle?.addEventListener('click', toggleCompare);
    compareLabel?.addEventListener('click', toggleCompare);

    const dbConfirmInput = el.querySelector('#db-reset-confirm');
    const dbResetBtn = el.querySelector('#db-reset-trigger');
    const dbResetResult = el.querySelector('#db-reset-result');
    const updateDbResetButton = () => {
        if (dbResetBtn) dbResetBtn.disabled = (dbConfirmInput?.value || '') !== 'DELETE';
    };
    dbConfirmInput?.addEventListener('input', updateDbResetButton);

    el.querySelector('#write-simulate-btn')?.addEventListener('click', async () => {
        const tenantId = el.querySelector('#write-tenant')?.value || '';
        const content = el.querySelector('#write-content')?.value?.trim() || '';
        if (!tenantId || !content) return;
        const button = el.querySelector('#write-simulate-btn');
        const resultBox = el.querySelector('#write-sim-result');
        button.disabled = true;
        button.textContent = 'Running...';
        resultBox.classList.add('hidden');
        try {
            const result = await simulateWrite({
                tenant_id: tenantId,
                content,
                session_id: el.querySelector('#write-session')?.value?.trim() || null,
                context_tags: splitTags(el.querySelector('#write-tags')?.value || ''),
                memory_type: el.querySelector('#write-memory-type')?.value?.trim() || null,
                compare_extractors: compareExtractors,
            });
            resultBox.innerHTML = buildWriteSimulation(result);
            resultBox.classList.remove('hidden');
            attachWriteSimulationListeners(resultBox);
        } catch (err) {
            resultBox.innerHTML = `<div class="card"><div class="empty-state" style="color:var(--danger)">Write simulation failed: ${escapeHtml(err.message)}</div></div>`;
            resultBox.classList.remove('hidden');
        } finally {
            button.disabled = false;
            button.textContent = 'Run Dry-Run Simulation';
        }
    });

    el.querySelector('#cons-trigger')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#cons-tenant')?.value || '';
        const user = el.querySelector('#cons-user')?.value || '';
        if (!tenant) return;
        const button = el.querySelector('#cons-trigger');
        const resultBox = el.querySelector('#cons-result');
        button.disabled = true;
        button.textContent = 'Running...';
        resultBox.classList.add('hidden');
        try {
            const result = await triggerConsolidate(tenant, user);
            resultBox.textContent = prettyJson(result);
            resultBox.classList.remove('hidden');
            showToast('Consolidation completed');
        } catch (err) {
            resultBox.textContent = `Error: ${err.message}`;
            resultBox.classList.remove('hidden');
            showToast(`Consolidation failed: ${err.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Run Consolidation';
        }
    });

    el.querySelector('#fgt-preview-btn')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#fgt-tenant')?.value || '';
        if (!tenant) return;
        const user = el.querySelector('#fgt-user')?.value || '';
        const maxMemories = parseInt(el.querySelector('#fgt-max')?.value || '200', 10);
        const resultBox = el.querySelector('#fgt-preview-result');
        resultBox.classList.remove('hidden');
        resultBox.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Building forgetting preview...</div>`;
        try {
            const result = await previewForgetting(tenant, user || null, maxMemories);
            resultBox.innerHTML = buildForgettingPreview(result);
            attachForgettingPreviewListeners(resultBox);
        } catch (err) {
            resultBox.innerHTML = `<div class="card"><div class="empty-state" style="color:var(--danger)">Failed to load forgetting preview: ${escapeHtml(err.message)}</div></div>`;
        }
    });

    el.querySelector('#fgt-trigger')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#fgt-tenant')?.value || '';
        if (!tenant) return;
        const user = el.querySelector('#fgt-user')?.value || '';
        const maxMemories = parseInt(el.querySelector('#fgt-max')?.value || '200', 10);
        const button = el.querySelector('#fgt-trigger');
        const resultBox = el.querySelector('#fgt-result');
        button.disabled = true;
        button.textContent = 'Running...';
        resultBox.classList.add('hidden');
        try {
            const result = await triggerForget(tenant, user, dryRun, maxMemories);
            resultBox.textContent = prettyJson(result);
            resultBox.classList.remove('hidden');
            showToast(`Forgetting completed${dryRun ? ' (dry run)' : ''}`);
        } catch (err) {
            resultBox.textContent = `Error: ${err.message}`;
            resultBox.classList.remove('hidden');
            showToast(`Forgetting failed: ${err.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Run Forgetting';
        }
    });

    el.querySelector('#rec-trigger')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#rec-tenant')?.value || '';
        if (!tenant) return;
        const user = el.querySelector('#rec-user')?.value || '';
        const button = el.querySelector('#rec-trigger');
        const resultBox = el.querySelector('#rec-result');
        button.disabled = true;
        button.textContent = 'Running...';
        resultBox.classList.add('hidden');
        try {
            const result = await triggerReconsolidate(tenant, user);
            resultBox.textContent = prettyJson(result);
            resultBox.classList.remove('hidden');
            showToast('Labile state released');
        } catch (err) {
            resultBox.textContent = `Error: ${err.message}`;
            resultBox.classList.remove('hidden');
            showToast(`Reconsolidation failed: ${err.message}`, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Release Labile State';
        }
    });

    dbResetBtn?.addEventListener('click', async () => {
        if ((dbConfirmInput?.value || '') !== 'DELETE') return;
        dbResetBtn.disabled = true;
        dbResetBtn.textContent = 'Running...';
        dbResetResult.classList.add('hidden');
        try {
            const result = await resetDatabase();
            dbResetResult.textContent = prettyJson(result);
            dbResetResult.classList.remove('hidden');
            showToast('Database reset completed');
        } catch (err) {
            dbResetResult.textContent = `Error: ${err.message}`;
            dbResetResult.classList.remove('hidden');
            showToast(`Database reset failed: ${err.message}`, 'error');
        } finally {
            dbConfirmInput.value = '';
            updateDbResetButton();
            dbResetBtn.textContent = 'Delete and recreate database';
        }
    });

    // Labile session expand/collapse
    el.querySelectorAll('.session-summary-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = row.dataset.sessionIdx;
            const detailRow = el.querySelector(`.session-detail-row[data-session-idx="${idx}"]`);
            const toggle = row.querySelector('.session-expand-toggle');
            if (!detailRow) return;
            const expanded = !detailRow.classList.contains('hidden');
            detailRow.classList.toggle('hidden', expanded);
            if (toggle) toggle.innerHTML = expanded ? '&#9654;' : '&#9660;';
            if (!expanded) {
                detailRow.querySelectorAll('.session-memory-link').forEach(link => {
                    link.addEventListener('click', event => {
                        event.preventDefault();
                        event.stopPropagation();
                        navigateTo('detail', { memoryId: link.dataset.memoryId });
                    });
                });
            }
        });
    });
}

function buildWriteSimulation(result) {
    const chunks = result.chunks || [];
    return `
        <div class="card">
            <div class="card-title">Write Simulation Summary</div>
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Chunks</div><div class="kpi-value">${formatNumber(result.summary.chunk_count || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Would Store</div><div class="kpi-value">${formatNumber(result.summary.would_store_count || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Skipped</div><div class="kpi-value">${formatNumber(result.summary.skipped_count || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Acceptance Rate</div><div class="kpi-value">${formatFloat(result.summary.write_gate_acceptance_rate || 0, 2)}%</div></div>
            </div>
            <div class="stack-list" style="margin-top:16px;">
                ${chunks.map((chunk, index) => `
                    <div class="stack-item">
                        <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                            <span>
                                <span class="badge badge-type">${escapeHtml(chunk.chunk_type)}</span>
                                <span class="badge ${chunk.would_store ? 'badge-ok' : 'badge-warning'}" style="margin-left:6px;">${escapeHtml(chunk.write_decision)}</span>
                                <span class="badge badge-unknown" style="margin-left:6px;">${escapeHtml(chunk.chosen_memory_type || 'unknown')}</span>
                            </span>
                            <span style="font-size:0.78rem;color:var(--text-muted)">chunk ${index + 1}</span>
                        </div>
                        <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(chunk.text)}</div>
                        <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);display:flex;gap:12px;flex-wrap:wrap;">
                            <span>salience ${formatFloat(chunk.salience, 3)}</span>
                            <span>novelty ${formatFloat(chunk.novelty, 3)}</span>
                            <span>importance ${formatFloat(chunk.importance, 3)}</span>
                            <span>confidence ${formatFloat(chunk.confidence, 3)}</span>
                            <span>decay ${chunk.decay_rate != null ? formatFloat(chunk.decay_rate, 3) : 'default'}</span>
                        </div>
                        ${chunk.risk_flags?.length ? `<div style="margin-top:8px;" class="tag-list">${chunk.risk_flags.map(flag => `<span class="tag">${escapeHtml(flag)}</span>`).join('')}</div>` : ''}
                        ${chunk.redaction_required ? `<div style="margin-top:8px;color:var(--warning);font-size:0.78rem;">Redacted output: ${escapeHtml(chunk.redacted_text)}</div>` : ''}
                        <div class="json-tree" style="margin-top:8px;">${escapeHtml(prettyJson({
                            key: chunk.key,
                            context_tags: chunk.context_tags,
                            extracted_constraints: chunk.extracted_constraints,
                            extracted_facts: chunk.extracted_facts,
                            entities: chunk.entities,
                            relations: chunk.relations,
                            extractor_outputs: chunk.extractor_outputs,
                        }))}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function buildForgettingPreview(result) {
    const items = result.items || [];
    return `
        <div class="card">
            <div class="card-title">Forgetting Preview</div>
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Scanned</div><div class="kpi-value">${formatNumber(result.scanned_count || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Duplicates</div><div class="kpi-value">${formatNumber(result.duplicates_found || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Planned Ops</div><div class="kpi-value">${formatNumber(result.operations_planned || 0)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Top Action</div><div class="kpi-value" style="font-size:0.95rem">${escapeHtml(topAction(result.summary || {}))}</div></div>
            </div>
            ${items.length ? `
                <div class="table-responsive" style="margin-top:16px;">
                    <table class="data-table">
                        <thead>
                            <tr><th>Memory</th><th>Type</th><th>Score</th><th>Suggested Action</th><th>Breakdown</th><th>Duplicates</th></tr>
                        </thead>
                        <tbody>
                            ${items.slice(0, 40).map(item => `
                                <tr>
                                    <td>
                                        <div class="text-preview" style="max-width:280px">${escapeHtml(item.text)}</div>
                                        <div style="margin-top:6px;font-size:0.78rem;color:var(--text-muted);">
                                            <a href="#detail/${item.memory_id}" class="forget-memory-link" data-memory-id="${escapeHtml(item.memory_id)}">open memory</a>
                                            ${item.protected ? '<span style="margin-left:8px;color:var(--warning)">protected</span>' : ''}
                                        </div>
                                    </td>
                                    <td><span class="badge badge-type">${escapeHtml(item.type)}</span></td>
                                    <td>${formatFloat(item.total_score, 4)}</td>
                                    <td><span class="badge ${item.suggested_action === 'keep' ? 'badge-ok' : item.suggested_action === 'delete' ? 'badge-error' : 'badge-warning'}">${escapeHtml(item.suggested_action)}</span></td>
                                    <td class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson({
                                        importance: item.importance_score,
                                        recency: item.recency_score,
                                        frequency: item.frequency_score,
                                        confidence: item.confidence_score,
                                        type_bonus: item.type_bonus_score,
                                        dependency: item.dependency_score,
                                        dependency_count: item.dependency_count,
                                    }))}</td>
                                    <td class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson(item.duplicate_matches || []))}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;margin-top:16px;">No preview items returned.</div>'}
        </div>
    `;
}

function attachWriteSimulationListeners(host) {
    host.querySelectorAll('.write-memory-link').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            navigateTo('detail', { memoryId: link.dataset.memoryId });
        });
    });
}

function attachForgettingPreviewListeners(host) {
    host.querySelectorAll('.forget-memory-link').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            navigateTo('detail', { memoryId: link.dataset.memoryId });
        });
    });
}

function splitTags(value) {
    return value
        .split(',')
        .map(item => item.trim())
        .filter(Boolean);
}

function topAction(summary) {
    const entries = Object.entries(summary);
    if (!entries.length) {
        return '—';
    }
    entries.sort((a, b) => b[1] - a[1]);
    return `${entries[0][0]} (${entries[0][1]})`;
}
