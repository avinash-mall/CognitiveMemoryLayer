/**
 * Retrieval Debugger Page
 * Interactive explain-mode query tool with saved investigations.
 */

import { explainRetrieval, getTenants } from '../api.js';
import { navigateTo } from '../app.js';
import {
    escapeHtml,
    formatDate,
    formatFloat,
    formatNumber,
    prettyJson,
} from '../utils/formatters.js';

const container = () => document.getElementById('page-retrieval');
const SAVED_KEY = 'cml_saved_retrieval_queries';

let tenants = [];

export async function renderRetrieval({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading retrieval debugger...</div>`;

    try {
        const data = await getTenants();
        tenants = data.tenants || [];
    } catch {
        tenants = [];
    }

    el.innerHTML = buildPage(tenantId);
    attachListeners(el);
    renderSavedQueries(el);
}

function buildPage(tenantId) {
    const tenantOptions = tenants.map((tenant) => (
        `<option value="${escapeHtml(tenant.tenant_id)}" ${tenant.tenant_id === tenantId ? 'selected' : ''}>${escapeHtml(tenant.tenant_id)}</option>`
    )).join('');

    return `
        <p class="page-desc">
            Inspect retrieval the same way the runtime executes it: query classification, planner choices,
            per-source execution, reranker contributions, packet warnings, and final LLM context.
        </p>

        <div class="detail-grid">
            <div class="card">
                <div class="card-title">Debugger</div>
                <div class="retrieval-form">
                    <div class="form-row">
                        <label for="ret-tenant">Tenant</label>
                        <select id="ret-tenant" class="select-sm">
                            ${tenantOptions || '<option value="">No tenants</option>'}
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="ret-query">Primary Query</label>
                        <textarea id="ret-query" class="input-sm" rows="3" placeholder="What should the assistant remember?" style="resize:vertical;min-height:84px"></textarea>
                    </div>
                    <div class="form-row">
                        <label for="ret-compare-query">Compare Query</label>
                        <textarea id="ret-compare-query" class="input-sm" rows="2" placeholder="Optional second query for side-by-side comparison" style="resize:vertical;min-height:64px"></textarea>
                    </div>
                    <div class="form-row-inline">
                        <div class="form-row">
                            <label for="ret-max">Max Results</label>
                            <input type="range" id="ret-max" min="1" max="30" value="10" style="width:160px">
                            <span id="ret-max-label" style="font-size:0.85rem;color:var(--text-secondary);margin-left:8px">10</span>
                        </div>
                        <div class="form-row">
                            <label for="ret-context">Context Tags</label>
                            <input type="text" id="ret-context" class="input-sm" placeholder="preferences, work">
                        </div>
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;">
                        <button id="ret-search-btn" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>Explain Retrieval</button>
                        <button id="ret-save-btn" class="btn btn-ghost" ${!tenants.length ? 'disabled' : ''}>Save Query</button>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Saved Investigations</div>
                <div id="ret-saved-list"></div>
            </div>
        </div>

        <div id="ret-results" style="margin-top:16px;"></div>
    `;
}

function attachListeners(el) {
    const maxSlider = el.querySelector('#ret-max');
    const maxLabel = el.querySelector('#ret-max-label');
    maxSlider?.addEventListener('input', () => {
        maxLabel.textContent = maxSlider.value;
    });

    el.querySelector('#ret-search-btn')?.addEventListener('click', async () => {
        await runDebugger(el);
    });

    el.querySelector('#ret-save-btn')?.addEventListener('click', () => {
        const tenantId = el.querySelector('#ret-tenant')?.value || '';
        const query = el.querySelector('#ret-query')?.value?.trim() || '';
        if (!tenantId || !query) {
            return;
        }
        saveQuery({
            tenant_id: tenantId,
            query,
            compare_query: el.querySelector('#ret-compare-query')?.value?.trim() || '',
            context_tags: el.querySelector('#ret-context')?.value?.trim() || '',
            max_results: parseInt(el.querySelector('#ret-max')?.value || '10', 10),
        });
        renderSavedQueries(el);
    });
}

async function runDebugger(el) {
    const tenantId = el.querySelector('#ret-tenant')?.value || '';
    const query = el.querySelector('#ret-query')?.value?.trim() || '';
    const compareQuery = el.querySelector('#ret-compare-query')?.value?.trim() || '';
    if (!tenantId || !query) {
        return;
    }

    const maxResults = parseInt(el.querySelector('#ret-max')?.value || '10', 10);
    const contextFilter = splitTags(el.querySelector('#ret-context')?.value || '');
    const resultsDiv = el.querySelector('#ret-results');
    const button = el.querySelector('#ret-search-btn');
    button.disabled = true;
    button.textContent = 'Explaining...';
    resultsDiv.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Running retrieval explain mode...</div>`;

    try {
        const requests = [
            explainRetrieval(tenantId, query, maxResults, contextFilter, null, 'list'),
        ];
        if (compareQuery) {
            requests.push(explainRetrieval(tenantId, compareQuery, maxResults, contextFilter, null, 'list'));
        }
        const [primary, secondary] = await Promise.all(requests);
        resultsDiv.innerHTML = compareQuery
            ? buildCompare(primary, secondary)
            : buildExplainPanel('Primary Query', primary);
        attachResultListeners(resultsDiv);
    } catch (err) {
        resultsDiv.innerHTML = `<div class="card"><div class="empty-state" style="color:var(--danger)">Error: ${escapeHtml(err.message)}</div></div>`;
    } finally {
        button.disabled = false;
        button.textContent = 'Explain Retrieval';
    }
}

function buildCompare(primary, secondary) {
    return `
        <div class="compare-grid">
            ${buildExplainPanel('Primary Query', primary)}
            ${buildExplainPanel('Compare Query', secondary)}
        </div>
    `;
}

function buildExplainPanel(title, data) {
    const analysis = data.analysis || {};
    const meta = data.retrieval_meta || {};
    const warnings = data.packet_warnings || [];
    const openQuestions = data.open_questions || [];

    return `
        <div class="compare-panel">
            <div class="kpi-grid" style="margin-bottom:16px;">
                <div class="kpi-card"><div class="kpi-label">${escapeHtml(title)}</div><div class="kpi-value" style="font-size:0.95rem">${escapeHtml(data.query)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Results</div><div class="kpi-value">${formatNumber(data.total_count)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Elapsed</div><div class="kpi-value">${formatFloat(data.elapsed_ms, 1)}ms</div></div>
                <div class="kpi-card"><div class="kpi-label">Intent</div><div class="kpi-value" style="font-size:0.95rem">${escapeHtml(analysis.intent || 'unknown')}</div></div>
            </div>

            <div class="detail-grid">
                <div class="card">
                    <div class="card-title">Query Analysis</div>
                    ${kv('Confidence', formatFloat(analysis.confidence || 0, 3))}
                    ${kv('Domain', analysis.query_domain || '—')}
                    ${kv('Constraint Dimensions', (analysis.constraint_dimensions || []).join(', ') || '—')}
                    ${kv('Entities', (analysis.entities || []).join(', ') || '—')}
                    ${kv('Key Phrases', (analysis.key_phrases || []).join(', ') || '—')}
                    ${kv('Suggested Sources', (analysis.suggested_sources || []).join(', ') || '—')}
                    ${kv('Decision Query', analysis.is_decision_query ? 'Yes' : 'No')}
                    ${(analysis.time_reference || analysis.time_start || analysis.time_end) ? `
                        <div class="json-tree" style="margin-top:12px;">${escapeHtml(prettyJson({
                            time_reference: analysis.time_reference,
                            time_start: analysis.time_start,
                            time_end: analysis.time_end,
                            metadata: analysis.metadata || {},
                        }))}</div>
                    ` : ''}
                </div>

                <div class="card">
                    <div class="card-title">Retrieval Meta</div>
                    ${kv('Sources Attempted', (meta.sources_attempted || []).join(', ') || '—')}
                    ${kv('Sources Completed', (meta.sources_completed || []).join(', ') || '—')}
                    ${kv('Timed Out', (meta.sources_timed_out || []).join(', ') || '—')}
                    ${kv('Failed', (meta.sources_failed || []).join(', ') || '—')}
                    ${kv('Total Elapsed', meta.total_elapsed_ms != null ? `${formatFloat(meta.total_elapsed_ms, 1)}ms` : '—')}
                </div>
            </div>

            <div class="card" style="margin-top:16px;">
                <div class="card-title">Planner Steps</div>
                ${buildPlanTable(data.plan_steps || [], data.parallel_groups || [])}
            </div>

            <div class="card" style="margin-top:16px;">
                <div class="card-title">Execution Trace</div>
                ${buildExecutionTable(data.execution_steps || [])}
            </div>

            <div class="detail-grid" style="margin-top:16px;">
                <div class="card">
                    <div class="card-title">Reranker Contributions</div>
                    ${buildRerankList(data.rerank || [])}
                </div>
                <div class="card">
                    <div class="card-title">Packet Warnings</div>
                    ${buildSimpleList(warnings, 'No packet warnings.')}
                    <div class="card-title" style="margin-top:16px;">Open Questions</div>
                    ${buildSimpleList(openQuestions, 'No open questions.')}
                </div>
            </div>

            <div class="detail-grid" style="margin-top:16px;">
                <div class="card">
                    <div class="card-title">Retrieved Memories</div>
                    ${buildResultsList(data.results || [])}
                </div>
                <div class="card">
                    <div class="card-title">Final LLM Context</div>
                    <pre class="result-box" style="white-space:pre-wrap;max-height:420px;overflow:auto">${escapeHtml(data.llm_context || 'No LLM context generated.')}</pre>
                </div>
            </div>
        </div>
    `;
}

function buildPlanTable(steps = [], groups = []) {
    if (!steps.length) {
        return '<div class="empty-state" style="padding:12px;">No planner steps.</div>';
    }
    const groupMap = {};
    groups.forEach((group, index) => {
        group.forEach((stepIndex) => {
            groupMap[stepIndex] = index + 1;
        });
    });
    return `
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr><th>#</th><th>Source</th><th>Priority</th><th>Parallel Group</th><th>Top K</th><th>Timeout</th><th>Filters</th></tr>
                </thead>
                <tbody>
                    ${steps.map((step, index) => `
                        <tr>
                            <td>${index}</td>
                            <td><span class="badge badge-type">${escapeHtml(step.source)}</span></td>
                            <td>${formatNumber(step.priority || 0)}</td>
                            <td>${groupMap[index] || '—'}</td>
                            <td>${formatNumber(step.top_k || 0)}</td>
                            <td>${formatNumber(step.timeout_ms || 0)}ms</td>
                            <td><div class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson({
                                key: step.key,
                                query: step.query,
                                seeds: step.seeds,
                                memory_types: step.memory_types,
                                time_filter: step.time_filter,
                                constraint_categories: step.constraint_categories,
                                skip_if_found: step.skip_if_found,
                            }))}</div></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function buildExecutionTable(steps = []) {
    if (!steps.length) {
        return '<div class="empty-state" style="padding:12px;">No execution data.</div>';
    }
    return `
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr><th>Source</th><th>Status</th><th>Elapsed</th><th>Results</th><th>Filters</th><th>Preview</th></tr>
                </thead>
                <tbody>
                    ${steps.map((step) => `
                        <tr>
                            <td><span class="badge badge-type">${escapeHtml(step.source)}</span></td>
                            <td>
                                <span class="badge ${step.success ? 'badge-ok' : step.timed_out ? 'badge-warning' : 'badge-error'}">
                                    ${step.success ? 'ok' : step.timed_out ? 'timeout' : 'failed'}
                                </span>
                                ${step.error ? `<div style="margin-top:6px;font-size:0.78rem;color:var(--text-muted)">${escapeHtml(step.error)}</div>` : ''}
                            </td>
                            <td>${formatFloat(step.elapsed_ms || 0, 1)}ms</td>
                            <td>${formatNumber(step.result_count || 0)}</td>
                            <td><div class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson(step.filters || {}))}</div></td>
                            <td><div class="mono" style="font-size:0.78rem;white-space:pre-wrap">${escapeHtml(prettyJson(step.result_preview || []))}</div></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function buildRerankList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No reranker output.</div>';
    }
    return `
        <div class="stack-list">
            ${items.map((item) => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span>
                            <span class="badge badge-type">${escapeHtml(item.source_type)}</span>
                            <span class="badge badge-ok" style="margin-left:6px;">${escapeHtml(item.retrieval_source || 'retrieved')}</span>
                        </span>
                        <span style="font-size:0.82rem;font-weight:600;">score ${formatFloat(item.final_score, 4)}</span>
                    </div>
                    <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.text)}</div>
                    <div class="json-tree" style="margin-top:8px;">${escapeHtml(prettyJson(item.breakdown || {}))}</div>
                    ${item.notes?.length ? `<div style="margin-top:8px;color:var(--text-muted);font-size:0.78rem;">${escapeHtml(item.notes.join(', '))}</div>` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

function buildResultsList(items = []) {
    if (!items.length) {
        return '<div class="empty-state" style="padding:12px;">No retrieval results.</div>';
    }
    return `
        <div class="stack-list">
            ${items.map((item) => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span>
                            <span class="badge badge-type">${escapeHtml(item.type)}</span>
                            ${item.retrieval_source ? `<span class="badge badge-ok" style="margin-left:6px;">${escapeHtml(item.retrieval_source)}</span>` : ''}
                        </span>
                        <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.timestamp)}</span>
                    </div>
                    <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.text)}</div>
                    <div style="margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;font-size:0.78rem;color:var(--text-muted);">
                        <span>confidence ${formatFloat(item.confidence, 3)}</span>
                        <span>relevance ${formatFloat(item.relevance_score, 3)}</span>
                        <a href="#detail/${item.id}" class="ret-memory-link" data-memory-id="${escapeHtml(item.id)}">open memory</a>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function buildSimpleList(items = [], emptyText) {
    if (!items.length) {
        return `<div class="empty-state" style="padding:12px;">${escapeHtml(emptyText)}</div>`;
    }
    return `<div class="stack-list">${items.map((item) => `<div class="stack-item">${escapeHtml(item)}</div>`).join('')}</div>`;
}

function kv(label, value) {
    return `
        <div class="detail-field">
            <span class="detail-field-label">${escapeHtml(label)}</span>
            <div class="detail-field-value">${escapeHtml(value)}</div>
        </div>
    `;
}

function splitTags(value) {
    return value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
}

function savedQueries() {
    try {
        return JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
    } catch {
        return [];
    }
}

function saveQuery(query) {
    const current = savedQueries();
    current.unshift({
        ...query,
        saved_at: new Date().toISOString(),
    });
    localStorage.setItem(SAVED_KEY, JSON.stringify(current.slice(0, 12)));
}

function renderSavedQueries(el) {
    const saved = savedQueries();
    const host = el.querySelector('#ret-saved-list');
    if (!saved.length) {
        host.innerHTML = '<div class="empty-state" style="padding:12px;">No saved queries yet.</div>';
        return;
    }
    host.innerHTML = `
        <div class="stack-list">
            ${saved.map((item, index) => `
                <div class="stack-item">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <span class="mono" style="font-size:0.78rem;">${escapeHtml(item.tenant_id)}</span>
                        <span style="font-size:0.78rem;color:var(--text-muted)">${formatDate(item.saved_at)}</span>
                    </div>
                    <div style="margin-top:8px;color:var(--text-primary)">${escapeHtml(item.query)}</div>
                    ${item.compare_query ? `<div style="margin-top:6px;color:var(--text-muted);font-size:0.78rem;">compare: ${escapeHtml(item.compare_query)}</div>` : ''}
                    <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
                        <button class="btn btn-ghost btn-xs ret-load-saved" data-index="${index}">Load</button>
                        <button class="btn btn-ghost btn-xs ret-delete-saved" data-index="${index}">Delete</button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;

    host.querySelectorAll('.ret-load-saved').forEach((button) => {
        button.addEventListener('click', () => {
            const item = saved[parseInt(button.dataset.index, 10)];
            if (!item) return;
            el.querySelector('#ret-tenant').value = item.tenant_id;
            el.querySelector('#ret-query').value = item.query;
            el.querySelector('#ret-compare-query').value = item.compare_query || '';
            el.querySelector('#ret-context').value = item.context_tags || '';
            el.querySelector('#ret-max').value = item.max_results || 10;
            el.querySelector('#ret-max-label').textContent = String(item.max_results || 10);
        });
    });

    host.querySelectorAll('.ret-delete-saved').forEach((button) => {
        button.addEventListener('click', () => {
            const next = saved.filter((_, index) => index !== parseInt(button.dataset.index, 10));
            localStorage.setItem(SAVED_KEY, JSON.stringify(next));
            renderSavedQueries(el);
        });
    });
}

function attachResultListeners(el) {
    el.querySelectorAll('.ret-memory-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            navigateTo('detail', { memoryId: link.dataset.memoryId });
        });
    });
}
