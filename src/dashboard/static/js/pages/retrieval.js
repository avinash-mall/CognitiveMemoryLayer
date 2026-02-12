/**
 * Retrieval Test Page
 * Interactive query tool to debug memory retrieval.
 */

import { testRetrieval, getTenants } from '../api.js';
import { formatNumber, formatFloat, formatDate, escapeHtml, prettyJson } from '../utils/formatters.js';

const container = () => document.getElementById('page-retrieval');

let tenants = [];

export async function renderRetrieval({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading...</div>`;

    try {
        const data = await getTenants();
        tenants = data.tenants || [];
    } catch { tenants = []; }

    el.innerHTML = buildPage(tenantId);
    attachListeners(el);
}

function buildPage(tenantId) {
    const tenantOptions = tenants.map(t =>
        `<option value="${escapeHtml(t.tenant_id)}" ${t.tenant_id === tenantId ? 'selected' : ''}>${escapeHtml(t.tenant_id)}</option>`
    ).join('');

    return `
        <p style="color:var(--text-secondary);margin-bottom:20px;">
            Test memory retrieval using the same read API the application uses.
            Useful to debug "why didn't the assistant remember X?"
        </p>

        <div class="card">
            <div class="card-title">Query</div>
            <div class="retrieval-form">
                <div class="form-row">
                    <label for="ret-tenant">Tenant</label>
                    <select id="ret-tenant" class="select-sm">
                        ${tenantOptions || '<option value="">No tenants</option>'}
                    </select>
                </div>
                <div class="form-row">
                    <label for="ret-query">Query</label>
                    <textarea id="ret-query" class="input-sm" rows="3" placeholder="What do you want to search for?" style="resize:vertical;min-height:60px"></textarea>
                </div>
                <div class="form-row-inline">
                    <div class="form-row">
                        <label for="ret-max">Max Results</label>
                        <input type="range" id="ret-max" min="1" max="50" value="10" style="width:150px">
                        <span id="ret-max-label" style="font-size:0.85rem;color:var(--text-secondary);margin-left:8px">10</span>
                    </div>
                    <div class="form-row">
                        <label for="ret-format">Format</label>
                        <select id="ret-format" class="select-sm">
                            <option value="list">List</option>
                            <option value="packet">Packet</option>
                            <option value="llm_context">LLM Context</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <label for="ret-context">Context Filter (comma-separated tags)</label>
                    <input type="text" id="ret-context" class="input-sm" placeholder="e.g. preferences, work">
                </div>
                <button id="ret-search-btn" class="btn btn-primary" ${!tenants.length ? 'disabled' : ''}>Search Memories</button>
            </div>
        </div>

        <div id="ret-results" class="hidden" style="margin-top:16px;"></div>
    `;
}

function attachListeners(el) {
    const maxSlider = el.querySelector('#ret-max');
    const maxLabel = el.querySelector('#ret-max-label');
    maxSlider?.addEventListener('input', () => { maxLabel.textContent = maxSlider.value; });

    el.querySelector('#ret-search-btn')?.addEventListener('click', async () => {
        const tenant = el.querySelector('#ret-tenant')?.value;
        const query = el.querySelector('#ret-query')?.value?.trim();
        if (!tenant || !query) return;

        const maxResults = parseInt(maxSlider?.value || '10', 10);
        const format = el.querySelector('#ret-format')?.value || 'list';
        const contextRaw = el.querySelector('#ret-context')?.value?.trim();
        const contextFilter = contextRaw ? contextRaw.split(',').map(s => s.trim()).filter(Boolean) : null;

        const btn = el.querySelector('#ret-search-btn');
        const resultsDiv = el.querySelector('#ret-results');
        btn.disabled = true;
        btn.textContent = 'Searching...';
        resultsDiv.classList.add('hidden');

        try {
            const data = await testRetrieval(tenant, query, maxResults, contextFilter, null, format);
            resultsDiv.innerHTML = buildResults(data);
            resultsDiv.classList.remove('hidden');
        } catch (err) {
            resultsDiv.innerHTML = `<div class="card"><div class="empty-state" style="color:var(--danger)">Error: ${escapeHtml(err.message)}</div></div>`;
            resultsDiv.classList.remove('hidden');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Search Memories';
        }
    });
}

function buildResults(data) {
    const summary = `
        <div class="kpi-grid" style="margin-bottom:16px;">
            <div class="kpi-card"><div class="kpi-label">Results</div><div class="kpi-value">${formatNumber(data.total_count)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Elapsed</div><div class="kpi-value">${formatFloat(data.elapsed_ms, 1)}ms</div></div>
            <div class="kpi-card"><div class="kpi-label">Query</div><div class="kpi-value" style="font-size:0.9rem">${escapeHtml(data.query.substring(0, 40))}</div></div>
        </div>
    `;

    if (data.llm_context) {
        return summary + `
            <div class="card"><div class="card-title">LLM Context Output</div>
                <pre class="result-box" style="white-space:pre-wrap;max-height:500px;overflow-y:auto">${escapeHtml(data.llm_context)}</pre>
            </div>`;
    }

    const resultCards = (data.results || []).map((r, i) => `
        <div class="card retrieval-result-card" style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div>
                    <span style="font-weight:600;color:var(--text-primary)">#${i + 1}</span>
                    <span class="badge badge-type" style="margin-left:8px">${escapeHtml(r.type)}</span>
                    ${r.retrieval_source ? `<span class="badge badge-ok" style="margin-left:4px">${escapeHtml(r.retrieval_source)}</span>` : ''}
                </div>
                <div style="text-align:right">
                    <div class="retrieval-score">
                        <div class="gauge-bar" style="width:80px;display:inline-flex;vertical-align:middle;margin-right:6px">
                            <div class="gauge-fill gauge-fill-accent" style="width:${Math.min(100, r.relevance_score * 100)}%"></div>
                        </div>
                        <span style="font-weight:600;font-size:0.85rem">${formatFloat(r.relevance_score, 3)}</span>
                    </div>
                </div>
            </div>
            <div style="color:var(--text-primary);font-size:0.9rem;line-height:1.6;margin-bottom:8px;">${escapeHtml(r.text)}</div>
            <div style="display:flex;gap:16px;font-size:0.8rem;color:var(--text-muted);">
                <span>Confidence: ${formatFloat(r.confidence, 3)}</span>
                <span>${formatDate(r.timestamp)}</span>
            </div>
        </div>
    `).join('');

    return summary + (resultCards || '<div class="card"><div class="empty-state">No results found</div></div>');
}
