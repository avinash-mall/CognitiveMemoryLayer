/**
 * Evaluation Page
 * Benchmark readiness, discovered artifacts, and latest summary/report.
 */

import { getEvaluationSummary } from '../api.js';
import { escapeHtml, formatDate, formatNumber, prettyJson } from '../utils/formatters.js';

const container = () => document.getElementById('page-evaluation');

export async function renderEvaluation() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading evaluation summary...</div>`;

    try {
        const data = await getEvaluationSummary();
        el.innerHTML = buildPage(data);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load evaluation summary: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildPage(data) {
    const readiness = data.readiness || {};
    const artifacts = data.artifacts || [];
    const latestSummary = data.latest_summary;

    return `
        <p class="page-desc">
            Evaluation readiness and discovered benchmark outputs. This page surfaces the repository’s existing
            LoCoMo/Locomo-Plus tooling and any output artifacts already generated on disk.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Artifacts</div><div class="kpi-value">${formatNumber(artifacts.length)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Samples Ready</div><div class="kpi-value">${readiness.unified_samples_present ? 'Yes' : 'No'}</div></div>
            <div class="kpi-card"><div class="kpi-label">Latest Summary</div><div class="kpi-value">${readiness.latest_summary_present ? 'Present' : 'Missing'}</div></div>
            <div class="kpi-card"><div class="kpi-label">Eval Model</div><div class="kpi-value" style="font-size:0.95rem">${escapeHtml(readiness.llm_eval_model || '—')}</div></div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Readiness</div>
                <div class="json-tree">${escapeHtml(prettyJson(readiness))}</div>
            </div>
            <div class="card">
                <div class="card-title">Latest Summary</div>
                ${latestSummary ? `<div class="json-tree">${escapeHtml(prettyJson(latestSummary))}</div>` : '<div class="empty-state" style="padding:12px;">No summary artifact found yet.</div>'}
            </div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Artifacts</div>
            ${artifacts.length ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Name</th><th>Kind</th><th>Size</th><th>Updated</th><th>Path</th></tr></thead>
                        <tbody>
                            ${artifacts.map(item => `
                                <tr>
                                    <td>${escapeHtml(item.name)}</td>
                                    <td><span class="badge badge-type">${escapeHtml(item.kind)}</span></td>
                                    <td>${formatNumber(item.size_bytes || 0)}</td>
                                    <td>${formatDate(item.updated_at)}</td>
                                    <td class="mono" style="font-size:0.78rem;">${escapeHtml(item.path)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;">No evaluation artifacts discovered in evaluation/outputs.</div>'}
        </div>

        ${data.latest_report ? `
            <div class="card" style="margin-top:16px;">
                <div class="card-title">Latest Comparison Report</div>
                <pre class="result-box" style="white-space:pre-wrap;max-height:420px;overflow:auto">${escapeHtml(data.latest_report)}</pre>
            </div>
        ` : ''}
    `;
}
