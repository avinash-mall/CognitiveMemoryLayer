/**
 * Evaluation Page
 * Benchmark readiness, LoCoMo / LoCoMo-Plus category breakdowns, run comparisons,
 * discovered artifacts, and latest evaluation report.
 */

import { getEvaluationSummary } from '../api.js';
import { escapeHtml, formatDate, formatFloat, formatNumber } from '../utils/formatters.js';

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

function formatPct(value) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return `${formatFloat(Number(value) * 100, 2)}%`;
}

function scoreBar(value) {
    const pct = Math.max(0, Math.min(100, (Number(value) || 0) * 100));
    const color = pct >= 70 ? 'var(--success, #22c55e)' : pct >= 45 ? 'var(--accent, #6366f1)' : 'var(--warning, #f59e0b)';
    return `
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="flex:1;min-width:90px;height:8px;background:var(--bg-tertiary, #1f2937);border-radius:4px;overflow:hidden;">
                <div style="width:${pct.toFixed(1)}%;height:100%;background:${color};border-radius:4px;"></div>
            </div>
            <span class="mono" style="font-size:0.82rem;min-width:56px;text-align:right;">${formatFloat(pct, 2)}%</span>
        </div>
    `;
}

function readinessBadge(ok, trueLabel = 'Ready', falseLabel = 'Missing') {
    return `<span class="badge ${ok ? 'badge-ok' : 'badge-warning'}">${escapeHtml(ok ? trueLabel : falseLabel)}</span>`;
}

function buildCategoryBreakdown(summary) {
    const byCategory = summary?.by_category;
    if (!byCategory || !Object.keys(byCategory).length) {
        return '<div class="empty-state" style="padding:12px;">No category breakdown available in latest summary.</div>';
    }
    const rows = Object.entries(byCategory).sort((a, b) => (b[1]?.avg || 0) - (a[1]?.avg || 0));
    return `
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr><th>Category</th><th>Average Score</th><th>Score / Total</th><th>Valid</th><th>Errors</th></tr>
                </thead>
                <tbody>
                    ${rows.map(([cat, stats]) => `
                        <tr>
                            <td><strong>${escapeHtml(cat)}</strong></td>
                            <td style="min-width:180px;">${scoreBar(stats.avg)}</td>
                            <td class="mono">${formatFloat(stats.score ?? 0, 1)} / ${formatNumber(stats.count ?? 0)}</td>
                            <td>${formatNumber(stats.valid ?? stats.count ?? 0)}</td>
                            <td><span class="badge ${(stats.errors || 0) === 0 ? 'badge-ok' : 'badge-error'}">${formatNumber(stats.errors || 0)}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function buildPage(data) {
    const readiness = data.readiness || {};
    const artifacts = data.artifacts || [];
    const latestSummary = data.latest_summary || null;
    const comparisons = data.comparisons || [];

    const overallAvg = latestSummary ? (latestSummary.overall_avg ?? latestSummary.score) : null;
    const validSamples = latestSummary?.valid_samples ?? latestSummary?.total_samples;
    const totalSamples = latestSummary?.total_samples;

    const readinessChecks = [
        ['LoCoMo-10 Dataset', readiness.locomo_data_present],
        ['LoCoMo-Plus Dataset', readiness.locomo_plus_data_present],
        ['Unified Input Samples', readiness.unified_samples_present],
        ['Eval Scripts', readiness.scripts_present],
        ['Results Directory', readiness.results_dir_present],
        ['Latest Summary JSON', readiness.latest_summary_present],
        ['Evaluation Report', readiness.eval_report_present],
    ];

    return `
        <p class="page-desc">
            Benchmark readiness, LoCoMo / LoCoMo-Plus judge summaries from <code>evaluation/results</code> and
            <code>evaluation/outputs</code>, and the latest evaluation report.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Overall Score</div>
                <div class="kpi-value">${formatPct(overallAvg)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Samples Scored</div>
                <div class="kpi-value">${totalSamples != null ? `${formatNumber(validSamples ?? totalSamples)} / ${formatNumber(totalSamples)}` : '—'}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Artifacts</div>
                <div class="kpi-value">${formatNumber(artifacts.length)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Eval Model</div>
                <div class="kpi-value" style="font-size:0.9rem;word-break:break-word;">${escapeHtml(readiness.llm_eval_model || '—')}</div>
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">Category Breakdown ${readiness.latest_summary_file ? `<span class="mono" style="font-size:0.74rem;color:var(--text-muted);font-weight:400;margin-left:8px;">${escapeHtml(readiness.latest_summary_file)}</span>` : ''}</div>
                ${buildCategoryBreakdown(latestSummary)}
            </div>
            <div class="card">
                <div class="card-title">Benchmark Readiness</div>
                <div class="stack-list">
                    ${readinessChecks.map(([label, ok]) => `
                        <div class="stack-item" style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;">
                            <span>${escapeHtml(label)}</span>
                            ${readinessBadge(Boolean(ok))}
                        </div>
                    `).join('')}
                </div>
                <div class="detail-field" style="margin-top:12px;">
                    <span class="detail-field-label">Eval Provider</span>
                    <div class="detail-field-value mono">${escapeHtml(readiness.llm_eval_provider || '—')}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Generated</span>
                    <div class="detail-field-value">${formatDate(data.generated_at)}</div>
                </div>
            </div>
        </div>

        ${comparisons.length > 1 ? `
            <div class="card" style="margin-top:16px;">
                <div class="card-title">Discovered Benchmark Runs (${comparisons.length})</div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Summary File</th><th>Overall</th><th>Samples</th><th>Errors</th><th>Updated</th></tr></thead>
                        <tbody>
                            ${comparisons.map(run => `
                                <tr>
                                    <td class="mono" style="font-size:0.8rem;">${escapeHtml(run.name)}</td>
                                    <td><strong>${formatPct(run.overall_avg)}</strong></td>
                                    <td>${run.total_samples != null ? formatNumber(run.total_samples) : '—'}</td>
                                    <td><span class="badge ${(run.error_count || 0) === 0 ? 'badge-ok' : 'badge-error'}">${formatNumber(run.error_count || 0)}</span></td>
                                    <td>${formatDate(run.updated_at)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        ` : ''}

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Discovered Artifacts</div>
            ${artifacts.length ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Name</th><th>Kind</th><th>Size (bytes)</th><th>Updated</th><th>Path</th></tr></thead>
                        <tbody>
                            ${artifacts.map(item => `
                                <tr>
                                    <td class="mono" style="font-size:0.8rem;">${escapeHtml(item.name)}</td>
                                    <td><span class="badge badge-type">${escapeHtml(item.kind)}</span></td>
                                    <td>${formatNumber(item.size_bytes || 0)}</td>
                                    <td>${formatDate(item.updated_at)}</td>
                                    <td class="mono" style="font-size:0.75rem;color:var(--text-muted);">${escapeHtml(item.path)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;">No evaluation artifacts discovered in evaluation/results or evaluation/outputs.</div>'}
        </div>

        ${data.latest_report ? `
            <div class="card" style="margin-top:16px;">
                <div class="card-title">Latest Evaluation Report ${readiness.eval_report_file ? `<span class="mono" style="font-size:0.74rem;color:var(--text-muted);font-weight:400;margin-left:8px;">${escapeHtml(readiness.eval_report_file)}</span>` : ''}</div>
                <pre class="result-box" style="white-space:pre-wrap;max-height:460px;overflow:auto">${escapeHtml(data.latest_report)}</pre>
            </div>
        ` : ''}
    `;
}
