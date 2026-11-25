/**
 * Quality Dashboard Page
 * Data quality, CML v3.0 subsystem hygiene (Hyperedges, DMN, Procedural), and Neo4j graph hygiene.
 */

import { getQualityOverview } from '../api.js';
import { navigateTo } from '../app.js';
import { escapeHtml, formatDate, formatFloat, formatNumber } from '../utils/formatters.js';

const container = () => document.getElementById('page-quality');

const ISSUE_LABELS = {
    low_confidence_memories: 'Low-Confidence Active Memories',
    low_evidence_facts: 'Single-Evidence Semantic Facts',
    orphan_facts: 'Orphan Semantic Facts (No Evidence IDs)',
    duplicate_content_hash_groups: 'Duplicate Content Hash Groups',
    broken_memory_lineage: 'Broken Memory Lineage',
    broken_fact_lineage: 'Broken Fact Lineage',
    missing_embeddings: 'Active Memories Missing Embeddings',
    low_confidence_hyperedges: 'Low-Confidence Hyperedge Engrams',
    failing_procedural_skills: 'Failing Procedural Skills',
};

const MEMORY_SAMPLE_ISSUES = new Set(['low_confidence_memories']);

export async function renderQuality({ tenantId } = {}) {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading quality overview...</div>`;

    try {
        const data = await getQualityOverview(tenantId);
        el.innerHTML = buildPage(data);
        attachListeners(el);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load quality overview: ${escapeHtml(err.message)}</p></div>`;
    }
}

function attachListeners(el) {
    el.querySelectorAll('.quality-memory-link').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            navigateTo('detail', { memoryId: link.dataset.memoryId });
        });
    });
}

function renderSampleIds(issue) {
    if (!issue.sample_ids || !issue.sample_ids.length) return '';
    if (MEMORY_SAMPLE_ISSUES.has(issue.label)) {
        const links = issue.sample_ids
            .map(id => `<a href="#detail/${escapeHtml(id)}" class="quality-memory-link mono" data-memory-id="${escapeHtml(id)}">${escapeHtml(id)}</a>`)
            .join(', ');
        return `<div style="margin-top:8px;font-size:0.78rem;">samples: ${links}</div>`;
    }
    return `<div class="mono" style="margin-top:8px;font-size:0.78rem;color:var(--text-muted);">samples: ${escapeHtml(issue.sample_ids.join(', '))}</div>`;
}

function buildPage(data) {
    const issues = data.issues || [];
    const hotspots = data.invalidation_hotspots || [];
    const graph = data.graph_hygiene || {};
    const v3 = data.v3_hygiene || {};
    const nonZeroIssues = issues.filter(i => (i.count || 0) > 0).length;

    return `
        <p class="page-desc">
            Inspect memory integrity, semantic fact evidence, CML v3.0 subsystem hygiene
            (hyperedge engrams, Conway DMN self-profiles, procedural playbooks, ecphory silent pool),
            invalidation hotspots, and Neo4j graph hygiene.
        </p>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Active Issues</div>
                <div class="kpi-value">${formatNumber(nonZeroIssues)} / ${formatNumber(issues.length)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Hyperedge Engrams</div>
                <div class="kpi-value">${formatNumber(v3.hyperedge_engrams_total || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">DMN Self-Profiles</div>
                <div class="kpi-value">${formatNumber(v3.dmn_self_profiles_total || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Procedural Skills</div>
                <div class="kpi-value">${formatNumber(v3.procedural_skills_total || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Silent (Ecphory) Pool</div>
                <div class="kpi-value">${formatNumber(v3.silent_ecphory_memories || 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Labile Sessions</div>
                <div class="kpi-value">${formatNumber(data.labile_sessions || 0)}</div>
            </div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Issue Breakdown</div>
            <div class="stack-list">
                ${issues.map(issue => `
                    <div class="stack-item">
                        <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                            <strong>${escapeHtml(ISSUE_LABELS[issue.label] || issue.label)}</strong>
                            <span class="badge ${(issue.count || 0) === 0 ? 'badge-ok' : 'badge-warning'}">${formatNumber(issue.count)}</span>
                        </div>
                        <div style="margin-top:6px;color:var(--text-muted);font-size:0.86rem;">${escapeHtml(issue.description || '')}</div>
                        ${renderSampleIds(issue)}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="detail-grid" style="margin-top:16px;">
            <div class="card">
                <div class="card-title">CML v3.0 Subsystem Hygiene</div>
                <div class="detail-field">
                    <span class="detail-field-label">Hyperedge Engrams</span>
                    <div class="detail-field-value">${formatNumber(v3.hyperedge_engrams_total || 0)} (${formatNumber(v3.low_confidence_hyperedges || 0)} low-conf)</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Avg Hyperedge Synaptic Weight</span>
                    <div class="detail-field-value">${v3.avg_hyperedge_synaptic_weight != null ? formatFloat(v3.avg_hyperedge_synaptic_weight, 3) : '—'}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">DMN Self-Profiles</span>
                    <div class="detail-field-value">${formatNumber(v3.dmn_self_profiles_total || 0)} (${formatNumber(v3.single_evidence_dmn_profiles || 0)} single-evidence)</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Procedural Skills</span>
                    <div class="detail-field-value">${formatNumber(v3.procedural_skills_total || 0)} (${formatNumber(v3.failing_procedural_skills || 0)} net-failing)</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Missing Embeddings (Active)</span>
                    <div class="detail-field-value">${formatNumber(v3.missing_embeddings || 0)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Silent Memories (Ecphory Eligible)</span>
                    <div class="detail-field-value">${formatNumber(v3.silent_ecphory_memories || 0)}</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Graph Hygiene</div>
                <div class="detail-field">
                    <span class="detail-field-label">Duplicate Entity Groups</span>
                    <div class="detail-field-value">${formatNumber(graph.duplicate_entity_groups || 0)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Stale / Unconnected Nodes</span>
                    <div class="detail-field-value">${formatNumber(graph.stale_nodes || 0)}</div>
                </div>
                <div class="detail-field">
                    <span class="detail-field-label">Generated</span>
                    <div class="detail-field-value">${formatDate(data.generated_at)}</div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-top:16px;">
            <div class="card-title">Fact Invalidation Hotspots</div>
            ${hotspots.length ? `
                <div class="table-responsive">
                    <table class="data-table">
                        <thead><tr><th>Fact Key</th><th>Invalidations</th></tr></thead>
                        <tbody>
                            ${hotspots.map(item => `
                                <tr>
                                    <td class="mono" style="font-size:0.8rem;">${escapeHtml(item.key)}</td>
                                    <td>${formatNumber(item.count)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<div class="empty-state" style="padding:12px;">No invalidation hotspots detected.</div>'}
        </div>
    `;
}
