/**
 * Components Status Page
 * Shows health and metrics for each system component.
 */

import { getComponents } from '../api.js';
import { formatNumber, formatLatency, formatMB, escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-components');

export async function renderComponents() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Checking components...</div>`;

    try {
        const data = await getComponents();
        el.innerHTML = buildComponents(data);
    } catch (err) {
        el.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">&#9888;</div>
            <p>Failed to load component status: ${escapeHtml(err.message)}</p>
        </div>`;
    }
}

function buildComponents(data) {
    if (!data?.components?.length) {
        return `<div class="empty-state"><div class="empty-state-icon">&#9881;</div><p>No component data available</p></div>`;
    }

    const cards = data.components.map(c => buildComponentCard(c)).join('');

    return `
        <p style="color:var(--text-secondary);margin-bottom:20px;">
            Real-time health status of all system components. Each card shows connection status, latency, and key metrics.
        </p>
        <div class="component-grid">${cards}</div>

        <!-- Architecture legend -->
        <div class="card" style="margin-top:24px;">
            <div class="card-title">Architecture</div>
            <div class="detail-grid" style="gap:12px;">
                <div>
                    <div class="component-detail">
                        <span class="component-detail-label">Sensory Buffer</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">In-memory token buffer for incoming text. Short-lived (TTL ~5min).</span>
                    </div>
                    <div class="component-detail">
                        <span class="component-detail-label">Working Memory</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">Semantic chunker producing salient chunks from token stream.</span>
                    </div>
                    <div class="component-detail">
                        <span class="component-detail-label">Hippocampal Store</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">Episodic memory in PostgreSQL/pgvector. Write-gated encoding.</span>
                    </div>
                </div>
                <div>
                    <div class="component-detail">
                        <span class="component-detail-label">Neocortical Store</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">Semantic facts in PostgreSQL + knowledge graph in Neo4j.</span>
                    </div>
                    <div class="component-detail">
                        <span class="component-detail-label">Consolidation</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">Episodic-to-semantic migration: sample, cluster, summarize, migrate.</span>
                    </div>
                    <div class="component-detail">
                        <span class="component-detail-label">Forgetting</span>
                        <span class="component-detail-value" style="font-family:var(--font-sans);font-weight:400;">Active decay, silencing, compression, and duplicate removal.</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function buildComponentCard(comp) {
    const statusClass = comp.status === 'ok' ? 'badge-ok' : comp.status === 'error' ? 'badge-error' : 'badge-unknown';
    const borderColor = comp.status === 'ok' ? 'var(--success)' : comp.status === 'error' ? 'var(--danger)' : 'var(--warning)';

    let detailsHtml = '';

    if (comp.name === 'PostgreSQL') {
        detailsHtml = `
            ${detailRow('Memory Records', formatNumber(comp.details?.memory_records))}
            ${detailRow('Semantic Facts', formatNumber(comp.details?.semantic_facts))}
            ${detailRow('Events', formatNumber(comp.details?.events))}
            ${detailRow('Embedding dimensions', comp.details?.embedding_dimensions ?? '—')}
            ${detailRow('Latency', formatLatency(comp.latency_ms))}
        `;
    } else if (comp.name === 'Neo4j') {
        detailsHtml = `
            ${detailRow('Nodes', formatNumber(comp.details?.nodes))}
            ${detailRow('Relationships', formatNumber(comp.details?.relationships))}
            ${detailRow('Latency', formatLatency(comp.latency_ms))}
        `;
    } else if (comp.name === 'Redis') {
        detailsHtml = `
            ${detailRow('Keys', formatNumber(comp.details?.keys))}
            ${detailRow('Memory Used', formatMB(comp.details?.used_memory_mb))}
            ${detailRow('Latency', formatLatency(comp.latency_ms))}
        `;
    } else {
        // Generic details
        if (comp.details && Object.keys(comp.details).length) {
            detailsHtml = Object.entries(comp.details).map(([k, v]) =>
                detailRow(k.replace(/_/g, ' '), typeof v === 'number' ? formatNumber(v) : String(v))
            ).join('');
        }
        if (comp.latency_ms !== null && comp.latency_ms !== undefined) {
            detailsHtml += detailRow('Latency', formatLatency(comp.latency_ms));
        }
    }

    if (comp.error) {
        detailsHtml += `
            <div class="component-detail" style="border-bottom:none;">
                <span class="component-detail-label" style="color:var(--danger)">Error</span>
                <span class="component-detail-value" style="color:var(--danger);font-size:0.82rem;">${escapeHtml(comp.error)}</span>
            </div>
        `;
    }

    return `
        <div class="component-card" style="border-left: 3px solid ${borderColor};">
            <div class="component-header">
                <span class="component-name">${escapeHtml(comp.name)}</span>
                <span class="badge ${statusClass}">${comp.status}</span>
            </div>
            ${detailsHtml}
        </div>
    `;
}

function detailRow(label, value) {
    return `
        <div class="component-detail">
            <span class="component-detail-label">${label}</span>
            <span class="component-detail-value">${value}</span>
        </div>
    `;
}
