/**
 * Configuration Page
 * Shows all settings grouped by section with inline editing for safe settings.
 */

import { getConfig, updateConfig } from '../api.js';
import { showToast } from '../app.js';
import { escapeHtml } from '../utils/formatters.js';

const container = () => document.getElementById('page-config');

export async function renderConfig() {
    const el = container();
    el.innerHTML = `<div class="loading-overlay"><div class="spinner"></div> Loading configuration...</div>`;

    try {
        const data = await getConfig();
        el.innerHTML = buildConfig(data);
        attachListeners(el, data);
    } catch (err) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load config: ${err.message}</p></div>`;
    }
}

function buildConfig(data) {
    return `
        <p style="color:var(--text-secondary);margin-bottom:20px;">
            Effective application configuration. Secret values are masked. Editable settings can be changed at runtime
            (stored in Redis, reset on restart if not persisted in .env).
        </p>

        ${(data.sections || []).map(section => `
            <div class="card config-section" style="margin-bottom:16px;">
                <div class="card-title">${escapeHtml(section.name)}</div>
                <div class="config-items">
                    ${(section.items || []).map(item => `
                        <div class="config-row" data-key="${escapeHtml(item.key)}">
                            <div class="config-key">
                                ${item.is_secret ? '<span class="config-lock" title="Secret - read only">&#128274;</span>' : ''}
                                <span class="config-key-name">${escapeHtml(item.key)}</span>
                                ${item.source !== 'default' ? `<span class="badge badge-${item.source === 'override' ? 'warning' : 'ok'}" style="margin-left:8px;font-size:0.7rem">${item.source}</span>` : ''}
                            </div>
                            <div class="config-value-row">
                                <span class="config-value" id="val-${cssKey(item.key)}">${escapeHtml(formatValue(item.value))}</span>
                                ${item.is_editable && !item.is_secret ? `
                                    <button class="btn btn-ghost btn-xs config-edit-btn" data-key="${escapeHtml(item.key)}" title="Edit">&#9998;</button>
                                ` : ''}
                            </div>
                            ${item.description ? `<div class="config-desc">${escapeHtml(item.description)}</div>` : ''}
                            ${item.default_value !== undefined && item.default_value !== null && !item.is_secret ? `<div class="config-default">Default: ${escapeHtml(formatValue(item.default_value))}</div>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('')}
    `;
}

function formatValue(v) {
    if (v === null || v === undefined) return '(not set)';
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) return JSON.stringify(v);
    return String(v);
}

function cssKey(key) {
    return key.replace(/[^a-zA-Z0-9]/g, '-');
}

function attachListeners(el, data) {
    el.querySelectorAll('.config-edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            const row = el.querySelector(`.config-row[data-key="${key}"]`);
            const valueSpan = row.querySelector('.config-value');
            const currentValue = valueSpan.textContent;

            // Replace value with input
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'input-sm config-edit-input';
            input.value = currentValue === '(not set)' ? '' : currentValue;
            input.style.width = '200px';

            const saveBtn = document.createElement('button');
            saveBtn.className = 'btn btn-primary btn-xs';
            saveBtn.textContent = 'Save';
            saveBtn.style.marginLeft = '8px';

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-ghost btn-xs';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.style.marginLeft = '4px';

            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.alignItems = 'center';
            wrapper.appendChild(input);
            wrapper.appendChild(saveBtn);
            wrapper.appendChild(cancelBtn);

            valueSpan.replaceWith(wrapper);
            btn.style.display = 'none';
            input.focus();

            cancelBtn.addEventListener('click', () => {
                wrapper.replaceWith(valueSpan);
                btn.style.display = '';
            });

            const doSave = async () => {
                let newValue = input.value.trim();
                // Try to parse as number or boolean
                if (newValue === 'true') newValue = true;
                else if (newValue === 'false') newValue = false;
                else if (!isNaN(newValue) && newValue !== '') newValue = Number(newValue);

                try {
                    await updateConfig({ [key]: newValue });
                    valueSpan.textContent = formatValue(newValue);
                    wrapper.replaceWith(valueSpan);
                    btn.style.display = '';
                    showToast(`Setting "${key}" updated`);
                } catch (err) {
                    showToast(`Failed to update: ${err.message}`, 'error');
                }
            };

            saveBtn.addEventListener('click', doSave);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') doSave();
                if (e.key === 'Escape') { wrapper.replaceWith(valueSpan); btn.style.display = ''; }
            });
        });
    });
}
