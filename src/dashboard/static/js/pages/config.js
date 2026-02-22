/**
 * Configuration Page
 * Shows all settings grouped by section with inline editing for editable settings.
 * Changes are persisted to .env; restart required for most settings.
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
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#9888;</div><p>Failed to load config: ${escapeHtml(err.message)}</p></div>`;
    }
}

function buildConfig(data) {
    return `
        <p style="color:var(--text-secondary);margin-bottom:20px;">
            Effective application configuration. Secret values are masked. Editable settings are persisted to <code>.env</code>.
            Restart the server for most changes to take effect.
        </p>

        ${(data.sections || []).map(section => `
            <div class="card config-section" style="margin-bottom:16px;">
                <div class="card-title">${escapeHtml(section.name)}</div>
                <div class="config-items">
                    ${(section.items || []).map(item => `
                        <div class="config-row" data-key="${escapeHtml(item.key)}" data-value-type="${getValueType(item)}" data-options="${item.options ? escapeHtml(JSON.stringify(item.options)) : ''}">
                            <div class="config-key">
                                ${item.is_secret ? '<span class="config-lock" title="Secret - read only">&#128274;</span>' : ''}
                                <span class="config-key-name">${escapeHtml(item.key)}</span>
                                ${item.is_required ? '<span class="badge badge-error" style="margin-left:6px;font-size:0.65rem" title="Required">required</span>' : ''}
                                ${item.requires_restart ? '<span class="badge" style="margin-left:6px;font-size:0.65rem;background:var(--bg-tertiary);color:var(--text-secondary)" title="Restart required">restart</span>' : ''}
                                ${item.source !== 'default' ? `<span class="badge badge-${item.source === 'override' ? 'warning' : 'ok'}" style="margin-left:6px;font-size:0.65rem">${item.source}</span>` : ''}
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

function getValueType(item) {
    const v = item.value;
    if (typeof v === 'boolean') return 'boolean';
    if (typeof v === 'number') return 'number';
    if (item.default_value !== undefined && item.default_value !== null) {
        if (typeof item.default_value === 'boolean') return 'boolean';
        if (typeof item.default_value === 'number') return 'number';
    }
    return 'string';
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

function findItemByKey(data, key) {
    for (const section of data.sections || []) {
        const item = (section.items || []).find(i => i.key === key);
        if (item) return item;
    }
    return null;
}

function attachListeners(el, data) {
    el.querySelectorAll('.config-edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            const item = findItemByKey(data, key);
            const row = el.querySelector(`.config-row[data-key="${key}"]`);
            const valueSpan = row.querySelector('.config-value');
            const valueType = row.dataset.valueType || 'string';
            let options = item?.options;
            if (!options && row.dataset.options) {
                try {
                    options = JSON.parse(row.dataset.options);
                } catch (_) {
                    options = null;
                }
            }
            const currentText = valueSpan.textContent;
            let currentValue = currentText === '(not set)' ? (valueType === 'boolean' ? false : '') : currentText;

            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.alignItems = 'center';
            wrapper.style.gap = '8px';
            wrapper.style.flexWrap = 'wrap';

            let input;
            if (options && options.length > 0) {
                input = document.createElement('select');
                input.className = 'select-sm config-edit-input';
                input.style.minWidth = '160px';
                const strVal = currentValue === '(uses llm)' || currentValue === '' || currentValue == null ? '' : String(currentValue);
                options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt === '(uses llm)' ? '' : opt;
                    option.textContent = opt;
                    const optVal = opt === '(uses llm)' ? '' : opt;
                    if (strVal === optVal || (strVal === '' && opt === '(uses llm)')) {
                        option.selected = true;
                    }
                    input.appendChild(option);
                });
                if (!input.value && options[0]) input.selectedIndex = 0;
            } else if (valueType === 'boolean') {
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = currentValue === 'true' || currentValue === true;
                input.className = 'config-edit-input';
            } else {
                input = document.createElement('input');
                input.type = valueType === 'number' ? 'number' : 'text';
                input.className = 'input-sm config-edit-input';
                input.value = currentValue;
                if (valueType === 'number') {
                    input.style.width = '120px';
                    const keyLower = key.toLowerCase();
                    if (keyLower.includes('percent') || keyLower.includes('weight') || keyLower.includes('threshold')) {
                        input.step = '0.01';
                        input.min = '0';
                        input.max = keyLower.includes('percent') || keyLower.includes('weight') || keyLower.includes('threshold') ? '1' : '';
                    } else {
                        input.min = '0';
                        input.step = '1';
                    }
                } else {
                    input.style.width = '200px';
                }
            }
            wrapper.appendChild(input);

            const saveBtn = document.createElement('button');
            saveBtn.className = 'btn btn-primary btn-xs';
            saveBtn.textContent = 'Save';
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-ghost btn-xs';
            cancelBtn.textContent = 'Cancel';
            wrapper.appendChild(saveBtn);
            wrapper.appendChild(cancelBtn);

            valueSpan.replaceWith(wrapper);
            btn.style.display = 'none';
            input.focus();

            const cancel = () => {
                valueSpan.textContent = currentText;
                wrapper.replaceWith(valueSpan);
                btn.style.display = '';
            };

            const doSave = async () => {
                let newValue;
                if (options && options.length > 0) {
                    const raw = input.value;
                    if (raw === 'true') newValue = true;
                    else if (raw === 'false') newValue = false;
                    else newValue = raw;
                } else if (valueType === 'boolean') {
                    newValue = input.checked;
                } else if (valueType === 'number') {
                    const raw = input.value.trim();
                    if (raw === '') {
                        showToast('Value cannot be empty', 'error');
                        return;
                    }
                    const num = Number(raw);
                    if (isNaN(num)) {
                        showToast('Invalid number', 'error');
                        return;
                    }
                    newValue = num;
                } else {
                    newValue = input.value.trim();
                }

                try {
                    const resp = await updateConfig({ [key]: newValue });
                    valueSpan.textContent = formatValue(newValue);
                    wrapper.replaceWith(valueSpan);
                    btn.style.display = '';
                    showToast(resp.message || `Setting "${key}" saved to .env`);
                } catch (err) {
                    showToast(`Failed: ${err.message}`, 'error');
                }
            };

            cancelBtn.addEventListener('click', cancel);
            saveBtn.addEventListener('click', doSave);
            if (input.type !== 'checkbox') {
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') doSave();
                    if (e.key === 'Escape') cancel();
                });
            }
        });
    });
}
