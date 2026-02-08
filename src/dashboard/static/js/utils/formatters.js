/**
 * Formatting utilities for the dashboard.
 */

/** Format a datetime string to locale display */
export function formatDate(dateStr) {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

/** Format a date to short form */
export function formatDateShort(dateStr) {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

/** Format a number with commas */
export function formatNumber(n) {
    if (n === null || n === undefined) return '—';
    return Number(n).toLocaleString();
}

/** Format a float to fixed decimal places */
export function formatFloat(n, decimals = 2) {
    if (n === null || n === undefined) return '—';
    return Number(n).toFixed(decimals);
}

/** Format bytes to human-readable MB */
export function formatMB(mb) {
    if (mb === null || mb === undefined) return '—';
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
    return `${Number(mb).toFixed(2)} MB`;
}

/** Format milliseconds to human readable */
export function formatLatency(ms) {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1) return `${(ms * 1000).toFixed(0)} us`;
    if (ms < 1000) return `${ms.toFixed(1)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
}

/** Truncate text to maxLen characters */
export function truncate(text, maxLen = 80) {
    if (!text) return '';
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + '...';
}

/** Truncate a UUID for display */
export function shortUuid(uuid) {
    if (!uuid) return '—';
    return uuid.slice(0, 8) + '...';
}

/** Get a CSS class for a status badge */
export function statusBadgeClass(status) {
    const s = (status || '').toLowerCase();
    return `badge badge-status-${s}`;
}

/** Get a CSS class for a component health status */
export function healthBadgeClass(status) {
    const s = (status || 'unknown').toLowerCase();
    if (s === 'ok') return 'badge badge-ok';
    if (s === 'error') return 'badge badge-error';
    return 'badge badge-unknown';
}

/** Pretty-print JSON with indentation */
export function prettyJson(obj) {
    if (obj === null || obj === undefined) return 'null';
    try {
        return JSON.stringify(obj, null, 2);
    } catch {
        return String(obj);
    }
}

/** Escape HTML entities */
export function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/** Format a percentage (0-1 range) */
export function formatPercent(n) {
    if (n === null || n === undefined) return '—';
    return `${(Number(n) * 100).toFixed(1)}%`;
}
