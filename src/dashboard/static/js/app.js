/**
 * Main application entry point.
 * Handles routing, authentication, theme toggling, and page initialization.
 */

import { getApiKey, setApiKey, clearApiKey, testAdminAuth, getTenants } from './api.js';
import { renderOverview } from './pages/overview.js';
import { renderMemories } from './pages/memories.js';
import { renderDetail } from './pages/detail.js';
import { renderComponents } from './pages/components.js';
import { renderEvents } from './pages/events.js';
import { renderManagement } from './pages/management.js';

// ---- State ----
let currentPage = 'overview';
let selectedTenantId = '';

// ---- DOM Refs ----
const loginOverlay = document.getElementById('login-overlay');
const appLayout = document.getElementById('app');
const loginForm = document.getElementById('login-form');
const apiKeyInput = document.getElementById('api-key-input');
const loginError = document.getElementById('login-error');
const pageTitle = document.getElementById('page-title');
const tenantSelector = document.getElementById('tenant-selector');
const refreshBtn = document.getElementById('refresh-btn');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const logoutBtn = document.getElementById('logout-btn');

// ---- Pages Config ----
const pages = {
    overview: { title: 'Overview', render: renderOverview },
    memories: { title: 'Memory Explorer', render: renderMemories },
    detail: { title: 'Memory Detail', render: renderDetail },
    components: { title: 'Components', render: renderComponents },
    events: { title: 'Event Log', render: renderEvents },
    management: { title: 'Management', render: renderManagement },
};

// ---- Theme ----
function initTheme() {
    const saved = localStorage.getItem('cml_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    themeIcon.textContent = saved === 'dark' ? '\u263E' : '\u2600';
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('cml_theme', next);
    themeIcon.textContent = next === 'dark' ? '\u263E' : '\u2600';
    // Re-render current page to refresh chart colors
    navigateTo(currentPage);
}

// ---- Auth ----
async function checkAuth() {
    const key = getApiKey();
    if (!key) {
        showLogin();
        return;
    }
    const ok = await testAdminAuth();
    if (ok) {
        showApp();
    } else {
        showLogin();
    }
}

function showLogin() {
    loginOverlay.classList.remove('hidden');
    appLayout.classList.add('hidden');
    apiKeyInput.focus();
}

function showApp() {
    loginOverlay.classList.add('hidden');
    appLayout.classList.remove('hidden');
    loadTenants();
    handleHashChange();
}

async function handleLogin(e) {
    e.preventDefault();
    const key = apiKeyInput.value.trim();
    if (!key) return;
    setApiKey(key);
    loginError.classList.add('hidden');

    const ok = await testAdminAuth();
    if (ok) {
        showApp();
    } else {
        loginError.textContent = 'Invalid API key or insufficient permissions. Admin key required.';
        loginError.classList.remove('hidden');
        clearApiKey();
    }
}

function handleLogout() {
    clearApiKey();
    showLogin();
}

// ---- Tenant selector ----
async function loadTenants() {
    try {
        const data = await getTenants();
        tenantSelector.innerHTML = '<option value="">All Tenants</option>';
        for (const t of data.tenants) {
            const opt = document.createElement('option');
            opt.value = t.tenant_id;
            opt.textContent = `${t.tenant_id} (${t.memory_count} memories)`;
            tenantSelector.appendChild(opt);
        }
    } catch {
        // Silently fail; tenant selector stays as "All"
    }
}

// ---- Routing ----
export function navigateTo(page, params = {}) {
    if (page === 'detail' && params.memoryId) {
        window.location.hash = `#detail/${params.memoryId}`;
        return;
    }
    window.location.hash = `#${page}`;
}

function handleHashChange() {
    const hash = window.location.hash.slice(1) || 'overview';
    const parts = hash.split('/');
    const page = parts[0];

    // Update active nav
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.dataset.page === page);
    });

    // Hide all pages, show target
    document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));

    const pageKey = page in pages ? page : 'overview';
    currentPage = pageKey;

    const pageEl = document.getElementById(`page-${pageKey}`);
    if (pageEl) {
        pageEl.classList.add('active');
    }

    // Update title
    pageTitle.textContent = pages[pageKey]?.title || 'Dashboard';

    // Render page
    const params = { tenantId: selectedTenantId };
    if (pageKey === 'detail' && parts[1]) {
        params.memoryId = parts[1];
    }
    pages[pageKey]?.render(params);
}

// ---- Toast Notifications ----
export function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ---- Init ----
function init() {
    initTheme();

    // Event listeners
    loginForm.addEventListener('submit', handleLogin);
    logoutBtn.addEventListener('click', handleLogout);
    themeToggle.addEventListener('click', toggleTheme);
    refreshBtn.addEventListener('click', () => handleHashChange());
    tenantSelector.addEventListener('change', (e) => {
        selectedTenantId = e.target.value;
        handleHashChange();
    });

    // Navigation
    document.querySelectorAll('.nav-item').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(el.dataset.page);
        });
    });

    window.addEventListener('hashchange', handleHashChange);

    // Check auth
    checkAuth();
}

// Export for pages to use
export function getSelectedTenantId() {
    return selectedTenantId;
}

init();
