/**
 * API client for the Cognitive Memory Layer dashboard.
 * Wraps fetch with auth headers and error handling.
 */

const API_BASE = '/api/v1/dashboard';

/** Get stored API key */
export function getApiKey() {
    return localStorage.getItem('cml_api_key') || '';
}

/** Store API key */
export function setApiKey(key) {
    localStorage.setItem('cml_api_key', key);
}

/** Clear stored API key */
export function clearApiKey() {
    localStorage.removeItem('cml_api_key');
}

/** Build default headers */
function headers() {
    const h = {
        'Content-Type': 'application/json',
    };
    const key = getApiKey();
    if (key) {
        h['X-API-Key'] = key;
    }
    return h;
}

/** Generic fetch wrapper */
async function request(method, path, { body = null, params = null } = {}) {
    let url = `${API_BASE}${path}`;
    if (params) {
        const searchParams = new URLSearchParams();
        for (const [k, v] of Object.entries(params)) {
            if (v !== null && v !== undefined && v !== '') {
                searchParams.append(k, v);
            }
        }
        const qs = searchParams.toString();
        if (qs) url += `?${qs}`;
    }

    const opts = {
        method,
        headers: headers(),
    };
    if (body !== null) {
        opts.body = JSON.stringify(body);
    }

    const resp = await fetch(url, opts);

    if (resp.status === 401) {
        clearApiKey();
        window.location.reload();
        throw new Error('Unauthorized');
    }

    if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${detail}`);
    }

    return resp.json();
}

/** Test if the API key is valid */
export async function testAuth() {
    try {
        const resp = await fetch('/api/v1/health', { headers: headers() });
        return resp.ok;
    } catch {
        return false;
    }
}

/** Test admin auth by calling a dashboard endpoint */
export async function testAdminAuth() {
    try {
        const resp = await fetch(`${API_BASE}/tenants`, { headers: headers() });
        return resp.ok;
    } catch {
        return false;
    }
}

// ---- Dashboard API Methods ----

export function getOverview(tenantId) {
    return request('GET', '/overview', { params: { tenant_id: tenantId } });
}

export function getMemories({ page = 1, perPage = 25, type, status, search, tenantId, sortBy, order } = {}) {
    return request('GET', '/memories', {
        params: {
            page,
            per_page: perPage,
            type,
            status,
            search,
            tenant_id: tenantId,
            sort_by: sortBy,
            order,
        },
    });
}

export function getMemoryDetail(memoryId) {
    return request('GET', `/memories/${memoryId}`);
}

export function getEvents({ page = 1, perPage = 25, eventType, operation, tenantId } = {}) {
    return request('GET', '/events', {
        params: {
            page,
            per_page: perPage,
            event_type: eventType,
            operation,
            tenant_id: tenantId,
        },
    });
}

export function getTimeline(days = 30, tenantId) {
    return request('GET', '/timeline', { params: { days, tenant_id: tenantId } });
}

export function getComponents() {
    return request('GET', '/components');
}

export function getTenants() {
    return request('GET', '/tenants');
}

export function triggerConsolidate(tenantId, userId) {
    return request('POST', '/consolidate', {
        body: { tenant_id: tenantId, user_id: userId || null },
    });
}

export function triggerForget(tenantId, userId, dryRun = true, maxMemories = 5000) {
    return request('POST', '/forget', {
        body: {
            tenant_id: tenantId,
            user_id: userId || null,
            dry_run: dryRun,
            max_memories: maxMemories,
        },
    });
}

export function resetDatabase() {
    return request('POST', '/database/reset');
}
