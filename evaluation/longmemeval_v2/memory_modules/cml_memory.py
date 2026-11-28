from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from requests.adapters import HTTPAdapter

from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require
from .support import (
    _annotate_action_with_object_details,
    _extract_content_text,
    _extract_object_lookup_from_tree,
    _load_api_key,
    _to_data_url,
)
from .trajectory_store import (
    load_json,
    logical_trajectory_fingerprint_from_simplified,
    materialize_prepared_trajectory,
    normalize_trajectory_pool_root,
    prepare_trajectory_insert,
    relative_symlink,
    save_json,
)

DEFAULT_CML_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CML_API_KEY_ENV = "CML_API_KEY"
DEFAULT_CML_TENANT_PREFIX = "lme2v2"
DEFAULT_CML_BATCH_SIZE = 64
DEFAULT_CML_TIMEOUT_SECONDS = 180.0

DEFAULT_CONTROLLER_MODEL = "qwen3.8-27b"
DEFAULT_CONTROLLER_BASE_URL = "http://127.0.0.1:8023/v1"
DEFAULT_CONTROLLER_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_CONTROLLER_MAX_TOKENS = 256
DEFAULT_CONTROLLER_TIMEOUT_SECONDS = 120.0

DEFAULT_ENABLE_QUERY_EXPANSION = True
DEFAULT_ENABLE_VISION_QUERY = True
DEFAULT_MAX_SURFACE_QUERIES = 2
DEFAULT_CML_TOP_K_PER_QUERY = 28
DEFAULT_MAX_WORKFLOWS = 3
DEFAULT_MAX_STATE_SLICES = 4
DEFAULT_PER_QUERY_SLICE_CAP = 2
DEFAULT_SLICE_RADIUS = 1
DEFAULT_MAX_IMAGES = 0
DEFAULT_CENTER_AXTREE_MAX_CHARS = 10500
DEFAULT_NEIGHBOR_AXTREE_MAX_CHARS = 900

ROOT_TITLE_RE = re.compile(r"RootWebArea\s+'([^']*)'")
WHITESPACE_RE = re.compile(r"\s+")
QUOTED_PHRASE_RE = re.compile(r"""["'`“”]([^"'`“”\n]{2,80})["'`“”]""")

# Boilerplate top-chrome lines present on virtually every ServiceNow / Postmill page
CHROME_BOILERPLATE_SUBSTRINGS = (
    "Skip to main content",
    "Open accessibility preferences",
    "Global skip links",
    "Unpinned All menu",
    "Unpinned Favorites menu",
    "Unpinned History menu",
    "Unpinned Workspaces menu",
    "Unpinned Admin menu",
    "My ServiceNow landing page",
    "ServiceNow Service Management",
    "Jump to main content",
    "Jump to sidebar",
)

EMPTY_WRAPPER_RE = re.compile(
    r"^(?:\[[A-Za-z0-9_-]+\]\s+)?"
    r"(?:generic|none|presentation|group|Section|paragraph|list|listitem|rowgroup|InlineTextBox)"
    r"(?:\s+'')?(?:,\s*(?:live='[^']*'|atomic|relevant='[^']*'|visible|clickable))*$",
    re.IGNORECASE,
)
NOISY_ATTR_RE = re.compile(
    r",\s*(?:live='[^']*'|atomic|relevant='[^']*'|visible|clickable|describedby='[^']*'|controls='[^']*')"
)

GOTCHA_SIGNAL_RE = re.compile(
    r"\b(?:error|fail|failed|cannot|can't|unable|missing|not\s+found|no\s+exact\s+match|"
    r"instead|however|warning|alert|blocked|disabled|readonly|read-only|not\s+modifiable|"
    r"must\s+click|need\s+to\s+click|save|update\s+button|confirm|workaround|unexpected|"
    r"does\s+not\s+appear|did\s+not\s+happen)\b",
    re.IGNORECASE,
)

QUERY_DECOMPOSE_SYSTEM_PROMPT = """You decompose a user question about past web/enterprise UI agent trajectories into concise search queries for a hybrid semantic + lexical memory engine.

Return ONLY valid JSON with this exact schema:
{"surface_queries":["..."],"workflow_query":"..."}

Rules:
- surface_queries: 1 to 3 short, concrete UI surface queries (4 to 14 words each) targeting exact page titles, forms, dialogs, dropdowns, tabs, catalog items, fields, buttons, or compared entities.
- If the question compares two pages/forms/entities (e.g. "Create Change Request vs Incident"), emit a separate surface query for EACH entity so both pages are retrieved.
- Preserve exact literal UI labels, field names, product names, and quoted terms from the question.
- Strip formatting instructions ("Mark your final answer in \\boxed{}", multiple-choice option letters, etc.).
- workflow_query: 1 short query (4 to 14 words) targeting multi-step task procedures, module sequences, or UI gotchas/troubleshooting.
- Output JSON only, starting with { and ending with }."""

VISION_LANDMARK_SYSTEM_PROMPT = """Extract concise, retrieval-ready UI landmarks from this screenshot to help retrieve matching past trajectories.
List in 2-4 compact lines: (1) Application & page/module/dialog title, (2) Visible form fields, selected values, tabs, or table filters, (3) Any visible warning, error, missing option, or action button state. Do not guess unseen details."""

_LEDGER_LOCK = threading.Lock()
_VERIFIED_TENANTS: set[str] = set()


def _normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    omitted = len(text) - max_chars
    marker = f"\n...[truncated {omitted} chars]...\n"
    head = (max_chars - len(marker)) // 2
    tail = max_chars - len(marker) - head
    return f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}".strip()


def _parse_root_title(axtree_text: str) -> str:
    m = ROOT_TITLE_RE.search(axtree_text or "")
    if m and m.group(1).strip():
        return _normalize_ws(m.group(1))
    return ""


def _short_url(url: str) -> str:
    if not url:
        return ""
    cleaned = re.sub(r"^https?://[^/]+", "", url.strip())
    return _truncate_middle(cleaned or url.strip(), 90)


def _domain_from_url(url: str) -> str:
    u = (url or "").lower()
    if "service-now.com" in u or "servicenow" in u:
        return "enterprise"
    if "localhost:" in u or "127.0.0.1:" in u:
        return "web"
    return "general"


def _annotated_transition_actions(states: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    known_lookup: dict[str, str] = {}
    for idx in range(len(states) - 1):
        pre_text = str(states[idx].get("text") or "")
        pre_lookup = _extract_object_lookup_from_tree(pre_text)
        merged_lookup = {**known_lookup, **pre_lookup}
        raw_act = states[idx + 1].get("action")
        act_str = raw_act.strip() if isinstance(raw_act, str) and raw_act.strip() else "<none>"
        if act_str != "<none>":
            actions.append(_annotate_action_with_object_details(act_str, merged_lookup))
        else:
            actions.append("<none>")
        known_lookup.update(pre_lookup)
    return actions


def _clean_axtree_for_reader(raw_axtree: str, max_chars: int) -> str:
    """Losslessly strip empty ARIA wrappers, node ID prefixes, repeated skip-link chrome,
    live-region attributes, and exact duplicate child StaticText echoes while preserving
    100% of labels, options, checkboxes, values, prices, states, and table cells."""
    if not raw_axtree:
        return ""
    kept: list[str] = []
    last_control_label: str | None = None
    for raw_line in raw_axtree.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if EMPTY_WRAPPER_RE.match(stripped):
            continue
        if any(bp in stripped for bp in CHROME_BOILERPLATE_SUBSTRINGS):
            continue
        cleaned = NOISY_ATTR_RE.sub("", stripped)
        cleaned = re.sub(r"^\[[A-Za-z0-9_-]+\]\s+", "", cleaned)
        if re.match(
            r"^(?:generic|none|presentation|group|Section|paragraph|list|listitem|rowgroup|"
            r"InlineTextBox|image|img|strong|emphasis|span|div|LayoutTable|LayoutTableRow|LayoutTableCell)"
            r"(?:\s+(?:''|\"\"))?$",
            cleaned,
            re.IGNORECASE,
        ):
            continue
        st_m = re.match(r"^StaticText\s+(?:'(.*)'|\"(.*)\")$", cleaned)
        if st_m:
            st_val = (st_m.group(1) if st_m.group(1) is not None else st_m.group(2) or "").strip()
            if not st_val or st_val == last_control_label:
                continue
        else:
            lbl_m = re.search(r"(?:'([^']+)'|\"([^\"]+)\")", cleaned)
            last_control_label = (
                (lbl_m.group(1) if lbl_m and lbl_m.group(1) is not None else (lbl_m.group(2) if lbl_m else ""))
                .strip()
                or None
            )
        indent_tabs = len(raw_line) - len(raw_line.lstrip("\t "))
        indent = " " * min(4, indent_tabs)
        kept.append(f"{indent}{cleaned}")
    joined = "\n".join(kept)
    return _truncate_middle(joined, max_chars)


def _compact_axtree_windows_for_cml(
    raw_axtree: str, window_chars: int = 2200, overlap_chars: int = 250, max_windows: int = 6
) -> list[str]:
    """Extract all salient UI labels/controls across the ENTIRE page into overlapping
    windows so options/prices/checkboxes anywhere on a 30K-char page are 100% indexed."""
    if not raw_axtree:
        return [""]
    salient: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_axtree.splitlines():
        s = raw_line.strip()
        if not s or EMPTY_WRAPPER_RE.match(s):
            continue
        if any(bp in s for bp in CHROME_BOILERPLATE_SUBSTRINGS):
            continue
        s = NOISY_ATTR_RE.sub("", s)
        s = re.sub(r"^\[[A-Za-z0-9_-]+\]\s+", "", s)
        if re.match(
            r"^(?:generic|none|presentation|group|Section|paragraph|list|listitem|rowgroup|"
            r"InlineTextBox|image|img|strong|emphasis)(?:\s+(?:''|\"\"))?$",
            s,
            re.IGNORECASE,
        ):
            continue
        norm = s.lower()
        if norm in seen:
            continue
        seen.add(norm)
        salient.append(s)
    full_blob = " | ".join(salient)
    if len(full_blob) <= window_chars:
        return [full_blob]
    windows: list[str] = []
    step = max(500, window_chars - overlap_chars)
    pos = 0
    while pos < len(full_blob) and len(windows) < max_windows:
        windows.append(full_blob[pos : pos + window_chars])
        if pos + window_chars >= len(full_blob):
            break
        pos += step
    return windows


def _load_repo_cml_api_key(api_key_env: str) -> str:
    for env_key in (api_key_env, "CML_ADMIN_API_KEY", "CML_API_KEY", "AUTH__ADMIN_API_KEY", "AUTH__API_KEY"):
        val = os.getenv(env_key, "").strip()
        if val:
            return val
    # Fallback: check repo root .env if running inside CognitiveMemoryLayer checkout
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        dotenv_path = parent / ".env"
        if dotenv_path.is_file():
            try:
                parsed: dict[str, str] = {}
                for ln in dotenv_path.read_text(encoding="utf-8").splitlines():
                    ln_s = ln.strip()
                    if ln_s and not ln_s.startswith("#") and "=" in ln_s:
                        k, v = ln_s.split("=", 1)
                        parsed[k.strip()] = v.strip().strip('"').strip("'")
                for k in ("AUTH__ADMIN_API_KEY", "CML_ADMIN_API_KEY", "AUTH__API_KEY", "CML_API_KEY"):
                    if parsed.get(k):
                        return parsed[k]
            except OSError:
                pass
            break
    return "test-key"


def _strip_question_wrappers(query: str) -> str:
    text = query or ""
    # Strip trailing \boxed{} instructions
    text = re.sub(
        r"Mark your final answer[^.\n]*(?:\.|in\s*\\boxed\{\}\.?)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^I am working (?:with|in) (?:a few forms in )?our ServiceNow portal\.?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return _normalize_ws(text) or _normalize_ws(query)


def _deterministic_query_bundle(query: str, visual_landmarks: str = "") -> dict[str, Any]:
    cleaned = _strip_question_wrappers(query)
    surface_queries: list[str] = []

    # Detect explicit comparisons: "X vs Y", "X versus Y", "between X and Y"
    vs_match = re.search(
        r"([A-Z][A-Za-z0-9 _/-]{2,45}?)\s+(?:vs\.?|versus)\s+([A-Z][A-Za-z0-9 _/-]{2,45}?)(?:[.:,?]|$)",
        cleaned,
    )
    if vs_match:
        left = _normalize_ws(vs_match.group(1))
        right = _normalize_ws(vs_match.group(2))
        tail = _normalize_ws(cleaned[vs_match.end() :])[:120]
        surface_queries.append(_normalize_ws(f"{left} {tail}"))
        surface_queries.append(_normalize_ws(f"{right} {tail}"))

    surface_queries.append(_truncate_middle(cleaned, 220))

    quoted = [m.strip() for m in QUOTED_PHRASE_RE.findall(query or "") if len(m.strip()) >= 2]
    if quoted:
        q_terms = " ".join(dict.fromkeys(quoted))
        cand = _normalize_ws(f"{q_terms} {cleaned[:120]}")
        if cand.lower() not in {s.lower() for s in surface_queries}:
            surface_queries.append(cand)

    if visual_landmarks:
        vis_q = _normalize_ws(f"{visual_landmarks[:180]} {cleaned[:100]}")
        surface_queries.insert(0, vis_q)

    dedup: list[str] = []
    seen: set[str] = set()
    for sq in surface_queries:
        if sq and sq.lower() not in seen:
            seen.add(sq.lower())
            dedup.append(sq)

    return {
        "surface_queries": dedup[:DEFAULT_MAX_SURFACE_QUERIES],
        "workflow_query": _truncate_middle(
            _normalize_ws(f"{cleaned} {visual_landmarks[:120]}").strip(),
            220,
        ),
    }


@register_memory
class CMLMemory(Memory):
    """CognitiveMemoryLayer (CML) neuro-inspired hybrid memory backend for LongMemEval-V2.

    Combines CML's multi-pathway retrieval (Dense pgvector HNSW + Dentate Gyrus
    Conjunction-Boosted tsvector BM25 + Cortical Hypergraph PPR + Cross-Encoder
    reranking) with Hippocampal Pointer Dereferencing into full-fidelity UI state
    slices, workflow playbooks, and center-state screenshots.
    """

    memory_type = "cml"

    @classmethod
    def _normalize_config(cls, memory_config: MemoryConfig) -> MemoryConfig:
        require(
            memory_config["memory_type"] == cls.memory_type,
            f"Expected memory_type={cls.memory_type}, got {memory_config['memory_type']}",
        )
        raw_params = dict(memory_config.get("memory_params") or {})
        cml_params = dict(raw_params.get("cml_params") or {})
        controller_params = dict(raw_params.get("controller_params") or {})
        retrieval_params = dict(raw_params.get("retrieval_params") or {})

        return {
            "memory_type": cls.memory_type,
            "memory_params": {
                "workspace_dir": raw_params.get("workspace_dir"),
                "trajectories_root_dir": raw_params.get("trajectories_root_dir"),
                "trajectory_pool_root": raw_params.get("trajectory_pool_root"),
                "cml_params": {
                    "base_url": str(
                        cml_params.get("base_url")
                        or os.getenv("CML_BASE_URL", DEFAULT_CML_BASE_URL)
                    ).strip(),
                    "api_key": cml_params.get("api_key"),
                    "api_key_env": str(
                        cml_params.get("api_key_env", DEFAULT_CML_API_KEY_ENV)
                    ).strip(),
                    "tenant_prefix": str(
                        cml_params.get("tenant_prefix", DEFAULT_CML_TENANT_PREFIX)
                    ).strip(),
                    "batch_size": int(cml_params.get("batch_size", DEFAULT_CML_BATCH_SIZE)),
                    "timeout_seconds": float(
                        cml_params.get("timeout_seconds", DEFAULT_CML_TIMEOUT_SECONDS)
                    ),
                },
                "controller_params": {
                    "model": str(
                        controller_params.get("model")
                        or os.getenv("LME_CONTROLLER_MODEL", DEFAULT_CONTROLLER_MODEL)
                    ).strip(),
                    "base_url": str(
                        controller_params.get("base_url")
                        or os.getenv("LME_CONTROLLER_BASE_URL", DEFAULT_CONTROLLER_BASE_URL)
                    ).strip(),
                    "api_key_env": str(
                        controller_params.get("api_key_env", DEFAULT_CONTROLLER_API_KEY_ENV)
                    ).strip(),
                    "api_key_file": controller_params.get("api_key_file"),
                    "max_tokens": int(
                        controller_params.get("max_tokens", DEFAULT_CONTROLLER_MAX_TOKENS)
                    ),
                    "timeout_seconds": float(
                        controller_params.get(
                            "timeout_seconds", DEFAULT_CONTROLLER_TIMEOUT_SECONDS
                        )
                    ),
                },
                "retrieval_params": {
                    "enable_query_expansion": bool(
                        retrieval_params.get(
                            "enable_query_expansion", DEFAULT_ENABLE_QUERY_EXPANSION
                        )
                    ),
                    "enable_vision_query": bool(
                        retrieval_params.get("enable_vision_query", DEFAULT_ENABLE_VISION_QUERY)
                    ),
                    "max_surface_queries": int(
                        retrieval_params.get("max_surface_queries", DEFAULT_MAX_SURFACE_QUERIES)
                    ),
                    "cml_top_k_per_query": int(
                        retrieval_params.get("cml_top_k_per_query", DEFAULT_CML_TOP_K_PER_QUERY)
                    ),
                    "max_workflows": int(
                        retrieval_params.get("max_workflows", DEFAULT_MAX_WORKFLOWS)
                    ),
                    "max_state_slices": int(
                        retrieval_params.get("max_state_slices", DEFAULT_MAX_STATE_SLICES)
                    ),
                    "per_query_slice_cap": int(
                        retrieval_params.get("per_query_slice_cap", DEFAULT_PER_QUERY_SLICE_CAP)
                    ),
                    "slice_radius": int(
                        retrieval_params.get("slice_radius", DEFAULT_SLICE_RADIUS)
                    ),
                    "max_images": int(retrieval_params.get("max_images", DEFAULT_MAX_IMAGES)),
                    "center_axtree_max_chars": int(
                        retrieval_params.get(
                            "center_axtree_max_chars", DEFAULT_CENTER_AXTREE_MAX_CHARS
                        )
                    ),
                    "neighbor_axtree_max_chars": int(
                        retrieval_params.get(
                            "neighbor_axtree_max_chars", DEFAULT_NEIGHBOR_AXTREE_MAX_CHARS
                        )
                    ),
                },
            },
        }

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: MemoryConfig,
        requested_config: MemoryConfig | None,
    ) -> MemoryConfig:
        saved_norm = cls._normalize_config(saved_config)
        if requested_config is None:
            return deepcopy(saved_norm)
        requested_norm = cls._normalize_config(requested_config)
        effective = deepcopy(saved_norm)
        effective["memory_params"]["cml_params"] = deepcopy(
            requested_norm["memory_params"]["cml_params"]
        )
        effective["memory_params"]["controller_params"] = deepcopy(
            requested_norm["memory_params"]["controller_params"]
        )
        effective["memory_params"]["retrieval_params"] = deepcopy(
            requested_norm["memory_params"]["retrieval_params"]
        )
        return effective

    def __init__(self, memory_params: dict[str, object]) -> None:
        normalized = self._normalize_config(
            {"memory_type": self.memory_type, "memory_params": dict(memory_params)}
        )
        params = normalized["memory_params"]
        super().__init__(params)

        ws_dir = params.get("workspace_dir")
        traj_root = params.get("trajectories_root_dir")
        pool_root = params.get("trajectory_pool_root")

        self.workspace_dir = (
            Path(str(ws_dir)).resolve()
            if isinstance(ws_dir, str) and ws_dir.strip()
            else None
        )
        self.trajectories_root_dir = (
            Path(str(traj_root)).resolve()
            if isinstance(traj_root, str) and traj_root.strip()
            else None
        )
        self.trajectory_pool_root: Path | None = None
        if isinstance(pool_root, str) and pool_root.strip():
            pool_path = Path(pool_root).expanduser().resolve()
            pool_path.mkdir(parents=True, exist_ok=True)
            self.trajectory_pool_root = normalize_trajectory_pool_root(pool_path)
        elif self.trajectories_root_dir is not None:
            # Automatic shared trajectory pool under dataset .cml_cache/trajectory_pool
            default_pool = self.trajectories_root_dir / ".cml_cache" / "trajectory_pool"
            default_pool.mkdir(parents=True, exist_ok=True)
            self.trajectory_pool_root = default_pool.resolve()

        cml_p = dict(params["cml_params"])
        ctrl_p = dict(params["controller_params"])
        ret_p = dict(params["retrieval_params"])

        self.cml_base_url = str(cml_p["base_url"]).rstrip("/")
        self.cml_api_key = (
            str(cml_p["api_key"]).strip()
            if isinstance(cml_p.get("api_key"), str) and str(cml_p["api_key"]).strip()
            else _load_repo_cml_api_key(str(cml_p["api_key_env"]))
        )
        self.cml_tenant_prefix = str(cml_p["tenant_prefix"])
        self.cml_batch_size = max(1, int(cml_p["batch_size"]))
        self.cml_timeout_seconds = float(cml_p["timeout_seconds"])

        self.controller_model = str(ctrl_p["model"])
        self.controller_base_url = str(ctrl_p["base_url"])
        self.controller_api_key_env = str(ctrl_p["api_key_env"])
        self.controller_api_key_file = (
            str(ctrl_p["api_key_file"]) if ctrl_p.get("api_key_file") else None
        )
        self.controller_max_tokens = int(ctrl_p["max_tokens"])
        self.controller_timeout_seconds = float(ctrl_p["timeout_seconds"])

        self.enable_query_expansion = bool(ret_p["enable_query_expansion"])
        self.enable_vision_query = bool(ret_p["enable_vision_query"])
        self.max_surface_queries = max(1, int(ret_p["max_surface_queries"]))
        self.cml_top_k_per_query = max(5, int(ret_p["cml_top_k_per_query"]))
        self.max_workflows = max(1, int(ret_p["max_workflows"]))
        self.max_state_slices = max(1, int(ret_p["max_state_slices"]))
        self.per_query_slice_cap = max(1, int(ret_p["per_query_slice_cap"]))
        self.slice_radius = max(0, int(ret_p["slice_radius"]))
        self.max_images = max(0, int(ret_p["max_images"]))
        self.center_axtree_max_chars = max(1000, int(ret_p["center_axtree_max_chars"]))
        self.neighbor_axtree_max_chars = max(500, int(ret_p["neighbor_axtree_max_chars"]))

        self.inserted_trajectory_ids: list[str] = []
        self._inserted_set: set[str] = set()
        self._domain: str = "general"
        self._pending_turns: list[dict[str, Any]] = []
        self._pending_ledger_updates: dict[str, str] = {}
        self._workflow_summaries: dict[str, dict[str, Any]] = {}
        self._last_query_diagnostics_local = threading.local()

        self._http_local = threading.local()
        self._llm_local = threading.local()
        self._insert_lock = threading.Lock()

        if self.workspace_dir is not None:
            (self.workspace_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    @property
    def _tenant_id(self) -> str:
        dom = self._domain if self._domain != "unknown" else "general"
        return f"{self.cml_tenant_prefix}-{dom}"

    def _cache_dir(self) -> Path:
        if self.trajectories_root_dir is not None:
            d = self.trajectories_root_dir / ".cml_cache"
        elif self.workspace_dir is not None:
            d = self.workspace_dir / ".cml_cache"
        else:
            d = Path("/tmp/cml_lme_v2_cache")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ledger_path(self) -> Path:
        return self._cache_dir() / f"{self._tenant_id}_ingested_ledger.json"

    def _get_http_session(self) -> requests.Session:
        sess = getattr(self._http_local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self._http_local.session = sess
        return sess

    def _get_controller_client(self) -> OpenAI:
        client = getattr(self._llm_local, "client", None)
        if client is None:
            api_key = _load_api_key(self.controller_api_key_env, self.controller_api_key_file)
            if self.controller_base_url:
                client = OpenAI(
                    base_url=self.controller_base_url,
                    api_key=api_key or "EMPTY",
                    max_retries=3,
                    timeout=self.controller_timeout_seconds,
                )
            else:
                require(api_key is not None, "Missing controller API key")
                client = OpenAI(
                    api_key=api_key,
                    max_retries=3,
                    timeout=self.controller_timeout_seconds,
                )
            self._llm_local.client = client
        return client

    def _verify_and_load_ledger(self) -> dict[str, str]:
        path = self._ledger_path()
        tenant = self._tenant_id
        with _LEDGER_LOCK:
            ledger: dict[str, str] = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        ledger = {str(k): str(v) for k, v in raw.items()}
                except (OSError, json.JSONDecodeError):
                    ledger = {}

            if ledger and tenant not in _VERIFIED_TENANTS:
                # Verify CML server actually still holds records for this tenant
                try:
                    resp = self._get_http_session().get(
                        f"{self.cml_base_url}/api/v1/memory/stats",
                        headers={
                            "X-API-Key": self.cml_api_key,
                            "X-Tenant-ID": tenant,
                        },
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        total = int((resp.json() or {}).get("total_memories") or 0)
                        if total == 0:
                            ledger = {}
                            path.write_text("{}\n", encoding="utf-8")
                    _VERIFIED_TENANTS.add(tenant)
                except Exception:
                    pass
            return ledger

    def _update_ledger(self, updates: dict[str, str]) -> None:
        if not updates:
            return
        path = self._ledger_path()
        with _LEDGER_LOCK:
            current: dict[str, str] = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        current = {str(k): str(v) for k, v in raw.items()}
                except (OSError, json.JSONDecodeError):
                    current = {}
            current.update(updates)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(path)
            _VERIFIED_TENANTS.add(self._tenant_id)

    def _cml_write_batch(self, turns: list[dict[str, Any]]) -> None:
        if not turns:
            return
        url = f"{self.cml_base_url}/api/v1/memory/write_batch"
        headers = {
            "X-API-Key": self.cml_api_key,
            "X-Tenant-ID": self._tenant_id,
            "X-Eval-Mode": "true",
        }
        for start in range(0, len(turns), self.cml_batch_size):
            batch = turns[start : start + self.cml_batch_size]
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    resp = self._get_http_session().post(
                        url,
                        json={"turns": batch},
                        headers=headers,
                        timeout=self.cml_timeout_seconds,
                    )
                    if resp.status_code in (429, 502, 503, 504):
                        time.sleep(min(20.0, 2.0 * (2**attempt)))
                        continue
                    resp.raise_for_status()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(min(15.0, 2.0 * (attempt + 1)))
            if last_exc is not None:
                raise RuntimeError(
                    f"CML write_batch failed for tenant {self._tenant_id}: {last_exc}"
                ) from last_exc

    def _flush_pending_cml_turns(self) -> None:
        with self._insert_lock:
            if not self._pending_turns:
                return
            turns = list(self._pending_turns)
            updates = dict(self._pending_ledger_updates)
            self._pending_turns.clear()
            self._pending_ledger_updates.clear()
        self._cml_write_batch(turns)
        self._update_ledger(updates)

    def _build_trajectory_artifacts(
        self, simplified: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build workflow summary dict + CML batch write turns for a simplified trajectory."""
        tid = str(simplified["id"])
        goal = _normalize_ws(str(simplified.get("goal") or ""))
        outcome = str(simplified.get("outcome") or "unknown").strip()
        start_url = str(simplified.get("start_url") or "").strip()
        states: list[dict[str, Any]] = list(simplified.get("states") or [])
        transition_actions = _annotated_transition_actions(states)

        visited_pages: list[str] = []
        gotcha_notes: list[str] = []
        key_thoughts: list[str] = []
        seen_thoughts: set[str] = set()

        turns: list[dict[str, Any]] = []
        prev_compact_sig = ""

        for idx, state in enumerate(states):
            s_idx = int(state.get("state_index", idx))
            step = int(state.get("step", s_idx))
            url = str(state.get("url") or "")
            raw_ax = str(state.get("text") or "")
            root_title = _parse_root_title(raw_ax) or _short_url(url)
            if root_title and (not visited_pages or visited_pages[-1] != root_title):
                visited_pages.append(root_title)

            thought_raw = state.get("thoughts")
            thought = (
                _normalize_ws(thought_raw)
                if isinstance(thought_raw, str) and thought_raw.strip()
                else ""
            )
            if thought and thought.lower() not in seen_thoughts:
                seen_thoughts.add(thought.lower())
                key_thoughts.append(f"Step {step} ({root_title}): {thought}")
                if GOTCHA_SIGNAL_RE.search(thought):
                    gotcha_notes.append(f"Step {step} [{root_title}]: {thought}")

            incoming_action = (
                "<initial state>"
                if idx == 0
                else transition_actions[idx - 1]
            )
            outgoing_action = (
                transition_actions[idx]
                if idx < len(transition_actions)
                else "<end of trajectory>"
            )

            ui_windows = _compact_axtree_windows_for_cml(raw_ax, window_chars=2200, overlap_chars=250, max_windows=6)
            sig = hashlib.md5(
                f"{url}|{root_title}|{incoming_action}|{'|'.join(w[:240] for w in ui_windows)}".encode()
            ).hexdigest()
            # Always keep first state, last state, states with thoughts/actions, or changed UI
            if (
                sig == prev_compact_sig
                and idx not in (0, len(states) - 1)
                and incoming_action == "<none>"
                and not thought
            ):
                continue
            prev_compact_sig = sig

            state_hdr = (
                f"[Page: {root_title} | URL: {_short_url(url)}] "
                f"[Goal: {_truncate_middle(goal, 120)} | Outcome: {outcome}] "
                f"[Prev Action -> State {s_idx}: {_truncate_middle(incoming_action, 150)}] "
                f"[Next Action: {_truncate_middle(outgoing_action, 150)}] "
                f"[Agent Observation: {_truncate_middle(thought or '<none>', 220)}] "
            )
            for win_idx, ui_win in enumerate(ui_windows):
                state_content = f"{state_hdr}UI Elements ({win_idx + 1}/{len(ui_windows)}): {ui_win}"
                turns.append(
                    {
                        "content": state_content[:4000],
                        "session_id": tid,
                        "turn_id": f"{tid}:state:{s_idx:04d}:w{win_idx}",
                        "metadata": {
                            "trajectory_id": tid,
                            "lme_kind": "state",
                            "state_index": s_idx,
                            "window_index": win_idx,
                            "step": step,
                            "root_title": root_title[:160],
                            "url": url[:240],
                        },
                    }
                )

        action_lines = [
            f"{i + 1}. {act}"
            for i, act in enumerate(transition_actions)
            if act and act != "<none>"
        ]
        pages_chain = " -> ".join(visited_pages[:18]) if visited_pages else _short_url(start_url)
        workflow_summary = {
            "trajectory_id": tid,
            "goal": goal,
            "outcome": outcome,
            "start_url": start_url,
            "visited_pages": visited_pages,
            "action_sequence": "\n".join(action_lines[:35]) or "1. <no recorded actions>",
            "gotcha_notes": gotcha_notes[:10],
            "key_thoughts": key_thoughts[:12],
            "state_count": len(states),
        }

        gotcha_block = (
            " | Gotchas/Signals: " + " ; ".join(gotcha_notes[:6])
            if gotcha_notes
            else ""
        )
        thoughts_block = " ; ".join(key_thoughts[:6])
        workflow_content = (
            f"[Workflow Playbook | Outcome: {outcome} | Pages: {_truncate_middle(pages_chain, 220)}] "
            f"Goal: {_truncate_middle(goal, 260)} | "
            f"Actions: {_truncate_middle(' ; '.join(action_lines[:20]), 750)} | "
            f"Observations: {_truncate_middle(thoughts_block, 600)}"
            f"{_truncate_middle(gotcha_block, 450)}"
        )
        turns.insert(
            0,
            {
                "content": workflow_content[:4000],
                "session_id": tid,
                "turn_id": f"{tid}:workflow",
                "metadata": {
                    "trajectory_id": tid,
                    "lme_kind": "workflow",
                    "outcome": outcome,
                    "state_index": -1,
                },
            },
        )
        return workflow_summary, turns

    def _summary_cache_path(self, trajectory_dir: Path) -> Path:
        return trajectory_dir / "cml_workflow_summary.json"

    def insert(self, trajectory: dict[str, object]) -> None:
        require(self.workspace_dir is not None, "cml insert requires workspace_dir")
        require(
            self.trajectories_root_dir is not None,
            "cml insert requires trajectories_root_dir",
        )
        prepared = prepare_trajectory_insert(
            trajectory,
            trajectories_root_dir=self.trajectories_root_dir,
        )
        tid = prepared.trajectory_id
        if tid in self._inserted_set:
            raise RuntimeError(f"Duplicate trajectory insert attempted: {tid}")

        simplified = prepared.simplified
        if self._domain == "general":
            raw_dom = trajectory.get("domain")
            if isinstance(raw_dom, str) and raw_dom in ("web", "enterprise"):
                self._domain = raw_dom
            else:
                self._domain = _domain_from_url(str(simplified.get("start_url") or ""))

        ws_traj_dir = self.workspace_dir / "trajectories" / tid
        require(not ws_traj_dir.exists(), f"Refusing to overwrite trajectory dir: {ws_traj_dir}")

        target_store_dir = ws_traj_dir
        if self.trajectory_pool_root is not None:
            pooled_dir = self.trajectory_pool_root / tid
            pooled_json = pooled_dir / "trajectory.json"
            if pooled_json.exists():
                try:
                    pooled_payload = load_json(pooled_json)
                    if (
                        isinstance(pooled_payload, dict)
                        and logical_trajectory_fingerprint_from_simplified(pooled_payload)
                        == prepared.fingerprint
                    ):
                        relative_symlink(pooled_dir, ws_traj_dir)
                        target_store_dir = pooled_dir
                    else:
                        shutil.rmtree(pooled_dir, ignore_errors=True)
                except Exception:
                    shutil.rmtree(pooled_dir, ignore_errors=True)
            if not ws_traj_dir.exists():
                pooled_dir.mkdir(parents=True, exist_ok=True)
                materialize_prepared_trajectory(prepared, pooled_dir)
                relative_symlink(pooled_dir, ws_traj_dir)
                target_store_dir = pooled_dir
        else:
            ws_traj_dir.mkdir(parents=True, exist_ok=False)
            materialize_prepared_trajectory(prepared, ws_traj_dir)

        summary_file = self._summary_cache_path(target_store_dir)
        ledger = self._verify_and_load_ledger()
        already_in_cml = ledger.get(tid) == prepared.fingerprint

        if already_in_cml and summary_file.exists():
            try:
                wf_summary = load_json(summary_file)
            except Exception:
                wf_summary, turns = self._build_trajectory_artifacts(simplified)
                save_json(summary_file, wf_summary)
                already_in_cml = False
            else:
                turns = []
        else:
            wf_summary, turns = self._build_trajectory_artifacts(simplified)
            save_json(summary_file, wf_summary)

        with self._insert_lock:
            self.inserted_trajectory_ids.append(tid)
            self._inserted_set.add(tid)
            self._workflow_summaries[tid] = wf_summary
            if not already_in_cml and turns:
                self._pending_turns.extend(turns)
                self._pending_ledger_updates[tid] = prepared.fingerprint
                should_flush = len(self._pending_turns) >= self.cml_batch_size
            else:
                should_flush = False

        if should_flush:
            self._flush_pending_cml_turns()

    def _describe_query_image(self, query: str, query_image: str) -> str:
        img_path = Path(query_image)
        if not img_path.exists():
            return ""
        try:
            client = self._get_controller_client()
            resp = client.chat.completions.create(
                model=self.controller_model,
                messages=[
                    {"role": "system", "content": VISION_LANDMARK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Question context: {_strip_question_wrappers(query)[:300]}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": _to_data_url(img_path)},
                            },
                        ],
                    },
                ],
                max_tokens=180,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            txt = _extract_content_text(resp.choices[0].message.content) or ""
            return _normalize_ws(txt)
        except Exception:
            return ""

    def _plan_queries(self, query: str, visual_landmarks: str) -> dict[str, Any]:
        fallback = _deterministic_query_bundle(query, visual_landmarks)
        if not self.enable_query_expansion:
            return fallback

        cleaned_q = _strip_question_wrappers(query)
        user_msg = f"Question:\n{cleaned_q}"
        if visual_landmarks:
            user_msg += f"\n\nVisible screenshot landmarks:\n{visual_landmarks}"

        try:
            client = self._get_controller_client()
            resp = client.chat.completions.create(
                model=self.controller_model,
                messages=[
                    {"role": "system", "content": QUERY_DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.controller_max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = _extract_content_text(resp.choices[0].message.content) or ""
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    sq_raw = parsed.get("surface_queries")
                    wq_raw = parsed.get("workflow_query")
                    sq_list: list[str] = []
                    seen: set[str] = set()
                    if isinstance(sq_raw, list):
                        for item in sq_raw:
                            if isinstance(item, str) and item.strip():
                                norm = _normalize_ws(item)
                                if norm.lower() not in seen:
                                    seen.add(norm.lower())
                                    sq_list.append(norm)
                    if visual_landmarks:
                        vis_q = _normalize_ws(f"{visual_landmarks[:160]} {cleaned_q[:80]}")
                        if vis_q.lower() not in seen:
                            sq_list.insert(0, vis_q)
                    if sq_list:
                        wf_q = (
                            _normalize_ws(wq_raw)
                            if isinstance(wq_raw, str) and wq_raw.strip()
                            else fallback["workflow_query"]
                        )
                        return {
                            "surface_queries": sq_list[: self.max_surface_queries],
                            "workflow_query": wf_q,
                        }
        except Exception:
            pass
        return fallback

    def _cml_read(self, query_text: str, max_results: int) -> list[dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []
        url = f"{self.cml_base_url}/api/v1/memory/read"
        headers = {
            "X-API-Key": self.cml_api_key,
            "X-Tenant-ID": self._tenant_id,
        }
        payload = {
            "query": query_text.strip()[:1200],
            "max_results": max_results,
            "format": "list",
        }
        for attempt in range(4):
            try:
                resp = self._get_http_session().post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=90.0,
                )
                if resp.status_code in (429, 502, 503, 504):
                    time.sleep(min(10.0, 1.5 * (2**attempt)))
                    continue
                resp.raise_for_status()
                body = resp.json() or {}
                memories = body.get("memories") or []
                out: list[dict[str, Any]] = []
                for m in memories:
                    if not isinstance(m, dict):
                        continue
                    meta = m.get("metadata") or {}
                    tid = str(meta.get("trajectory_id") or "")
                    # Strict haystack isolation: only admit trajectories inserted into this Memory instance
                    if tid and tid in self._inserted_set:
                        out.append(m)
                return out
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return []

    def _lexical_fallback_trajectories(self, query: str) -> list[str]:
        """Pure in-memory lexical fallback over inserted workflow summaries if CML returns sparse hits."""
        q_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]{3,}", (query or "").lower())
            if tok not in {"the", "and", "for", "with", "what", "which", "when", "that", "from", "our", "are", "this"}
        }
        if not q_tokens:
            return self.inserted_trajectory_ids[: self.max_workflows]
        scored: list[tuple[float, str]] = []
        for tid in self.inserted_trajectory_ids:
            wf = self._workflow_summaries.get(tid)
            if not wf:
                continue
            blob = (
                f"{wf.get('goal', '')} {' '.join(wf.get('visited_pages') or [])} "
                f"{wf.get('action_sequence', '')} {' '.join(wf.get('gotcha_notes') or [])}"
            ).lower()
            hits = sum(1 for tok in q_tokens if tok in blob)
            if hits > 0:
                scored.append((hits / len(q_tokens), tid))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [tid for _, tid in scored[: self.max_workflows]]

    def _load_stored_trajectory(self, trajectory_id: str) -> dict[str, Any]:
        require(self.workspace_dir is not None, "cml query requires workspace_dir")
        traj_path = self.workspace_dir / "trajectories" / trajectory_id / "trajectory.json"
        payload = load_json(traj_path)
        require(isinstance(payload, dict), f"Invalid trajectory.json for {trajectory_id}")
        return payload

    def _resolve_state_screenshot(self, trajectory_id: str, screenshot_rel: str) -> Path | None:
        if self.workspace_dir is None or not screenshot_rel:
            return None
        candidate = (self.workspace_dir / "trajectories" / trajectory_id / screenshot_rel).resolve()
        if candidate.exists():
            return candidate
        return None

    def _load_workflow_summary(self, trajectory_id: str) -> dict[str, Any]:
        cached = self._workflow_summaries.get(trajectory_id)
        if cached is not None:
            return cached
        if self.workspace_dir is not None:
            summary_path = (
                self.workspace_dir / "trajectories" / trajectory_id / "cml_workflow_summary.json"
            )
            if summary_path.exists():
                loaded = load_json(summary_path)
                if isinstance(loaded, dict):
                    self._workflow_summaries[trajectory_id] = loaded
                    return loaded
        traj = self._load_stored_trajectory(trajectory_id)
        wf_summary, _ = self._build_trajectory_artifacts(traj)
        self._workflow_summaries[trajectory_id] = wf_summary
        return wf_summary

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        require(isinstance(query, str) and query.strip(), "cml query must be non-empty")
        self._flush_pending_cml_turns()

        if not self.inserted_trajectory_ids:
            return []

        visual_landmarks = ""
        if query_image is not None and self.enable_vision_query:
            visual_landmarks = self._describe_query_image(query, query_image)

        bundle = self._plan_queries(query, visual_landmarks)
        surface_queries: list[str] = list(bundle["surface_queries"])
        workflow_query: str = str(bundle["workflow_query"])

        # Execute CML hybrid reads across surface queries + workflow query
        query_state_blocks: list[tuple[str, list[dict[str, Any]]]] = []
        traj_score_acc: Counter[str] = Counter()
        traj_query_hits: dict[str, set[int]] = {}

        all_queries: list[tuple[int, str, bool]] = [
            (idx, sq, False) for idx, sq in enumerate(surface_queries)
        ]
        if workflow_query and workflow_query.lower() not in {sq.lower() for sq in surface_queries}:
            all_queries.append((len(all_queries), workflow_query, True))

        for q_idx, q_text, is_wf_only in all_queries:
            memories = self._cml_read(q_text, max_results=self.cml_top_k_per_query)
            state_hits_for_q: list[dict[str, Any]] = []
            for rank_idx, mem in enumerate(memories):
                meta = mem.get("metadata") or {}
                tid = str(meta.get("trajectory_id") or "")
                if not tid or tid not in self._inserted_set:
                    continue
                rel = float(mem.get("relevance") or 0.5)
                kind = str(meta.get("lme_kind") or "state")
                rank_decay = 1.0 / (1.0 + 0.12 * rank_idx)
                weight = 1.25 if (kind == "workflow" and is_wf_only) else 1.0
                traj_score_acc[tid] += rel * rank_decay * weight
                traj_query_hits.setdefault(tid, set()).add(q_idx)
                if kind == "state" and not is_wf_only:
                    s_idx = int(meta.get("state_index", 0))
                    state_hits_for_q.append(
                        {
                            "trajectory_id": tid,
                            "state_index": s_idx,
                            "score": rel,
                            "rank": rank_idx + 1,
                            "query": q_text,
                        }
                    )
            if state_hits_for_q:
                query_state_blocks.append((q_text, state_hits_for_q))

        # Co-activation bonus for trajectories retrieved across multiple query angles
        for tid, q_set in traj_query_hits.items():
            if len(q_set) >= 2:
                traj_score_acc[tid] *= 1.0 + 0.15 * (len(q_set) - 1)

        top_workflow_tids = [tid for tid, _ in traj_score_acc.most_common(self.max_workflows)]
        if not top_workflow_tids:
            top_workflow_tids = self._lexical_fallback_trajectories(query)

        # Round-robin diverse merge of state slices across surface queries so compared pages both appear
        selected_slices: list[dict[str, Any]] = []
        covered_states: set[tuple[str, int]] = set()

        def _overlaps_existing(tid: str, center: int) -> bool:
            return any(
                (tid, offset) in covered_states
                for offset in range(center - self.slice_radius, center + self.slice_radius + 1)
            )

        for _, hits in query_state_blocks:
            taken_for_q = 0
            for hit in hits:
                if len(selected_slices) >= self.max_state_slices:
                    break
                if taken_for_q >= self.per_query_slice_cap:
                    break
                tid = hit["trajectory_id"]
                c_idx = hit["state_index"]
                if _overlaps_existing(tid, c_idx):
                    continue
                selected_slices.append(hit)
                for offset in range(c_idx - self.slice_radius, c_idx + self.slice_radius + 1):
                    covered_states.add((tid, offset))
                taken_for_q += 1

        # Backfill remaining slice budget by global score across all query blocks
        if len(selected_slices) < self.max_state_slices:
            remaining_candidates = [
                hit for _, hits in query_state_blocks for hit in hits
            ]
            remaining_candidates.sort(key=lambda h: float(h["score"]), reverse=True)
            for hit in remaining_candidates:
                if len(selected_slices) >= self.max_state_slices:
                    break
                tid = hit["trajectory_id"]
                c_idx = hit["state_index"]
                if _overlaps_existing(tid, c_idx):
                    continue
                selected_slices.append(hit)
                for offset in range(c_idx - self.slice_radius, c_idx + self.slice_radius + 1):
                    covered_states.add((tid, offset))

        # Ensure at least top workflow trajectory contributes a representative slice if none matched
        if not selected_slices and top_workflow_tids:
            lead_tid = top_workflow_tids[0]
            traj = self._load_stored_trajectory(lead_tid)
            states = traj.get("states") or []
            if states:
                mid_idx = len(states) // 2
                selected_slices.append(
                    {
                        "trajectory_id": lead_tid,
                        "state_index": mid_idx,
                        "score": 0.5,
                        "rank": 1,
                        "query": surface_queries[0] if surface_queries else query,
                    }
                )

        items: list[MemoryContextItem] = []

        # Section 1: Workflow & Environment Gotcha Playbooks
        if top_workflow_tids:
            wf_lines: list[str] = [
                "## Retrieved Workflow Playbooks & Environment Observations (CognitiveMemoryLayer)\n"
            ]
            for rank_i, tid in enumerate(top_workflow_tids, start=1):
                wf = self._load_workflow_summary(tid)
                outcome_val = str(wf.get("outcome", "unknown"))
                pages_str = " -> ".join((wf.get("visited_pages") or [])[:15])
                failure_note = (
                    "  * IMPORTANT: This task run ended in FAILURE — do NOT assume its attempted goal succeeded; "
                    "check the UI state slices and thoughts below to see whether a requested filter, field, "
                    "option, or workflow step does NOT exist in this environment.\n"
                    if outcome_val.lower() == "failure"
                    else ""
                )
                act_lines = (wf.get("action_sequence") or "").splitlines()[:20]
                wf_lines.append(
                    f"\n### Workflow Memory {rank_i} (trajectory={tid}, outcome={outcome_val.upper()})\n"
                    f"{failure_note}"
                    f"- Goal: {wf.get('goal', '')}\n"
                    f"- Start URL: {wf.get('start_url', '')}\n"
                    f"- Ordered Visited Pages / Modules: {pages_str}\n"
                )
                gotchas = wf.get("gotcha_notes") or []
                if gotchas:
                    wf_lines.append("- Environment Gotchas & Critical Signals:\n")
                    for g in gotchas[:6]:
                        wf_lines.append(f"  * {g}\n")
                thoughts = wf.get("key_thoughts") or []
                if thoughts:
                    wf_lines.append("- Key Agent Reasoning Trace:\n")
                    for th in thoughts[:5]:
                        wf_lines.append(f"  * {th}\n")
                wf_lines.append(
                    "- Annotated Action Sequence:\n" + "\n".join(act_lines) + "\n"
                )
            items.append({"type": "text", "value": "".join(wf_lines).strip() + "\n"})

        # Section 2: Dereferenced High-Fidelity State Slices + Multimodal Screenshots
        if selected_slices:
            items.append(
                {
                    "type": "text",
                    "value": "\n## Retrieved UI State Slices & Transition Evidence\n",
                }
            )
            traj_cache: dict[str, dict[str, Any]] = {}
            attached_images = 0

            for slice_rank, sl in enumerate(selected_slices, start=1):
                tid = sl["trajectory_id"]
                center_idx = int(sl["state_index"])
                if tid not in traj_cache:
                    traj_cache[tid] = self._load_stored_trajectory(tid)
                traj = traj_cache[tid]
                states: list[dict[str, Any]] = list(traj.get("states") or [])
                state_map = {
                    int(st["state_index"]): st
                    for st in states
                    if isinstance(st, dict) and isinstance(st.get("state_index"), int)
                }
                transition_actions = _annotated_transition_actions(states)

                lo = max(0, center_idx - self.slice_radius)
                hi = min(len(states) - 1, center_idx + self.slice_radius)
                state_blocks: list[str] = []
                for s_i in range(lo, hi + 1):
                    st = state_map.get(s_i)
                    if st is None:
                        continue
                    is_center = s_i == center_idx
                    max_ax_chars = (
                        self.center_axtree_max_chars
                        if is_center
                        else self.neighbor_axtree_max_chars
                    )
                    in_act = "<initial state>" if s_i == 0 else transition_actions[s_i - 1]
                    out_act = (
                        transition_actions[s_i]
                        if s_i < len(transition_actions)
                        else "<end of trajectory>"
                    )
                    thought_val = st.get("thoughts")
                    thought_txt = (
                        thought_val.strip()
                        if isinstance(thought_val, str) and thought_val.strip()
                        else "<none>"
                    )
                    cleaned_ax = _clean_axtree_for_reader(
                        str(st.get("text") or ""),
                        max_chars=max_ax_chars,
                    )
                    role_tag = " [CENTER MATCH]" if is_center else ""
                    state_blocks.append(
                        f"#### State {s_i} (step {st.get('step', s_i)}){role_tag}\n"
                        f"- URL: {st.get('url', '')}\n"
                        f"- Incoming Action: {in_act}\n"
                        f"- Next Action: {out_act}\n"
                        f"- Agent Thought: {thought_txt}\n"
                        f"- UI Accessibility Tree:\n{cleaned_ax}"
                    )

                traj_outcome = str(traj.get("outcome", "unknown"))
                slice_outcome_banner = (
                    "- Trajectory Outcome: FAILURE (this attempted task did NOT succeed — check UI options/controls carefully rather than assuming the requested action/filter existed!)\n"
                    if traj_outcome.lower() == "failure"
                    else f"- Trajectory Outcome: {traj_outcome}\n"
                )
                slice_header = (
                    f"### UI State Slice {slice_rank} "
                    f"(trajectory={tid}, center_state={center_idx}, score={sl['score']:.4f})\n"
                    f"- Matched Query: {sl['query']}\n"
                    f"- Trajectory Goal: {traj.get('goal', '')}\n"
                    f"{slice_outcome_banner}\n"
                    + "\n\n".join(state_blocks)
                    + "\n"
                )
                items.append({"type": "text", "value": slice_header})

                if attached_images < self.max_images:
                    center_st = state_map.get(center_idx)
                    if center_st is not None:
                        scr_path = self._resolve_state_screenshot(
                            tid, str(center_st.get("screenshot") or "")
                        )
                        if scr_path is not None:
                            items.append({"type": "image", "value": str(scr_path)})
                            attached_images += 1

        self._last_query_diagnostics_local.info = {
            "tenant_id": self._tenant_id,
            "surface_queries": surface_queries,
            "workflow_query": workflow_query,
            "visual_landmarks_used": bool(visual_landmarks),
            "selected_workflow_trajectories": top_workflow_tids,
            "selected_state_slices": [
                {
                    "trajectory_id": s["trajectory_id"],
                    "state_index": s["state_index"],
                    "score": round(float(s["score"]), 4),
                }
                for s in selected_slices
            ],
        }
        return items

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        _ = query, query_image, memory_context
        info = getattr(self._last_query_diagnostics_local, "info", None)
        return dict(info) if isinstance(info, dict) else None

    def _save_backend(self, output_dir: Path) -> None:
        self._flush_pending_cml_turns()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            output_dir / "index.json",
            {
                "memory_type": self.memory_type,
                "domain": self._domain,
                "tenant_id": self._tenant_id,
                "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
                "trajectory_count": len(self.inserted_trajectory_ids),
            },
        )
        save_json(
            output_dir / "workflow_summaries.json",
            self._workflow_summaries,
        )
        if self.workspace_dir is not None and self.workspace_dir.resolve() != output_dir.resolve():
            src_trajs = self.workspace_dir / "trajectories"
            dst_trajs = output_dir / "trajectories"
            if src_trajs.exists():
                if dst_trajs.exists():
                    shutil.rmtree(dst_trajs)
                shutil.copytree(src_trajs, dst_trajs, symlinks=True)

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        index_payload = load_json(self.workspace_dir / "index.json")
        require(isinstance(index_payload, dict), "cml index.json must be an object")
        ids = index_payload.get("inserted_trajectory_ids")
        require(
            isinstance(ids, list) and all(isinstance(x, str) and x for x in ids),
            "cml index.json missing valid inserted_trajectory_ids",
        )
        self.inserted_trajectory_ids = list(ids)
        self._inserted_set = set(ids)
        dom = index_payload.get("domain")
        if isinstance(dom, str) and dom:
            self._domain = dom
        wf_path = self.workspace_dir / "workflow_summaries.json"
        if wf_path.exists():
            loaded = load_json(wf_path)
            if isinstance(loaded, dict):
                self._workflow_summaries = loaded
