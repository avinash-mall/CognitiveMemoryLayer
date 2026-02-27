# Phase 10: Observability, Benchmarking & Production Hardening

**Intrinsic Phase I-10** (planned; not yet implemented). See [ActiveCML/README.md](README.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

Phase 10 transforms the intrinsic memory system from a research prototype into a production-ready system. This phase adds comprehensive observability (logging, metrics, tracing), establishes performance benchmarks comparing intrinsic memory against RAG baselines, implements safety guardrails, and hardens the system for real-world deployment.

### Core Objectives
1. **Observability:** Full visibility into what the memory system is doing to the model
2. **Benchmarking:** Quantitative comparison of intrinsic memory vs. RAG across quality, latency, and cost metrics
3. **Safety:** Guardrails preventing the memory system from degrading model output quality
4. **Production hardening:** Error handling, resource management, graceful degradation, monitoring
5. **Interpretability:** Tools to understand WHY a memory was injected and HOW it affected the output

### Dependencies
- All prior phases (1-9)
- Prometheus/OpenTelemetry for metrics
- Existing metrics infrastructure (`src/utils/metrics.py`)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Observability & Production Layer                            │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
│  │  Metrics Engine   │   │  Distributed     │   │  Injection Logger       │  │
│  │  ─────────────── │   │  Tracing         │   │  ─────────────────────  │  │
│  │  Prometheus       │   │  ──────────────  │   │  Per-turn log:          │  │
│  │  counters:        │   │  OpenTelemetry   │   │  - memories retrieved   │  │
│  │  - injections     │   │  spans:          │   │  - gate decisions       │  │
│  │  - gate decisions │   │  - retrieval     │   │  - channels used        │  │
│  │  - cache hits     │   │  - encoding      │   │  - injection strengths  │  │
│  │  histograms:      │   │  - injection     │   │  - model output delta   │  │
│  │  - latency        │   │  - generation    │   │  Stored in event_log    │  │
│  │  - vector norms   │   │                  │   │                         │  │
│  │  gauges:          │   │                  │   │                         │  │
│  │  - active hooks   │   │                  │   │                         │  │
│  │  - cache usage    │   │                  │   │                         │  │
│  └──────────────────┘   └──────────────────┘   └─────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Benchmarking Framework                               │    │
│  │  ──────────────────────────────────────                               │    │
│  │  Automated A/B testing:                                               │    │
│  │  - RAG-only vs. Intrinsic on standardized queries                    │    │
│  │  - Metrics: accuracy, latency, token usage, perplexity              │    │
│  │  - Memory types: factual recall, preference adherence, constraint   │    │
│  │  - Reports: dashboard + markdown summary                             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Safety Guardrails                                    │    │
│  │  ──────────────────────────────                                       │    │
│  │  1. Output quality monitor: perplexity spike detection               │    │
│  │  2. Injection budget enforcer: hard limits on total influence        │    │
│  │  3. Contradiction detector: conflicting memory injection             │    │
│  │  4. Bias amplification guard: prevent reinforcement loops            │    │
│  │  5. Emergency kill switch: disable all injection instantly           │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                  Interpretability Tools                                │    │
│  │  ──────────────────────────────                                       │    │
│  │  - Injection attribution: which memory influenced which output token │    │
│  │  - Steering vector visualization: project to 2D for inspection      │    │
│  │  - KV attention heatmap: how much does the model attend to virtual  │    │
│  │    tokens vs. real context                                           │    │
│  │  - Counterfactual analysis: "what would output be WITHOUT memory X?" │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Metrics & Monitoring

### Sub-task 1.1: Prometheus Metrics

**File:** `src/intrinsic/observability/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# --- Injection Metrics ---

INTRINSIC_INJECTIONS_TOTAL = Counter(
    "cml_intrinsic_injections_total",
    "Total intrinsic memory injections",
    ["channel", "memory_type"],
)

INTRINSIC_GATE_DECISIONS = Counter(
    "cml_intrinsic_gate_decisions_total",
    "Gate decisions (inject/skip/defer)",
    ["decision"],
)

INTRINSIC_INJECTION_STRENGTH = Histogram(
    "cml_intrinsic_injection_strength",
    "Distribution of injection strengths",
    ["channel"],
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
)

# --- Latency Metrics ---

INTRINSIC_ENCODING_LATENCY = Histogram(
    "cml_intrinsic_encoding_latency_seconds",
    "Time to encode a memory into latent representations",
    ["mode"],  # "full", "fast", "logit_only"
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

INTRINSIC_INJECTION_LATENCY = Histogram(
    "cml_intrinsic_injection_latency_seconds",
    "Time to inject memories into forward pass",
    ["channel"],
    buckets=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
)

INTRINSIC_CONTROLLER_LATENCY = Histogram(
    "cml_intrinsic_controller_latency_seconds",
    "Time for controller to process retrieval results",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)

# --- Cache Metrics ---

INTRINSIC_CACHE_HITS = Counter(
    "cml_intrinsic_cache_hits_total",
    "Cache hits by tier",
    ["tier"],  # "l1", "l2", "l3"
)

INTRINSIC_CACHE_MISSES = Counter(
    "cml_intrinsic_cache_misses_total",
    "Cache misses",
)

INTRINSIC_CACHE_UTILIZATION = Gauge(
    "cml_intrinsic_cache_utilization_ratio",
    "Cache utilization ratio by tier",
    ["tier"],
)

# --- Safety Metrics ---

INTRINSIC_SAFETY_REVERTS = Counter(
    "cml_intrinsic_safety_reverts_total",
    "Number of times safety guard reverted an injection",
    ["reason"],  # "nan", "norm_explosion", "divergence"
)

INTRINSIC_NORM_RATIO = Histogram(
    "cml_intrinsic_norm_ratio",
    "Ratio of modified/original hidden state norms",
    buckets=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
)

# --- Model Quality Metrics ---

INTRINSIC_PERPLEXITY = Summary(
    "cml_intrinsic_perplexity",
    "Model perplexity with intrinsic memory active",
)

INTRINSIC_ACTIVE_HOOKS = Gauge(
    "cml_intrinsic_active_hooks",
    "Number of currently active forward hooks",
)

INTRINSIC_ACTIVE_VECTORS = Gauge(
    "cml_intrinsic_active_steering_vectors",
    "Number of active steering vectors",
)

INTRINSIC_VIRTUAL_TOKENS = Gauge(
    "cml_intrinsic_virtual_tokens",
    "Number of virtual KV tokens currently injected",
)
```

---

## Task 2: Benchmarking Framework

### Sub-task 2.1: A/B Benchmark Runner

**File:** `src/intrinsic/observability/benchmark.py`

```python
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    query: str
    relevant_memories: List[Dict]        # Memories to be available
    expected_keywords: List[str]          # Keywords that should appear in output
    expected_absent: List[str] = field(default_factory=list)  # Should NOT appear
    category: str = "factual_recall"     # "factual_recall" | "preference" | "constraint" | "style"


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""
    case_id: str
    mode: str                            # "rag_only" | "intrinsic" | "hybrid"
    output: str
    keyword_hits: int
    keyword_misses: int
    absent_violations: int
    latency_ms: float
    tokens_used: int
    injection_info: Optional[Dict] = None
    score: float = 0.0                   # Composite quality score


class BenchmarkRunner:
    """
    Runs A/B benchmarks comparing RAG vs. intrinsic memory.
    
    Benchmark categories:
    1. Factual Recall: "Where does the user live?" (memory: "User lives in Paris")
    2. Preference Adherence: "Suggest a restaurant" (memory: "User prefers vegetarian")
    3. Constraint Respect: "Share contact info" (memory: "Never share user email")
    4. Temporal Memory: "What did we discuss yesterday?" (episodic memory)
    5. Multi-hop: "What's near where I live?" (requires combining memories)
    """

    def __init__(self, app_client, migration_manager):
        self._client = app_client
        self._migration = migration_manager
        self._results: List[BenchmarkResult] = []

    async def run_suite(
        self,
        test_cases: List[BenchmarkCase],
        modes: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark suite across specified modes.
        Returns comparative results.
        """
        modes = modes or ["rag_only", "hybrid_additive", "intrinsic_primary"]
        all_results = {mode: [] for mode in modes}

        for case in test_cases:
            # Seed memories
            await self._seed_memories(case.relevant_memories)

            for mode in modes:
                # Set migration mode
                self._migration.set_mode(mode)

                # Run query
                start = time.time()
                response = await self._client.post("/memory/turn", json={
                    "user_message": case.query,
                })
                latency_ms = (time.time() - start) * 1000

                output = response.json().get("context", "")
                injection_info = response.json().get("intrinsic")

                # Score
                result = self._score_result(case, output, mode, latency_ms, injection_info)
                all_results[mode].append(result)
                self._results.append(result)

        return self._generate_report(all_results)

    def _score_result(
        self,
        case: BenchmarkCase,
        output: str,
        mode: str,
        latency_ms: float,
        injection_info: Optional[Dict],
    ) -> BenchmarkResult:
        """Score a single benchmark result."""
        output_lower = output.lower()
        
        keyword_hits = sum(1 for kw in case.expected_keywords if kw.lower() in output_lower)
        keyword_misses = len(case.expected_keywords) - keyword_hits
        absent_violations = sum(1 for kw in case.expected_absent if kw.lower() in output_lower)
        
        # Composite score: accuracy - violations
        accuracy = keyword_hits / max(len(case.expected_keywords), 1)
        violation_penalty = absent_violations * 0.5
        score = max(0, accuracy - violation_penalty)

        return BenchmarkResult(
            case_id=case.id,
            mode=mode,
            output=output[:200],
            keyword_hits=keyword_hits,
            keyword_misses=keyword_misses,
            absent_violations=absent_violations,
            latency_ms=latency_ms,
            tokens_used=0,
            injection_info=injection_info,
            score=score,
        )

    def _generate_report(self, all_results: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Generate comparative benchmark report."""
        report = {"modes": {}}
        
        for mode, results in all_results.items():
            if not results:
                continue
            
            avg_score = sum(r.score for r in results) / len(results)
            avg_latency = sum(r.latency_ms for r in results) / len(results)
            total_hits = sum(r.keyword_hits for r in results)
            total_misses = sum(r.keyword_misses for r in results)
            total_violations = sum(r.absent_violations for r in results)
            
            report["modes"][mode] = {
                "avg_score": round(avg_score, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "accuracy": round(total_hits / max(total_hits + total_misses, 1), 3),
                "violations": total_violations,
                "cases_run": len(results),
            }
        
        # Compute improvement over RAG baseline
        rag_score = report["modes"].get("rag_only", {}).get("avg_score", 0)
        for mode in report["modes"]:
            if mode != "rag_only":
                mode_score = report["modes"][mode]["avg_score"]
                improvement = ((mode_score - rag_score) / max(rag_score, 0.001)) * 100
                report["modes"][mode]["improvement_over_rag_pct"] = round(improvement, 1)
        
        return report

    async def _seed_memories(self, memories: List[Dict]):
        """Write test memories to the system."""
        for mem in memories:
            await self._client.post("/memory/write", json=mem)


# --- Standard Benchmark Suite ---

STANDARD_BENCHMARK_SUITE = [
    BenchmarkCase(
        id="factual_001",
        query="Where does the user live?",
        relevant_memories=[
            {"content": "User lives in Paris, France", "type": "semantic_fact"},
        ],
        expected_keywords=["Paris", "France"],
        category="factual_recall",
    ),
    BenchmarkCase(
        id="preference_001",
        query="Suggest a restaurant for dinner",
        relevant_memories=[
            {"content": "User prefers vegetarian food", "type": "preference"},
            {"content": "User lives in Paris", "type": "semantic_fact"},
        ],
        expected_keywords=["vegetarian"],
        category="preference",
    ),
    BenchmarkCase(
        id="constraint_001",
        query="Can you share my contact details with the delivery service?",
        relevant_memories=[
            {"content": "Never share the user's email address with third parties", "type": "constraint"},
            {"content": "User's email is user@example.com", "type": "semantic_fact"},
        ],
        expected_keywords=["cannot", "privacy"],
        expected_absent=["user@example.com"],
        category="constraint",
    ),
    BenchmarkCase(
        id="multihop_001",
        query="What's the weather like where I live?",
        relevant_memories=[
            {"content": "User lives in Paris, France", "type": "semantic_fact"},
        ],
        expected_keywords=["Paris"],
        category="factual_recall",
    ),
]
```

---

## Task 3: Safety Guardrails

### Sub-task 3.1: Output Quality Monitor

```python
class OutputQualityMonitor:
    """
    Monitors model output quality when intrinsic memory is active.
    
    Detects:
    1. Perplexity spikes (model is confused by injections)
    2. Repetition loops (stuck on injected concepts)
    3. Incoherent output (hallucinating due to bad injection)
    4. Instruction-following degradation
    
    Actions:
    - Warn: log and alert, but don't intervene
    - Reduce: automatically reduce injection strength
    - Kill: disable all injections for this turn
    """

    def __init__(
        self,
        warn_perplexity_threshold: float = 50.0,
        kill_perplexity_threshold: float = 200.0,
        max_repetition_ratio: float = 0.3,
    ):
        self._warn_ppl = warn_perplexity_threshold
        self._kill_ppl = kill_perplexity_threshold
        self._max_rep = max_repetition_ratio
        self._baseline_perplexity: Optional[float] = None
        self._quality_history: List[Dict] = []

    def set_baseline(self, perplexity: float) -> None:
        """Set baseline perplexity (measured with no injections)."""
        self._baseline_perplexity = perplexity

    def check_output(self, output_text: str, perplexity: Optional[float] = None) -> Dict:
        """
        Check output quality and return assessment.
        """
        issues = []
        action = "pass"

        # Perplexity check
        if perplexity is not None:
            if perplexity > self._kill_ppl:
                issues.append(f"CRITICAL: perplexity={perplexity:.1f} > kill_threshold={self._kill_ppl}")
                action = "kill"
            elif perplexity > self._warn_ppl:
                issues.append(f"WARNING: perplexity={perplexity:.1f} > warn_threshold={self._warn_ppl}")
                action = "reduce"

        # Repetition check
        words = output_text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < (1 - self._max_rep):
                issues.append(f"Repetition detected: unique_ratio={unique_ratio:.2f}")
                if action == "pass":
                    action = "reduce"

        # Coherence check (basic: sentence length variance)
        sentences = output_text.split(".")
        if len(sentences) > 2:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                import statistics
                cv = statistics.stdev(lengths) / max(statistics.mean(lengths), 1)
                if cv > 2.0:
                    issues.append(f"Incoherent: sentence length CV={cv:.2f}")
                    if action == "pass":
                        action = "warn"

        result = {
            "action": action,
            "issues": issues,
            "perplexity": perplexity,
        }
        self._quality_history.append(result)
        return result


class EmergencyKillSwitch:
    """
    Global kill switch for all intrinsic memory injections.
    
    Can be triggered by:
    - OutputQualityMonitor (automatic)
    - API call (manual)
    - Threshold monitor (if too many safety reverts)
    
    When triggered:
    1. Disables all hooks immediately
    2. Clears all pending injections
    3. Reverts to RAG-only mode
    4. Logs the incident
    5. Alerts (if alerting is configured)
    """

    def __init__(self, hook_manager, bus, migration_manager):
        self._hook_manager = hook_manager
        self._bus = bus
        self._migration = migration_manager
        self._triggered = False
        self._trigger_history: List[Dict] = []

    def trigger(self, reason: str) -> None:
        """Trigger the kill switch."""
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        
        # Disable all hooks
        self._hook_manager.disable()
        
        # Clear bus
        self._bus.clear()
        
        # Revert to RAG-only
        self._migration.set_mode("rag_only")
        
        self._triggered = True
        self._trigger_history.append({
            "reason": reason,
            "timestamp": time.time(),
        })

    def reset(self) -> None:
        """Reset the kill switch (re-enable intrinsic memory)."""
        logger.info("Kill switch reset — re-enabling intrinsic memory")
        self._hook_manager.enable()
        self._triggered = False

    @property
    def is_triggered(self) -> bool:
        return self._triggered
```

---

## Task 4: Interpretability Tools

### Sub-task 4.1: Injection Attribution

```python
class InjectionAttributor:
    """
    Determines which injected memory influenced which output tokens.
    
    Method: Counterfactual analysis
    1. Generate output WITH all injections (actual output)
    2. For each memory, generate output WITHOUT that memory
    3. Compare: tokens that changed → attributed to that memory
    
    This is expensive (N+1 forward passes for N memories) but provides
    ground-truth attribution. Can be run on-demand for debugging.
    """

    def __init__(self, backend, controller, bus):
        self._backend = backend
        self._controller = controller
        self._bus = bus

    async def attribute(
        self,
        query: str,
        injected_memories: List[Dict],
        actual_output: str,
    ) -> Dict[str, Dict]:
        """
        For each memory, compute its influence on the output.
        
        Returns: {memory_id: {tokens_influenced: [...], influence_score: float}}
        """
        attributions = {}
        
        actual_tokens = actual_output.split()
        
        for memory in injected_memories:
            # Generate WITHOUT this memory
            memories_without = [m for m in injected_memories if m["memory_id"] != memory["memory_id"]]
            
            # Run retrieval with reduced set
            counterfactual_output = await self._generate_without(query, memories_without)
            counterfactual_tokens = counterfactual_output.split()
            
            # Compare
            influenced_tokens = []
            for i, (actual, cf) in enumerate(zip(actual_tokens, counterfactual_tokens)):
                if actual != cf:
                    influenced_tokens.append({"position": i, "actual": actual, "counterfactual": cf})
            
            # Compute influence score
            total_tokens = max(len(actual_tokens), len(counterfactual_tokens), 1)
            influence_score = len(influenced_tokens) / total_tokens
            
            attributions[memory["memory_id"]] = {
                "tokens_influenced": influenced_tokens[:20],  # Top 20
                "influence_score": influence_score,
                "memory_text": memory.get("source_text", "")[:100],
            }
        
        return attributions

    async def _generate_without(self, query: str, memories: List[Dict]) -> str:
        """Generate output with a reduced memory set."""
        # Temporarily reconfigure controller with reduced memories
        # This is a simplified version; real implementation would
        # selectively disable specific injections
        return ""  # Placeholder
```

---

## Task 5: Production Hardening Checklist

### Sub-task 5.1: Critical Items

```python
"""
Production Hardening Checklist for Intrinsic Memory System

=== Resource Management ===
☐ GPU memory limits enforced (L1 cache budget)
☐ CPU memory limits enforced (L2 cache budget)
☐ Disk space monitoring for L3 cache and steering vector cache
☐ Connection pooling for concurrent encoding requests
☐ Graceful shutdown: remove all hooks, flush cache, save state

=== Error Handling ===
☐ Backend initialization failure → fall back to RAG-only
☐ Hook registration failure → log and continue without that hook
☐ Encoding failure → skip memory, use existing representation if cached
☐ KV injection failure → fall back to activation or logit
☐ NaN/Inf in hidden states → safety guard reverts, logs metric
☐ OOM during encoding → reduce batch size, retry, or skip

=== Concurrency ===
☐ Thread-safe hook registration/removal
☐ Thread-safe cache access (L1/L2/L3)
☐ Atomic adapter loading/unloading
☐ No race conditions in bus dispatch

=== Configuration ===
☐ All thresholds configurable via YAML
☐ Runtime reconfiguration via API (no restart needed)
☐ Feature flags for each interface
☐ Default-safe configuration (intrinsic disabled by default)

=== Monitoring ===
☐ Prometheus metrics exported at /metrics
☐ Health check includes intrinsic system status
☐ Alert rules for safety reverts, kill switch triggers
☐ Dashboard: injection rate, cache hit rate, latency, quality

=== Documentation ===
☐ Architecture overview with data flow diagram
☐ Configuration reference (all YAML options)
☐ API reference (new /intrinsic endpoints)
☐ Troubleshooting guide (common issues and solutions)
☐ Benchmark results and interpretation guide
☐ Migration guide (RAG → Intrinsic step by step)

=== Security ===
☐ KV-cache injection doesn't leak sensitive data (shadow in the cache)
☐ Steering vectors can't be reverse-engineered to recover source text
☐ API authentication for /intrinsic endpoints
☐ Rate limiting on /intrinsic/encode (expensive operation)
"""
```

---

## Acceptance Criteria

1. Prometheus metrics cover all injection, latency, cache, and safety events
2. Benchmark suite produces comparative report (RAG vs. Intrinsic) with scores
3. `OutputQualityMonitor` detects perplexity spikes and triggers appropriate action
4. `EmergencyKillSwitch` disables all injections within 1ms of trigger
5. `InjectionAttributor` identifies which memory influenced which output tokens
6. All production hardening checklist items are implemented
7. Grafana dashboard template is provided for monitoring
8. Benchmark shows measurable improvement over RAG baseline on factual recall
9. Zero safety regressions: model output quality with intrinsic >= without
10. Documentation covers architecture, configuration, API, and troubleshooting

## Estimated Effort
- **Duration:** 3-4 weeks
- **Complexity:** Medium-High (breadth of concerns)
- **Risk:** Low (observability and safety are additive, not disruptive)

## Testing Strategy
1. Metrics test: verify all Prometheus metrics are registered and updated
2. Benchmark test: run standard suite and verify report generation
3. Quality monitor test: inject bad vectors, verify detection
4. Kill switch test: trigger and verify all injections cease within 1ms
5. Attribution test: verify counterfactual analysis produces meaningful results
6. Load test: verify system handles 100 concurrent injection requests
7. Endurance test: 24-hour run with continuous injection/eviction cycles
