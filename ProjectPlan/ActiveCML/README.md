# ActiveCML — intrinsic memory roadmap

**Status: designed, not started.** None of this exists in `src/`. Every phase below was
specified in detail (10 documents, ~8,300 lines) and none was implemented; the specs were
compressed into this index so the intent survives without a wall of unbuilt pseudo-code.
The full text is in git history if a phase is ever picked up — start from the commit that
removed `Phase1_Foundation_ModelAccessLayer.md` and its siblings.

For what the system **actually does** today, see [Usage Documentation](../../docs/usage.md)
and [docs/STATE.md](../../docs/STATE.md).

## The idea

CML today is an *extrinsic* memory system: it retrieves memories as text and puts them in
the prompt. That costs context window, incurs O(n²) attention on every retrieved token, and
leaves the model free to ignore what it was given.

ActiveCML proposed making memory *intrinsic* — injecting it into the model's own computation
graph at four increasing depths:

| Depth | Interface | What it modifies | Works with |
|---|---|---|---|
| Shallowest | Logit | the output distribution before sampling | any provider (`logit_bias`) |
| Primary | Activation | the residual stream between layers | local models |
| Deepest non-weight | Synaptic | the attention KV-cache | local models |
| Deepest | Weight | the weights themselves, via LoRA | local models |

## Phases

**I-1 — Foundation: Model Access Layer.** The abstraction every later phase depends on: a
unified interface over model internals (hidden states, KV-cache, logits, optionally weights),
a forward-pass hook system, and model introspection. Deliberately implements no injection
strategy of its own. *Nothing downstream could start without it, and it was never begun.*

**I-2 — Logit Interface.** The safest and most portable mechanism, operating at the very end
of the forward pass. Two sub-interfaces: simple logit bias (boost specific tokens — the only
approach that works through API-only providers), and kNN-LM interpolation (blend the model's
parametric distribution with a non-parametric one from nearest-neighbour memory lookup,
requiring full logit access). Sequenced first because it is reversible and provider-agnostic.

**I-3 — Activation Interface.** The intended *primary* mechanism. Rests on the linear
representation hypothesis: high-level concepts correspond to directions in activation space,
so adding a steering vector to hidden states shifts the model's topic focus, tone, and
factual priors without touching weights. Deeper leverage than logit bias because it affects
every subsequent layer's computation rather than only the final distribution.

**I-4 — Synaptic Interface (KV-cache).** Pre-compute the KV pairs for memory content and
inject them into the attention cache, so the model behaves as though it had already read the
memory — without those tokens consuming context or attention cost. A noted side benefit: the
injected pairs are opaque vectors rather than readable tokens, so sensitive memory content is
not trivially recoverable by inspecting the cache.

**I-5 — Controller & Gating Unit.** The part that makes the above a system rather than a set
of capabilities: given a query and retrieved memories, decide which memories to inject,
through which interface, and at what strength — relevance gating, interface routing, strength
calibration.

**I-6 — Memory Encoding Pipeline.** Phases I-2..I-4 each need memories in a different latent
format, and deriving them independently duplicates work. This phase unifies encoding into a
single pass producing every required representation — a redesign of the hippocampal encoder.

**I-7 — Memory Hierarchy & Cache Management.** Latent representations need tiered storage:
steering vectors are small (~32 KB at 4096-dim fp16) but numerous, KV pairs are large. An
LMCache-inspired hierarchy keeping hot memories on GPU, warm in CPU RAM, cold on disk.

**I-8 — Weight Adaptation Interface (dynamic LoRA).** The deepest mechanism: load
task-specific LoRA adapters at runtime so the model temporarily *learns* from memory —
synaptic plasticity rather than inference-time steering. The Controller classifies the query
and selects adapters.

**I-9 — Integration & Migration.** Wire I-1..I-8 into the existing application lifecycle
while the REST API, memory stores, and background workers keep working unchanged. Explicitly
a backward-compatibility phase, not a rewrite.

**I-10 — Observability, Benchmarking & Hardening.** Make it production-shaped: visibility
into what the memory system is doing to the model, benchmarks against the RAG baseline,
safety guardrails.

## Known risks (from the phase risk register)

The four cross-cutting concerns that applied to the whole programme, worth re-reading before
starting any phase:

1. **Request concurrency and hook-state isolation** — forward-pass hooks are global to a
   model instance, so concurrent requests can contaminate each other's injections. This was
   assessed as the hardest problem in the design.
2. **Single point of failure in the model backend** — intrinsic injection couples CML to one
   serving stack, unlike today's provider-agnostic HTTP calls.
3. **GPU memory exhaustion** — latent representations compete with the model and KV-cache
   for VRAM.
4. **Latency accumulation** — each interface adds per-token or per-request overhead; the
   combined budget is what decides whether this beats prompt stuffing at all.

## Why it stalled

Intrinsic injection requires owning the serving stack. CML currently talks to an
OpenAI-compatible endpoint over HTTP (`LLM_INTERNAL__*`), which is what makes it portable
across vLLM, Ollama, and hosted APIs — and precisely what these phases would give up. The
extrinsic path also kept improving in the meantime (see the measured retrieval quality in
[docs/STATE.md](../../docs/STATE.md)). Reviving this is a strategic choice about owning
inference, not an incremental next step.
