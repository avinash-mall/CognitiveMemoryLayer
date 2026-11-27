"""Run full LoCoMo-Plus evaluation against CML using OpenRouter's Batch API.

Supports ``:batch`` models such as ``z-ai/glm-5.3-flash:batch`` (50% cheaper than
sync endpoints, inline JSON submission via ``POST /api/v1/batches``).

Pipeline:
1. Phase A: Ingest deduplicated conversations into the running CML server.
2. Phase B: Retrieve CML memory contexts locally, then submit QA prompts to
   OpenRouter ``/api/v1/batches`` and poll until completed.
3. Phase C: Submit LLM-as-Judge prompts to OpenRouter ``/api/v1/batches``, poll
   until completed, and write standard predictions / judged / summary JSONs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure repo & evaluation paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "locomo_plus"))

from task_eval.llm_as_judge import (
    _compute_summary,
    _parse_judge_response,
    get_judge_prompt,
    label_to_score,
)

from cml.eval.locomo import (
    QA_PROMPT,
    _build_conversation_groups,
    _cml_read,
    _extract_answer,
    _ingest_sample,
    format_cognitive_trigger_and_prompt,
    phase_a_ingestion,
)


def _or_request(
    method: str,
    path: str,
    api_key: str,
    body: dict | None = None,
    *,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_openrouter_batch(
    api_key: str,
    model: str,
    requests_list: list[dict],
    *,
    poll_interval: int = 20,
    chunk_size: int = 350,
    label: str = "batch",
) -> dict[str, str]:
    """Submit requests to OpenRouter /api/v1/batches (chunked to stay within pre-auth budget).

    Returns a mapping of ``custom_id -> assistant_content``.
    """
    all_outputs: dict[str, str] = {}
    total = len(requests_list)
    idx = 0
    cur_chunk_size = chunk_size

    while idx < total:
        chunk = requests_list[idx : idx + cur_chunk_size]
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "requests": chunk,
        }
        try:
            created = _or_request("POST", "/batches", api_key, payload)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 402 and cur_chunk_size > 100:
                cur_chunk_size = max(100, cur_chunk_size // 2)
                print(
                    f"[{label}] 402 pre-auth budget limit; reducing chunk_size to {cur_chunk_size}",
                    flush=True,
                )
                continue
            raise RuntimeError(f"Batch create HTTP {exc.code}: {err_body}") from exc

        batch_id = created["id"]
        print(
            f"[{label}] Submitted batch {batch_id} ({len(chunk)} reqs, {idx + len(chunk)}/{total})",
            flush=True,
        )

        # Poll until terminal status
        while True:
            time.sleep(poll_interval)
            st = _or_request("GET", f"/batches/{batch_id}", api_key)
            status = st.get("status")
            counts = st.get("request_counts", {})
            print(
                f"[{label}] {batch_id}: status={status} counts={counts}",
                flush=True,
            )
            if status in ("completed", "failed", "expired", "cancelled"):
                break

        if st.get("status") != "completed":
            raise RuntimeError(
                f"Batch {batch_id} ended with status={st.get('status')}: {st.get('error')}"
            )

        usage = st.get("usage") or {}
        print(
            f"[{label}] {batch_id} completed! cost=${usage.get('cost', 0):.6f} "
            f"tokens={usage.get('total_tokens', 0)}",
            flush=True,
        )

        for item in st.get("results") or []:
            cid = item.get("custom_id") or item.get("id")
            resp_body = (item.get("response") or {}).get("body") or {}
            choices = resp_body.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
            else:
                content = ""
            if cid is not None:
                all_outputs[str(cid)] = str(content)

        idx += len(chunk)

    return all_outputs


def _extra_body_for_model(model: str) -> dict:
    m = model.lower()
    if "glm-5" in m or "gpt-5" in m:
        return {"reasoning": {"effort": "low"}}
    if "gemini-2.5" in m:
        return {"reasoning": {"enabled": False}}
    if "qwen3" in m:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _run_openrouter_sync(
    api_key: str,
    model: str,
    requests_list: list[dict],
    *,
    workers: int = 24,
    checkpoint_file: Path | None = None,
    stats_out: dict | None = None,
    label: str = "sync",
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, str]:
    """Run chat completions concurrently against OpenRouter /v1/chat/completions."""
    all_outputs: dict[str, str] = {}
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            loaded = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                all_outputs.update({str(k): str(v) for k, v in loaded.items() if v})
        except (json.JSONDecodeError, OSError):
            pass

    pending = [r for r in requests_list if str(r["custom_id"]) not in all_outputs]
    total = len(requests_list)
    if not pending:
        print(f"[{label}] All {total} responses already cached in {checkpoint_file}", flush=True)
        return all_outputs

    print(
        f"[{label}] Running {len(pending)}/{total} requests against {model} ({workers} workers)...",
        flush=True,
    )
    lock = threading.Lock()
    done_count = len(all_outputs)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0

    def _call_one(req_item: dict) -> tuple[str, str, int, int, float]:
        cid = str(req_item["custom_id"])
        body = dict(req_item["body"])
        body["model"] = model
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                resp = _or_request("POST", "/chat/completions", api_key, body, base_url=base_url)
                choices = resp.get("choices") or []
                content = ""
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                usage = resp.get("usage") or {}
                return (
                    cid,
                    str(content),
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    float(usage.get("cost") or 0.0),
                )
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {exc.code}: {err_body[:300]}")
                if exc.code == 402:
                    cur_mt = int(body.get("max_tokens") or 200)
                    if cur_mt > 80:
                        body["max_tokens"] = max(80, cur_mt // 2)
                    if "in_flight_budget_exhausted" in err_body or cur_mt > 80 or attempt < 4:
                        time.sleep(min(25, 2.0 * (2**attempt)))
                        continue
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(30, 2.0 * (2**attempt)))
                    continue
                raise last_err from exc
            except Exception as exc:
                last_err = exc
                time.sleep(min(20, 1.5 * (2**attempt)))
        raise RuntimeError(f"Failed {cid} after retries: {last_err}")

    def _save_ckpt() -> None:
        if checkpoint_file is not None:
            try:
                tmp_path = checkpoint_file.with_suffix(checkpoint_file.suffix + ".tmp")
                tmp_path.write_text(json.dumps(all_outputs, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp_path, checkpoint_file)
            except OSError:
                pass

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_call_one, r) for r in pending]
            for fut in as_completed(futures):
                cid, content, pt, ct, cost = fut.result()
                with lock:
                    if content:
                        all_outputs[cid] = content
                    done_count += 1
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    total_cost_usd += cost
                    if done_count % 25 == 0 or done_count == total:
                        _save_ckpt()
                    if done_count % 100 == 0 or done_count == total:
                        print(
                            f"[{label}] progress: {done_count}/{total} "
                            f"(prompt_tok={total_prompt_tokens:,} comp_tok={total_completion_tokens:,} "
                            f"cost=${total_cost_usd:.4f})",
                            flush=True,
                        )
    finally:
        _save_ckpt()
        if stats_out is not None:
            stats_out["prompt_tokens"] = stats_out.get("prompt_tokens", 0) + total_prompt_tokens
            stats_out["completion_tokens"] = (
                stats_out.get("completion_tokens", 0) + total_completion_tokens
            )
            stats_out["cost_usd"] = stats_out.get("cost_usd", 0.0) + total_cost_usd

    return all_outputs


def _run_requests(
    api_key: str,
    model: str,
    requests_list: list[dict],
    *,
    poll_interval: int,
    chunk_size: int,
    sync_workers: int,
    checkpoint_file: Path | None,
    stats_out: dict | None = None,
    label: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, str]:
    if model.endswith(":batch"):
        return _run_openrouter_batch(
            api_key,
            model,
            requests_list,
            poll_interval=poll_interval,
            chunk_size=chunk_size,
            label=label,
        )
    return _run_openrouter_sync(
        api_key,
        model,
        requests_list,
        workers=sync_workers,
        checkpoint_file=checkpoint_file,
        stats_out=stats_out,
        label=label,
        base_url=base_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo-Plus Evaluator (OpenRouter Batch/Sync or Local OpenAI-Compatible)"
    )
    parser.add_argument("--unified-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cml-url", default="http://localhost:8000")
    parser.add_argument("--cml-api-key", default=os.getenv("AUTH__API_KEY", "test-key"))
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="z-ai/glm-5.3-flash:batch")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--tenant-prefix", default="glm53-batch-full")
    parser.add_argument("--ingestion-workers", type=int, default=4)
    parser.add_argument("--sync-workers", type=int, default=24)
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument(
        "--stratified-per-cat",
        type=int,
        default=0,
        help="Take first N samples per category (preserves original conversation tenant mapping)",
    )
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Exit after Phase A1 ingestion (and optional Phase A2 consolidation) so server flags can switch for Phase B+C",
    )
    parser.add_argument(
        "--consolidate-windowed",
        action="store_true",
        help="Run Phase A2 Windowed Micro-Sleep Consolidation across all ingested conversation tenants",
    )
    parser.add_argument(
        "--consolidation-workers",
        type=int,
        default=2,
        help="Concurrent tenant workers for Phase A2 Windowed Consolidation",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=3,
        help="Concurrent CML /memory/read workers during Phase B",
    )
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument(
        "--qa-max-tokens",
        type=int,
        default=512,
        help="max_tokens for Phase B QA calls (lower reduces OpenRouter pre-auth hold)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=384,
        help="max_tokens for Phase C Judge calls",
    )
    parser.add_argument(
        "--reuse-contexts-from",
        default="",
        help="Optional path to existing predictions JSON to reuse retrieved_context by orig_index",
    )
    args = parser.parse_args()
    judge_model = args.judge_model or args.model

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_EVAL__API_KEY")
        or ""
    )
    if not api_key and "openrouter.ai" in args.base_url:
        raise RuntimeError("Missing OPENROUTER_API_KEY / OPENAI_API_KEY")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.unified_file, encoding="utf-8") as f:
        all_samples = json.load(f)
    if args.limit_samples > 0:
        all_samples = all_samples[: args.limit_samples]

    sample_to_conv, unique_convs = _build_conversation_groups(all_samples)
    if args.stratified_per_cat > 0:
        cat_counts: dict[str, int] = {}
        selected_pairs: list[tuple[int, dict]] = []
        for idx, s in enumerate(all_samples):
            c = s.get("category", "")
            if cat_counts.get(c, 0) < args.stratified_per_cat:
                cat_counts[c] = cat_counts.get(c, 0) + 1
                selected_pairs.append((idx, s))
    else:
        selected_pairs = list(enumerate(all_samples))

    samples = [s for _, s in selected_pairs]
    print(
        f"Loaded {len(samples)} samples (from {len(all_samples)} total) in {args.unified_file}",
        flush=True,
    )

    needed_convs = sorted({sample_to_conv[oidx] for oidx, _ in selected_pairs})
    # Phase A: Ingest unique conversations into CML
    print(
        f"[Phase A] Dedup: {len(samples)} selected samples -> {len(needed_convs)} unique conversations "
        f"(out of {len(unique_convs)} total)",
        flush=True,
    )
    if not args.skip_ingestion:
        if args.stratified_per_cat > 0:
            ing_ckpt = out_dir / "locomo_ingestion_checkpoint.json"
            done_ing: set[int] = set()
            if ing_ckpt.exists():
                try:
                    done_ing = set(
                        json.loads(ing_ckpt.read_text(encoding="utf-8")).get(
                            "completed_indices", []
                        )
                    )
                except Exception:
                    done_ing = set()
            pending_ing = [cid for cid in needed_convs if cid not in done_ing]
            if pending_ing:
                print(
                    f"[Phase A] Ingesting {len(pending_ing)} stratified conversations ({args.ingestion_workers} workers)...",
                    flush=True,
                )
                ing_lock = threading.Lock()
                with ThreadPoolExecutor(max_workers=max(1, args.ingestion_workers)) as ex:
                    futs = {
                        ex.submit(
                            _ingest_sample,
                            args.cml_url,
                            args.cml_api_key,
                            cid,
                            all_samples[cid],
                            0.0,
                            None,
                            args.tenant_prefix,
                        ): cid
                        for cid in pending_ing
                    }
                    for fut in as_completed(futs):
                        cid_done = futs[fut]
                        fut.result()
                        with ing_lock:
                            done_ing.update(unique_convs.get(cid_done, [cid_done]))
                            ing_ckpt.write_text(
                                json.dumps(
                                    {
                                        "completed_indices": sorted(done_ing),
                                        "conversation_dedup": True,
                                    }
                                ),
                                encoding="utf-8",
                            )
        else:
            phase_a_ingestion(
                samples=all_samples,
                cml_url=args.cml_url,
                cml_api_key=args.cml_api_key,
                limit=None,
                ingestion_workers=args.ingestion_workers,
                checkpoint_file=out_dir / "locomo_ingestion_checkpoint.json",
                tenant_prefix=args.tenant_prefix,
            )

    # Phase A2: Windowed Micro-Sleep Consolidation (Hyperedges + Proustian Cues + Conway + DMN)
    if args.consolidate_windowed:
        cons_ckpt = out_dir / "locomo_consolidation_checkpoint.json"
        done_convs: set[int] = set()
        if cons_ckpt.exists():
            try:
                done_convs = {int(x) for x in json.loads(cons_ckpt.read_text(encoding="utf-8"))}
            except Exception:
                done_convs = set()

        # Sort conversations by sample reuse frequency descending so the 10 shared base
        # conversations warm the deterministic window cache before the 401 single-sample variants.
        conv_freq: dict[int, int] = {}
        for c_id in sample_to_conv.values():
            conv_freq[c_id] = conv_freq.get(c_id, 0) + 1
        target_convs = needed_convs if args.stratified_per_cat > 0 else list(unique_convs.keys())
        ordered_convs = sorted(target_convs, key=lambda cid: -conv_freq.get(cid, 0))
        pending_convs = [cid for cid in ordered_convs if cid not in done_convs]
        print(
            f"[Phase A2] Windowed Micro-Sleep Consolidation: {len(pending_convs)}/{len(ordered_convs)} conversations pending...",
            flush=True,
        )

        def _consolidate_one(cid: int) -> tuple[int, dict]:
            tid = f"{args.tenant_prefix}-{cid}"
            url = f"{args.cml_url.rstrip('/')}/api/v1/memory/consolidate_windowed"
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": args.cml_api_key,
                "X-Tenant-ID": tid,
            }
            payload = json.dumps({"use_llm": True, "max_episodes": 2500}).encode("utf-8")
            for attempt in range(5):
                try:
                    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=1800) as resp:
                        return cid, json.loads(resp.read().decode("utf-8"))
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f"Consolidation failed for {tid}: {exc}") from exc
                    time.sleep(2.0 * (attempt + 1))
            return cid, {}

        # Run high-frequency base conversations first (warming shared windows), then variants concurrently
        base_convs = [cid for cid in pending_convs if conv_freq.get(cid, 0) > 1]
        variant_convs = [cid for cid in pending_convs if conv_freq.get(cid, 0) <= 1]
        cons_lock = threading.Lock()

        for b_cid in base_convs:
            _, st = _consolidate_one(b_cid)
            done_convs.add(b_cid)
            cons_ckpt.write_text(json.dumps(sorted(done_convs)), encoding="utf-8")
            print(
                f"  [Phase A2 base] conv {b_cid} consolidated ({len(done_convs)}/{len(ordered_convs)}): {st}",
                flush=True,
            )

        if variant_convs:
            with ThreadPoolExecutor(max_workers=max(1, args.consolidation_workers)) as ex:
                futs = [ex.submit(_consolidate_one, cid) for cid in variant_convs]
                for fut in as_completed(futs):
                    v_cid, _st = fut.result()
                    with cons_lock:
                        done_convs.add(v_cid)
                        if len(done_convs) % 25 == 0 or len(done_convs) == len(ordered_convs):
                            cons_ckpt.write_text(json.dumps(sorted(done_convs)), encoding="utf-8")
                            print(
                                f"  [Phase A2 progress] {len(done_convs)}/{len(ordered_convs)} consolidated",
                                flush=True,
                            )
            cons_ckpt.write_text(json.dumps(sorted(done_convs)), encoding="utf-8")

    if args.ingest_only:
        print("[Phase A] --ingest-only complete; exiting cleanly before Phase B.", flush=True)
        return

    usage_file = out_dir / "usage_stats.json"
    usage_stats: dict = {"qa": {}, "judge": {}}
    if usage_file.exists():
        try:
            usage_stats = json.loads(usage_file.read_text(encoding="utf-8"))
        except Exception:
            usage_stats = {"qa": {}, "judge": {}}

    # Phase B: Fetch CML contexts locally & submit QA
    pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
    loaded_complete_preds = False
    records: list[dict] = []
    if pred_file.exists():
        existing_preds = json.loads(pred_file.read_text(encoding="utf-8"))
        if len(existing_preds) == len(samples) and all(
            not (r.get("prediction") or "").startswith("[Error:") for r in existing_preds
        ):
            records = existing_preds
            loaded_complete_preds = True
            print(
                f"[Phase B] Loaded {len(records)} complete predictions from {pred_file}",
                flush=True,
            )

    if not loaded_complete_preds:
        reused_ctx_map: dict[int, str] = {}
        read_ctx_ckpt = out_dir / "cml_read_contexts_checkpoint.json"
        if args.reuse_contexts_from:
            src_path = Path(args.reuse_contexts_from)
            src_rows = json.loads(src_path.read_text(encoding="utf-8"))
            for row in src_rows:
                if "orig_index" in row and "retrieved_context" in row:
                    reused_ctx_map[int(row["orig_index"])] = str(row["retrieved_context"] or "")
            print(
                f"[Phase B] Reusing {len(reused_ctx_map)} retrieved contexts from {src_path}",
                flush=True,
            )
        elif read_ctx_ckpt.exists():
            try:
                loaded_ctx = json.loads(read_ctx_ckpt.read_text(encoding="utf-8"))
                if isinstance(loaded_ctx, dict):
                    reused_ctx_map = {int(k): str(v) for k, v in loaded_ctx.items() if v}
                    print(
                        f"[Phase B] Resuming {len(reused_ctx_map)}/{len(samples)} cached CML contexts from {read_ctx_ckpt}",
                        flush=True,
                    )
            except Exception:
                reused_ctx_map = {}

        if len(reused_ctx_map) < len(samples):
            print(
                f"[Phase B] Reading CML contexts for {len(samples) - len(reused_ctx_map)}/{len(samples)} samples ({args.read_workers} workers)...",
                flush=True,
            )

        def _read_sample(pair_idx_and_item: tuple[int, tuple[int, dict]]) -> tuple[int, dict, dict]:
            i, (orig_idx, sample) = pair_idx_and_item
            tenant_id = f"{args.tenant_prefix}-{sample_to_conv.get(orig_idx, orig_idx)}"
            trigger = (sample.get("trigger") or "").strip()
            category = sample.get("category", "")
            read_query = (
                format_cognitive_trigger_and_prompt(sample, "")[0]
                if category == "Cognitive"
                else trigger
            )
            if orig_idx in reused_ctx_map:
                llm_context = reused_ctx_map[orig_idx]
            else:
                llm_context = _cml_read(
                    args.cml_url, args.cml_api_key, tenant_id, read_query, args.max_results
                )
            effective_context = (
                llm_context
                if len(llm_context or "") > 30
                else "(No relevant memories found for this query.)"
            )
            if category == "Cognitive":
                _, prompt = format_cognitive_trigger_and_prompt(sample, effective_context)
            else:
                prompt = effective_context + "\n\n" + QA_PROMPT.format(trigger)
            qa_body: dict = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.qa_max_tokens,
                "temperature": 0,
            }
            qa_body.update(_extra_body_for_model(args.model))
            req_obj = {"custom_id": f"qa-{orig_idx}", "body": qa_body}
            meta_obj = {
                "orig_index": orig_idx,
                "question_input": trigger,
                "evidence": sample.get("cue_dialogue", sample.get("evidence", "")),
                "category": category,
                "ground_truth": sample.get("answer", sample.get("ground_truth", "")),
                "context_length": len(llm_context or ""),
                "retrieved_context": llm_context or "",
            }
            return i, req_obj, meta_obj

        indexed_results: dict[int, tuple[dict, dict]] = {}
        if reused_ctx_map and len(reused_ctx_map) >= len(samples):
            for item in enumerate(selected_pairs):
                idx_i, req_obj, meta_obj = _read_sample(item)
                indexed_results[idx_i] = (req_obj, meta_obj)
        else:
            read_done = 0
            read_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=max(1, args.read_workers)) as ex:
                futs = [ex.submit(_read_sample, item) for item in enumerate(selected_pairs)]
                for fut in as_completed(futs):
                    idx_i, req_obj, meta_obj = fut.result()
                    with read_lock:
                        indexed_results[idx_i] = (req_obj, meta_obj)
                        reused_ctx_map[int(meta_obj["orig_index"])] = str(
                            meta_obj.get("retrieved_context") or ""
                        )
                        read_done += 1
                        if read_done % 60 == 0 or read_done == len(samples):
                            try:
                                tmp_rc = read_ctx_ckpt.with_suffix(".json.tmp")
                                tmp_rc.write_text(
                                    json.dumps(
                                        {str(k): v for k, v in reused_ctx_map.items()},
                                        ensure_ascii=False,
                                    ),
                                    encoding="utf-8",
                                )
                                os.replace(tmp_rc, read_ctx_ckpt)
                            except OSError:
                                pass
                            print(f"  CML read progress: {read_done}/{len(samples)}", flush=True)

        qa_requests = [indexed_results[i][0] for i in range(len(samples))]
        meta_rows = [indexed_results[i][1] for i in range(len(samples))]

        qa_stats = dict(usage_stats.get("qa") or {})
        try:
            qa_outputs = _run_requests(
                api_key,
                args.model,
                qa_requests,
                poll_interval=args.poll_interval,
                chunk_size=args.chunk_size,
                sync_workers=args.sync_workers,
                checkpoint_file=out_dir / "qa_sync_checkpoint.json",
                stats_out=qa_stats,
                label="Phase B QA",
                base_url=args.base_url,
            )
        finally:
            usage_stats["qa"] = qa_stats
            usage_stats["total_cost_usd"] = float(qa_stats.get("cost_usd", 0.0)) + float(
                (usage_stats.get("judge") or {}).get("cost_usd", 0.0)
            )
            usage_file.write_text(json.dumps(usage_stats, indent=2), encoding="utf-8")

        records = []
        for i, meta in enumerate(meta_rows):
            oidx = meta.get("orig_index", i)
            raw_pred = qa_outputs.get(f"qa-{oidx}", qa_outputs.get(f"qa-{i}", ""))
            clean_pred = _extract_answer(raw_pred) if raw_pred else "[Error: empty response]"
            rec = dict(meta)
            rec["prediction"] = clean_pred
            rec["model"] = f"{args.model}_cml"
            records.append(rec)

        pred_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Phase B] Saved {len(records)} predictions to {pred_file}", flush=True)

    # Phase C: Submit LLM-as-Judge
    judge_requests_by_id: dict[str, dict] = {}
    for i, rec in enumerate(records):
        pred = rec.get("prediction", "")
        if pred.startswith("[Error:") or pred.startswith("[API Error:"):
            continue
        prompt = get_judge_prompt(
            rec.get("category") or "default",
            rec.get("evidence", ""),
            pred,
            rec.get("ground_truth") or "",
        )
        judge_body: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.judge_max_tokens,
            "temperature": 0,
        }
        judge_body.update(_extra_body_for_model(judge_model))
        oidx = rec.get("orig_index", i)
        cid = f"judge-{oidx}"
        judge_requests_by_id[cid] = {
            "custom_id": cid,
            "body": judge_body,
        }

    judge_ckpt = out_dir / "judge_sync_checkpoint.json"
    judge_stats = dict(usage_stats.get("judge") or {})
    judge_outputs: dict[str, str] = {}
    for judge_pass in range(3):
        req_batch = list(judge_requests_by_id.values())
        try:
            judge_outputs = _run_requests(
                api_key,
                judge_model,
                req_batch,
                poll_interval=args.poll_interval,
                chunk_size=args.chunk_size,
                sync_workers=args.sync_workers,
                checkpoint_file=judge_ckpt,
                stats_out=judge_stats,
                label=f"Phase C Judge (pass {judge_pass + 1})",
                base_url=args.base_url,
            )
        finally:
            usage_stats["judge"] = judge_stats
            usage_stats["total_cost_usd"] = float(
                (usage_stats.get("qa") or {}).get("cost_usd", 0.0)
            ) + float(judge_stats.get("cost_usd", 0.0))
            usage_file.write_text(json.dumps(usage_stats, indent=2), encoding="utf-8")

        failed_cids = [
            cid
            for cid in judge_requests_by_id
            if _parse_judge_response(judge_outputs.get(cid, ""))[0] == "_parse_failed"
        ]
        if not failed_cids or judge_pass == 2 or judge_model.endswith(":batch"):
            break
        print(
            f"[Phase C] Retrying {len(failed_cids)} unparseable judge responses (pass {judge_pass + 2})...",
            flush=True,
        )
        for cid in failed_cids:
            judge_outputs.pop(cid, None)
        if judge_ckpt.exists():
            judge_ckpt.write_text(json.dumps(judge_outputs, ensure_ascii=False), encoding="utf-8")

    judged: list[dict] = []
    for i, rec in enumerate(records):
        r = dict(rec)
        oidx = rec.get("orig_index", i)
        raw = judge_outputs.get(f"judge-{oidx}", judge_outputs.get(f"judge-{i}"))
        if raw is None:
            r["judge_label"] = "_parse_failed"
            r["judge_reason"] = "Skipped or missing batch response"
            r["judge_score"] = None
        else:
            label, reason = _parse_judge_response(raw)
            r["judge_label"] = label
            r["judge_reason"] = reason
            r["judge_score"] = label_to_score(label)
        judged.append(r)

    judged_file = out_dir / "locomo_plus_qa_cml_judged.json"
    summary_file = out_dir / "locomo_plus_qa_cml_judge_summary.json"
    judged_file.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _compute_summary(judged)
    summary["usage_stats"] = usage_stats
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"FULL BATCH EVAL SUMMARY ({args.model}, judge={judge_model})", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
