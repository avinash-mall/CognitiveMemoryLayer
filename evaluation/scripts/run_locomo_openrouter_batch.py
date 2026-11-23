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
    COGNITIVE_PROMPT,
    QA_PROMPT,
    _build_conversation_groups,
    _cml_read,
    _extract_answer,
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

    def _call_one(req_item: dict) -> tuple[str, str, int, int]:
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
                )
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {exc.code}: {err_body[:300]}")
                if exc.code in (429, 500, 502, 503, 504) or (
                    exc.code == 402 and "in_flight_budget_exhausted" in err_body
                ):
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
                checkpoint_file.write_text(
                    json.dumps(all_outputs, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                pass

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_call_one, r) for r in pending]
        for fut in as_completed(futures):
            cid, content, pt, ct = fut.result()
            with lock:
                all_outputs[cid] = content
                done_count += 1
                total_prompt_tokens += pt
                total_completion_tokens += ct
                if done_count % 100 == 0 or done_count == total:
                    print(
                        f"[{label}] progress: {done_count}/{total} "
                        f"(prompt_tok={total_prompt_tokens:,} comp_tok={total_completion_tokens:,})",
                        flush=True,
                    )
                    _save_ckpt()

    _save_ckpt()
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
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--poll-interval", type=int, default=20)
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

    # Phase A: Ingest unique conversations into CML
    print(
        f"[Phase A] Dedup: {len(all_samples)} samples -> {len(unique_convs)} unique conversations",
        flush=True,
    )
    if not args.skip_ingestion:
        phase_a_ingestion(
            samples=all_samples,
            cml_url=args.cml_url,
            cml_api_key=args.cml_api_key,
            limit=None,
            ingestion_workers=args.ingestion_workers,
            checkpoint_file=out_dir / "locomo_ingestion_checkpoint.json",
            tenant_prefix=args.tenant_prefix,
        )

    # Phase B: Fetch CML contexts locally & submit QA
    pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
    if pred_file.exists():
        records = json.loads(pred_file.read_text(encoding="utf-8"))
        print(f"[Phase B] Loaded {len(records)} existing predictions from {pred_file}", flush=True)
    else:
        print(f"[Phase B] Reading CML contexts for {len(samples)} samples...", flush=True)
        qa_requests: list[dict] = []
        meta_rows: list[dict] = []
        for i, (orig_idx, sample) in enumerate(selected_pairs):
            tenant_id = f"{args.tenant_prefix}-{sample_to_conv.get(orig_idx, orig_idx)}"
            trigger = (sample.get("trigger") or "").strip()
            category = sample.get("category", "")
            llm_context = _cml_read(
                args.cml_url, args.cml_api_key, tenant_id, trigger, args.max_results
            )
            effective_context = (
                llm_context
                if len(llm_context or "") > 30
                else "(No relevant memories found for this query.)"
            )
            if category == "Cognitive":
                prompt = effective_context + "\n\n" + COGNITIVE_PROMPT.format(trigger)
            else:
                prompt = effective_context + "\n\n" + QA_PROMPT.format(trigger)
            qa_body: dict = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0,
            }
            qa_body.update(_extra_body_for_model(args.model))
            qa_requests.append(
                {
                    "custom_id": f"qa-{i}",
                    "body": qa_body,
                }
            )
            meta_rows.append(
                {
                    "orig_index": orig_idx,
                    "question_input": trigger,
                    "evidence": sample.get("cue_dialogue", sample.get("evidence", "")),
                    "category": category,
                    "ground_truth": sample.get("answer", sample.get("ground_truth", "")),
                    "context_length": len(llm_context or ""),
                    "retrieved_context": llm_context or "",
                }
            )
            if (i + 1) % 60 == 0 or (i + 1) == len(samples):
                print(f"  CML read progress: {i + 1}/{len(samples)}", flush=True)

        qa_outputs = _run_requests(
            api_key,
            args.model,
            qa_requests,
            poll_interval=args.poll_interval,
            chunk_size=args.chunk_size,
            sync_workers=args.sync_workers,
            checkpoint_file=out_dir / "qa_sync_checkpoint.json",
            label="Phase B QA",
            base_url=args.base_url,
        )

        records = []
        for i, meta in enumerate(meta_rows):
            raw_pred = qa_outputs.get(f"qa-{i}", "")
            clean_pred = _extract_answer(raw_pred) if raw_pred else "[Error: empty response]"
            rec = dict(meta)
            rec["prediction"] = clean_pred
            rec["model"] = f"{args.model}_cml"
            records.append(rec)

        pred_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Phase B] Saved {len(records)} predictions to {pred_file}", flush=True)

    # Phase C: Submit LLM-as-Judge
    judge_requests: list[dict] = []
    valid_indices: list[int] = []
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
            "max_tokens": 384,
            "temperature": 0,
        }
        judge_body.update(_extra_body_for_model(judge_model))
        judge_requests.append(
            {
                "custom_id": f"judge-{i}",
                "body": judge_body,
            }
        )
        valid_indices.append(i)

    judge_outputs = _run_requests(
        api_key,
        judge_model,
        judge_requests,
        poll_interval=args.poll_interval,
        chunk_size=args.chunk_size,
        sync_workers=args.sync_workers,
        checkpoint_file=out_dir / "judge_sync_checkpoint.json",
        label="Phase C Judge",
        base_url=args.base_url,
    )

    judged: list[dict] = []
    for i, rec in enumerate(records):
        r = dict(rec)
        raw = judge_outputs.get(f"judge-{i}")
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
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"FULL BATCH EVAL SUMMARY ({args.model})", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
