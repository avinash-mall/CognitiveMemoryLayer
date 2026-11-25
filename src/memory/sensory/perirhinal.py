"""Perirhinal & Parahippocampal Multimodal Sensory Engram Stream.

Extracts dense visual/document/perceptual descriptors from attached media items
(images, screenshots, shared photos, diagrams) via the configured OpenAI-compatible
vision endpoint (e.g. local ``qwen3.8-27b`` with ``--mmproj``) or structured captions,
binding perceptual features into both Hippocampal episodes and Cortical Hyperedges.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any

from ...utils.logging_config import get_logger

logger = get_logger(__name__)

_VISION_PROMPT = (
    "Describe this image concisely in 1-2 factual sentences for episodic memory indexing: "
    "include key objects, people/activities, setting, colors, and any visible text."
)


def _to_data_url(item: dict[str, Any]) -> str | None:
    if item.get("url") and str(item["url"]).startswith(("http://", "https://", "data:")):
        return str(item["url"])
    if item.get("base64"):
        mime = str(item.get("mime_type") or "image/jpeg")
        raw_b64 = str(item["base64"]).strip()
        if raw_b64.startswith("data:"):
            return raw_b64
        return f"data:{mime};base64,{raw_b64}"
    if item.get("path"):
        p = Path(str(item["path"]))
        if p.is_file() and p.stat().st_size <= 15_000_000:
            mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            encoded = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return None


def _call_vision_endpoint_sync(
    base_url: str,
    api_key: str,
    model: str,
    data_url: str,
) -> str | None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 140,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }
        ],
    }
    if "qwen3" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
            return str(content).strip() or None
    return None


class PerirhinalSensoryStream:
    """Encodes multimodal media attachments into Perirhinal perceptual descriptors."""

    @classmethod
    async def extract_descriptors(
        cls,
        media: list[dict[str, Any]] | None,
        *,
        llm_client: Any | None = None,
    ) -> list[str]:
        if not media:
            return []

        is_mock = os.environ.get("EMBEDDING_INTERNAL__PROVIDER", "").strip().lower() == "mock"
        base_url = getattr(llm_client, "base_url", None) if llm_client is not None else None
        model = getattr(llm_client, "model", None) if llm_client is not None else None
        api_key = getattr(llm_client, "api_key", "") if llm_client is not None else ""

        descriptors: list[str] = []
        for item in media[:6]:
            if not isinstance(item, dict):
                continue
            explicit_desc = (
                item.get("description")
                or item.get("caption")
                or item.get("alt_text")
                or item.get("ocr_text")
            )
            vlm_desc: str | None = None
            data_url = _to_data_url(item)
            if data_url and base_url and model and not is_mock:
                try:
                    vlm_desc = await asyncio.to_thread(
                        _call_vision_endpoint_sync,
                        str(base_url),
                        str(api_key or ""),
                        str(model),
                        data_url,
                    )
                except Exception as exc:
                    logger.debug("perirhinal_vlm_fallback", extra={"error": str(exc)})

            combined_parts = [str(p).strip() for p in (vlm_desc, explicit_desc) if p]
            if not combined_parts and item.get("path"):
                combined_parts.append(f"Visual media file: {Path(str(item['path'])).name}")
            elif not combined_parts and item.get("url"):
                combined_parts.append(f"Visual media URL: {str(item['url'])[:120]}")

            if combined_parts:
                descriptors.append(" — ".join(dict.fromkeys(combined_parts))[:320])

        return descriptors

    @staticmethod
    def enrich_content_with_sensory(content: str, descriptors: list[str]) -> str:
        """Bind Perirhinal visual descriptors onto episodic content without losing verbatim text."""
        if not descriptors:
            return content
        joined = "; ".join(descriptors)
        return f"{content.rstrip()}\n[Shared Media / Visual Context: {joined}]"
