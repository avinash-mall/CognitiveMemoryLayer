"""Phase I-2: Logit Interface (Logit-Bias Synthesis & Superseded-Belief Suppression).

Operates at the output distribution before sampling. Works with any OpenAI-compatible
provider (vLLM, OpenAI, OpenRouter, SGLang) as well as local logits processors.
- Boosts high-confidence retrieved entity, date, and constraint anchor tokens.
- Suppresses superseded/invalidated fact tokens and CA1 unverified presupposition
  tokens so the model does not hallucinate outdated or false premises.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{2,}\b")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "can",
        "just",
        "don",
        "now",
        "user",
        "said",
        "mentioned",
        "earlier",
        "prefers",
        "preference",
        "constraint",
        "fact",
    }
)


class LogitBiasSynthesizer:
    """Synthesizes OpenAI/vLLM-compatible token logit_bias maps from retrieved memories."""

    def __init__(self, encoding_name: str = "cl100k_base", max_tokens: int = 48) -> None:
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self._encoder: Any = None
        self._encoder_loaded = False

    def _get_encoder(self) -> Any:
        if not self._encoder_loaded:
            self._encoder_loaded = True
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding(self.encoding_name)
            except Exception:
                self._encoder = None
        return self._encoder

    def _encode_term(self, term: str) -> list[int]:
        enc = self._get_encoder()
        if enc is not None:
            try:
                # Encode with leading space (standard BPE word-initial token in GPT/Qwen/Llama)
                ids = enc.encode(f" {term}")
                return list(ids[:2]) if ids else []
            except Exception:
                pass
        # Deterministic fallback token ID hash in valid BPE range [1000, 99000]
        h = sum((i + 1) * ord(c) for i, c in enumerate(term)) % 98000 + 1000
        return [h]

    def synthesize(
        self,
        packet: Any,
        *,
        gate_strength: float = 0.7,
        unverified_presuppositions: list[str] | None = None,
        superseded_values: list[str] | None = None,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Return (logit_bias_map, boosted_terms, suppressed_terms)."""
        if gate_strength <= 0.05:
            return {}, [], []

        boost_candidates: dict[str, float] = {}
        suppress_candidates: dict[str, float] = {}

        # 1. Extract high-confidence anchor terms from facts, constraints, preferences, top episodes
        all_mems = (
            list(getattr(packet, "constraints", [])[:4])
            + list(getattr(packet, "facts", [])[:5])
            + list(getattr(packet, "preferences", [])[:4])
            + list(getattr(packet, "recent_episodes", [])[:4])
        )
        for mem in all_mems:
            rec = getattr(mem, "record", None)
            if rec is None:
                continue
            conf = float(getattr(rec, "confidence", 0.5) or 0.5)
            rel = float(getattr(mem, "relevance_score", 0.5) or 0.5)
            weight = min(3.5, round(gate_strength * (1.2 + 1.5 * conf * max(0.4, rel)), 2))

            # Explicit entities first
            for ent in getattr(rec, "entities", []) or []:
                ent_text = getattr(ent, "text", None) or getattr(ent, "normalized", None) or ""
                for w in _WORD_RE.findall(str(ent_text)):
                    if w.lower() not in _STOPWORDS and len(w) >= 3:
                        boost_candidates[w] = max(boost_candidates.get(w, 0.0), weight)

            # Capitalized proper nouns & numbers/dates in text
            text = str(getattr(rec, "text", "") or "")
            for w in _WORD_RE.findall(text):
                w_low = w.lower()
                if w_low in _STOPWORDS:
                    continue
                if w[0].isupper() or w.isdigit():
                    boost_candidates[w] = max(boost_candidates.get(w, 0.0), weight)

        # 2. Negative bias on superseded values & CA1 unverified presuppositions
        neg_weight = -round(min(4.5, 2.5 + 2.0 * gate_strength), 2)
        for term in (unverified_presuppositions or []) + (superseded_values or []):
            for w in _WORD_RE.findall(str(term)):
                if w.lower() not in _STOPWORDS and w not in boost_candidates:
                    suppress_candidates[w] = neg_weight

        logit_bias: dict[str, float] = {}
        boosted_terms: list[str] = []
        suppressed_terms: list[str] = []

        # Allocate up to 2/3 of budget to positive anchors and 1/3 to negative suppressions
        pos_budget = (self.max_tokens * 2) // 3
        for term, bias in sorted(boost_candidates.items(), key=lambda kv: kv[1], reverse=True):
            if len(logit_bias) >= pos_budget:
                break
            tids = self._encode_term(term)
            for tid in tids:
                if len(logit_bias) < pos_budget:
                    logit_bias[str(tid)] = bias
            if tids:
                boosted_terms.append(term)

        for term, bias in suppress_candidates.items():
            if len(logit_bias) >= self.max_tokens:
                break
            tids = self._encode_term(term)
            for tid in tids:
                tid_str = str(tid)
                if tid_str not in logit_bias and len(logit_bias) < self.max_tokens:
                    logit_bias[tid_str] = bias
            if tids:
                suppressed_terms.append(term)

        return logit_bias, boosted_terms[:15], suppressed_terms[:10]
