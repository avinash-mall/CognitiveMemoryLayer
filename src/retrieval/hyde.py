"""HyDE (Hypothetical Document Embedding) for cognitive memory queries.

For LoCoMo-Plus queries where the cue and trigger have low semantic overlap,
HyDE generates a hypothetical memory that *would* answer the question, then
searches with that hypothetical document.  This bridges the semantic gap
between the query and the stored memories.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..utils.llm import LLMClient

logger = structlog.get_logger(__name__)

_HYDE_PROMPT = """Given this query from a user in a conversation:
"{query}"

Imagine you have perfect memory of all past conversations with this user.
Write what a relevant past memory entry would look like that would help
answer this query. Focus on implicit constraints, goals, preferences,
emotional states, or behavioral patterns the user may have expressed earlier.

Write 1-2 sentences as if you are recalling the memory. Do NOT answer the
question — just describe what the relevant memory would contain.

Hypothetical memory:"""


_COGNITIVE_CUE_PROMPT = """A person just made this remark in a conversation:
"{query}"

This remark may be implicitly connected to a past personal experience, habit,
hard-learned lesson, boundary, fear, allergy, goal, or value they mentioned weeks ago
using completely different words.

Write 3 distinct, concrete 1-sentence past statements the speaker might have said
earlier that explain their current remark (focus on causal events, personal rules,
past scares, or boundaries — avoid repeating the exact words of the remark).
Return exactly 3 lines, no numbering or bullets."""


_BRIDGE_QUERY_PROMPT = """Question: "{query}"

Retrieved memories so far:
{snippets}

If these memories already contain the complete, specific answer (including exact entity names, counts, or items), reply ONLY with: NONE
If they mention an unresolved reference (e.g., "my home country", "a friend", "that event", "the painting") OR only partially answer a multi-part/count question ("how many times", "what types"), write ONE short search query (4 to 12 words) to retrieve the missing fact or additional occurrences. No explanation.

Follow-up query or NONE:"""

_COGNITIVE_SELECT_PROMPT = """Current speaker remark:
"{query}"

Candidate past memories (numbered):
{candidates}

Which 1 to 3 numbered memories above have the strongest hidden causal, habit, boundary, fear, allergy, goal, or past-experience connection to the speaker's current remark (even if they use completely different words)?
Reply ONLY with comma-separated numbers in order of best match (e.g. `14, 3`), or `NONE` if none are relevant."""

_MULTI_ANGLE_COGNITIVE_PROMPT = """A speaker just made this remark in an ongoing conversation:
"{query}"

Generate 4 distinct, short memory search queries (4 to 12 words each) to retrieve an EARLIER past conversation turn that implicitly explains or motivates this remark across a semantic gap:
1. A past mishap, accident, injury, illness, scare, or loss that could have caused this behavior.
2. Specific physical objects, tools, health/financial/home items, or places tied to that root cause (avoid repeating words from the remark).
3. A personal rule, boundary, habit change, goal, or value the speaker may have adopted earlier.
4. A realistic 1-sentence first-person statement the speaker might have said weeks ago.

Return ONLY the 4 search queries, one per line, no numbering or bullets."""

_MULTI_ANGLE_FACTUAL_PROMPT = """Question about past conversations:
"{query}"

Generate 3 distinct, targeted search queries (4 to 12 words each) to retrieve ALL relevant evidence turns (including indirect mentions and shared photo captions):
1. Core entity + specific event/attribute keywords (strip question filler).
2. Synonyms, related objects, or photo-caption descriptions (e.g. "married" -> "wedding dress anniversary husband"; "books" -> "novel reading book cover photo").
3. Broader background, hobby, motivation, family, or second-hop bridge terms for the person(s) asked about.

Return ONLY the 3 search queries, one per line, no numbering or bullets."""

_JUDGED_DELIVERY_PROMPT = """Evaluate numbered conversation memories for answering or responding to the input below.

Input ({mode_label}):
"{query}"

Candidate Memories:
{candidates}

Instructions:
1. "selected": Pick ALL 1-based candidate numbers that provide direct facts, partial multi-hop pieces, shared-photo details, persona context (for Would/Likely/common-sense questions), or hidden past causal/habit/boundary/scare experiences explaining a conversational remark. Prefer specific verbatim speaker turns over broad [Chapter Summary] paragraphs when both exist. For list/multi-hop questions, include EVERY candidate mentioning a distinct item/activity/book/place/date (up to 12 numbers, ordered most important first).
2. "grounding": Classify how well the Candidate Memories support answering the input:
   - "supported": Direct factual or episodic evidence exists in the candidates (including via nicknames like Mel=Melanie, timestamps, or photo descriptions), OR this is an open conversational remark with a relevant past experience.
   - "inferable": Exact conclusion is not stated verbatim, but clues, persona, breakups/family plans, goals, or events in the candidates allow inferring the answer (e.g. "relationship status" inferred from breakup + single parent; "Would/Likely" common-sense questions).
   - "speaker_swap": Input asks about a specific event/action/item of Person A, and the candidates clearly show ONLY a DIFFERENT person (Person B) did/had it with zero evidence for Person A.
   - "unsupported_premise": Input asks about a specific concrete event, object, trip, injury, or quote that is completely absent/unmentioned across all candidates (and you selected [] as relevant).
3. "swap_note": If grounding is "speaker_swap", briefly state who actually did it vs who was asked (else "").
4. "bridge_query": ONLY if a selected memory leaves an unresolved multi-hop reference (e.g. says "moved from my home country" without naming the country, or mentions a book/photo without its title), write ONE 4-8 word search query to find that missing fact; otherwise "".

Return ONLY valid JSON:
{{"selected": [1, 4], "grounding": "supported", "swap_note": "", "bridge_query": ""}}"""


class HyDEGenerator:
    """Generates hypothetical documents and query expansions for retrieval."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def expand_multi_angle_queries(
        self,
        query: str,
        *,
        is_cognitive: bool = False,
        routing_intent: str = "single_hop",
    ) -> list[str]:
        """Generate 3-4 orthogonal search angles via LLM_INTERNAL to bridge lexical & multi-hop gaps."""
        if not query or not query.strip():
            return []
        _ = routing_intent
        prompt = (
            _MULTI_ANGLE_COGNITIVE_PROMPT.format(query=query.strip())
            if is_cognitive
            else _MULTI_ANGLE_FACTUAL_PROMPT.format(query=query.strip())
        )
        max_angles = 4 if is_cognitive else 3
        temp = 0.4 if is_cognitive else 0.2
        try:
            raw = await self._llm.complete(prompt, temperature=temp, max_tokens=220)
            seen: set[str] = {query.strip().lower()}
            angles: list[str] = []
            for line in (raw or "").splitlines():
                cleaned = line.strip().lstrip("0123456789.-•*) ").strip().strip('"')
                if len(cleaned) < 6:
                    continue
                norm = " ".join(cleaned.lower().split())
                if norm in seen:
                    continue
                seen.add(norm)
                angles.append(cleaned)
                if len(angles) >= max_angles:
                    break
            return angles
        except Exception as exc:
            logger.warning("multi_angle_expansion_failed", error=str(exc))
            return []

    async def evaluate_candidates_and_grounding(
        self,
        query: str,
        candidates: list[str],
        *,
        is_cognitive: bool = False,
        routing_intent: str = "single_hop",
    ) -> dict[str, Any]:
        """Kumiho-inspired Stage-2 batched LLM candidate selector + reality-monitoring verdict."""
        import json
        import re

        if not query or not candidates:
            return {}
        capped = candidates[:36]
        numbered = "\n".join(f"{i + 1}. {c[:240]}" for i, c in enumerate(capped))
        mode_label = "conversational remark" if is_cognitive else f"{routing_intent} question"
        prompt = _JUDGED_DELIVERY_PROMPT.format(
            mode_label=mode_label,
            query=query.strip(),
            candidates=numbered,
        )
        try:
            data = await self._llm.complete_json(prompt, temperature=0.0, max_tokens=320)
            if not isinstance(data, dict):
                return {}
        except Exception:
            try:
                raw = await self._llm.complete(prompt, temperature=0.0, max_tokens=320)
                m = re.search(r"\{.*\}", raw or "", re.DOTALL)
                data = json.loads(m.group(0)) if m else {}
            except Exception as exc:
                logger.warning("judged_delivery_eval_failed", error=str(exc))
                return {}

        raw_sel = data.get("selected") or []
        selected: list[int] = []
        seen: set[int] = set()
        if isinstance(raw_sel, list):
            for item in raw_sel:
                try:
                    idx = int(item) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(capped) and idx not in seen:
                    seen.add(idx)
                    selected.append(idx)

        grounding = str(data.get("grounding") or "").strip().lower()
        if grounding not in {"supported", "inferable", "speaker_swap", "unsupported_premise"}:
            grounding = ""
        if is_cognitive and grounding in {"unsupported_premise", "speaker_swap"}:
            grounding = "inferable"

        swap_note = str(data.get("swap_note") or "").strip()[:200]
        bridge_q = str(data.get("bridge_query") or "").strip().strip('"')[:120]
        if bridge_q.upper() == "NONE" or len(bridge_q) < 5:
            bridge_q = ""

        return {
            "selected_indices": selected[:12],
            "grounding": grounding,
            "swap_note": swap_note,
            "bridge_query": bridge_q,
        }

    async def select_cognitive_cues(self, query: str, candidates: list[str]) -> list[int]:
        """Select top 1-3 0-based candidate indices with hidden cognitive connection to query."""
        import re

        if not query or not candidates:
            return []
        numbered = "\n".join(f"{i + 1}. {c[:180]}" for i, c in enumerate(candidates[:22]))
        try:
            response = await self._llm.complete(
                _COGNITIVE_SELECT_PROMPT.format(query=query.strip(), candidates=numbered),
                temperature=0.0,
                max_tokens=32,
            )
            txt = response.strip()
            if not txt or txt.upper().startswith("NONE"):
                return []
            nums = [int(n) - 1 for n in re.findall(r"\b(\d{1,2})\b", txt)]
            seen: set[int] = set()
            valid: list[int] = []
            for idx in nums:
                if 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    valid.append(idx)
            return valid[:3]
        except Exception as exc:
            logger.warning("cognitive_cue_select_failed", error=str(exc))
            return []

    async def generate_bridge_query(self, query: str, snippets: list[str]) -> str | None:
        """Generate a targeted 2nd-hop bridge search query when 1st-pass snippets leave a gap."""
        if not query or not snippets:
            return None
        joined = "\n".join(f"- {s[:160]}" for s in snippets[:12])
        try:
            response = await self._llm.complete(
                _BRIDGE_QUERY_PROMPT.format(query=query.strip(), snippets=joined),
                temperature=0.0,
                max_tokens=48,
            )
            cleaned = response.strip().splitlines()[0].strip().strip('"').strip("'")
            if not cleaned or cleaned.upper().startswith("NONE") or len(cleaned) < 6:
                return None
            return cleaned
        except Exception as exc:
            logger.warning("bridge_query_generation_failed", error=str(exc))
            return None

    async def generate_cognitive_cues(self, query: str) -> list[str]:
        """Generate 3 hypothetical past cue statements to bridge semantic disconnect."""
        if not query or not query.strip():
            return []
        try:
            response = await self._llm.complete(
                _COGNITIVE_CUE_PROMPT.format(query=query.strip()),
                temperature=0.6,
                max_tokens=240,
            )
            lines = [
                line.strip().lstrip("0123456789.-•* )")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 12
            ]
            return lines[:3]
        except Exception as exc:
            logger.warning("cognitive_hyde_generation_failed", error=str(exc))
            return []

    async def generate_hypothetical_memory(self, query: str) -> str | None:
        """Generate a hypothetical memory that would answer the query.

        Returns None if generation fails.
        """
        if not query or not query.strip():
            return None

        try:
            response = await self._llm.complete(
                _HYDE_PROMPT.format(query=query.strip()),
                temperature=0.7,
                max_tokens=200,
            )
            result = response.strip()
            if result and len(result) > 10:
                return result
        except Exception as exc:
            logger.warning("hyde_generation_failed", error=str(exc))

        return None
