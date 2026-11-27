"""Memory packet builder for LLM consumption."""

import json
from datetime import datetime

from ..core.enums import MemoryType
from ..core.schemas import MemoryPacket, RetrievedMemory, source_label

# Fallback constants when config unavailable (BUG-02: avoid diluting constraints).
# Keep EPISODE_RELEVANCE_THRESHOLD in step with RetrievalSettings.episode_relevance_
# threshold — two defaults for one knob that disagree is worse than either value.
EPISODE_RELEVANCE_THRESHOLD = 0.4
MAX_EPISODES_WHEN_CONSTRAINTS = 18
MAX_EPISODES_DEFAULT = 22
MAX_CONSTRAINT_TOKENS = 400


class MemoryPacketBuilder:
    """Builds structured memory packets from retrieved memories."""

    def build(
        self,
        memories: list[RetrievedMemory],
        query: str,
        include_provenance: bool = True,
        llm_grounding: dict | None = None,
    ) -> MemoryPacket:
        """Build a memory packet from retrieved memories."""
        _ = include_provenance
        packet = MemoryPacket(query=query, ranked_memories=list(memories))
        for mem in memories:
            mem_type = mem.record.type
            if mem_type == MemoryType.SEMANTIC_FACT:
                packet.facts.append(mem)
            elif mem_type == MemoryType.PREFERENCE:
                packet.preferences.append(mem)
            elif mem_type == MemoryType.PROCEDURE:
                packet.procedures.append(mem)
            elif mem_type == MemoryType.CONSTRAINT:
                packet.constraints.append(mem)
            else:
                packet.recent_episodes.append(mem)
        packet.constraints, constraint_warnings = self._resolve_constraint_conflicts(
            packet.constraints
        )
        conflicts = self._detect_conflicts(packet)
        if conflicts:
            packet.warnings.extend(conflicts)
        if constraint_warnings:
            packet.warnings.extend(constraint_warnings)

        # CA1 Comparator & Orbitofrontal Reality Monitoring: use LLM_INTERNAL grounding
        # verdict when available (eliminating regex presupposition false positives) combined
        # with tightened Speaker-Attribution checking; fall back to CA1RealityMonitor when offline.
        try:
            from ..core.config import get_settings

            if getattr(get_settings().features, "reality_monitoring_enabled", True) and memories:
                from .reality_monitor import CA1RealityMonitor

                verdict = CA1RealityMonitor.inspect(query, memories)
                g_verdict = (
                    str((llm_grounding or {}).get("grounding") or "").strip().lower()
                    if llm_grounding
                    else ""
                )
                swap_warnings = [w for w in verdict.warnings if "Speaker-Attribution Alert" in w]
                if g_verdict:
                    if g_verdict == "speaker_swap":
                        if swap_warnings:
                            for w in swap_warnings:
                                if w not in packet.warnings:
                                    packet.warnings.append(w)
                        else:
                            snote = str((llm_grounding or {}).get("swap_note") or "").strip()
                            detail = f" ({snote})" if snote else ""
                            w_msg = (
                                f"Reality Monitoring (CA1 Speaker-Attribution Alert): Retrieved "
                                f"memories attribute this event/item to a DIFFERENT speaker{detail}. "
                                f"Do NOT substitute another person's details; if unrecorded for the "
                                f'asked person, reply ONLY: "I don\'t have information about that from our previous conversations."'
                            )
                            packet.warnings.append(w_msg)
                    elif g_verdict == "unsupported_premise":
                        for w in swap_warnings:
                            if w not in packet.warnings:
                                packet.warnings.append(w)
                        sel_cnt = int((llm_grounding or {}).get("selected_count") or 0)
                        if not swap_warnings and sel_cnt == 0:
                            w_unv = (
                                "Reality Monitoring (CA1 Evidence-Grounding Alert): Retrieved "
                                "memories do NOT confirm this specific premise, event, or item. "
                                "Do NOT guess or fabricate details; if genuinely unmentioned in the "
                                'context, reply ONLY: "I don\'t have information about that from our previous conversations."'
                            )
                            packet.warnings.append(w_unv)
                    elif g_verdict == "inferable":
                        for w in swap_warnings:
                            if w not in packet.warnings:
                                packet.warnings.append(w)
                    # When g_verdict == "supported", suppress noisy regex Presupposition Notes!
                else:
                    for w in verdict.warnings:
                        if w not in packet.warnings:
                            packet.warnings.append(w)
        except Exception:
            pass

        for mem in memories:
            if mem.record.confidence < 0.5:
                packet.open_questions.append(
                    f"Uncertain: {mem.record.text} (confidence: {mem.record.confidence:.2f})"
                )
        return packet

    def _resolve_constraint_conflicts(
        self,
        constraints: list[RetrievedMemory],
    ) -> tuple[list[RetrievedMemory], list[str]]:
        """Resolve retrieval-time supersession conflicts among constraints.

        Keeps the most recent surviving constraints and drops older constraints
        when a newer one supersedes them (same key).
        """
        if len(constraints) <= 1:
            return constraints, []

        def _ts(mem: RetrievedMemory) -> datetime:
            ts = mem.record.timestamp
            return ts if isinstance(ts, datetime) else datetime.min

        ordered = sorted(constraints, key=_ts, reverse=True)
        kept: list[RetrievedMemory] = []
        warnings: list[str] = []

        for candidate in ordered:
            cand_key = getattr(candidate.record, "key", None) or ""
            suppressed = False
            for newer in kept:
                newer_key = getattr(newer.record, "key", None) or ""
                if cand_key and newer_key and cand_key == newer_key:
                    warnings.append(
                        f"Suppressed older constraint for key '{cand_key}' in favor of newer statement."
                    )
                    suppressed = True
                    break

            if not suppressed:
                kept.append(candidate)

        return kept, warnings

    def _detect_conflicts(self, packet: MemoryPacket) -> list[str]:
        """Detect potential conflicts in retrieved memories."""
        conflicts: list[str] = []
        preference_values: dict = {}
        for pref in packet.preferences:
            key = getattr(pref.record, "key", None)
            if key in preference_values:
                if preference_values[key] != pref.record.text:
                    conflicts.append(
                        f"Conflicting preferences for {key}: "
                        f"'{preference_values[key]}' vs '{pref.record.text}'"
                    )
            elif key:
                preference_values[key] = pref.record.text
        fact_values: dict = {}
        for fact in packet.facts:
            key = getattr(fact.record, "key", None)
            if key and key in fact_values:
                if fact_values[key] != fact.record.text:
                    conflicts.append(f"Conflicting facts for {key}")
            elif key:
                fact_values[key] = fact.record.text
        return conflicts

    def to_llm_context(
        self,
        packet: MemoryPacket,
        max_tokens: int = 3000,
        format: str = "markdown",
    ) -> str:
        """Format packet for LLM context injection."""
        if format == "markdown":
            return self._format_markdown(packet, max_tokens)
        if format == "json":
            return self._format_json(packet, max_tokens)
        return packet.to_context_string(max_chars=max_tokens * 4)

    @staticmethod
    def _source_label(mem: RetrievedMemory) -> str:
        """Origin marker for a retrieved memory; see ``schemas.source_label``.

        Suppressed for the fact prong: ``semantic_facts`` has no provenance column,
        so ``_fact_to_record`` stamps every row AGENT_INFERRED as a placeholder.
        Rendering that would label every fact identically — no signal, and it would
        weaken constraint wording on the strength of a value nobody recorded.
        """
        if mem.retrieval_source == "facts":
            return ""
        return source_label(mem.record)

    @staticmethod
    def _constraint_provenance(mem: RetrievedMemory) -> str:
        """Extract compact provenance string from a constraint memory."""
        meta = mem.record.metadata or {}
        # Check for structured constraints in metadata
        constraints_meta = meta.get("constraints", [])
        if constraints_meta and isinstance(constraints_meta, list):
            first = constraints_meta[0] if constraints_meta else {}
            ctype = first.get("constraint_type", "")
            prov = first.get("provenance", [])
            prov_str = ", ".join(prov) if prov else ""
            label = f"[{ctype.title()}]" if ctype else ""
            src = f" (from {prov_str})" if prov_str else ""
            return f"{label}{src}"
        # Fallback: use source_turn_id from evidence refs
        turn_id = meta.get("source_turn_id", "")
        if turn_id:
            return f"(from {turn_id})"
        return ""

    def _format_markdown(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as markdown with constraint-first token budget.

        Constraints get a reserved budget so they are never truncated.
        Facts, preferences, and episodes share the remaining budget.
        """
        max_chars = max_tokens * 4
        main_header = "# Retrieved Memory Context\n\n"
        # There is deliberately no "weak evidence, consider refusing" note here. One was
        # written, then removed on measurement: across 25 queries per category on the
        # full2 corpus, no retrieval-score signal separates unanswerable questions from
        # answerable ones. Median top cosine 0.624 for adversarial against 0.638 for
        # answerable, mean-of-top-5 0.596 against 0.586 (pointing the wrong way), and
        # top1-minus-top5 margin 0.063 against 0.065 — all three deltas are noise. A
        # threshold on any of them fires on correct answers as often as on absent ones.
        # `packet.sufficiency` is still reported to callers; it just does not steer the
        # prompt (see gist-demotion tests in tests/unit/test_gist_demotion.py).
        sections_budget = max_chars - len(main_header)
        try:
            from ..core.config import get_settings

            settings = get_settings()
            constraint_budget = min(settings.retrieval.max_constraint_tokens * 4, sections_budget)
            threshold = settings.retrieval.episode_relevance_threshold
            episode_limit = (
                settings.retrieval.max_episodes_when_constraints
                if packet.constraints
                else settings.retrieval.max_episodes_default
            )
        except Exception:
            constraint_budget = min(MAX_CONSTRAINT_TOKENS * 4, sections_budget)
            threshold = EPISODE_RELEVANCE_THRESHOLD
            episode_limit = (
                MAX_EPISODES_WHEN_CONSTRAINTS if packet.constraints else MAX_EPISODES_DEFAULT
            )

        sections: list[str] = []
        used = 0

        # 1. Constraints first (reserved budget; never truncate mid-constraint)
        if packet.constraints:
            must_follow = []
            consider = []
            # Keep constraints focused so off-topic [Value] turns don't crowd factual context
            c_slice_cap = 6
            try:
                from ..core.laya_system1 import LayaSystem1Engine

                if (
                    packet.query
                    and not LayaSystem1Engine.classify_query_routing(packet.query).is_cognitive_cue
                ):
                    c_slice_cap = 3
            except Exception:
                pass
            for c in packet.constraints[:c_slice_cap]:
                meta = c.record.metadata or {}
                cmeta = meta.get("constraints", [])
                ctype = cmeta[0].get("constraint_type", "") if cmeta else ""
                if ctype in ("value", "policy", "preference"):
                    must_follow.append(c)
                else:
                    consider.append(c)

            constraint_lines: list[str] = []
            if must_follow:
                constraint_lines.append("## Constraints (Must Follow)\n")
                for c in must_follow:
                    prov = self._constraint_provenance(c)
                    line = f'- Earlier you said: "{c.record.text}" {prov}\n'
                    if (
                        used + sum(len(x) for x in constraint_lines) + len(line)
                        <= constraint_budget
                    ):
                        constraint_lines.append(line)
            if consider:
                constraint_lines.append("## Other Constraints to Consider\n")
                for c in consider:
                    prov = self._constraint_provenance(c)
                    line = f'- You also mentioned: "{c.record.text}" {prov}\n'
                    if (
                        used + sum(len(x) for x in constraint_lines) + len(line)
                        <= constraint_budget
                    ):
                        constraint_lines.append(line)

            if constraint_lines:
                block = "".join(constraint_lines) + "\n"
                sections.append(block)
                used += len(block)

        remaining = sections_budget - used

        # 1.5. Warnings & CA1 Reality Monitoring (high-priority, never truncated by episodes)
        if packet.warnings and remaining > 80:
            header = "## Warnings\n"
            warn_lines = [f"- {w}\n" for w in packet.warnings]
            s = header + "".join(warn_lines) + "\n"
            if len(s) <= remaining:
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 2. Facts (prioritize crisp atomic facts; cap long Chapter Summaries so episodes keep budget)
        if packet.facts and remaining > 100:
            header = "## Known Facts\n"
            fact_lines: list[str] = []
            atomic_f = [
                f
                for f in packet.facts
                if not str(f.record.text or "").startswith("[Chapter Summary")
            ]
            chapter_f = [
                f
                for f in packet.facts
                if str(f.record.text or "").startswith("[Chapter Summary")
                and "] Session dialogue (" not in str(f.record.text or "")
            ]
            ordered_facts = atomic_f[:8] + chapter_f[:2]
            for f in ordered_facts[:10]:
                conf = f"[{f.record.confidence:.0%}]" if f.record.confidence < 1.0 else ""
                ftxt = str(f.record.text or "")
                if ftxt.startswith("[Chapter Summary") and len(ftxt) > 380:
                    ftxt = ftxt[:377].rstrip() + "..."
                line = f"- {ftxt}{self._source_label(f)} {conf}\n"
                if len(header) + sum(len(x) for x in fact_lines) + len(line) <= remaining:
                    fact_lines.append(line)
                else:
                    break
            if fact_lines:
                s = header + "".join(fact_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 3. Preferences
        if packet.preferences and remaining > 100:
            header = "## User Preferences\n"
            pref_lines: list[str] = []
            for p in packet.preferences[:8]:
                line = f"- {p.record.text}{self._source_label(p)}\n"
                if len(header) + sum(len(x) for x in pref_lines) + len(line) <= remaining:
                    pref_lines.append(line)
                else:
                    break
            if pref_lines:
                s = header + "".join(pref_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 4. Recent episodes (with chronological timestamps for temporal reasoning)
        # Adaptive episode budget & Cognitive cue protection via System 1 routing
        effective_episode_limit = episode_limit
        effective_threshold = threshold
        is_cognitive_cue = False
        routing_intent = "single_hop"
        try:
            from ..core.config import get_settings as _get_route_settings

            _route_features = _get_route_settings().features
            if getattr(_route_features, "adaptive_episode_budget_enabled", True) and packet.query:
                from ..core.laya_system1 import LayaSystem1Engine

                routing = LayaSystem1Engine.classify_query_routing(
                    packet.query,
                    default_wide=episode_limit,
                    default_focused=min(16, episode_limit),
                )
                effective_episode_limit = min(episode_limit, routing.recommended_max_episodes)
                is_cognitive_cue = routing.is_cognitive_cue
                routing_intent = routing.intent
                if is_cognitive_cue:
                    # Semantically disconnected Cognitive cues often score 0.22-0.39 raw cosine
                    effective_threshold = min(threshold, 0.20)
        except Exception:
            pass

        # Include any verbatim conversational turns that were typed CONSTRAINT at write time
        # into relevant_episodes as well so they retain [YYYY-MM-DD] timestamps and Cognitive cue slots.
        seen_ep_ids = {str(e.record.id) for e in packet.recent_episodes}
        verbatim_constraint_eps = [
            c
            for c in packet.constraints
            if str(c.record.id) not in seen_ep_ids
            and c.retrieval_source != "constraints"
            and not (c.record.metadata or {}).get("proustian_cue_pointer")
            and (
                "turn_idx" in (c.record.metadata or {})
                or str((c.record.metadata or {}).get("source_turn_id") or "").startswith("turn_")
            )
        ]
        candidate_episodes = list(packet.recent_episodes) + verbatim_constraint_eps
        relevant_episodes = [
            e
            for e in candidate_episodes
            if getattr(e, "relevance_score", 1.0) > effective_threshold
        ]

        def _fmt_speaker_prefix(spk_val: str, raw_txt: str) -> str:
            s_clean = (spk_val or "").strip()
            if not s_clean:
                return ""
            t_low = (raw_txt or "").lstrip().lower()
            s_low = s_clean.lower()
            if t_low.startswith((f"{s_low}:", f"{s_low} said")):
                return ""
            return f"({s_clean}) "

        # On Cognitive empathy cues, protect verbatim episodes from synthetic prospective
        # crowding and preserve LLM-selected / Proustian / HyDE-promoted cues first.
        if is_cognitive_cue and relevant_episodes:
            verbatim = [
                e
                for e in relevant_episodes
                if not (e.record.metadata or {}).get("prospective_implication")
                and not (e.record.metadata or {}).get("proustian_cue_pointer")
                and not str(e.record.text or "").startswith(("[Prospective", "[Proustian Cue"))
            ]
            if verbatim:
                verbatim.sort(
                    key=lambda e: (
                        3 if (e.record.metadata or {}).get("_llm_selected") else 0,
                        2 if (e.record.metadata or {}).get("_proustian_promoted") else 0,
                        1 if (e.record.metadata or {}).get("_hyde_promoted") else 0,
                        getattr(e, "relevance_score", 0.0),
                    ),
                    reverse=True,
                )
                relevant_episodes = verbatim

        # Promote top 3 verbatim cues into a high-visibility section on Cognitive triggers
        # so the answering model explicitly sees and applies latent personal lessons/boundaries.
        promoted_count = 0
        if is_cognitive_cue and relevant_episodes and remaining > 120:
            cog_header = "## Relevant Past Experiences & Implicit Constraints\n"
            cog_lines: list[str] = []
            for e in relevant_episodes[:3]:
                meta = e.record.metadata or {}
                speaker_prefix = _fmt_speaker_prefix(meta.get("speaker", ""), e.record.text)
                line = f"- {speaker_prefix}{e.record.text}\n"
                if len(cog_header) + sum(len(x) for x in cog_lines) + len(line) <= remaining:
                    cog_lines.append(line)
            if cog_lines:
                s = cog_header + "".join(cog_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)
                promoted_count = len(cog_lines)

        if relevant_episodes and remaining > 100:
            selected_episodes = list(relevant_episodes[promoted_count:effective_episode_limit])
            if routing_intent == "temporal":
                header = "## Recent Events (Chronological Order)\n"
                selected_episodes.sort(
                    key=lambda e: (
                        e.record.timestamp.timestamp()
                        if hasattr(e.record.timestamp, "timestamp")
                        else float("inf")
                    )
                )
            else:
                header = "## Recent Events\n"
            ep_lines: list[str] = []
            for e in selected_episodes:
                # Include temporal metadata when available for temporal reasoning
                meta = e.record.metadata or {}
                said = (
                    e.record.timestamp.strftime("%Y-%m-%d")
                    if hasattr(e.record.timestamp, "strftime")
                    else ""
                )
                event_disp = str(meta.get("event_date_display") or "").strip()
                event_date = event_disp or str(meta.get("event_date") or "")[:10]
                if said and event_date and event_date != said:
                    date_str = f" [said {said}, refers to {event_date}]"
                elif said:
                    date_str = f" [{said}]"
                else:
                    date_str = f" [{event_date}]" if event_date else ""
                speaker_prefix = _fmt_speaker_prefix(meta.get("speaker", ""), e.record.text)
                line = (
                    f"- {speaker_prefix}{e.record.text}{self._source_label(e)}{date_str} "
                    f"(confidence: {e.record.confidence:.2f})\n"
                )
                if len(header) + sum(len(x) for x in ep_lines) + len(line) <= remaining:
                    ep_lines.append(line)
                else:
                    break
            if ep_lines:
                s = header + "".join(ep_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

            # 4.2. Hippocampal Time Cell deterministic event timeline on temporal queries
            if routing_intent == "temporal" and remaining > 160:
                try:
                    from .temporal_engine import HippocampalTimeCellEngine

                    tl_block = HippocampalTimeCellEngine.build_timeline_block(selected_episodes)
                    if tl_block and len(tl_block) <= remaining:
                        sections.append(tl_block)
                        used += len(tl_block)
                        remaining -= len(tl_block)
                except Exception:
                    pass

        # 4.3. Conway Tier-3 Persistent DMN Self-Profiles & Procedural Playbooks
        rmeta = getattr(packet, "retrieval_meta", None) or {}
        dmn_self_lines = rmeta.get("dmn_self_profiles") or []
        if (
            dmn_self_lines
            and remaining > 120
            and (is_cognitive_cue or routing_intent in ("common_sense", "cognitive"))
        ):
            s_hdr = "## Autobiographical Self-Model (Conway DMN)\n"
            s_body = "".join(f"- {ln}\n" for ln in dmn_self_lines[:3])
            s_block = s_hdr + s_body + "\n"
            if len(s_block) <= remaining:
                sections.append(s_block)
                used += len(s_block)
                remaining -= len(s_block)

        proc_lines = rmeta.get("procedural_skills") or []
        if (packet.procedures or proc_lines) and remaining > 120:
            p_hdr = "## Procedural Skill Playbooks\n"
            p_items = list(proc_lines[:3]) + [p.record.text for p in packet.procedures[:3]]
            p_body = "".join(f"- {ln}\n" for ln in p_items[:4])
            p_block = p_hdr + p_body + "\n"
            if len(p_block) <= remaining:
                sections.append(p_block)
                used += len(p_block)
                remaining -= len(p_block)

        preplay_notes = rmeta.get("preplay_alerts") or []
        if preplay_notes and remaining > 100:
            pp_hdr = "## Hippocampal Preplay & Constraint Check\n"
            pp_body = "".join(f"- {ln}\n" for ln in preplay_notes[:3])
            pp_block = pp_hdr + pp_body + "\n"
            if len(pp_block) <= remaining:
                sections.append(pp_block)
                used += len(pp_block)
                remaining -= len(pp_block)

        # 4.5. Default Mode Network (DMN / mPFC) Theory-of-Mind & Mentalizing Context
        try:
            from ..core.config import get_settings as _get_dmn_settings

            dmn_on = getattr(_get_dmn_settings().features, "dmn_theory_of_mind_enabled", True)
        except Exception:
            dmn_on = True
        if dmn_on and remaining > 120:
            from ..memory.dmn import DefaultModeNetwork

            if DefaultModeNetwork.is_mentalizing_query(packet.query):
                dmn_lines = DefaultModeNetwork.synthesize_mental_model(packet)
                if dmn_lines:
                    header = "## Theory-of-Mind & Psychological Context\n"
                    formatted_dmn = [f"- {line}\n" for line in dmn_lines]
                    s = header + "".join(formatted_dmn) + "\n"
                    if len(s) <= remaining:
                        sections.append(s)
                        used += len(s)
                        remaining -= len(s)

        result = main_header + "".join(sections)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return result

    def _format_json(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as JSON string."""
        data = {
            "facts": [
                {
                    "text": f.record.text,
                    "confidence": f.record.confidence,
                    "source": self._source_label(f).strip(" []") or "user",
                }
                for f in packet.facts[:8]
            ],
            "preferences": [
                {"text": p.record.text, "source": self._source_label(p).strip(" []") or "user"}
                for p in packet.preferences[:8]
            ],
            "recent": [
                {
                    "text": e.record.text,
                    "source": self._source_label(e).strip(" []") or "user",
                    "date": (
                        e.record.timestamp.isoformat()
                        if hasattr(e.record.timestamp, "isoformat")
                        else str(e.record.timestamp)
                    ),
                }
                for e in packet.recent_episodes[:8]
            ],
            "constraints": [
                {
                    "text": c.record.text,
                    "confidence": c.record.confidence,
                    "source": self._source_label(c).strip(" []") or "user",
                    "provenance": self._constraint_provenance(c),
                }
                for c in packet.constraints[:6]
            ],
            "warnings": packet.warnings,
        }
        return json.dumps(data, indent=2)
