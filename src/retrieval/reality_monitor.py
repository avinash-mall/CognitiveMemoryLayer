"""CA1 Comparator & Orbitofrontal Reality Monitoring (False-Premise Immune System).

In the hippocampal-prefrontal circuit, region CA1 acts as a comparator between
incoming cues/presuppositions and reinstated CA3/neocortical traces, while the
orbitofrontal cortex performs reality monitoring (distinguishing externally
experienced facts from fabricated or presupposed premises).

Instead of thresholding scalar retrieval scores (which docs/STATE.md proved
cannot separate adversarial from answerable queries: 0.624 vs 0.638 cosine),
this module performs structural presupposition verification:
1. Detects when a query anchors on a known subject/topic present in retrieved
   memories but presupposes a specific proper noun, location, year, or quoted
   entity that is completely absent from the retrieved evidence.
2. Detects explicit negation/antonym contradictions between query presuppositions
   and high-confidence retrieved facts/constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Question words and sentence-initial auxiliaries that are capitalized by position
# rather than because they are proper-noun entities.
_QUESTION_STARTERS = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "why",
        "how",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "can",
        "has",
        "have",
        "had",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "with",
        "about",
        "if",
        "tell",
        "describe",
        "explain",
        "list",
        "name",
        "give",
    }
)

_STOPWORDS = _QUESTION_STARTERS | frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "by",
        "as",
        "into",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "against",
        "during",
        "without",
        "before",
        "under",
        "around",
        "among",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "me",
        "you",
        "him",
        "she",
        "he",
        "they",
        "them",
        "we",
        "us",
        "this",
        "that",
        "these",
        "those",
        "any",
        "some",
        "all",
        "both",
        "each",
        "ever",
        "much",
        "many",
        "kind",
        "type",
        "time",
        "day",
        "year",
        "week",
        "month",
        "user",
    }
)

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{3,40})['\"]")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RealityMonitorVerdict:
    """Result of CA1 / Orbitofrontal presupposition verification."""

    has_mismatch: bool = False
    grounded_anchors: list[str] = field(default_factory=list)
    unverified_presuppositions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CA1RealityMonitor:
    """Verifies query presuppositions against retrieved CA3/Neocortical memory traces."""

    @classmethod
    def inspect(cls, query: str, memories: list[Any]) -> RealityMonitorVerdict:
        """Compare query presuppositions against retrieved memories."""
        verdict = RealityMonitorVerdict()
        if not query or not memories or len(query.strip()) < 12:
            return verdict

        corpus_texts = [str(getattr(getattr(m, "record", m), "text", "") or "") for m in memories]
        combined_lower = " \n ".join(corpus_texts).lower()
        corpus_words = set(_WORD_RE.findall(combined_lower))
        if not corpus_words:
            return verdict

        # 1. Extract specific presupposed anchors from the query:
        #    - non-initial capitalized proper nouns (or initial ones that are not question words)
        #    - 4-digit years (e.g., 2022, 2024)
        #    - quoted phrases
        proper_candidates: list[str] = []
        tokens_raw = query.strip().split()
        for idx, match in enumerate(_PROPER_NOUN_RE.finditer(query)):
            word = match.group(0)
            w_low = word.lower()
            if w_low in _STOPWORDS:
                continue
            # If it's the very first word of the query, only treat it as a proper anchor
            # if it isn't a common verb/noun starter
            if match.start() == 0 and tokens_raw and w_low in _QUESTION_STARTERS:
                continue
            if word not in proper_candidates:
                proper_candidates.append(word)

        years = _YEAR_RE.findall(query)
        quoted = [q.strip() for q in _QUOTED_RE.findall(query) if len(q.strip()) >= 3]

        specific_anchors = list(dict.fromkeys([*proper_candidates, *years, *quoted]))
        if not specific_anchors:
            return verdict

        grounded: list[str] = []
        unverified: list[str] = []
        for anchor in specific_anchors:
            a_low = anchor.lower()
            anchor_tokens = _WORD_RE.findall(a_low)
            if a_low in combined_lower or (
                anchor_tokens and all(t in corpus_words for t in anchor_tokens)
            ):
                grounded.append(anchor)
            else:
                unverified.append(anchor)

        # Also check general content-word grounding so a query like
        # "When did I visit Paris?" (where "I" is implicit user and "Paris" is unverified,
        # while "visit" or travel memories exist) is caught when the tenant has rich memories.
        q_content_words = [
            w for w in _WORD_RE.findall(query.lower()) if w not in _STOPWORDS and len(w) > 3
        ]
        grounded_content = [w for w in q_content_words if w in corpus_words]

        # A CA1 presupposition mismatch occurs when:
        # - We have at least one unverified specific anchor (proper noun, year, or quoted term)
        # - AND the query has grounded context in the retrieved memories (either another
        #   specific anchor is grounded, or >=1 content words + >=2 retrieved memories exist).
        if unverified and (grounded or (grounded_content and len(memories) >= 2)):
            verdict.has_mismatch = True
            verdict.grounded_anchors = grounded or grounded_content[:3]
            verdict.unverified_presuppositions = unverified
            unv_str = ", ".join(f"'{u}'" for u in unverified[:3])
            grd_str = ", ".join(f"'{g}'" for g in verdict.grounded_anchors[:3])
            verdict.warnings.append(
                f"Reality Monitoring (CA1 Presupposition Alert): Query presupposes {unv_str}, "
                f"which is absent from retrieved memories matching {grd_str}. "
                f"Do not fabricate or confirm {unv_str} unless explicitly supported by evidence; "
                f"if unsupported, reply concisely that you don't have information about that."
            )

        # 2. Hippocampal Relational Speaker-Topic Binding (detects speaker-swap traps where
        #    both Person A and Person B appear in the conversation, the query asks about
        #    Person A doing Topic X, but every retrieved memory about Topic X is spoken by
        #    or attributed to Person B rather than Person A).
        speaker_prefix_re = re.compile(r"^\s*([A-Z][a-zA-Z0-9]{1,20})\s*:")
        query_persons = [p for p in proper_candidates if len(p) >= 3]
        topic_words = [
            w
            for w in q_content_words
            if w not in {p.lower() for p in query_persons} and len(w) >= 4
        ]
        if len(query_persons) == 1 and topic_words:
            target_person = query_persons[0]
            target_low = target_person.lower()
            topic_hits_target = 0
            topic_hits_other_speakers: dict[str, int] = {}
            matched_topics: set[str] = set()

            for text in corpus_texts:
                t_low = text.lower()
                hit_kws = [w for w in topic_words if w in t_low]
                if not hit_kws:
                    continue
                matched_topics.update(hit_kws)
                m_spk = speaker_prefix_re.match(text)
                speaker = m_spk.group(1) if m_spk else ""
                if speaker.lower() == target_low:
                    topic_hits_target += 1
                elif speaker and target_low not in t_low:
                    topic_hits_other_speakers[speaker] = (
                        topic_hits_other_speakers.get(speaker, 0) + 1
                    )

            if topic_hits_target == 0 and sum(topic_hits_other_speakers.values()) >= 2:
                other_spk = max(topic_hits_other_speakers.items(), key=lambda kv: kv[1])[0]
                topics_str = ", ".join(f"'{t}'" for t in sorted(matched_topics)[:3])
                verdict.has_mismatch = True
                verdict.contradictions.append(f"speaker_swap:{target_person}->{other_spk}")
                verdict.warnings.append(
                    f"Reality Monitoring (CA1 Speaker-Attribution Alert): Retrieved memories "
                    f"about {topics_str} are attributed to {other_spk}, NOT {target_person}. "
                    f"The query asks about {target_person}; do not attribute {other_spk}'s "
                    f"actions or plans to {target_person}. Reply concisely that you don't have "
                    f"information about that for {target_person} without speculating."
                )

        # 3. System 1 Laya Calibrated Reality Verification (catches subtle ungrounded
        #    premises and speaker-swaps that structural proper-noun rules miss, while
        #    automatically skipping empathetic Cognitive dialogue cues).
        if not verdict.has_mismatch:
            try:
                from ..core.laya_system1 import LayaSystem1Engine

                routing = LayaSystem1Engine.classify_query_routing(query)
                laya_v = LayaSystem1Engine.verify_reality_grounding(
                    query,
                    memories,
                    is_cognitive_cue=routing.is_cognitive_cue,
                )
                if laya_v.is_ungrounded:
                    verdict.has_mismatch = True
                    verdict.contradictions.append(f"laya_system1:{laya_v.reason}")
                    verdict.warnings.append(
                        "Reality Monitoring (CA1 System-1 Alert): The retrieved memories do "
                        "not appear to support the factual premise or speaker-action "
                        "attribution assumed by this question. Verify strictly against the "
                        "evidence; if the assumed event or attribution is not present, reply "
                        "concisely that you don't have information about that from previous "
                        "conversations — do not substitute unrelated details."
                    )
            except Exception:
                pass

        return verdict
