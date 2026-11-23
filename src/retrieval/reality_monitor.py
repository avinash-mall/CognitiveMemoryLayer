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
_ACRONYM_RE = re.compile(r"^[A-Z0-9]{2,}$")
_HYPOTHETICAL_RE = re.compile(
    r"(?:^\s*(?:would|could|might|should|is\s+it\s+likely|do\s+you\s+think)\b|\blikely\b)",
    re.IGNORECASE,
)
_COGNITIVE_CUE_RE = re.compile(r"^\s*(?:[AB]|Speaker\s*[AB]|User|Assistant)\s*:", re.IGNORECASE)
_SPEAKER_TEXT_RE = re.compile(r"^\s*([A-Z][a-zA-Z0-9]{1,20})\s*(?::|said\b)")
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

        # Never emit false-premise warnings on open-ended Cognitive dialogue cues
        if _COGNITIVE_CUE_RE.match(query) and not query.strip().endswith("?"):
            return verdict

        corpus_texts = [str(getattr(getattr(m, "record", m), "text", "") or "") for m in memories]
        combined_lower = " \n ".join(corpus_texts).lower()
        corpus_words = set(_WORD_RE.findall(combined_lower))
        if not corpus_words:
            return verdict

        is_hypothetical = bool(_HYPOTHETICAL_RE.search(query))

        # 1. Extract specific presupposed anchors from the query:
        #    - non-initial capitalized proper nouns (excluding acronyms like LGBTQ/PhD)
        #    - 4-digit years (e.g., 2022, 2024)
        #    - quoted phrases
        proper_candidates: list[str] = []
        tokens_raw = query.strip().split()
        for match in _PROPER_NOUN_RE.finditer(query):
            word = match.group(0)
            w_low = word.lower()
            if w_low in _STOPWORDS or _ACRONYM_RE.match(word):
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
            # Match exact, token-subset, or nickname prefix (e.g. "Mel" -> "Melanie")
            is_prefix_match = len(a_low) >= 3 and any(
                w.startswith(a_low) and len(w) <= len(a_low) + 6 for w in corpus_words
            )
            if (
                a_low in combined_lower
                or is_prefix_match
                or (anchor_tokens and all(t in corpus_words for t in anchor_tokens))
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

        # Skip Stage 1 presupposition warnings on hypothetical / common-sense questions
        # ("Would Caroline likely have Dr. Seuss books?") where the entity is intentionally
        # introduced by the question to test world-knowledge inference rather than a trap.
        if (
            not is_hypothetical
            and unverified
            and (grounded or (grounded_content and len(memories) >= 2))
        ):
            verdict.has_mismatch = True
            verdict.grounded_anchors = grounded or grounded_content[:3]
            verdict.unverified_presuppositions = unverified
            unv_str = ", ".join(f"'{u}'" for u in unverified[:3])
            grd_str = ", ".join(f"'{g}'" for g in verdict.grounded_anchors[:3])
            verdict.warnings.append(
                f"Reality Monitoring (CA1 Presupposition Note): {unv_str} is not explicitly "
                f"named in retrieved memories matching {grd_str}. Verify carefully before "
                f"confirming {unv_str}; if related evidence supports an answer, use it without "
                f"fabricating unsupported details."
            )

        # 2. Hippocampal Relational Speaker-Topic Binding (detects speaker-swap traps where
        #    both Person A and Person B appear in the conversation, the query asks about
        #    Person A doing Topic X, but every retrieved memory about Topic X is spoken by
        #    or attributed to Person B rather than Person A).
        query_persons = [p for p in proper_candidates if len(p) >= 3]
        topic_words = [
            w
            for w in q_content_words
            if w not in {p.lower() for p in query_persons} and len(w) >= 4
        ]
        if not is_hypothetical and len(query_persons) == 1 and topic_words:
            target_person = query_persons[0]
            target_low = target_person.lower()
            topic_hits_target = 0
            topic_hits_other_speakers: dict[str, int] = {}
            matched_topics: set[str] = set()

            for mem in memories:
                rec = getattr(mem, "record", mem)
                text = str(getattr(rec, "text", "") or "")
                meta = getattr(rec, "metadata", None) or {}
                t_low = text.lower()
                hit_kws = [w for w in topic_words if w in t_low]
                if not hit_kws:
                    continue
                matched_topics.update(hit_kws)
                # Extract speaker from metadata first (Postgres records store speaker in metadata
                # and/or prefix text with "Melanie said, ..."), then fall back to text prefix.
                speaker = str(meta.get("speaker") or "").strip()
                if not speaker:
                    m_spk = _SPEAKER_TEXT_RE.match(text)
                    speaker = m_spk.group(1) if m_spk else ""
                spk_low = speaker.lower()
                # Check if target person is the speaker (including nickname prefix Mel -> Melanie)
                is_target_speaker = bool(
                    spk_low
                    and (
                        spk_low == target_low
                        or (len(target_low) >= 3 and spk_low.startswith(target_low))
                        or (len(spk_low) >= 3 and target_low.startswith(spk_low))
                    )
                )
                # Also check if text talks ABOUT target_person (ignoring salutation "Thanks, Caroline!"
                # or "Hey Caroline," at the start/end of another speaker's turn)
                body_no_salutation = re.sub(
                    r"\b(?:hey|hi|thanks|thank you|wow|yep|yeah|haha|oh)\s*,?\s*"
                    + re.escape(target_low)
                    + r"\b",
                    "",
                    t_low,
                )
                # Strip leading "Speaker said," before checking body mentions
                body_clean = re.sub(r"^\s*[a-z]+\s+said\s*,\s*", "", body_no_salutation)
                mentions_target_substantively = target_low in body_clean
                if is_target_speaker:
                    topic_hits_target += 1
                elif speaker and not mentions_target_substantively:
                    topic_hits_other_speakers[speaker] = (
                        topic_hits_other_speakers.get(speaker, 0) + 1
                    )

            if topic_hits_target == 0 and sum(topic_hits_other_speakers.values()) >= 2:
                other_spk = max(topic_hits_other_speakers.items(), key=lambda kv: kv[1])[0]
                topics_str = ", ".join(f"'{t}'" for t in sorted(matched_topics)[:3])
                verdict.has_mismatch = True
                verdict.contradictions.append(f"speaker_swap:{target_person}->{other_spk}")
                verdict.warnings.append(
                    f"Reality Monitoring (CA1 Speaker-Attribution Alert): Every retrieved memory "
                    f"about {topics_str} is by {other_spk}, NOT {target_person}. "
                    f"Do NOT answer using {other_spk}'s actions or plans as if they were "
                    f"{target_person}'s; if there is no evidence for {target_person} doing this, "
                    f"simply state that you don't have information about that for {target_person}."
                )

        return verdict
