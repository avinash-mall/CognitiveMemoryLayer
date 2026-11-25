"""Rostral Prefrontal Cortex (BA10) Prospective Memory & Causal Intention Triggers.

Models human prospective memory (remembering to perform intended actions or
applying forward-looking causal constraints when a future cue or event occurs).
Supports both fast System-1 deterministic causal-implication induction and
System-2 LLM prospective indexing, plus read-time parent-episode ecphory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProspectiveTrigger:
    """A forward-looking intention or causal implication derived from an episode."""

    condition_cue: str
    implication: str
    trigger_type: str  # "causal_constraint" | "event_intention" | "temporal_plan"
    confidence: float = 0.72


_CAUSAL_IMPLICATION_RULES: list[tuple[re.Pattern[str], str, str, str]] = [
    # Physical injury / mobility limitation -> sports / strenuous activity constraint
    (
        re.compile(
            r"\b(?:broke|broken|fractured|sprained|injured|surgery on|tore|torn)\b.*?\b(ankle|leg|knee|foot|ACL|achilles|back|hip|arm|wrist|shoulder)\b",
            re.IGNORECASE,
        ),
        "physical activity, sports, running, hiking, marathon, gym, heavy lifting, travel walking",
        "Physical limitation ({match}): cannot participate in strenuous physical activities, running, or high-impact sports until recovered.",
        "causal_constraint",
    ),
    # Allergy / dietary restriction -> dining, recipe, gift, restaurant selection
    (
        re.compile(
            r"\b(?:allergic to|allergy to|intolerant to|lactose intolerant|celiac|vegan|vegetarian|pescatarian|avoid eating|can't eat|cannot eat)\s+([a-zA-Z0-9\s,\-]+?)(?:[.!?;]|$)",
            re.IGNORECASE,
        ),
        "food, restaurant, dinner, meal, cooking, snack, gift, catering, menu",
        "Dietary/health constraint ({match}): must avoid incompatible foods or ingredients when choosing restaurants, meals, or food gifts.",
        "causal_constraint",
    ),
    # Financial / budget constraint -> spending, shopping, travel, gift
    (
        re.compile(
            r"\b(?:saving up for|tight budget|paying off debt|cut back on spending|laid off|lost (?:my|his|her) job|strict budget)\b([^.!?;]*)",
            re.IGNORECASE,
        ),
        "expensive purchase, luxury travel, dining out, budgeting, financial decision, shopping",
        "Financial constraint ({match}): prefers cost-effective, budget-conscious options and avoiding unnecessary expenses.",
        "causal_constraint",
    ),
    # Future intention / scheduled commitment -> planning & reminder trigger
    (
        re.compile(
            r"\b(?:planning to|going to|scheduled to|will be|have a flight to|moving to|starting a new job at|visiting)\s+([A-Za-z0-9\s,\-]+?)\s+(?:next\s+\w+|in\s+[A-Z][a-z]+|on\s+[A-Z][a-z]+|tomorrow|soon)\b",
            re.IGNORECASE,
        ),
        "schedule, availability, upcoming plans, preparation, timeline",
        "Upcoming scheduled plan ({match}): affects future availability, location, and preparation needs.",
        "temporal_plan",
    ),
    # Pet / caregiving responsibility -> travel, housing, schedule
    (
        re.compile(
            r"\b(?:adopted a|got a new|taking care of (?:my|our))\s+(dog|puppy|cat|kitten|baby|newborn|elderly parent)\b",
            re.IGNORECASE,
        ),
        "travel, accommodation, daily routine, pet-friendly, schedule, weekend plans",
        "Caregiving/household responsibility ({match}): requires pet/family-friendly arrangements and affects travel or schedule flexibility.",
        "event_intention",
    ),
]


class ProspectiveMemoryEngine:
    """Synthesizes BA10 prospective triggers and resolves prospective hits to parent episodes."""

    @classmethod
    def induce_triggers(cls, text: str, max_count: int = 4) -> list[ProspectiveTrigger]:
        """Fast System-1 deterministic induction of prospective triggers from text."""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return []

        triggers: list[ProspectiveTrigger] = []
        seen_implications: set[str] = set()
        for pattern, cue, template, ttype in _CAUSAL_IMPLICATION_RULES:
            if len(triggers) >= max_count:
                break
            match = pattern.search(text)
            if match:
                snippet = match.group(0).strip(" .,!?;")
                implication = template.format(match=snippet)
                if implication not in seen_implications:
                    seen_implications.add(implication)
                    triggers.append(
                        ProspectiveTrigger(
                            condition_cue=cue,
                            implication=implication,
                            trigger_type=ttype,
                        )
                    )
        return triggers[:max_count]

    @classmethod
    def induce_implication_strings(cls, text: str, max_count: int = 4) -> list[str]:
        """Return formatted implication strings suitable for prospective embedding records."""
        return [
            f"{t.implication} (Relevant for: {t.condition_cue})"
            for t in cls.induce_triggers(text, max_count=max_count)
        ]
