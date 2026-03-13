"""
Multilingual prompt definitions for synthetic data generation.

Used by prepare.py to generate training data in the top 15 languages
by global internet usage. Language selection is weighted (English ~30%,
others distributed) so models remain strong in English while covering
multiple languages.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """A supported language for synthetic generation."""

    code: str  # ISO 639-1
    name: str  # English name
    native_name: str  # Name in the language itself
    weight: float  # Relative frequency for weighted random selection


# Top 15 languages by global internet usage. Weights: English ~30%, rest ~5% each.
SUPPORTED_LANGUAGES: list[Language] = [
    Language("en", "English", "English", 0.30),
    Language("zh", "Chinese (Simplified)", "简体中文", 0.05),
    Language("es", "Spanish", "Español", 0.05),
    Language("ar", "Arabic", "العربية", 0.05),
    Language("hi", "Hindi", "हिन्दी", 0.05),
    Language("pt", "Portuguese", "Português", 0.05),
    Language("fr", "French", "Français", 0.05),
    Language("ja", "Japanese", "日本語", 0.05),
    Language("ru", "Russian", "Русский", 0.05),
    Language("de", "German", "Deutsch", 0.05),
    Language("ko", "Korean", "한국어", 0.05),
    Language("tr", "Turkish", "Türkçe", 0.05),
    Language("id", "Indonesian", "Bahasa Indonesia", 0.05),
    Language("vi", "Vietnamese", "Tiếng Việt", 0.05),
    Language("it", "Italian", "Italiano", 0.05),
]


def pick_language(rng: random.Random) -> Language:
    """Weighted random selection from SUPPORTED_LANGUAGES."""
    weights = [lang.weight for lang in SUPPORTED_LANGUAGES]
    return rng.choices(SUPPORTED_LANGUAGES, weights=weights, k=1)[0]


def system_prompt_single(lang: Language) -> str:
    """System prompt for single-text synthetic classification generation."""
    return (
        f"Generate synthetic classification data in {lang.name}. "
        "Return STRICT JSON only, no markdown fences. "
        "Do not explain, think aloud, or add commentary before or after the JSON."
    )


def system_prompt_pair(lang: Language) -> str:
    """System prompt for text-pair synthetic classification generation."""
    return (
        f"Generate synthetic text-pair classification data in {lang.name}. "
        "Return STRICT JSON only, no markdown fences. "
        "Do not explain, think aloud, or add commentary before or after the JSON."
    )


def user_prompt_single(
    task: str,
    label: str,
    n: int,
    seed_text: str,
    lang: Language,
) -> str:
    """User prompt for single-text generation. Includes language instruction when not English."""
    base = (
        f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
        f"Seed example from related dataset (do not copy): {seed_text}\n\n"
        'Return exactly: {"samples":[{"text":"..."}]}\n'
        "Output JSON only. Do not include explanations.\n"
        "Each sample must match target label exactly, be diverse, "
        "and be one concise sentence (8-22 words)."
    )
    if lang.code != "en":
        lang_instruction = (
            f"\n\nIMPORTANT: Generate ALL text samples in {lang.name} ({lang.native_name}). "
            "Do not use English."
        )
        return base + lang_instruction
    return base


def user_prompt_pair(
    task: str,
    label: str,
    n: int,
    seed_a: str,
    seed_b: str,
    lang: Language,
) -> str:
    """User prompt for text-pair generation. Includes language instruction when not English."""
    base = (
        f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
        f"Seed pair from related dataset (do not copy): A={seed_a} | B={seed_b}\n\n"
        'Return exactly: {"samples":[{"text_a":"...","text_b":"..."}]}\n'
        "Output JSON only. Do not include explanations.\n"
        "Every pair must match target label exactly and be diverse. "
        "Each field should be one concise sentence (6-20 words)."
    )
    if lang.code != "en":
        lang_instruction = (
            f"\n\nIMPORTANT: Generate ALL text samples (text_a and text_b) in {lang.name} "
            f"({lang.native_name}). Do not use English."
        )
        return base + lang_instruction
    return base
