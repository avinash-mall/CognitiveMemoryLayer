"""Interference detection between memories (duplicates, overlap)."""
from dataclasses import dataclass
from typing import List, Optional

from ..core.schemas import MemoryRecord


@dataclass
class InterferenceResult:
    """Result of interference detection."""

    memory_id: str
    interfering_memory_id: str
    similarity: float
    interference_type: str  # "duplicate", "conflicting", "overlapping"
    recommendation: str  # "merge", "keep_newer", "keep_higher_confidence"


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity in pure Python (no numpy)."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 * norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class InterferenceDetector:
    """
    Detects interference between memories (duplicates, overlapping content).
    """

    def __init__(
        self,
        embedding_client: Optional[object] = None,
        similarity_threshold: float = 0.9,
        conflict_threshold: float = 0.7,
    ) -> None:
        self.embeddings = embedding_client
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold

    def detect_duplicates(
        self,
        records: List[MemoryRecord],
    ) -> List[InterferenceResult]:
        """Detect near-duplicate memories using embeddings."""
        results: List[InterferenceResult] = []
        with_embedding = [(r, r.embedding) for r in records if r.embedding]
        ids = [str(r.id) for r, _ in with_embedding]
        vecs = {str(r.id): emb for r, emb in with_embedding}

        for i, id1 in enumerate(ids):
            for id2 in ids[i + 1 :]:
                sim = _cosine_similarity(vecs[id1], vecs[id2])
                if sim >= self.similarity_threshold:
                    r1 = next(r for r in records if str(r.id) == id1)
                    r2 = next(r for r in records if str(r.id) == id2)
                    results.append(
                        InterferenceResult(
                            memory_id=id1,
                            interfering_memory_id=id2,
                            similarity=sim,
                            interference_type="duplicate",
                            recommendation=self._recommend_resolution(r1, r2),
                        )
                    )
        return results

    def detect_overlapping(
        self,
        records: List[MemoryRecord],
        text_overlap_threshold: float = 0.7,
    ) -> List[InterferenceResult]:
        """Detect memories with significant text overlap."""
        results: List[InterferenceResult] = []
        for i, r1 in enumerate(records):
            for r2 in records[i + 1 :]:
                overlap = self._text_overlap(r1.text, r2.text)
                if overlap >= text_overlap_threshold:
                    results.append(
                        InterferenceResult(
                            memory_id=str(r1.id),
                            interfering_memory_id=str(r2.id),
                            similarity=overlap,
                            interference_type="overlapping",
                            recommendation=self._recommend_resolution(r1, r2),
                        )
                    )
        return results

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Word-level Jaccard overlap."""
        w1 = set(text1.lower().split())
        w2 = set(text2.lower().split())
        if not w1 or not w2:
            return 0.0
        inter = len(w1 & w2)
        union = len(w1 | w2)
        return inter / union if union else 0.0

    def _recommend_resolution(
        self,
        r1: MemoryRecord,
        r2: MemoryRecord,
    ) -> str:
        """Recommend how to resolve interference."""
        if abs(r1.confidence - r2.confidence) > 0.2:
            return "keep_higher_confidence"
        if r1.timestamp > r2.timestamp or r2.timestamp > r1.timestamp:
            return "keep_newer"
        return "merge"
