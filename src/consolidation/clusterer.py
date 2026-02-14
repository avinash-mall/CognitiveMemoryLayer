"""Semantic clustering of episodes (pure Python, no sklearn)."""

from dataclasses import dataclass, field

from ..core.schemas import MemoryRecord


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    """Average of vectors."""
    if not embeddings:
        return []
    n = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / len(embeddings) for i in range(n)]


@dataclass
class EpisodeCluster:
    """A cluster of related episodes. BUG-11: use default_factory for mutable default."""

    cluster_id: int
    episodes: list[MemoryRecord]
    centroid: list[float] | None = None

    common_entities: list[str] = field(default_factory=list)
    dominant_type: str = "unknown"
    avg_confidence: float = 0.0


class SemanticClusterer:
    """Clusters episodes by semantic similarity using greedy cosine clustering."""

    def __init__(
        self,
        min_cluster_size: int = 2,
        max_clusters: int = 20,
        similarity_threshold: float = 0.7,
    ):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold

    def cluster(self, episodes: list[MemoryRecord]) -> list[EpisodeCluster]:
        """Cluster episodes by semantic similarity."""
        if not episodes:
            return []

        valid = [ep for ep in episodes if ep.embedding]
        if not valid:
            return [
                EpisodeCluster(
                    cluster_id=0,
                    episodes=episodes,
                    avg_confidence=sum(e.confidence for e in episodes) / len(episodes),
                )
            ]

        n_clusters = min(
            self.max_clusters,
            max(1, len(valid) // self.min_cluster_size),
        )
        if len(valid) <= n_clusters:
            return [
                EpisodeCluster(
                    cluster_id=i,
                    episodes=[ep],
                    centroid=ep.embedding,
                    avg_confidence=ep.confidence,
                )
                for i, ep in enumerate(valid)
            ]

        # Greedy: assign each episode to nearest cluster by centroid similarity
        clusters: list[EpisodeCluster] = []
        for i, ep in enumerate(valid):
            emb = ep.embedding
            if not emb:
                continue
            best_idx = -1
            best_sim = self.similarity_threshold - 0.01
            for idx, c in enumerate(clusters):
                if c.centroid:
                    sim = _cosine_sim(emb, c.centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx
            if best_idx >= 0:
                clusters[best_idx].episodes.append(ep)
                clusters[best_idx].centroid = _centroid(
                    [e.embedding for e in clusters[best_idx].episodes if e.embedding]
                )
                clusters[best_idx].avg_confidence = sum(
                    e.confidence for e in clusters[best_idx].episodes
                ) / len(clusters[best_idx].episodes)
            else:
                clusters.append(
                    EpisodeCluster(
                        cluster_id=len(clusters),
                        episodes=[ep],
                        centroid=emb,
                        avg_confidence=ep.confidence,
                    )
                )

        # Merge small clusters into nearest
        small = [c for c in clusters if len(c.episodes) < self.min_cluster_size]
        large = [c for c in clusters if len(c.episodes) >= self.min_cluster_size]
        for c in small:
            reassigned: set = set()
            for ep in c.episodes:
                nearest = self._find_nearest(ep, large)
                if nearest:
                    nearest.episodes.append(ep)
                    reassigned.add(id(ep))
                    nearest.centroid = _centroid(
                        [e.embedding for e in nearest.episodes if e.embedding]
                    )
                    nearest.avg_confidence = sum(e.confidence for e in nearest.episodes) / len(
                        nearest.episodes
                    )
            # Only add episodes that were not reassigned (no duplication)
            unassigned = [ep for ep in c.episodes if id(ep) not in reassigned]
            if unassigned:
                large.append(
                    EpisodeCluster(
                        cluster_id=c.cluster_id,
                        episodes=unassigned,
                        centroid=_centroid([e.embedding for e in unassigned if e.embedding])
                        or c.centroid,
                        avg_confidence=sum(e.confidence for e in unassigned) / len(unassigned),
                    )
                )
        clusters = [c for c in large if len(c.episodes) >= self.min_cluster_size]
        if not clusters and large:
            clusters = large

        for i, c in enumerate(clusters):
            c.cluster_id = i
            c.common_entities = self._common_entities(c.episodes)
            c.dominant_type = self._dominant_type(c.episodes)
        return clusters

    def _common_entities(self, episodes: list[MemoryRecord]) -> list[str]:
        entity_counts: dict[str, int] = {}
        for ep in episodes:
            for ent in ep.entities:
                key = getattr(ent, "normalized", str(ent))
                entity_counts[key] = entity_counts.get(key, 0) + 1
        threshold = len(episodes) * 0.5
        return [e for e, count in entity_counts.items() if count >= threshold]

    def _dominant_type(self, episodes: list[MemoryRecord]) -> str:
        type_counts: dict[str, int] = {}
        for ep in episodes:
            t = ep.type.value if hasattr(ep.type, "value") else str(ep.type)
            type_counts[t] = type_counts.get(t, 0) + 1
        return max(type_counts, key=type_counts.get) if type_counts else "unknown"

    def _find_nearest(
        self, episode: MemoryRecord, clusters: list[EpisodeCluster]
    ) -> EpisodeCluster | None:
        if not clusters:
            return None
        if not episode.embedding:
            return None  # Cannot compute similarity without embedding; caller handles explicitly
        best = None
        best_sim = -1.0
        for c in clusters:
            if c.centroid:
                sim = _cosine_sim(episode.embedding, c.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best = c
        return best
