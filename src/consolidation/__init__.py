"""Phase 7: Consolidation engine (episodic → semantic)."""

from .clusterer import EpisodeCluster, SemanticClusterer
from .migrator import ConsolidationMigrator, MigrationResult
from .sampler import EpisodeSampler, SamplingConfig
from .schema_aligner import AlignmentResult, SchemaAligner
from .summarizer import ExtractedGist, GistExtractor
from .triggers import (
    ConsolidationScheduler,
    ConsolidationTask,
    TriggerCondition,
    TriggerType,
)
from .worker import ConsolidationReport, ConsolidationWorker

__all__ = [
    "AlignmentResult",
    "ConsolidationMigrator",
    "ConsolidationReport",
    "ConsolidationScheduler",
    "ConsolidationTask",
    "ConsolidationWorker",
    "EpisodeCluster",
    "EpisodeSampler",
    "ExtractedGist",
    "GistExtractor",
    "MigrationResult",
    "SamplingConfig",
    "SchemaAligner",
    "SemanticClusterer",
    "TriggerCondition",
    "TriggerType",
]
