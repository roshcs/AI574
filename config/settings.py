"""
Central configuration for the Multi-Domain Intelligent Assistant.
All model paths, hyperparameters, and constants live here.
Adjust for your Colab environment (GPU type, memory, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_EMBEDDING_MODEL = "thenlper/gte-large"

# ── Model Configuration ──────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for the primary language model."""
    preset: str = "gemma3_instruct_12b"             # KerasHub preset (~24GB bfloat16, fits A100 80GB)
    backend: str = "jax"                             # jax | torch | tensorflow
    dtype: str = "bfloat16"                          # bfloat16 for A100/L4
    max_new_tokens: int = 1024                       # full-generation budget (answers)
    short_max_new_tokens: int = 256                  # budget for routing, grading, hallucination checks
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    fallback_preset: Optional[str] = None             # None = no fallback (avoids double OOM)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = 1024
    batch_size: int = 64
    max_seq_length: int = 512
    device: str = "cuda"
    trust_remote_code: bool = False


# ── Vector Store Configuration ────────────────────────────────────────────────

@dataclass
class VectorStoreConfig:
    """Configuration for ChromaDB vector store."""
    persist_directory: str = "./chroma_db"
    collections: dict = field(default_factory=lambda: {
        "industrial": "industrial_knowledge",
        "recipe": "recipe_knowledge",
        "scientific": "scientific_knowledge",
    })
    search_top_k: int = 5
    similarity_metric: str = "cosine"                # cosine | l2 | ip


# ── Chunking Configuration ────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 768                            # tokens (target midpoint)
    chunk_overlap_pct: float = 0.15                  # 15% overlap
    min_chunk_size: int = 512
    max_chunk_size: int = 1024
    separators: list = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "
    ])


# ── RAG / CRAG Configuration ─────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """Configuration for the Corrective RAG pipeline."""
    max_rewrite_attempts: int = 2
    relevance_threshold: float = 0.7                 # doc grading cutoff
    confidence_threshold: float = 0.6                # hallucination check
    grading_labels: tuple = ("relevant", "irrelevant", "ambiguous")
    skip_hallucination_check: bool = False            # True = skip validation (saves one LLM call per query)


# ── Supervisor / Routing Configuration ────────────────────────────────────────

@dataclass
class DomainSpec:
    """Specification for a single domain (used for config-driven wiring)."""
    name: str
    agent_class: str          # dotted import path, e.g. "agents.industrial_agent.IndustrialAgent"
    prompt_key: str           # attribute name in config.prompts, e.g. "INDUSTRIAL_SYSTEM_PROMPT"
    collection: str           # vector store collection name


@dataclass
class SupervisorConfig:
    """Configuration for the supervisor/router agent."""
    routing_confidence_threshold: float = 0.75       # below → clarify
    domains: list = field(default_factory=lambda: [
        "industrial", "recipe", "scientific"
    ])
    domain_registry: list = field(default_factory=lambda: [
        DomainSpec("industrial", "agents.industrial_agent.IndustrialAgent",
                   "INDUSTRIAL_SYSTEM_PROMPT", "industrial_knowledge"),
        DomainSpec("recipe", "agents.recipe_agent.RecipeAgent",
                   "RECIPE_SYSTEM_PROMPT", "recipe_knowledge"),
        DomainSpec("scientific", "agents.scientific_agent.ScientificAgent",
                   "SCIENTIFIC_SYSTEM_PROMPT", "scientific_knowledge"),
    ])
    fallback_tool: str = "tavily_search"


# ── Aggregate Configuration ───────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    """Top-level configuration combining all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)


# Default config instance — import this everywhere
CONFIG = ProjectConfig()
