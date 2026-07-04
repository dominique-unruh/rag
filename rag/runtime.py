"""Wire the global LlamaIndex Settings from config: LLM, embeddings, splitter.

All three are stock LlamaIndex components pointed at one OpenAI-compatible
endpoint. Configuring Settings once means every downstream index/retriever/
chat-engine picks them up implicitly.
"""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from . import errors
from .config import Config


def build_llm(cfg: Config) -> OpenAILike:
    return OpenAILike(
        model=cfg.models.chat_model,
        api_base=cfg.models.api_base,
        api_key=cfg.models.api_key,
        temperature=cfg.models.temperature,
        is_chat_model=True,
        # Local servers often lack a /models function-calling advertisement;
        # keep default tool behavior off so condense_plus_context works everywhere.
        is_function_calling_model=False,
    )


def build_embedding(cfg: Config) -> OpenAILikeEmbedding:
    # Send `dimensions` ONLY when explicitly opted in (embed_dimensions): many
    # models reject the param entirely (no Matryoshka support). When omitted the
    # model returns its native width, which verify_embedding_dim then checks.
    return OpenAILikeEmbedding(
        model_name=cfg.models.embed_model,
        api_base=cfg.models.api_base,
        api_key=cfg.models.api_key,
        dimensions=cfg.models.embed_dimensions,
    )


def build_splitter(cfg: Config) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )


def configure(cfg: Config) -> None:
    """Install LLM, embedding, and node parser into the global Settings."""
    Settings.llm = build_llm(cfg)
    Settings.embed_model = build_embedding(cfg)
    Settings.node_parser = build_splitter(cfg)
    Settings.chunk_size = cfg.chunking.chunk_size
    Settings.chunk_overlap = cfg.chunking.chunk_overlap


class EmbeddingDimError(errors.UserError):
    """The provider's embedding length disagrees with configured embed_dim."""


def verify_embedding_dim(cfg: Config) -> int:
    """Embed a probe string and confirm the provider returns `embed_dim` values.

    A mismatch silently corrupts the Qdrant collection (wrong-width vectors), so
    fail loudly and early — before any collection is created or queried.
    Requires Settings.embed_model to be installed (call configure() first).
    Returns the observed dimension.
    """
    with errors.endpoint_errors(f"({cfg.models.api_base}, embedding)"):
        probe = Settings.embed_model.get_text_embedding("dimension check")
    actual = len(probe)
    if actual != cfg.models.embed_dim:
        raise EmbeddingDimError(
            f"Embedding dimension mismatch: config models.embed_dim={cfg.models.embed_dim}, "
            f"but model '{cfg.models.embed_model}' at {cfg.models.api_base} returned {actual}. "
            f"Set models.embed_dim to {actual} and (re)ingest into a fresh "
            f"qdrant.collection, since the stored vector width cannot change."
        )
    return actual
