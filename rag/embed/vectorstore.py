"""Vector store wrapper over Qdrant (PLAN §6).

Two deployment modes, chosen by config (``vector_store`` subtree):

- **Server** (recommended for a real corpus): a running Qdrant server addressed
  by ``vector_store.url``. Collections may keep vectors **on disk** with optional
  **quantization**, so RAM holds only the HNSW graph (+ quantized vectors), not
  every raw vector — necessary once the corpus outgrows RAM.
- **Embedded-local** (default when no ``url`` is set): the pure-Python
  ``QdrantClient(path=...)`` store under ``storage/qdrant``. It loads the whole
  collection into RAM; fine for small/dev corpora (the original prototype mode),
  but RAM-bound and single-writer.

One store holds a separate **collection per embedding model**. Authentication in
server mode is an optional API key (or a signed JWT for scoped/RBAC access),
resolved through the secrets mechanism (never read from YAML); TLS is enabled by
using an ``https://`` URL.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models

from ..config import Config
from ..errors import ConfigError, SecretNotFoundError, VectorStoreBusyError, VectorStoreError
from ..secrets import Secrets

# Map our config distance names to Qdrant's enum.
_DISTANCES: dict[str, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

# Named sparse vector used for sparse collections (Qdrant sparse vectors are
# always named). IDF weighting is applied by the collection at query time.
SPARSE_VECTOR_NAME: str = "text"

# Subdirectory (under the storage dir) holding the embedded-local Qdrant store.
_LOCAL_SUBDIR: str = "qdrant"

# Default per-request timeout (seconds) for a server client if config omits it.
_DEFAULT_TIMEOUT: int = 60

# Prefix for the keyring secret holding the Qdrant server API key / JWT, in the
# shared "rag" keyring service (alongside the email plugin's secrets). The full
# name is suffixed with the server's host:port (see :func:`_api_key_secret`) so a
# key stored for one server is never sent to a different one by accident. Store it
# with e.g. `keyring set rag vector_store_api_key:localhost:6333`; an absent key
# means the server needs no auth (a keyless localhost-only server).
_API_KEY_SECRET_PREFIX: str = "vector_store_api_key"

# Default Qdrant port assumed when a URL omits one (keeps the derived secret name
# stable and identical to the Makefile's derivation).
_DEFAULT_PORT: int = 6333


def _api_key_secret(url: str) -> str:
    """Return the host:port-scoped keyring secret name for the server at ``url``.

    Binding the secret to ``host:port`` means the API key resolved for one server
    can never be sent to another (a different URL yields a different name, hence a
    miss rather than a wrong-target leak). Must stay in lock-step with the same
    derivation in the Makefile's ``qdrant-server`` target.
    """
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    # ``or`` (not a conditional) so a missing port falls back to the Qdrant default,
    # matching the Makefile. Consistency of the id matters more than port semantics.
    port = parts.port or _DEFAULT_PORT
    return f"{_API_KEY_SECRET_PREFIX}:{host}:{port}"


def _resolve_api_key(secret_name: str, secrets: Secrets | None) -> str | None:
    """Return the stored Qdrant API key for ``secret_name``, or ``None`` if unset.

    Looks under the host:port-scoped ``secret_name`` (keyring/file) and **never
    prompts**: an absent key simply means "no auth", so a keyless localhost server
    just works. If the server *does* require a key and none is stored, the
    connection ping fails later with an actionable :class:`VectorStoreError`.
    """
    # Force a non-prompting resolver: the key is optional, so a missing one must
    # not block a batch run or pop a prompt for a server that needs no auth.
    resolver = secrets if secrets is not None else Secrets(allow_prompt=False)
    try:
        return resolver.get(secret_name)
    except SecretNotFoundError:
        return None


def _quantization_config(settings: Mapping[str, Any]) -> models.QuantizationConfig | None:
    """Build a Qdrant quantization config from the ``quantization`` subsection.

    ``None``/absent (the default) stores full-precision vectors. ``{type: scalar}``
    selects int8 scalar quantization (~4x smaller than fp32; ``always_ram`` keeps
    the quantized vectors resident for fast scoring while the originals stay on
    disk). Only ``scalar`` is supported for now; the shape leaves room for
    ``binary``/``product`` later without a config break.
    """
    raw = settings.get("quantization")
    # ``None``/``False``/absent all mean "no quantization" — full precision.
    if raw in (None, False):
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError("vector_store.quantization must be a mapping")
    qtype = str(raw.get("type", "scalar")).lower()
    if qtype == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                # Default false: keep the int8 copy on disk (mmap) too, minimising
                # resident RAM. Set true to pin it in RAM for faster scoring when
                # memory allows. Cheaply flippable later via update_collection.
                always_ram=bool(raw.get("always_ram", False)),
            )
        )
    raise ConfigError(
        f"unsupported vector_store.quantization type {qtype!r} (only 'scalar' is implemented)"
    )


def open_qdrant_client(config: Config, *, secrets: Secrets | None = None) -> QdrantClient:
    """Open a raw Qdrant client per the ``vector_store`` config.

    Server mode when ``vector_store.url`` is set; otherwise the embedded-local
    store under ``storage/qdrant``. The API key is **never** in YAML: it is
    resolved from the keyring under a host:port-scoped name (see
    :func:`_api_key_secret`), and an absent key means no auth. TLS follows an
    ``https://`` URL. Fails fast with a clear :class:`VectorStoreError` on an
    unreachable/denied server, rather than surfacing a bare transport error on the
    first operation.
    """
    settings = config.vector_store
    url = settings.get("url")
    if url:
        # Optional server auth: an API key / JWT stored under a secret name scoped
        # to this server's host:port (so it can't be sent to a different server).
        # None (not stored) => no auth header, for a keyless localhost server.
        secret_name = _api_key_secret(str(url))
        api_key = _resolve_api_key(secret_name, secrets)
        try:
            client = QdrantClient(
                url=str(url),
                api_key=api_key,
                prefer_grpc=bool(settings.get("prefer_grpc", False)),
                timeout=int(settings.get("timeout", _DEFAULT_TIMEOUT)),
            )
            # Cheap round-trip so a bad URL / wrong key / down server is reported
            # here with actionable text instead of deep in an upsert call.
            client.get_collections()
        except Exception as exc:
            raise VectorStoreError(
                f"cannot reach Qdrant server at {url}: {exc}. Check the server is "
                f"running and reachable, and that the API key (keyring secret "
                f"{secret_name!r} in service 'rag') is set and correct if the "
                f"server requires auth."
            ) from exc
        return client

    # Embedded-local fallback (original prototype mode): one directory, one writer.
    path = config.storage_dir / _LOCAL_SUBDIR
    # Directory is created by Qdrant; ensure the parent exists first.
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as exc:
        # Qdrant's local client raises a bare RuntimeError when the directory is
        # already locked by another client — turn it into an actionable one.
        raise VectorStoreBusyError(
            f"the embedded vector store at {path} is in use by another process "
            f"(embedded Qdrant allows only one at a time). Close the db-browser "
            f"or any other running 'rag embed'/chat and try again."
        ) from exc


def open_vector_store(config: Config, *, secrets: Secrets | None = None) -> "VectorStore":
    """Open the configured vector store, wrapped with collection-creation settings.

    Reads ``on_disk`` (defaults to ``True`` in server mode, ignored by the
    embedded-local store which is always in RAM) and the optional ``quantization``
    subsection, so new collections are created with the right storage profile.
    """
    settings = config.vector_store
    url = settings.get("url")
    # on_disk / quantization are server-only collection settings; the embedded-local
    # store is always in RAM and would silently ignore them. Refuse the combination
    # so a misconfiguration is caught loudly instead of quietly giving no benefit.
    if not url:
        if settings.get("on_disk"):
            raise ConfigError(
                "vector_store.on_disk requires a server (set vector_store.url); the "
                "embedded-local store is always in RAM and cannot store vectors on disk."
            )
        if settings.get("quantization") not in (None, False):
            raise ConfigError(
                "vector_store.quantization requires a server (set vector_store.url); the "
                "embedded-local store does not support quantization."
            )
    client = open_qdrant_client(config, secrets=secrets)
    # on_disk is only meaningful with a server; default it on there (the reason to
    # move off embedded-local), off otherwise. The embedded store ignores it.
    on_disk = bool(settings.get("on_disk", bool(url)))
    return VectorStore(client, on_disk=on_disk, quantization=_quantization_config(settings))


class VectorStore:
    """Thin wrapper around a Qdrant client, one collection per model."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        on_disk: bool = False,
        quantization: models.QuantizationConfig | None = None,
    ) -> None:
        """Wrap an already-open ``client``.

        ``on_disk`` and ``quantization`` are applied when *creating* a collection
        (server mode); they do not alter existing collections and are ignored by
        the embedded-local store. Use :func:`open_vector_store` to build one from
        config rather than constructing directly.
        """
        # The underlying Qdrant client (server or embedded-local).
        self._client: QdrantClient = client
        # Whether new collections store vectors on disk (server mode).
        self._on_disk: bool = on_disk
        # Optional quantization applied to new dense collections (server mode).
        self._quantization: models.QuantizationConfig | None = quantization

    def ensure_collection(self, collection: str, *, kind: str, dim: int, distance: str) -> None:
        """Create ``collection`` for a dense or sparse model if it is absent.

        ``kind`` is ``"dense"`` or ``"sparse"``. A dense collection is checked for
        a matching vector size on re-open (mixing dims corrupts retrieval, PLAN
        §6); a sparse collection uses a named sparse vector with an IDF modifier.
        New collections honour the store's ``on_disk``/``quantization`` settings.
        """
        if self._client.collection_exists(collection):
            if kind == "dense":
                info = self._client.get_collection(collection)
                existing = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing != dim:
                    raise ValueError(
                        f"collection {collection!r} has dim {existing}, expected {dim}; "
                        "use a fresh collection for a different model/dim"
                    )
            return
        if kind == "sparse":
            # Sparse vectors: no fixed size; IDF applied at query time. The sparse
            # index can also live on disk when the store is on-disk.
            self._client.create_collection(
                collection,
                vectors_config={},
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                        index=models.SparseIndexParams(on_disk=self._on_disk),
                    )
                },
            )
        else:
            metric = _DISTANCES.get(distance.lower(), models.Distance.COSINE)
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=metric, on_disk=self._on_disk
                ),
                quantization_config=self._quantization,
            )

    def replace_document(
        self, collection: str, source_type: str, doc_id: str, points: list[models.PointStruct]
    ) -> None:
        """Delete a document's existing points, then upsert its new ``points``.

        Deleting first means a re-embed (e.g. after the active set changed) never
        leaves stale points from the old set behind.
        """
        self._delete_document(collection, source_type, doc_id)
        if points:
            self._client.upsert(collection, points=points)

    def _delete_document(self, collection: str, source_type: str, doc_id: str) -> None:
        """Delete all points of one document from ``collection`` (by payload filter)."""
        self._client.delete(
            collection,
            points_selector=models.FilterSelector(filter=_doc_filter(source_type, doc_id)),
        )

    def delete_document(self, collection: str, source_type: str, doc_id: str) -> None:
        """Public: remove all of a document's points from ``collection`` (GC/cleanup)."""
        if self._client.collection_exists(collection):
            self._delete_document(collection, source_type, doc_id)

    def count(self, collection: str) -> int:
        """Return the number of points in ``collection`` (for tests/stats)."""
        return self._client.count(collection).count

    def close(self) -> None:
        """Release the Qdrant client (and, in embedded mode, its directory lock)."""
        self._client.close()


def _doc_filter(source_type: str, doc_id: str) -> models.Filter:
    """Build a Qdrant payload filter matching one document's points."""
    return models.Filter(
        must=[
            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        ]
    )
