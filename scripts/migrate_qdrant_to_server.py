#!/usr/bin/env python3
"""One-time migration: embedded-local Qdrant -> Qdrant server.

The embedded-local store (``QdrantClient(path=storage/qdrant)``) and the Qdrant
server use *different, incompatible* on-disk formats, so you cannot point the
server at the old directory. This script copies every point (vectors + payloads)
out of the local store and upserts it into the server, creating each collection
on the server with the **on-disk / quantization** profile from your config.

Both endpoints are taken from ``config.yaml``:
  * source  = ``storage.dir`` / ``qdrant``  (the embedded-local store)
  * dest    = ``vector_store.url`` (+ ``on_disk`` / ``quantization``; API key from
              the keyring secret ``vector_store_api_key`` if the server needs auth)

So the intended workflow is:
  1. Stand up the Qdrant server (e.g. ``docker run -p 6333:6333 qdrant/qdrant``).
  2. Add the ``vector_store`` section to ``config.yaml`` (see doc/operations.md).
  3. Run this script once.
  4. From then on, ``rag embed`` / db-browser talk to the server.

Idempotent: point ids are deterministic, so a re-run overwrites rather than
duplicates. Safe: it only reads the local store and writes the server; it never
deletes the local directory (remove it yourself once you've verified the server).

Usage:
    python scripts/migrate_qdrant_to_server.py --config config.yaml
    python scripts/migrate_qdrant_to_server.py --config config.yaml --batch 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the in-repo ``rag`` package importable when run directly from a checkout.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402

from rag.config import Config  # noqa: E402
from rag.embed.vectorstore import _LOCAL_SUBDIR, open_vector_store  # noqa: E402


def _collection_kind_and_params(info: object) -> tuple[str, int, str]:
    """Classify a source collection, returning ``(kind, dim, distance)``.

    ``kind`` is ``"sparse"`` or ``"dense"``. For a sparse collection ``dim`` is 0
    and ``distance`` is unused; for a dense collection they are read from the
    stored vector params so the destination is created with a matching size/metric.
    """
    params = info.config.params  # type: ignore[attr-defined]
    # A sparse collection carries a non-empty ``sparse_vectors`` config and no
    # (or empty) dense vector params.
    sparse = getattr(params, "sparse_vectors", None)
    if sparse:
        return "sparse", 0, "cosine"
    vectors = params.vectors
    # Our collections use a single unnamed dense vector, so ``vectors`` is a
    # VectorParams with ``.size`` / ``.distance`` (a Distance enum).
    dim = int(vectors.size)
    # Distance enum values are "Cosine"/"Dot"/"Euclid" — lower-cased they match
    # the config names our VectorStore understands.
    distance = str(vectors.distance.value).lower()
    return "dense", dim, distance


def _copy_collection(
    src: QdrantClient, dst_client: QdrantClient, name: str, batch: int
) -> int:
    """Scroll all points of ``name`` from ``src`` into ``dst_client``; return count.

    Vectors and payloads are copied verbatim (deterministic ids mean a re-run
    overwrites), in batches of ``batch`` points to bound memory.
    """
    copied = 0
    offset: object | None = None
    while True:
        points, offset = src.scroll(
            name,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,  # required — we are copying the vectors themselves
        )
        if not points:
            break
        dst_client.upsert(
            name,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        copied += len(points)
        # ``scroll`` returns ``None`` as the next offset once exhausted.
        if offset is None:
            break
    return copied


def main() -> None:
    """Parse arguments, open both stores, and copy every collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="path to config.yaml"
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="points copied per upsert (default: 256)"
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if not config.vector_store.get("url"):
        parser.error(
            "config vector_store.url is not set — nothing to migrate to. Add the "
            "server section to config.yaml first (see doc/operations.md)."
        )

    # Source: the embedded-local store on disk (independent of vector_store.url).
    src_path = config.storage_dir / _LOCAL_SUBDIR
    if not src_path.exists():
        print(f"no embedded-local store at {src_path}; nothing to migrate.")
        return
    src = QdrantClient(path=str(src_path))

    # Destination: the configured server, wrapped so ensure_collection applies the
    # on-disk / quantization profile from config when creating each collection.
    # The API key (if the server needs one) is read from the keyring under the
    # fixed secret name — store it beforehand with `keyring set rag ...`.
    dst = open_vector_store(config)

    try:
        collections = [c.name for c in src.get_collections().collections]
        if not collections:
            print("source store has no collections; nothing to migrate.")
            return
        for name in collections:
            info = src.get_collection(name)
            kind, dim, distance = _collection_kind_and_params(info)
            # Create the destination collection with the on-disk/quant profile.
            dst.ensure_collection(name, kind=kind, dim=dim, distance=distance)
            copied = _copy_collection(src, dst._client, name, args.batch)
            print(f"  {name}: copied {copied} point(s) [{kind}, dim={dim}].")
        print("migration complete. Verify the server, then delete", src_path)
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
