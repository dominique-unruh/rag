"""rag — a local, single-user retrieval-augmented-generation pipeline.

The package is split into the pipeline stages described in ``PLAN.md``:
extraction (module 1, implemented here), processing/translation, embedding and
chat. This top-level module only carries package metadata; the concrete
functionality lives in the submodules (``rag.extract``, ``rag.db``, …).
"""

# Single source of truth for the package version, mirrored in pyproject.toml.
__version__: str = "0.1.0"
