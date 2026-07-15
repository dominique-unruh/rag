"""Module 1 — extraction (PLAN §4).

Pulls data from sources into the ``documents`` table and the blob store. The
package has a *general part* (this package's ``base``/``store``/``discovery``/
``runner`` modules, which own the DB and blob store) and a *plugin system* under
``rag.extract.plugins`` — one subpackage per source type. A plugin only ever
talks to the system through the :class:`~rag.extract.base.Store` handle it is
given; it never touches the database directly.
"""

from __future__ import annotations

from .base import Extractor, Store, UpsertResult, UpsertStatus

# Public surface of the extraction package for plugin authors.
__all__ = ["Extractor", "Store", "UpsertResult", "UpsertStatus"]
