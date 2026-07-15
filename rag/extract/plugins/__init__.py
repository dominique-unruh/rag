"""Container package for extraction plugins (PLAN §4).

Each direct subpackage here is one plugin; its directory name is the plugin's
``source_type``. Discovery (:mod:`rag.extract.discovery`) enumerates these
subpackages, so adding a source is just dropping in a new subdirectory whose
``__init__`` defines an :class:`~rag.extract.base.Extractor` subclass.
"""
