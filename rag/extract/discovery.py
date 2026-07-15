"""Plugin discovery for extraction (PLAN §4).

Plugins live one-per-subpackage under ``rag.extract.plugins``. The subpackage
directory name *is* the ``source_type``; the framework derives and enforces it
here rather than trusting any attribute on the plugin class.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from ..errors import PluginError
from . import plugins as plugins_package
from .base import Extractor


def _extractor_in_module(module_name: str) -> type[Extractor]:
    """Import ``module_name`` and return the single ``Extractor`` subclass in it.

    Raises :class:`PluginError` if the module defines no concrete extractor, or
    more than one (ambiguous).
    """
    module = importlib.import_module(module_name)
    # Collect concrete Extractor subclasses *defined in this module* (ignore
    # imported base classes and re-exports from elsewhere).
    found: list[type[Extractor]] = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Extractor)
        and obj is not Extractor
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]
    if not found:
        raise PluginError(f"plugin module {module_name!r} defines no Extractor subclass")
    if len(found) > 1:
        # Ambiguous: which one is the plugin? Force one class per plugin package.
        names = ", ".join(cls.__name__ for cls in found)
        raise PluginError(f"plugin module {module_name!r} defines multiple Extractors: {names}")
    return found[0]


def discover_plugins() -> dict[str, type[Extractor]]:
    """Return a mapping of ``source_type`` -> plugin class for all plugins.

    Scans the immediate subpackages of ``rag.extract.plugins``; each subpackage
    name becomes a ``source_type`` bound to the ``Extractor`` its ``__init__``
    module defines.
    """
    result: dict[str, type[Extractor]] = {}
    # Walk only the direct children of the plugins package directory.
    for module_info in pkgutil.iter_modules(plugins_package.__path__):
        # Consider packages only (each plugin is its own subdirectory).
        if not module_info.ispkg:
            continue
        source_type = module_info.name
        result[source_type] = _extractor_in_module(f"{plugins_package.__name__}.{source_type}")
    return result


def load_plugin(source_type: str) -> type[Extractor]:
    """Return the plugin class for ``source_type``.

    Raises :class:`PluginError` if no plugin subpackage of that name exists.
    """
    plugins = discover_plugins()
    if source_type not in plugins:
        available = ", ".join(sorted(plugins)) or "(none)"
        raise PluginError(f"unknown source type {source_type!r}; available: {available}")
    return plugins[source_type]
