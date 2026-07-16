"""Translator discovery (PLAN §5.3).

Translators live one-per-module under ``rag.translate.translators``; the **module
name is the ``--translator`` value**, and each module exposes a module-level
``translator`` instance (e.g. ``translator = Dispatch()``).
"""

from __future__ import annotations

import importlib
import pkgutil

from ..errors import RagError
from . import translators as translators_package
from .base import Translator


class TranslatorError(RagError):
    """Raised for translator discovery/loading problems."""


def load_translator(name: str) -> Translator:
    """Return the ``translator`` instance from module ``name``.

    Raises :class:`TranslatorError` if the module is missing or does not expose a
    valid ``translator`` instance.
    """
    module_name = f"{translators_package.__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        available = ", ".join(list_translators()) or "(none)"
        raise TranslatorError(f"unknown translator {name!r}; available: {available}") from exc
    instance = getattr(module, "translator", None)
    if not isinstance(instance, Translator):
        raise TranslatorError(
            f"translator module {name!r} must define a module-level 'translator' "
            f"instance of Translator"
        )
    return instance


def list_translators() -> list[str]:
    """Return the names of all available translator modules (sorted)."""
    # Enumerate the non-package modules directly under the translators package.
    return sorted(
        info.name
        for info in pkgutil.iter_modules(translators_package.__path__)
        if not info.ispkg
    )
