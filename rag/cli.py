"""Top-level command-line entry point (PLAN §8).

Dispatches subcommands to the pipeline modules. Only ``extract`` (module 1) is
implemented so far; ``translate`` / ``embed`` / ``chat`` / ``maintenance`` will
be added as those modules land.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .errors import RagError
from .extract.runner import run_extraction

# Default config file path used when ``--config`` is not supplied.
_DEFAULT_CONFIG: str = "./config.yaml"


def _parse_kv_options(pairs: list[str]) -> dict[str, str]:
    """Parse ``key=value`` CLI tokens into a plugin-options dict.

    Plugin-specific options are passed as ``key=value`` pairs so the core does not
    need to know each plugin's flags in advance (PLAN §8). Raises
    :class:`SystemExit` via argparse-style error on a malformed token.
    """
    options: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            # Enforce the key=value shape so mistakes fail fast and clearly.
            raise SystemExit(f"invalid option {pair!r}; expected key=value")
        key, value = pair.split("=", 1)
        options[key] = value
    return options


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG pipeline.")
    # Global option: which config file to load.
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, type=Path,
        help="path to the YAML config file (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract subcommand ----------------------------------------------
    extract = subparsers.add_parser("extract", help="run a source extraction plugin")
    extract.add_argument("source_type", help="plugin id (its subdirectory name), e.g. 'email'")
    extract.add_argument(
        "options", nargs="*", metavar="key=value",
        help="plugin-specific options (e.g. since='1w ago' count=100 folder=INBOX)",
    )
    return parser


def _cmd_extract(args: argparse.Namespace) -> None:
    """Handle ``rag extract`` — load config and run the chosen plugin."""
    config = Config.load(args.config)
    # Convert the raw key=value tokens into the plugin options mapping.
    options = _parse_kv_options(args.options)
    run_extraction(args.source_type, dict(options), config)


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns a process exit code.

    ``argv`` defaults to ``sys.argv[1:]``; domain errors (:class:`RagError`) are
    reported concisely rather than as tracebacks.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        # Dispatch to the selected subcommand handler.
        if args.command == "extract":
            _cmd_extract(args)
        else:  # pragma: no cover - argparse enforces valid commands
            parser.error(f"unknown command {args.command!r}")
    except RagError as exc:
        # Expected, domain-level failure: print a clean message, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Allow ``python -m rag.cli`` in addition to the installed ``rag`` script.
    raise SystemExit(main())
