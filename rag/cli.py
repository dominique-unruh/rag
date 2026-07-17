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
from .interrupt import graceful_interrupts
from .secrets import Secrets

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
    # Global verbose flag: plugins print a one-liner per item when set.
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print a line for each extracted item",
    )
    # Re-verify stored passwords: before using a keyring secret, ask whether to
    # keep it, allowing a stale one to be replaced without deleting it by hand.
    parser.add_argument(
        "--reauth", action="store_true",
        help="for each stored secret, ask whether to keep it or enter a new one",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract subcommand ----------------------------------------------
    extract = subparsers.add_parser("extract", help="run a source extraction plugin")
    extract.add_argument("source_type", help="plugin id (its subdirectory name), e.g. 'email'")
    extract.add_argument(
        "options", nargs="*", metavar="key=value",
        help="plugin-specific options (e.g. since='1w ago' count=100 folder=INBOX)",
    )

    # --- translate subcommand --------------------------------------------
    translate = subparsers.add_parser("translate", help="translate documents to Markdown")
    translate.add_argument(
        "--translator", default="dispatch",
        help="translator to use (module name in translate.translators; default: %(default)s)",
    )
    translate.add_argument("--source", dest="source_type", help="restrict to one source_type")
    translate.add_argument("--doc", dest="doc_id", help="restrict to one doc_id")
    translate.add_argument("--limit", type=int, help="max number of documents to process")
    translate.add_argument(
        "--force", action="store_true",
        help="retranslate even documents whose active translation is up to date",
    )
    translate.add_argument(
        "--dry-run", action="store_true",
        help="translate but do not write to the DB (dirty repo allowed)",
    )
    translate.add_argument(
        "--inspect", action="store_true",
        help="print translations for review instead of storing them (dirty repo allowed)",
    )

    # --- embed subcommand ------------------------------------------------
    embed = subparsers.add_parser("embed", help="embed translations into a model's index")
    embed.add_argument("--model", required=True, help="embedding model id (from config)")
    embed.add_argument("--source", dest="source_type", help="restrict to one source_type")
    embed.add_argument("--doc", dest="doc_id", help="restrict to one doc_id")
    embed.add_argument("--limit", type=int, help="max number of documents to process")
    embed.add_argument(
        "--force", action="store_true",
        help="re-embed even documents already up to date for this model",
    )

    # --- maintenance subcommand (with its own sub-tools) -----------------
    maintenance = subparsers.add_parser("maintenance", help="run a maintenance tool")
    tools = maintenance.add_subparsers(dest="tool", required=True)

    # db-browser
    dbb = tools.add_parser("db-browser", help="local web UI for browsing the database")
    dbb.add_argument("--host", default="127.0.0.1", help="bind host (default: %(default)s)")
    dbb.add_argument("--port", type=int, default=8765, help="bind port (default: %(default)s)")
    dbb.add_argument("--no-open", action="store_true", help="do not open a browser")

    # delete: remove one item via the proper (cascading) store API
    delete = tools.add_parser("delete", help="delete an item using the cascading store API")
    delete.add_argument("table", choices=["extract"], help="which table/domain to delete from")
    delete.add_argument("source_type", help="the item's source_type (plugin id)")
    delete.add_argument("doc_id", help="the item's doc_id")
    return parser


def _cmd_extract(args: argparse.Namespace) -> None:
    """Handle ``rag extract`` — load config and run the chosen plugin."""
    config = Config.load(args.config)
    # Convert the raw key=value tokens into the plugin options mapping.
    options: dict[str, object] = dict(_parse_kv_options(args.options))
    # Pass the verbose flag through to the plugin as a reserved option.
    options["verbose"] = args.verbose
    # Build the secret resolver here so the global --reauth flag takes effect.
    secrets = Secrets(reauth=args.reauth)
    run_extraction(args.source_type, options, config, secrets=secrets)


def _cmd_translate(args: argparse.Namespace) -> None:
    """Handle ``rag translate`` — run the translation pipeline."""
    from .translate.pipeline import TranslateOptions, run_translation

    config = Config.load(args.config)
    options = TranslateOptions(
        translator=args.translator,
        source_type=args.source_type,
        doc_id=args.doc_id,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
        inspect=args.inspect,
        verbose=args.verbose,
    )
    count = run_translation(config, options)
    # Report the outcome (inspection mode already printed the translations).
    if not args.inspect:
        verb = "would translate" if args.dry_run else "translated"
        print(f"{verb} {count} document(s)")


def _cmd_embed(args: argparse.Namespace) -> None:
    """Handle ``rag embed`` — run the embedding pipeline for one model."""
    from .embed.pipeline import EmbedOptions, run_embed

    config = Config.load(args.config)
    options = EmbedOptions(
        model=args.model,
        source_type=args.source_type,
        doc_id=args.doc_id,
        limit=args.limit,
        force=args.force,
        verbose=args.verbose,
    )
    count = run_embed(config, options)
    print(f"embedded {count} document(s) into {args.model}")


def _cmd_maintenance(args: argparse.Namespace) -> None:
    """Handle ``rag maintenance`` — dispatch to the selected maintenance tool."""
    config = Config.load(args.config)
    if args.tool == "db-browser":
        # Imported lazily so the HTTP server code loads only when needed.
        from .maintenance.dbbrowser import serve

        serve(config, host=args.host, port=args.port, open_browser=not args.no_open)
    elif args.tool == "delete":
        _maintenance_delete(args, config)


def _maintenance_delete(args: argparse.Namespace, config: Config) -> None:
    """Delete one document via the store API so deletions cascade properly."""
    from .blobs import BlobStore
    from .db import Database
    from .extract.store import SqliteStore

    storage = config.storage_dir
    # ``table`` is currently always "extract" (the documents table); the store is
    # scoped to the given source_type.
    db = Database.open(storage / "metadata.db")
    try:
        store = SqliteStore(db.conn, BlobStore(storage / "blobs"), args.source_type)
        # missing_ok=False so a wrong id reports an error rather than silently
        # doing nothing. delete() cascades to translations/embeddings via FKs.
        store.delete(args.doc_id, missing_ok=False)
        print(f"deleted {args.source_type}:{args.doc_id}")
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns a process exit code.

    ``argv`` defaults to ``sys.argv[1:]``; domain errors (:class:`RagError`) are
    reported concisely rather than as tracebacks.
    """
    parser = _build_parser()
    # Enable shell tab-completion when argcomplete is installed and active. This
    # is a no-op during normal runs; it only acts when the shell invokes us in
    # completion mode (via the argcomplete registration).
    try:
        import argcomplete
    except ImportError:
        pass
    else:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    # Subcommand handlers; the long-running pipeline commands run under a graceful
    # SIGINT handler so a first Ctrl-C stops at the next item, not mid-write.
    handlers = {
        "extract": _cmd_extract,
        "translate": _cmd_translate,
        "embed": _cmd_embed,
        "maintenance": _cmd_maintenance,
    }
    graceful = {"extract", "translate", "embed"}
    handler = handlers.get(args.command)
    try:
        if handler is None:  # pragma: no cover - argparse enforces valid commands
            parser.error(f"unknown command {args.command!r}")
        elif args.command in graceful:
            # First Ctrl-C requests a checkpointed stop; a second aborts at once.
            with graceful_interrupts():
                handler(args)
        else:
            handler(args)
    except RagError as exc:
        # Expected, domain-level failure: print a clean message, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Allow ``python -m rag.cli`` in addition to the installed ``rag`` script.
    raise SystemExit(main())
