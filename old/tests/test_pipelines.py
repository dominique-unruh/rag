"""Pipeline CLI parsers: --max / --docs / --delay / --verbose."""

from __future__ import annotations

from rag import embed, graph


def test_embed_parser():
    args = embed.build_parser().parse_args(["--max", "5", "--docs", "notes:*", "--delay", "2"])
    assert args.max == 5 and args.docs == "notes:*" and args.delay == 2.0
    d = embed.build_parser().parse_args([])
    assert d.max is None and d.docs is None and d.verbose is False


def test_graph_parser():
    args = graph.build_parser().parse_args(["--max", "3", "--docs", "*@h", "-v"])
    assert args.max == 3 and args.docs == "*@h" and args.verbose is True
    d = graph.build_parser().parse_args([])
    assert d.max is None and d.docs is None
