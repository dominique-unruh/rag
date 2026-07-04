"""trace: retriever wrapping prints store queries + results (no model thinking)."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag import chat, trace


def _make_sub(clsname, nodes):
    """Build a retriever instance whose class name drives the trace label."""
    def retrieve(self, query, **kwargs):
        return nodes

    return type(clsname, (), {"retrieve": retrieve})()


def _pg(subs):
    r = type("PGRetriever", (), {})()
    r.sub_retrievers = subs
    return r


def test_abbrev_collapses_and_truncates():
    assert trace._abbrev("a\n  b   c") == "a b c"
    long = "x" * 200
    out = trace._abbrev(long, width=50)
    assert len(out) == 50 and out.endswith("…")


def test_enable_traces_vector_query_and_results(capsys):
    nodes = [
        NodeWithScore(
            node=TextNode(text="Alice met Bob in Berlin", id_="abcd1234efgh"),
            score=0.87,
        )
    ]
    sub = _make_sub("VectorContextRetriever", nodes)
    pg = _pg([sub])

    assert trace.enable(pg) == 1
    out = sub.retrieve(QueryBundle(query_str="berlin trip"))
    assert out == nodes  # passthrough

    err = capsys.readouterr().err
    assert "[db:vector]" in err
    assert "berlin trip" in err
    assert "Alice met Bob" in err
    assert "0.87" in err
    assert "abcd1234" in err  # node id, truncated to 8


def test_enable_labels_graph_and_unknown(capsys):
    g = _make_sub("LLMSynonymRetriever", [])
    u = _make_sub("WeirdRetriever", [])
    assert trace.enable(_pg([g, u])) == 2
    g.retrieve(QueryBundle(query_str="q"))
    u.retrieve(QueryBundle(query_str="q"))
    err = capsys.readouterr().err
    assert "[db:graph/synonym]" in err
    assert "[db:WeirdRetriever]" in err  # unknown -> raw class name


def test_enable_wraps_single_retriever_without_subs(capsys):
    r = _make_sub("VectorContextRetriever", [])
    assert trace.enable(r) == 1  # no sub_retrievers -> wrap the retriever itself
    r.retrieve(QueryBundle(query_str="hi"))
    assert "[db:vector]" in capsys.readouterr().err


def test_chat_parser_trace_flag():
    assert chat.build_parser().parse_args([]).trace is False
    assert chat.build_parser().parse_args(["--trace"]).trace is True
