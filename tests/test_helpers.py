# tests/test_helpers.py
# -*- coding: utf-8 -*-
import types

from api import _nodes_for_log


def test_nodes_for_log_truncates_text_and_rounds_score():
    node = types.SimpleNamespace(text="x" * 500, score=0.123456, metadata={"id": 7})
    out = _nodes_for_log([node], text_len=10)
    assert out == [{"doc_id": 7, "score": 0.1235, "text": "x" * 10}]


def test_nodes_for_log_handles_missing_metadata_and_score():
    node = types.SimpleNamespace(text="abc", score=None, metadata=None)
    out = _nodes_for_log([node])
    assert out[0]["doc_id"] is None
    assert out[0]["score"] is None