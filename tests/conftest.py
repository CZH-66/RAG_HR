# tests/conftest.py
# -*- coding: utf-8 -*-
"""测试夹具：用假的 engine 模块替换真实实现。

真实 engine 在导入时会加载 HuggingFace embedding 模型并连接 Chroma，
既慢又依赖本地索引与模型缓存；单元测试只验证 API 逻辑，因此整体打桩。
端到端验证请使用项目根目录的 smoke_test.py（需要真实服务）。
"""
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class FakeNode:
    def __init__(self, text: str = "fake policy text", score: float | None = 0.8, doc_id: int = 1):
        self.text = text
        self.score = score
        self.metadata = {"id": doc_id, "source": "hr_dataset.json"}


class FakeQueryResponse:
    def __init__(self, question: str):
        self._question = question
        self.source_nodes = [FakeNode()]

    def __str__(self) -> str:
        return f"fake answer for: {self._question}"


class FakeQueryEngine:
    def query(self, question: str) -> FakeQueryResponse:
        return FakeQueryResponse(question)


class FakeChatEngine:
    """模拟 CondenseQuestionChatEngine：记录改写结果 + 产出固定 token 流。"""

    def __init__(self):
        self.last_condensed_question: str | None = None

    def stream_chat(self, message: str):
        self.last_condensed_question = f"rewritten: {message}"
        response = types.SimpleNamespace()
        response.response_gen = iter(["你好", "，我是 HR 助手"])
        response.source_nodes = [FakeNode(score=0.75)]
        return response


def _install_fake_engine() -> None:
    if "engine" in sys.modules:
        return
    engine = types.ModuleType("engine")
    engine.DEFAULT_LANG = "zh"
    engine.PROMPT_VERSION = "test"
    engine.normalize_lang = lambda lang: lang if lang in ("zh", "en") else "zh"
    engine.get_query_engine = lambda lang="zh": FakeQueryEngine()
    engine.create_chat_engine = lambda lang="zh": FakeChatEngine()
    sys.modules["engine"] = engine


_install_fake_engine()


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """返回 (TestClient, 日志文件路径)，并把日志重定向到临时目录。"""
    import api

    log_file = tmp_path / "api_log.jsonl"
    monkeypatch.setattr(api, "LOG_FILE", log_file)
    monkeypatch.setattr(api, "sessions", {})
    return TestClient(api.app), log_file





