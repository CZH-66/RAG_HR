import os

# 必须在导入 llama_index / huggingface 之前设置（engine.py 里也有，双保险）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import time
import threading
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import BASE_DIR, LLM_MODEL
# 复用 engine.py 的全局初始化（Embedding/LLM/索引只加载一次）
from engine import (
    get_query_engine, create_chat_engine,
    DEFAULT_LANG, PROMPT_VERSION, normalize_lang,
)

app = FastAPI(title="HR Policy RAG API")

# ---------- JSONL 请求日志 ----------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api_log.jsonl"
_log_lock = threading.Lock()


def write_log(record: dict):
    """追加一行 JSON 日志（线程安全）。"""
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nodes_for_log(source_nodes, text_len: int = 200) -> list[dict]:
    """检索片段 -> 可落日志结构：命中哪条文档、分数、原文摘要。

    有了 doc_id 才能区分"没检索到"和"检索到了但模型判定不相关"，
    这是 RAG badcase 复盘的关键证据。
    """
    items = []
    for node in source_nodes:
        meta = getattr(node, "metadata", None) or {}
        items.append({
            "doc_id": meta.get("id"),
            "score": round(node.score, 4) if node.score is not None else None,
            "text": (node.text or "")[:text_len],
        })
    return items


# ---------- 单轮问答（保留原接口，行为不变） ----------
class QueryRequest(BaseModel):
    question: str
    lang: str = DEFAULT_LANG      # zh=中文回答，en=英文回答

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()
    try:
        response = get_query_engine(req.lang).query(req.question)
        sources = _nodes_for_log(response.source_nodes)
        result = QueryResponse(answer=str(response), sources=sources)
        write_log({
            "ts": _now_iso(),
            "endpoint": "/query",
            "session_id": None,
            "question": req.question,
            "rewritten_question": None,          # 单轮接口没有改写步骤
            "lang": normalize_lang(req.lang),
            "answer": result.answer,
            "sources": sources,                  # 命中片段明细
            "num_sources": len(sources),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "model": LLM_MODEL,
            "prompt_version": PROMPT_VERSION,
            "status": "ok",
        })
        return result
    except Exception as e:
        write_log({
            "ts": _now_iso(),
            "endpoint": "/query",
            "session_id": None,
            "question": req.question,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "num_sources": 0,
            "model": LLM_MODEL,
            "prompt_version": PROMPT_VERSION,
            "status": "error",
            "error": str(e),
        })
        raise

# ---------- 多轮流式会话（SSE） ----------
# 会话管理：session_id -> chat_engine（单进程演示版；多 worker 部署时需换成 Redis 存历史）
sessions: dict[str, object] = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    lang: str = DEFAULT_LANG


def _session_key(session_id: str, lang: str) -> str:
    return f"{session_id}|{normalize_lang(lang)}"


def get_engine(session_id: str, lang: str):
    """同一 session_id 下，中英文各自维护一个引擎，回忆互不干扰。"""
    key = _session_key(session_id, lang)
    if key not in sessions:
        sessions[key] = create_chat_engine(lang)
    return sessions[key]


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    engine = get_engine(session_id, req.lang)
    t0 = time.perf_counter()

    # stream_chat 立即返回，token 通过 response_gen 逐个产出
    response = engine.stream_chat(req.message)
    # 改写（condense）在 stream_chat 内同步完成，立刻取回，避免并发请求覆盖
    rewritten_question = getattr(engine, "last_condensed_question", None)

    def event_stream():
        answer_parts = []
        try:
            # 第一个事件：回传 session_id，客户端后续追问需携带
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"

            # 追问改写是同步的一步（一次 LLM 调用），先告知客户端
            yield f"data: {json.dumps({'status': 'condensing'})}\n\n"

            for token in response.response_gen:
                if token:
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'delta': token})}\n\n"

            # 流结束后 source_nodes 才填充完整，作为收尾事件发给前端
            sources = _nodes_for_log(response.source_nodes)
            yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

            write_log({
                "ts": _now_iso(),
                "endpoint": "/chat",
                "session_id": session_id,
                "question": req.message,
                "rewritten_question": rewritten_question,
                "lang": normalize_lang(req.lang),
                "answer": "".join(answer_parts),
                "sources": sources,
                "num_sources": len(sources),
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
                "status": "ok",
            })
        except Exception as e:
            write_log({
                "ts": _now_iso(),
                "endpoint": "/chat",
                "session_id": session_id,
                "question": req.message,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "num_sources": 0,
                "model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
                "status": "error",
                "error": str(e),
            })
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，保证逐字到达
        },
    )


class ResetRequest(BaseModel):
    session_id: str | None = None


@app.post("/reset")
def reset(req: ResetRequest):
    removed = 0
    if req.session_id:
        prefix = f"{req.session_id}|"
        for key in [k for k in sessions if k.startswith(prefix)]:
            sessions.pop(key, None)
            removed += 1
    return {"ok": True, "reset": removed > 0, "removed": removed}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn

    # 直接传 app 对象：不依赖当前工作目录，也无需 "src.api:app" 这样的导入路径
    uvicorn.run(app, host="127.0.0.1", port=8000)
