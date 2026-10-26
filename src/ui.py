
# -*- coding: utf-8 -*-
"""HR 政策问答 —— Gradio 前端界面。

与旧版相比的主要改动：
1. 走 /chat 流式接口（SSE），回答逐字输出；流式不可用时自动回退 /query 单轮接口；
2. session_id 用 gr.State 保存，"新会话"会调用 /reset 释放服务端会话历史；
3. 回答正文与参考片段分离，片段放进可折叠面板并展示相似度；
4. 布局改为「主对话区 + 侧边助手栏（提示 / 示例问题 / 服务状态）」；
5. 兼容 Gradio 6：theme、css 只能在 launch() 中传入，Chatbot 不传 type、复制按钮用 buttons=["copy"]。
"""
import json
import os
import time

import gradio as gr
import requests

print("gradio version:", gr.__version__)

API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000").rstrip("/")
CHAT_URL = f"{API_BASE}/chat"
QUERY_URL = f"{API_BASE}/query"
RESET_URL = f"{API_BASE}/reset"
HEALTH_URL = f"{API_BASE}/health"
TIMEOUT = (5, 180)

START_CMD = "python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000"

# 检索语料与向量模型都是英文：建议英文提问，回答语言可切换
LANGUAGE_CHOICES = [("中文", "zh"), ("English", "en")]
DEFAULT_LANG = "zh"

NO_SOURCES = "_提问后，这里会显示本次回答引用的政策片段及相似度。_"

# 示例问题按语言分组：中文示例保留英文关键词，便于命中英文向量索引
EXAMPLE_QUESTIONS = {
    "zh": [
        "公司的标准工作时间是几点到几点？（working hours）",
        "公司的远程办公政策是怎样的？（remote work policy）",
        "混合办公模式具体怎么执行？（hybrid work model）",
        "我每年有多少天年假？（vacation days）",
        "公司的着装要求是什么？（dress code）",
        "如何申请费用报销？（reimbursement）",
    ],
    "en": [
        "What are the official working hours at the company?",
        "What is the company's remote work policy?",
        "How does the hybrid work model function here?",
        "How many vacation days do I get annually?",
        "What is the dress code policy at the company?",
        "How do I claim a reimbursement?",
    ],
}

PLACEHOLDER = {
    "zh": "输入 HR 问题（中文 / English 均可），回车发送…",
    "en": "Ask an HR question in English, then press Enter…",
}

HEADER_MD = """
# 🧑‍💼 HR 政策问答助手
基于公司 HR 政策文档构建的检索增强问答（RAG）。回答只依据检索到的政策片段，文档未覆盖的问题会明确说明无法作答。
"""

SIDE_NOTE = """
### 💡 使用提示
- 右侧「🌐 回答语言」同时决定**回答语言**和**示例问题语言**，默认中文；中英文各自维护独立会话记忆。
- 检索语料与向量模型都是英文，**建议用英文提问**；中文示例里的括号英文关键词就是为此保留的。
- 支持**多轮追问**：先问「remote work policy」，再问「那需要审批吗？」，后一句会被自动改写为完整问题。
- 回答下方的 **参考片段** 可展开，查看相似度与英文原文摘要。
- 文档之外的问题会明确回复「根据现有 HR 政策，无法回答该问题。」
"""
CSS = """
#app-header { padding: 6px 2px 10px; }
#app-header h1 { font-size: 1.5rem; margin: 0 0 6px; }
#app-header p { margin: 0; color: var(--body-text-color-subdued); font-size: .92rem; line-height: 1.6; }
#status-bar { min-height: 20px; margin: 2px 0 6px; font-size: .82rem; color: var(--body-text-color-subdued); }
#status-bar p { margin: 0; }
#sources-panel { font-size: .88rem; line-height: 1.6; }
#sources-panel blockquote { margin: 4px 0 12px; padding-left: 10px; border-left: 3px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
#side-hint { font-size: .88rem; line-height: 1.6; }
#example-zh button, #example-en button { text-align: left !important; justify-content: flex-start !important; font-size: .82rem; line-height: 1.4; white-space: normal; height: auto; padding: 6px 10px; }
"""


def _format_sources(sources):
    """把检索片段渲染成 Markdown，避免直接拼进回答正文。"""
    if not sources:
        return NO_SOURCES
    blocks = []
    for i, item in enumerate(sources[:3], 1):
        score = item.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
        text = " ".join((item.get("text") or "").split())
        blocks.append(f"**片段 {i}** ｜ 相似度 `{score_text}`\n\n> {text}")
    return "\n\n".join(blocks)


def _error_text(exc):
    return (
        "⚠️ **请求失败，未能获取回答**\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "请确认后端 API 已启动（在项目根目录执行）：\n\n"
        f"```powershell\n{START_CMD}\n```"
    )


def _query_once(question, lang=DEFAULT_LANG):
    """单轮接口兜底（当 /chat 流式接口不可用时）。"""
    resp = requests.post(
        QUERY_URL, json={"question": question, "lang": lang}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("answer", "")).strip(), data.get("sources") or []


def ask(question, history, session_id, lang):
    question = (question or "").strip()
    if not question:
        raise gr.Error("请先输入问题再发送～")

    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    started = time.perf_counter()
    yield "", history, NO_SOURCES, "🔍 正在检索相关政策片段…", session_id

    answer_parts, sources, stream_error = [], [], None
    try:
        payload = {"message": question, "lang": lang or DEFAULT_LANG}
        if session_id:
            payload["session_id"] = session_id
        with requests.post(
            CHAT_URL,
            json=payload,
            stream=True,
            timeout=TIMEOUT,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue

                if event.get("session_id"):
                    session_id = event["session_id"]
                if event.get("status") == "condensing":
                    yield "", history, NO_SOURCES, "🧠 正在结合上下文理解问题…", session_id
                elif "delta" in event:
                    answer_parts.append(event["delta"])
                    history[-1]["content"] = "".join(answer_parts)
                    yield "", history, NO_SOURCES, "✍️ 正在生成回答…", session_id
                elif "sources" in event:
                    sources = event.get("sources") or []
                if event.get("done"):
                    break
    except Exception as exc:
        stream_error = exc

    elapsed = time.perf_counter() - started
    answer = "".join(answer_parts).strip()

    if stream_error is not None:
        if answer:
            answer += f"\n\n> ⚠️ 流式输出中断：{stream_error}"
            status = "⚠️ 流式输出中断（已保留部分回答）"
        else:
            try:
                answer, sources = _query_once(question, lang)
                status = f"✅ 已完成（回退单轮接口）· {elapsed:.1f}s"
            except Exception as exc:
                history[-1]["content"] = _error_text(exc)
                yield "", history, NO_SOURCES, "❌ 请求失败", session_id
                return
    elif not answer:
        answer = "（模型未返回内容，请换个问法重试，或查看 `logs/api_log.jsonl` 排查。）"
        status = f"⚠️ 空回答 · {elapsed:.1f}s"
    else:
        status = f"✅ 已完成 · {elapsed:.1f}s · 引用 {len(sources)} 条片段"

    history[-1]["content"] = answer
    yield "", history, _format_sources(sources), status, session_id


def new_session(session_id):
    """清空界面，并通知后端释放该会话历史。"""
    if session_id:
        try:
            requests.post(RESET_URL, json={"session_id": session_id, "message": ""}, timeout=5)
        except requests.RequestException:
            pass
    return "", [], NO_SOURCES, "", None


def check_service():
    try:
        resp = requests.get(HEALTH_URL, timeout=3)
        if resp.ok:
            data = resp.json()
            return f"🟢 **服务在线** · 活跃会话 {data.get('sessions', 0)} · `{API_BASE}`"
    except (requests.RequestException, ValueError):
        pass
    return (
        "🔴 **服务未连接**\n\n"
        "请先启动后端（在项目根目录执行）：\n\n"
        f"```powershell\n{START_CMD}\n```"
    )


def switch_language(lang):
    """切换语言时同步示例问题分组与输入框提示语。"""
    lang = lang if lang in EXAMPLE_QUESTIONS else DEFAULT_LANG
    return (
        gr.update(visible=lang == "zh"),
        gr.update(visible=lang == "en"),
        gr.update(placeholder=PLACEHOLDER[lang]),
    )


with gr.Blocks(title="HR 政策问答助手") as demo:
    gr.Markdown(HEADER_MD, elem_id="app-header")
    session_id = gr.State(None)

    with gr.Row():
        with gr.Column(scale=7, min_width=420):
            chatbot = gr.Chatbot(
                value=[],
                label="对话",
                show_label=False,
                height=500,
                buttons=["copy"],
                placeholder="还没有对话，输入一个问题开始吧 👇",
            )
            status = gr.Markdown("", elem_id="status-bar")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder=PLACEHOLDER[DEFAULT_LANG],
                    show_label=False,
                    container=False,
                    autofocus=True,
                    lines=1,
                    max_lines=6,
                    scale=9,
                )
                send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                clear = gr.Button("🆕 新会话", size="sm")
                refresh = gr.Button("🔄 检测服务", size="sm")
            with gr.Accordion("📎 参考片段（Top-3）", open=False):
                sources_md = gr.Markdown(NO_SOURCES, elem_id="sources-panel")

        with gr.Column(scale=3, min_width=280):
            lang = gr.Radio(
                choices=LANGUAGE_CHOICES,
                value=DEFAULT_LANG,
                label="🌐 回答语言",
            )
            gr.Markdown(SIDE_NOTE, elem_id="side-hint")
            gr.Markdown("### 💬 示例问题\n_点击填入输入框，不会自动发送_")

            zh_examples = gr.Column(visible=(DEFAULT_LANG == "zh"), elem_id="example-zh")
            with zh_examples:
                for q in EXAMPLE_QUESTIONS["zh"]:
                    btn = gr.Button(q, size="sm")
                    btn.click(lambda q=q: q, None, msg)

            en_examples = gr.Column(visible=(DEFAULT_LANG == "en"), elem_id="example-en")
            with en_examples:
                for q in EXAMPLE_QUESTIONS["en"]:
                    btn = gr.Button(q, size="sm")
                    btn.click(lambda q=q: q, None, msg)

            service_md = gr.Markdown("⏳ 正在检测服务…")

    outputs = [msg, chatbot, sources_md, status, session_id]
    inputs = [msg, chatbot, session_id, lang]
    msg.submit(ask, inputs, outputs)
    send.click(ask, inputs, outputs)
    clear.click(new_session, [session_id], outputs)
    lang.change(switch_language, lang, [zh_examples, en_examples, msg])
    refresh.click(check_service, None, service_md)
    demo.load(check_service, None, service_md)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        footer_links=[],
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
