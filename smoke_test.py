# smoke_test.py
# -*- coding: utf-8 -*-
"""端到端冒烟测试：验证真实 API 服务可用。

前置条件：API 已启动（python run_local.py 或 uvicorn ...），且 .env 中配置了 API_KEY。
用法：
    python smoke_test.py
    python smoke_test.py --base http://127.0.0.1:8001
"""
import argparse
import json
import sys

import requests


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}")
    if not ok and detail:
        print(f"       {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="HR RAG 服务端到端冒烟测试")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument("--timeout", type=float, default=90, help="单次请求超时（秒）")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    results: list[bool] = []

    # 1. 健康检查
    try:
        resp = requests.get(f"{base}/health", timeout=5)
        results.append(check("GET /health 返回 200", resp.status_code == 200, resp.text[:200]))
    except requests.RequestException as exc:
        print(f"[FAIL] 无法连接 {base}/health：{exc}")
        print("       请先启动 API：python run_local.py")
        return 1

    # 2. 单轮问答
    try:
        resp = requests.post(
            f"{base}/query",
            json={"question": "What are the official working hours at the company?", "lang": "zh"},
            timeout=args.timeout,
        )
        data = resp.json()
        ok = resp.status_code == 200 and bool(data.get("answer")) and isinstance(data.get("sources"), list)
        results.append(check("POST /query 返回 answer + sources", ok, resp.text[:300]))
    except (requests.RequestException, ValueError) as exc:
        results.append(check("POST /query", False, str(exc)))

    # 3. 流式多轮问答
    session_id = None
    try:
        events = []
        with requests.post(
            f"{base}/chat",
            json={"message": "What is the dress code policy at the company?", "lang": "zh"},
            stream=True,
            timeout=(5, args.timeout),
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))

        session_id = next((e["session_id"] for e in events if "session_id" in e), None)
        results.append(check("POST /chat 回传 session_id", bool(session_id)))
        results.append(check("POST /chat 输出 delta", any("delta" in e for e in events)))
        results.append(
            check(
                "POST /chat 以 done + sources 收尾",
                bool(events) and events[-1].get("done") is True and isinstance(events[-1].get("sources"), list),
                str(events[-1] if events else "无事件"),
            )
        )
    except (requests.RequestException, ValueError) as exc:
        results.append(check("POST /chat 流式请求", False, str(exc)))

    # 4. 会话回收
    if session_id:
        try:
            resp = requests.post(f"{base}/reset", json={"session_id": session_id}, timeout=5)
            results.append(check("POST /reset 清理会话", resp.status_code == 200, resp.text[:200]))
        except requests.RequestException as exc:
            results.append(check("POST /reset", False, str(exc)))

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} 项通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())