# -*- coding: utf-8 -*-
"""本地一键启动 HR RAG 服务（API + UI），并统一管理子进程生命周期。

用法：
    python run_local.py           # 端口被占用时给出提示并退出
    python run_local.py --kill    # 自动结束占用 8000 / 7860 的残留进程后再启动
    python run_local.py --timeout 300

特点：
    · 先起 API 并轮询 /health，就绪后才起 UI；
    · 按 Ctrl+C 同时停止两个子进程，不会留下"忘了关"的僵尸进程；
    · 任一子进程意外退出时，另一个也会被一并回收。

注意：两个子进程共用当前终端，日志会交错输出（前缀分别为 uvicorn 与 gradio 自身日志）。
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
API_PORT = 8000
UI_PORT = 7860
API_HEALTH_URL = f"http://{HOST}:{API_PORT}/health"

IS_WINDOWS = os.name == "nt"
CONSOLE_ENCODING = "mbcs" if IS_WINDOWS else "utf-8"

# src 下的模块使用 "from config import ..." 这种以 src 为根的导入，必须带 --app-dir src
API_CMD = [
    sys.executable, "-m", "uvicorn", "api:app",
    "--app-dir", "src", "--host", HOST, "--port", str(API_PORT),
]
UI_CMD = [sys.executable, os.path.join("src", "ui.py")]


def log(msg: str) -> None:
    print(f"[run_local] {msg}", flush=True)


# ---------- 端口 / 进程工具 ----------
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((HOST, port)) == 0


def pids_on_port(port: int) -> list[int]:
    """解析 netstat 找出监听指定端口的 PID（仅 Windows）。"""
    if not IS_WINDOWS:
        return []
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True, encoding=CONSOLE_ENCODING, errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: set[int] = set()
    suffix = f":{port}"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].upper().startswith("TCP"):
            continue
        if parts[1].endswith(suffix) and parts[3].upper() == "LISTENING":
            try:
                pids.add(int(parts[4]))
            except ValueError:
                continue
    return sorted(pids)


def process_name(pid: int) -> str:
    if not IS_WINDOWS:
        return "unknown"
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True, encoding=CONSOLE_ENCODING, errors="replace",
        )
        if out.strip():
            return out.strip().split(",")[0].strip('"')
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def kill_pid(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )


def resolve_port(port: int, auto_kill: bool) -> bool:
    """返回端口是否可用；auto_kill=True 时尝试结束占用者。"""
    if not port_in_use(port):
        return True

    pids = pids_on_port(port)
    detail = "、".join(f"PID {p}（{process_name(p)}）" for p in pids) or "未知进程"
    log(f"端口 {port} 已被占用：{detail}")

    if not auto_kill:
        log("  · 手动结束：taskkill /PID <PID> /T /F")
        log("  · 自动处理：python run_local.py --kill")
        return False

    if not pids:
        log(f"  无法确定占用 PID，请手动排查：netstat -ano | findstr :{port}")
        return False
    for pid in pids:
        log(f"  结束 {process_name(pid)}（PID {pid}）")
        kill_pid(pid)
    time.sleep(0.8)
    return not port_in_use(port)


# ---------- 环境自检 ----------
def preflight_env() -> None:
    if not os.path.exists(os.path.join(ROOT, ".env")) and not os.getenv("API_KEY"):
        log("警告：未找到 .env，且环境变量 API_KEY 未设置，调用大模型时可能报 api_key 缺失。")
        log("      在项目根创建 .env 并写入：API_KEY=sk-xxx")
    if not os.path.exists(os.path.join(ROOT, "chroma_db", "chroma.sqlite3")):
        log("警告：未找到 chroma_db/chroma.sqlite3，可能尚未构建索引。")
        log("      先执行：cd src && python ingest.py")


# ---------- 子进程管理 ----------
def spawn(cmd: list[str], name: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    log(f"启动 {name}：{' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=ROOT, env=env)


def stop(proc: subprocess.Popen | None, name: str, timeout: int = 10) -> None:
    if proc is None or proc.poll() is not None:
        return
    log(f"停止 {name}（PID {proc.pid}）…")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{name} 未在 {timeout}s 内退出，强制结束。")
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def wait_api_ready(proc: subprocess.Popen, timeout: int) -> bool:
    log("等待 API 就绪（首次启动要加载向量模型，通常 10~60 秒）…")
    started = time.time()
    while time.time() - started < timeout:
        if proc.poll() is not None:
            log(f"API 进程已退出（exit code {proc.returncode}），放弃启动。")
            return False
        try:
            with urllib.request.urlopen(API_HEALTH_URL, timeout=1) as resp:
                if resp.status == 200:
                    log(f"API 就绪，用时 {time.time() - started:.1f}s")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    log(f"等待 API 超时（{timeout}s），请查看上面的 uvicorn 日志。")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="一键启动 HR RAG 本地服务（API + UI）")
    parser.add_argument("--kill", action="store_true", help="自动结束占用 8000/7860 端口的残留进程")
    parser.add_argument("--timeout", type=int, default=180, help="等待 API 就绪的秒数，默认 180")
    args = parser.parse_args()

    os.chdir(ROOT)
    preflight_env()

    for port in (API_PORT, UI_PORT):
        if not resolve_port(port, args.kill):
            return 1

    api: subprocess.Popen | None = None
    ui: subprocess.Popen | None = None
    try:
        api = spawn(API_CMD, "API")
        if not wait_api_ready(api, args.timeout):
            return 1

        ui = spawn(UI_CMD, "UI")
        time.sleep(1.0)
        if ui.poll() is not None:
            log(f"UI 启动后立即退出（exit code {ui.returncode}），请查看上面的 gradio 日志。")
            return ui.returncode or 1

        log("=" * 68)
        log(f"API  {API_HEALTH_URL}   PID {api.pid}")
        log(f"UI   http://{HOST}:{UI_PORT}          PID {ui.pid}")
        log("按 Ctrl+C 同时停止两个服务")
        log("=" * 68)

        while True:
            if api.poll() is not None:
                log(f"API 意外退出（exit code {api.returncode}）")
                return api.returncode or 1
            if ui.poll() is not None:
                log(f"UI 已退出（exit code {ui.returncode}）")
                return ui.returncode or 0
            time.sleep(0.5)

    except KeyboardInterrupt:
        log("收到 Ctrl+C，开始关闭服务…")
        return 0
    finally:
        stop(ui, "UI")
        stop(api, "API")
        log("已全部停止。")


if __name__ == "__main__":
    sys.exit(main())