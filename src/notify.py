"""Bark push notifications (https://github.com/Finb/Bark) and tap-to-choose replies via an ntfy.sh topic.

The device key is read from the BARK_KEY environment variable; set it once with
    setx BARK_KEY <key>
Optional BARK_SERVER overrides https://api.day.app. The reply topic is read from NOTIFY_REPLY_TOPIC or
from ~/.emr_kg_notify.json {"reply_topic": ...}; neither the key nor the topic may be stored in the repo.

ask() sends one Bark push per option; tapping one opens https://ntfy.sh/<topic>/publish?message=<request>:<option>,
and ask() polls the topic until a reply for this request arrives. With multi=True every tap adds an option and
a separate "完成" push submits the current set. Only the listed option ids are accepted and nothing else is ever
executed, so a stranger who guessed the topic could at most pick among the offered options.

    python src/notify.py "标题" "内容"                       # send one message
    python src/notify.py --run -- python src/eval_sapbert.py   # run a command, push when it ends
    python src/notify.py --ask "标题" "问题" A="选项一" B="选项二" [--multi] [--timeout 3600] [--default A]
        (prints the chosen option id, or comma-separated ids with --multi; the default / nothing on timeout)
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

CONFIG = Path.home() / ".emr_kg_notify.json"
REPLY_SERVER = "https://ntfy.sh"
DONE = "__done__"
RESET = "__reset__"


def env_var(name: str) -> str:
    """Process environment first, then the Windows user environment (terminals opened before setx lack it)."""
    value = os.environ.get(name, "").strip()
    if value or os.name != "nt":
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip()
    except OSError:
        return ""


def bark(title: str, body: str, group: str = "emr-kg", url: str | None = None) -> bool:
    key = env_var("BARK_KEY")
    if not key:
        print("[notify] BARK_KEY not set; skip push", file=sys.stderr)
        return False
    server = os.environ.get("BARK_SERVER", "https://api.day.app").rstrip("/")
    payload = {"title": title, "body": body[:900], "group": group}
    if url:
        payload["url"] = url
    req = urllib.request.Request(f"{server}/{key}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = json.load(resp).get("code") == 200
    except Exception as exc:  # a failed push must never break the experiment
        print(f"[notify] push failed: {exc}", file=sys.stderr)
        return False
    if not ok:
        print("[notify] push rejected by server", file=sys.stderr)
    return ok


def reply_topic() -> str:
    topic = env_var("NOTIFY_REPLY_TOPIC")
    if not topic and CONFIG.exists():
        topic = json.loads(CONFIG.read_text(encoding="utf-8")).get("reply_topic", "")
    return topic


def _replies(topic: str) -> list[tuple[str, str]]:
    """(message id, text) in publication order."""
    # since=all instead of a timestamp: the local clock is not trusted; requests are told apart by their random id
    url = f"{REPLY_SERVER}/{topic}/json?poll=1&since=all"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            lines = r.read().decode("utf-8").splitlines()
    except Exception as e:
        print(f"[notify] poll failed: {e}", file=sys.stderr)
        return []
    out = []
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message":
            out.append((str(m.get("id", "")), str(m.get("message", "")).strip()))
    return out


def ask(title: str, question: str, options: dict[str, str], timeout_s: int = 3600,
        default=None, poll_s: int = 15, multi: bool = False):
    """Return the chosen option id (multi=True: list of ids, possibly empty), or default on timeout."""
    topic = reply_topic()
    if not topic:
        print("[notify] no reply topic configured; returning default", file=sys.stderr)
        return default
    rid = secrets.token_hex(4)
    base = f"{REPLY_SERVER}/{topic}/publish"
    link = lambda k: f"{base}?message={urllib.parse.quote(f'{rid}:{k}')}"
    lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    wait = f"{timeout_s // 60} 分钟" if timeout_s < 7200 else f"{timeout_s / 3600:.1f} 小时"
    dflt = ",".join(default) if isinstance(default, (list, tuple)) else default
    fallback = f"，超时默认选 {dflt}" if dflt else ""
    how = ("可以多选：点一条就选上（重复点不会取消）；点错了点「清空重选」；选好后点「完成」提交" if multi
           else "点下面对应选项的那条通知即可")
    bark(f"需要你选择：{title}", f"{question}\n{lines}\n\n{how}（{wait}内有效{fallback}）。")
    for k, v in options.items():
        bark(f"选 {k}：{v}"[:60], f"点这条 = 选择 {k}（{title}）", url=link(k))
    if multi:
        bark(f"清空重选（{title}）"[:60], "点错了就点这条，已选的全部清空（只能用一次）", url=link(RESET))
        bark(f"完成：提交选择（{title}）"[:60], "选好后点这条提交", url=link(DONE))

    # Taps are add-only and every distinct link counts once: iOS may re-open (reload) a link page later,
    # which publishes the same message again.
    chosen: list[str] = []
    seen: set[str] = set()
    used: set[str] = set()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for mid, msg in _replies(topic):
            if mid in seen:
                continue
            seen.add(mid)
            r, _, opt = msg.partition(":")
            if r != rid:
                continue
            if not multi and opt in options:
                bark(f"已收到你的选择：{opt}", f"{title}：{options[opt]}")
                return opt
            if not multi or msg in used:
                continue
            used.add(msg)
            if opt in options or opt == RESET:
                if opt == RESET:
                    chosen.clear()
                    used = {m for m in used if m.partition(":")[2] not in options}
                elif opt not in chosen:
                    chosen.append(opt)
                now = "、".join(f"{k}（{options[k]}）" for k in chosen) or "（还没选）"
                bark(f"当前已选：{'、'.join(chosen) or '无'}", f"{title}\n{now}\n选好后点「完成」。")
            elif opt == DONE:
                picked = "、".join(f"{k}（{options[k]}）" for k in chosen) or "一个都不选"
                bark(f"已提交：{'、'.join(chosen) or '无'}", f"{title}：{picked}")
                return chosen
        time.sleep(poll_s)
    bark(f"选择超时：{title}", f"没有收到{'「完成」' if multi else '选择'}，"
                              f"{'按默认选 ' + dflt if dflt else '暂不处理，等你回到电脑前'}。")
    return default


def _cli_ask(argv: list[str]) -> int:
    timeout, default, multi, pos, opts = 3600, None, False, [], {}
    it = iter(argv)
    for a in it:
        if a == "--timeout":
            timeout = int(next(it))
        elif a == "--default":
            default = next(it)
        elif a == "--multi":
            multi = True
        elif "=" in a and len(pos) >= 2:
            k, v = a.split("=", 1)
            opts[k] = v
        else:
            pos.append(a)
    if len(pos) < 2 or not opts:
        print('usage: notify.py --ask "标题" "问题" A="选项一" B="选项二" [--multi] [--timeout 秒] [--default A]',
              file=sys.stderr)
        return 2
    if multi and default is not None:
        default = [d for d in default.split(",") if d]
    choice = ask(pos[0], pos[1], opts, timeout, default, multi=multi)
    if choice is None:
        return 1
    print(",".join(choice) if multi else choice)
    return 0


def run_and_notify(cmd: list[str]) -> int:
    start = time.monotonic()
    tail: deque[str] = deque(maxlen=6)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.strip():
            tail.append(line.rstrip()[:150])
    code = proc.wait()
    minutes, seconds = divmod(int(time.monotonic() - start), 60)
    name = next((a for a in cmd if a.endswith(".py")), cmd[0])
    status = "完成" if code == 0 else f"失败（退出码 {code}）"
    bark(f"{os.path.basename(name)} {status}", f"用时 {minutes} 分 {seconds} 秒\n" + "\n".join(tail))
    return code


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--run"]:
        cmd = args[1:]
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        if not cmd:
            sys.exit("usage: python src/notify.py --run -- <command ...>")
        sys.exit(run_and_notify(cmd))
    if args[:1] == ["--ask"]:
        sys.exit(_cli_ask(args[1:]))
    if not args:
        sys.exit(__doc__)
    sys.exit(0 if bark(args[0], args[1] if len(args) > 1 else "") else 1)


if __name__ == "__main__":
    main()
