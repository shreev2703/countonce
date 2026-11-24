#!/usr/bin/env python3
import glob
import json
import os
import re
import sys
import time

REPO = os.path.expanduser("~/countonce")
TASKS_FILE = os.path.join(REPO, "TASKS_OVERNIGHT.md")
BAR = "=========================================================================="

def latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None

def parse_stream(log_path):
    info = {
        "model": "detecting...", "in_tok": 0, "cache_read": 0,
        "out_tok": 0, "cost": 0.0, "turns": 0, "tools": [], "subagents": []
    }
    if not log_path or not os.path.exists(log_path):
        return info
    with open(log_path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            etype = ev.get("type")
            if etype == "system" and ev.get("model"):
                info["model"] = ev["model"]
            elif etype == "assistant":
                msg = ev.get("message", {})
                if msg.get("model"):
                    info["model"] = msg["model"]
                u = msg.get("usage", {})
                if u:
                    info["in_tok"] = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                    info["cache_read"] = u.get("cache_read_input_tokens", 0)
                    info["out_tok"] += u.get("output_tokens", 0)
                    info["turns"] += 1
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use":
                        name = block.get("name", "tool")
                        inp = block.get("input", {})
                        detail = (
                            inp.get("file_path") or inp.get("command")
                            or inp.get("pattern") or inp.get("description") or ""
                        )
                        detail = str(detail).replace("\n", " ")[:52]
                        if name in ("Agent", "Task"):
                            desc = str(inp.get("description") or inp.get("prompt") or "subagent")[:48]
                            info["subagents"].append(f"{inp.get('subagent_type', 'worker')}: {desc}")
                        info["tools"].append(f"{name:<10} {detail}")
            elif etype == "result":
                info["cost"] = ev.get("total_cost_usd", info["cost"])
    return info

def render():
    orch_log = latest_file(os.path.join(REPO, "logs/overnight_*/orchestrator.log"))
    status, account = "Idle / Finished", "N/A"
    if orch_log and os.path.exists(orch_log):
        with open(orch_log, errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            status = lines[-1]
            m = re.search(r"Acc (\d+)", status)
            if m:
                account = f"Account {m.group(1)} ({'Personal' if m.group(1)=='1' else 'Hopscotch'})"

    task_logs = [
        p for p in glob.glob(os.path.join(REPO, "logs/overnight_*/*.log"))
        if not p.endswith("orchestrator.log")
    ]
    active_log = max(task_logs, key=os.path.getmtime) if task_logs else None
    s = parse_stream(active_log)

    done, todo, blocked = [], [], []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE) as f:
            for l in f:
                l = l.strip()
                if l.startswith("- [x] "): done.append(l[6:])
                elif l.startswith("- [ ] "): todo.append(l[6:])
                elif l.startswith("- [!] "): blocked.append(l[6:])

    ctx_used = s["in_tok"] + s["cache_read"]
    ctx_max = 200000
    pct = min(100, int((ctx_used / ctx_max) * 100))
    filled = int(pct / 5)
    bar = "#" * filled + "-" * (20 - filled)

    out = ["\033[H\033[J"]
    out.append(BAR)
    out.append(f" COUNTONCE LIVE MISSION CONTROL  |  {time.strftime('%H:%M:%S')}")
    out.append(BAR)
    out.append(f" ACCOUNT : {account}")
    out.append(f" MODEL   : {s['model']}   (Turn {s['turns']}/40)")
    out.append(f" CONTEXT : [{bar}] {pct}% ({ctx_used:,} / 200k tokens) | Out: {s['out_tok']:,}")
    out.append(f" QUEUE   : {len(done)} Done  |  {len(todo)} Remaining  |  {len(blocked)} Blocked")
    out.append(f" STATUS  : {status[:78]}")
    out.append(BAR)
    out.append(" CURRENT / NEXT TASKS:")
    if todo:
        for i, t in enumerate(todo[:4]):
            tag = "[ACTIVE]" if i == 0 else f"[NEXT {i}]"
            out.append(f"  {tag:<8} {t[:66]}")
    else:
        out.append("  All queued tasks complete (waiting on Opus audit or finished).")
    out.append(BAR)
    out.append(f" SUBAGENTS SPAWNED THIS TASK ({len(s['subagents'])}):")
    if s["subagents"]:
        for sa in s["subagents"][-3:]:
            out.append(f"  -> {sa}")
    else:
        out.append("  (Direct execution mode - no subagents spawned yet)")
    out.append(BAR)
    out.append(" LIVE TOOL FEED (WHAT CLAUDE IS DOING RIGHT NOW):")
    for t in (s["tools"][-7:] or ["  Waiting for first tool call..."]):
        out.append(f"  > {t}")
    out.append(BAR)
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        while True:
            render()
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nExited HUD (background tmux orchestrator is still running).")
