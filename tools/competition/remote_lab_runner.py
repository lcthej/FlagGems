#!/usr/bin/env python3
"""Small token-protected HTTP runner for FlagOS lab evidence jobs.

This is intentionally limited to a fixed job allowlist. It is not a general
remote shell.
"""

from __future__ import annotations

import html
import json
import os
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(os.environ.get("FG_ROOT", str(Path.home() / "FlagGems"))).expanduser()
VENVDIR = Path(
    os.environ.get("FG_VENV", str(Path.home() / "fg-iluvatar-venv"))
).expanduser()
EVIDENCE_DIR = Path(os.environ.get("FG_EVIDENCE_DIR", "/tmp/fg_iluvatar_evidence"))
HOST = os.environ.get("FG_RUNNER_HOST", "0.0.0.0")
PORT = int(os.environ.get("FG_RUNNER_PORT", "30000"))
TOKEN = os.environ.get("FG_RUNNER_TOKEN") or secrets.token_urlsafe(24)
COREX = Path(
    os.environ.get("FG_COREX_HOME", "/usr/local/corex-4.4.0.rc.11.20251201")
)


OPS = {
    "median": {
        "branch": "competition/median-operator-round13",
        "test": "tests/test_median.py",
        "bench": "benchmark/test_median.py",
        "marker": "median",
    },
    "scatter_reduce": {
        "branch": "competition/scatter-reduce",
        "test": "tests/test_scatter_reduce.py",
        "bench": "benchmark/test_scatter_reduce.py",
        "marker": "scatter_reduce",
    },
    "conv_transpose2d": {
        "branch": "competition/conv-transpose2d-operator",
        "test": "tests/test_conv_transpose2d.py",
        "bench": "benchmark/test_conv_transpose2d.py",
        "marker": "conv_transpose2d",
    },
    "svd": {
        "branch": "competition/svd-operator",
        "test": "tests/test_svd.py",
        "bench": "benchmark/test_svd.py",
        "marker": "svd",
        "smoke": (
            "python -c \"import torch, flag_gems; "
            "shapes=[(2,2),(8,2),(2,8),(16,8),(8,16),(5,3),(3,5)]; "
            "ctx=flag_gems.use_gems(include=['svd']); ctx.__enter__(); "
            "[print(s, torch.svd(torch.randn(s, device=flag_gems.device))[1][:3]) "
            "for s in shapes]; ctx.__exit__(None,None,None)\""
        ),
    },
    "ctc_loss": {
        "branch": "competition/ctc-loss-operator",
        "test": "tests/test_ctc_loss.py",
        "bench": "benchmark/test_ctc_loss.py",
        "marker": "ctc_loss",
    },
    "chunk_gated_delta_rule": {
        "branch": "competition/chunk-gated-delta-rule-operator",
        "test": "tests/test_chunk_gated_delta_rule.py",
        "bench": "benchmark/test_chunk_gated_delta_rule.py",
        "marker": "chunk_gated_delta_rule",
    },
}


LOCK = threading.Lock()
CURRENT: dict[str, object] = {
    "running": False,
    "job": None,
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "log": None,
}


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def env_preamble() -> str:
    lib64 = COREX / "lib64"
    return "\n".join(
        [
            "set -uo pipefail",
            f"cd {sh_quote(str(ROOT))}",
            f"source {sh_quote(str(VENVDIR / 'bin' / 'activate'))}",
            f"export LIBRARY_PATH={sh_quote(str(lib64))}:$LIBRARY_PATH",
            f"export LD_LIBRARY_PATH={sh_quote(str(lib64))}:$LD_LIBRARY_PATH",
            "export GEMS_VENDOR=iluvatar",
            "export PYTHONUNBUFFERED=1",
        ]
    )


def checkout_cmd(op: str) -> str:
    branch = OPS[op]["branch"]
    return "\n".join(
        [
            f"git fetch origin {sh_quote(branch)}:{sh_quote(branch)} --depth 1 || "
            f"git fetch origin {sh_quote(branch)}:{sh_quote(branch)}",
            f"git checkout {sh_quote(branch)}",
            f"git pull --ff-only origin {sh_quote(branch)} || true",
            "git log --oneline -1",
        ]
    )


def build_command(job: str) -> tuple[str, Path]:
    op = None
    kind = None
    for candidate in sorted(OPS, key=len, reverse=True):
        prefix = candidate + "_"
        if job.startswith(prefix):
            op = candidate
            kind = job[len(prefix) :]
            break
    if op is None or kind is None:
        raise ValueError("job must be '<op>_<kind>', for example svd_pytest")
    if kind not in {"env", "smoke", "pytest", "benchmark", "all"}:
        raise ValueError(f"unknown kind: {kind}")

    op_dir = EVIDENCE_DIR / op
    op_dir.mkdir(parents=True, exist_ok=True)
    log = op_dir / f"{job}_{int(time.time())}.log"
    cfg = OPS[op]

    commands = [env_preamble(), checkout_cmd(op)]
    if kind == "env":
        commands += [
            "pwd",
            "git branch --show-current",
            "python --version",
            "python -c \"import torch, triton, flag_gems; "
            "print('torch', torch.__version__); "
            "print('triton', triton.__version__); "
            "print('device', flag_gems.device); "
            "print('vendor', flag_gems.vendor_name); "
            "print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"",
            "ixsmi || true",
        ]
    elif kind == "smoke":
        commands.append(cfg.get("smoke") or f"pytest -q {cfg['test']} -m {cfg['marker']} --tb=short -x")
    elif kind == "pytest":
        commands.append(f"pytest -q {cfg['test']} -m {cfg['marker']} --tb=short -x")
    elif kind == "benchmark":
        ixsmi_log = op_dir / f"ixsmi_{op}_benchmark_{int(time.time())}.log"
        commands.append(
            "\n".join(
                [
                    f"(while true; do date '+%F %T'; ixsmi; sleep 1; done) > {sh_quote(str(ixsmi_log))} 2>&1 &",
                    "IXSMI_PID=$!",
                    "set +e",
                    f"pytest -q -s {cfg['bench']} --level core --warmup 5 --iter 10 --record log -m {cfg['marker']} --tb=short",
                    "RC=$?",
                    "kill $IXSMI_PID || true",
                    f"echo ixsmi_log={sh_quote(str(ixsmi_log))}",
                    "exit $RC",
                ]
            )
        )
    else:
        commands += [
            cfg.get("smoke") or f"pytest -q {cfg['test']} -m {cfg['marker']} --tb=short -x",
            f"pytest -q {cfg['test']} -m {cfg['marker']} --tb=short -x",
            f"pytest -q -s {cfg['bench']} --level core --warmup 5 --iter 10 --record log -m {cfg['marker']} --tb=short",
        ]
    return "\n".join(commands), log


def run_job(job: str) -> None:
    command, log = build_command(job)
    with LOCK:
        CURRENT.update(
            {
                "running": True,
                "job": job,
                "started_at": time.time(),
                "finished_at": None,
                "returncode": None,
                "log": str(log),
            }
        )
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"$ {job}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=os.environ.copy(),
            start_new_session=True,
        )
        rc = proc.wait()
    with LOCK:
        CURRENT.update(
            {"running": False, "finished_at": time.time(), "returncode": rc}
        )


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode()


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


class Handler(BaseHTTPRequestHandler):
    server_version = "FlagGemsRemoteRunner/1.0"

    def authorized(self, query: dict[str, list[str]]) -> bool:
        header = self.headers.get("X-Token")
        qtoken = query.get("token", [None])[0]
        return secrets.compare_digest(header or qtoken or "", TOKEN)

    def send(self, status: int, content: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in {"/", ""}:
            body = "<h3>FlagGems remote runner</h3><p>Use /jobs, /status, /run?job=svd_pytest.</p>"
            self.send(200, body.encode(), "text/html; charset=utf-8")
            return
        if not self.authorized(query):
            self.send(403, b'{"error":"forbidden"}')
            return
        if parsed.path == "/jobs":
            jobs = [f"{op}_{kind}" for op in OPS for kind in ["env", "smoke", "pytest", "benchmark", "all"]]
            self.send(200, json_bytes({"jobs": jobs, "ops": OPS}))
            return
        if parsed.path == "/status":
            with LOCK:
                payload = dict(CURRENT)
            log_path = Path(str(payload["log"])) if payload.get("log") else None
            payload["tail"] = tail(log_path) if log_path else ""
            self.send(200, json_bytes(payload))
            return
        if parsed.path == "/run":
            job = query.get("job", [None])[0]
            if not job:
                self.send(400, b'{"error":"missing job"}')
                return
            with LOCK:
                if CURRENT["running"]:
                    self.send(409, json_bytes({"error": "job already running", "current": CURRENT}))
                    return
            try:
                build_command(job)
            except Exception as exc:
                self.send(400, json_bytes({"error": str(exc)}))
                return
            thread = threading.Thread(target=run_job, args=(job,), daemon=True)
            thread.start()
            self.send(202, json_bytes({"started": job}))
            return
        if parsed.path == "/log":
            with LOCK:
                log_path = CURRENT.get("log")
            if not log_path:
                self.send(404, b"no log", "text/plain; charset=utf-8")
                return
            self.send(200, html.escape(tail(Path(str(log_path)), 1000)).encode(), "text/plain; charset=utf-8")
            return
        if parsed.path == "/stop":
            os.kill(os.getpid(), signal.SIGTERM)
            self.send(200, b'{"stopping":true}')
            return
        self.send(404, b'{"error":"not found"}')


def main() -> None:
    print(f"FlagGems remote runner listening on {HOST}:{PORT}", flush=True)
    print(f"FG_RUNNER_TOKEN={TOKEN}", flush=True)
    print("Jobs: /jobs, /run?job=svd_pytest, /status, /log, /stop", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
