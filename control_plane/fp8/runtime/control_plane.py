from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO


try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("PyYAML is required. Run `uv sync` or install PyYAML>=6.0.2.") from exc

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROL_ROOT = REPO_ROOT / "control_plane" / "fp8"
STATE_DIR = CONTROL_ROOT / "state"
RUNTIME_DIR = CONTROL_ROOT / "runtime"
CONFIG_TEMPLATE_PATH = RUNTIME_DIR / "config_template.yaml"
DEFAULT_DASHBOARD_HOST = "0.0.0.0"
DEFAULT_DASHBOARD_PORT = 8233
PROMPT_DIR = RUNTIME_DIR / "generated_prompts"
LOG_DIR = RUNTIME_DIR / "logs"
MANAGER_REPORT = CONTROL_ROOT / "reports" / "manager_report.md"
SESSION_STATE = RUNTIME_DIR / "session_state.json"
CONTROL_PLANE_RUNTIME = "uv run --no-project --with 'PyYAML>=6.0.2' python"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def yaml_text(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, shell=True, text=True, capture_output=True)


def format_command(template: Any, values: dict[str, str]) -> list[str]:
    if isinstance(template, str):
        return shlex.split(template.format(**values))
    return [str(part).format(**values) for part in template]


def is_placeholder_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return False
    return normalized.startswith("/absolute/path/") or normalized in {"unassigned", "none"}


@dataclass
class WorkerProcess:
    agent: str
    resource_pool: str
    provider: str
    model: str
    command: list[str]
    worktree_path: Path
    log_path: Path
    log_handle: TextIO
    process: subprocess.Popen[str]
    started_at: float


class DualStackThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        if hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        super().server_bind()


class IPv6OnlyThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        if hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except OSError:
                pass
        super().server_bind()


def bind_server(
    server_cls: type[ThreadingHTTPServer], address: Any, handler: type[BaseHTTPRequestHandler]
) -> ThreadingHTTPServer:
    server = server_cls(address, handler, bind_and_activate=False)
    server.server_bind()
    server.server_activate()
    return server


def create_http_servers(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> list[ThreadingHTTPServer]:
    attempts: list[str] = []
    servers: list[ThreadingHTTPServer] = []

    if host == "0.0.0.0":
        try:
            servers.append(bind_server(ThreadingHTTPServer, (host, port), handler))
        except OSError as exc:
            attempts.append(f"{host}:{port} ({exc})")
        try:
            servers.append(bind_server(IPv6OnlyThreadingHTTPServer, ("::", port, 0, 0), handler))
        except OSError as exc:
            attempts.append(f"[::]:{port} ({exc})")
        if servers:
            return servers
        detail = "; ".join(attempts) if attempts else "no compatible address families"
        raise OSError(f"failed to bind control plane on {host}:{port}: {detail}")

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as exc:
        raise OSError(f"failed to resolve control plane host {host}:{port}: {exc}") from exc

    for family, _, _, _, sockaddr in infos:
        try:
            if family == socket.AF_INET6:
                return [bind_server(IPv6OnlyThreadingHTTPServer, sockaddr, handler)]
            if family == socket.AF_INET:
                return [bind_server(ThreadingHTTPServer, sockaddr, handler)]
        except OSError as exc:
            attempts.append(f"{sockaddr} ({exc})")

    detail = "; ".join(attempts) if attempts else "no compatible address families"
    raise OSError(f"failed to bind control plane on {host}:{port}: {detail}")


def browser_open_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def format_endpoint(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def load_session_state_file() -> dict[str, Any]:
    if not SESSION_STATE.exists():
        return {}
    return json.loads(SESSION_STATE.read_text(encoding="utf-8"))


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_pid(pid: int, sig: int = signal.SIGTERM) -> None:
    os.kill(pid, sig)


def control_plane_base_url(args: argparse.Namespace, session_state: dict[str, Any]) -> str:
    server = session_state.get("server", {})
    host = args.host or server.get("host") or DEFAULT_DASHBOARD_HOST
    port = args.port or int(server.get("port") or DEFAULT_DASHBOARD_PORT)
    return f"http://{browser_open_host(str(host))}:{port}"


def post_control_plane(url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError(body or str(exc)) from exc
        raise RuntimeError(data.get("error") or "\n".join(data.get("errors", [])) or str(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


class ControlPlaneService:
    def __init__(
        self,
        config_path: Path,
        host_override: str | None = None,
        port_override: int | None = None,
        persist_config_path: Path | None = None,
        force_dry_run: bool = False,
    ):
        self.config_path = config_path
        self.persist_config_path = persist_config_path or config_path
        self.host_override = host_override
        self.port_override = port_override
        self.force_dry_run = force_dry_run
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.server_threads: list[threading.Thread] = []
        self.http_servers: list[ThreadingHTTPServer] = []
        self.processes: dict[str, WorkerProcess] = {}
        self.simulated_processes: dict[str, dict[str, Any]] = {}
        self.last_event = "initialized"
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.config: dict[str, Any] = {}
        self.project: dict[str, Any] = {}
        self.providers: dict[str, Any] = {}
        self.resource_pools: dict[str, Any] = {}
        self.workers: list[dict[str, Any]] = []
        self.provider_stats: dict[str, dict[str, Any]] = {}
        self.dry_run = False
        self.dry_run_reason = ""
        self.listen_host = ""
        self.listen_port = 0
        self.listen_endpoints: list[str] = []
        self.reload_config()

    def refresh_runtime_mode(self) -> None:
        using_template = self.config_path.resolve() == CONFIG_TEMPLATE_PATH.resolve()
        self.dry_run = self.force_dry_run or using_template
        reasons: list[str] = []
        if using_template:
            reasons.append(f"using template config {self.config_path}")
        if self.persist_config_path != self.config_path:
            reasons.append(f"save target is {self.persist_config_path}")
        if self.force_dry_run:
            reasons.append("--dry-run is enabled")
        self.dry_run_reason = "; ".join(reasons)

    def reload_config(self) -> None:
        with self.lock:
            self.config = load_yaml(self.config_path)
            self.project = self.config.get("project", {})
            self.providers = self.config.get("providers", {})
            self.resource_pools = self.config.get("resource_pools", {})
            self.workers = self.config.get("workers", [])
            self.refresh_runtime_mode()
            self.provider_stats = self.provider_stats or {
                pool_name: {
                    "launch_successes": 0,
                    "launch_failures": 0,
                    "clean_exits": 0,
                    "failed_exits": 0,
                    "last_failure": "",
                    "last_latency_ms": None,
                    "last_probe_ok": False,
                    "last_work_quality": 0.0,
                }
                for pool_name in self.resource_pools
            }
            for pool_name in self.resource_pools:
                self.provider_stats.setdefault(
                    pool_name,
                    {
                        "launch_successes": 0,
                        "launch_failures": 0,
                        "clean_exits": 0,
                        "failed_exits": 0,
                        "last_failure": "",
                        "last_latency_ms": None,
                        "last_probe_ok": False,
                        "last_work_quality": 0.0,
                    },
                )

    def validation_errors(self, config: dict[str, Any] | None = None) -> list[str]:
        cfg = config or self.config
        errors: list[str] = []
        project = cfg.get("project", {})
        providers = cfg.get("providers", {})
        resource_pools = cfg.get("resource_pools", {})
        workers = cfg.get("workers", [])

        if not project.get("repository_name"):
            errors.append("project.repository_name is required")
        if not project.get("local_repo_root"):
            errors.append("project.local_repo_root is required")
        elif is_placeholder_path(project.get("local_repo_root")):
            errors.append("project.local_repo_root must be replaced with a real path")
        if not project.get("paddle_repo_path"):
            errors.append("project.paddle_repo_path is required")
        elif is_placeholder_path(project.get("paddle_repo_path")):
            errors.append("project.paddle_repo_path must be replaced with a real path")
        dashboard = project.get("dashboard", {})
        if not dashboard.get("host"):
            errors.append("project.dashboard.host is required")
        if not dashboard.get("port"):
            errors.append("project.dashboard.port is required")
        if project.get("manager_git_identity"):
            manager_identity = project.get("manager_git_identity", {})
            if not str(manager_identity.get("name", "")).strip():
                errors.append("project.manager_git_identity.name is required when manager_git_identity is set")
            if not str(manager_identity.get("email", "")).strip():
                errors.append("project.manager_git_identity.email is required when manager_git_identity is set")

        seen_agents: set[str] = set()
        seen_branches: set[str] = set()
        seen_worktrees: set[str] = set()

        for pool_name, pool in resource_pools.items():
            provider_name = pool.get("provider")
            if provider_name not in providers:
                errors.append(f"resource_pools.{pool_name}.provider references unknown provider {provider_name}")
            if not pool.get("model"):
                errors.append(f"resource_pools.{pool_name}.model is required")
            priority = pool.get("priority", 100)
            if not isinstance(priority, int):
                errors.append(f"resource_pools.{pool_name}.priority must be an integer")

        for worker in workers:
            agent = str(worker.get("agent", "")).strip()
            if not agent:
                errors.append("worker.agent is required")
                continue
            if agent in seen_agents:
                errors.append(f"duplicate worker agent {agent}")
            seen_agents.add(agent)

            pool_name = worker.get("resource_pool")
            pool_queue = worker.get("resource_pool_queue", [])
            if pool_name and pool_name not in resource_pools:
                errors.append(f"worker {agent} references unknown resource_pool {pool_name}")
            if not pool_name and not pool_queue:
                errors.append(f"worker {agent} must define resource_pool or resource_pool_queue")
            if pool_queue and not isinstance(pool_queue, list):
                errors.append(f"worker {agent} resource_pool_queue must be a list")
            for candidate_pool in pool_queue if isinstance(pool_queue, list) else []:
                if candidate_pool not in resource_pools:
                    errors.append(f"worker {agent} resource_pool_queue references unknown pool {candidate_pool}")
            branch = str(worker.get("branch", "")).strip()
            if not branch:
                errors.append(f"worker {agent} branch is required")
            elif branch in seen_branches:
                errors.append(f"duplicate worker branch {branch}")
            else:
                seen_branches.add(branch)

            worktree = str(worker.get("worktree_path", "")).strip()
            if not worktree:
                errors.append(f"worker {agent} worktree_path is required")
            elif is_placeholder_path(worktree):
                errors.append(f"worker {agent} worktree_path must be replaced with a real path")
            elif worktree in seen_worktrees:
                errors.append(f"duplicate worker worktree_path {worktree}")
            else:
                seen_worktrees.add(worktree)

            environment_path = worker.get("environment_path")
            if worker.get("environment_type") not in {"none", None} and is_placeholder_path(environment_path):
                errors.append(f"worker {agent} environment_path must be replaced with a real path")

            if not worker.get("test_command"):
                errors.append(f"worker {agent} test_command is required")
            if not worker.get("submit_strategy"):
                errors.append(f"worker {agent} submit_strategy is required")
            git_identity = worker.get("git_identity")
            if git_identity is not None:
                if not isinstance(git_identity, dict):
                    errors.append(f"worker {agent} git_identity must be a mapping")
                else:
                    if not str(git_identity.get("name", "")).strip():
                        errors.append(f"worker {agent} git_identity.name is required when git_identity is set")
                    if not str(git_identity.get("email", "")).strip():
                        errors.append(f"worker {agent} git_identity.email is required when git_identity is set")

        return errors

    def save_config_text(self, raw_text: str) -> list[str]:
        parsed = yaml.safe_load(raw_text) or {}
        errors = self.validation_errors(parsed)
        if errors:
            return errors
        target_path = self.persist_config_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml_text(parsed), encoding="utf-8")
        self.config_path = target_path
        self.reload_config()
        self.last_event = f"config_saved:{now_iso()}"
        return []

    def configured_api_key(self, provider: dict[str, Any], pool: dict[str, Any]) -> str:
        api_env_name = provider.get("api_key_env_name")
        configured_value = str(pool.get("api_key", ""))
        if configured_value and configured_value != "replace_me_or_use_api_key_env":
            return configured_value
        if api_env_name:
            return str(os.environ.get(api_env_name, ""))
        return ""

    def score_work_quality(self, stats: dict[str, Any], active_workers: int) -> float:
        successes = int(stats.get("launch_successes", 0))
        launch_failures = int(stats.get("launch_failures", 0))
        clean_exits = int(stats.get("clean_exits", 0))
        failed_exits = int(stats.get("failed_exits", 0))
        numerator = successes + clean_exits + active_workers
        denominator = numerator + launch_failures + failed_exits + 1
        return round(numerator / denominator, 3)

    def evaluate_resource_pool(self, pool_name: str) -> dict[str, Any]:
        pool = self.resource_pools[pool_name]
        provider_name = str(pool.get("provider", "unassigned"))
        provider = self.providers.get(provider_name, {})
        stats = self.provider_stats.setdefault(
            pool_name,
            {
                "launch_successes": 0,
                "launch_failures": 0,
                "clean_exits": 0,
                "failed_exits": 0,
                "last_failure": "",
                "last_latency_ms": None,
                "last_probe_ok": False,
                "last_work_quality": 0.0,
            },
        )
        start = time.perf_counter()
        binary = None
        template = provider.get("command_template")
        if isinstance(template, str):
            parts = shlex.split(template)
            binary = parts[0] if parts else None
        elif isinstance(template, list) and template:
            binary = str(template[0])
        binary_path = shutil.which(binary) if binary else None
        has_api_key = bool(self.configured_api_key(provider, pool))
        probe_ok = bool(binary_path) and has_api_key
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        stats["last_latency_ms"] = latency_ms
        stats["last_probe_ok"] = probe_ok

        active_workers = sum(
            1
            for worker in self.processes.values()
            if worker.resource_pool == pool_name and worker.process.poll() is None
        )
        work_quality = self.score_work_quality(stats, active_workers)
        stats["last_work_quality"] = work_quality

        connection_quality = 1.0 if probe_ok else 0.0
        if probe_ok and latency_ms < 25:
            connection_quality = 1.0
        elif probe_ok and latency_ms < 100:
            connection_quality = 0.9
        elif probe_ok:
            connection_quality = 0.8

        base_priority = int(pool.get("priority", 100))
        score = round(base_priority * 100 + connection_quality * 30 + work_quality * 70, 3)

        return {
            "resource_pool": pool_name,
            "provider": provider_name,
            "model": pool.get("model", "unassigned"),
            "priority": base_priority,
            "binary": binary or "unassigned",
            "binary_found": bool(binary_path),
            "api_key_present": has_api_key,
            "connection_quality": connection_quality,
            "work_quality": work_quality,
            "score": score,
            "latency_ms": latency_ms,
            "active_workers": active_workers,
            "last_failure": stats.get("last_failure", ""),
        }

    def provider_queue(self) -> list[dict[str, Any]]:
        evaluations = [self.evaluate_resource_pool(pool_name) for pool_name in self.resource_pools]
        return sorted(evaluations, key=lambda item: (-item["score"], -item["priority"], item["resource_pool"]))

    def candidate_pools_for_worker(self, worker: dict[str, Any]) -> list[str]:
        configured_queue = worker.get("resource_pool_queue")
        if isinstance(configured_queue, list) and configured_queue:
            return configured_queue
        if worker.get("resource_pool"):
            return [str(worker["resource_pool"])]
        return [item["resource_pool"] for item in self.provider_queue()]

    def best_pool_for_worker(self, worker: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        queue = self.provider_queue()
        evaluations = {item["resource_pool"]: item for item in queue}
        ordered_candidates = []
        for pool_name in self.candidate_pools_for_worker(worker):
            if pool_name in evaluations:
                ordered_candidates.append(evaluations[pool_name])
        ordered_candidates.sort(key=lambda item: (-item["score"], -item["priority"], item["resource_pool"]))
        for item in ordered_candidates:
            if item["binary_found"] and item["api_key_present"]:
                return item["resource_pool"], item
        if ordered_candidates:
            return ordered_candidates[0]["resource_pool"], ordered_candidates[0]
        raise RuntimeError(f"worker {worker['agent']} has no eligible resource pool candidates")

    def write_session_state(self) -> None:
        worker_payload = {
            agent: {
                "pid": worker.process.pid,
                "resource_pool": worker.resource_pool,
                "provider": worker.provider,
                "model": worker.model,
                "command": worker.command,
                "worktree_path": str(worker.worktree_path),
                "log_path": str(worker.log_path),
                "alive": worker.process.poll() is None,
                "returncode": worker.process.poll(),
                "simulated": False,
            }
            for agent, worker in self.processes.items()
        }
        worker_payload.update(self.simulated_processes)
        payload = {
            "updated_at": now_iso(),
            "last_event": self.last_event,
            "server": {
                "pid": os.getpid(),
                "host": self.listen_host,
                "port": self.listen_port,
                "endpoints": self.listen_endpoints,
                "config_path": str(self.config_path),
                "persist_config_path": str(self.persist_config_path),
                "dry_run": self.dry_run,
                "alive": not self.stop_event.is_set(),
            },
            "workers": worker_payload,
        }
        SESSION_STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_cli_commands(self) -> dict[str, str]:
        host = self.host_override or self.project.get("dashboard", {}).get("host", "127.0.0.1")
        port = self.port_override or int(self.project.get("dashboard", {}).get("port", 8233))
        config = str(self.config_path)
        serve = f"{CONTROL_PLANE_RUNTIME} control_plane/fp8/runtime/control_plane.py serve --config {config} --host {host} --port {port} --open-browser"
        up = f"{CONTROL_PLANE_RUNTIME} control_plane/fp8/runtime/control_plane.py up --config {config} --host {host} --port {port} --open-browser"
        if self.dry_run:
            serve += " --dry-run"
            up += " --dry-run"
        return {"serve": serve, "up": up}

    def task_title(self, task_id: str) -> str:
        backlog = load_yaml(STATE_DIR / "backlog.yaml")
        for item in backlog.get("items", []):
            if item.get("id") == task_id:
                return str(item.get("title", task_id))
        return task_id

    def integration_branch(self) -> str:
        return str(self.project.get("integration_branch") or self.project.get("base_branch") or "main")

    def worker_git_identity(self, worker: dict[str, Any]) -> dict[str, str]:
        identity = worker.get("git_identity") or {}
        return {
            "name": str(identity.get("name", "")).strip(),
            "email": str(identity.get("email", "")).strip(),
        }

    def manager_git_identity(self) -> dict[str, str]:
        identity = self.project.get("manager_git_identity") or {}
        return {
            "name": str(identity.get("name", "")).strip(),
            "email": str(identity.get("email", "")).strip(),
        }

    def configure_git_identity(self, worker: dict[str, Any]) -> None:
        identity = self.worker_git_identity(worker)
        worktree_path = Path(worker["worktree_path"])
        if identity["name"]:
            result = subprocess.run(
                ["git", "config", "user.name", identity["name"]],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "failed to set git user.name")
        if identity["email"]:
            result = subprocess.run(
                ["git", "config", "user.email", identity["email"]],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "failed to set git user.email")

    def merge_queue(self) -> list[dict[str, Any]]:
        runtime_workers = {
            str(item.get("agent")): item for item in load_yaml(STATE_DIR / "agent_runtime.yaml").get("workers", [])
        }
        heartbeats = {
            str(item.get("agent")): item for item in load_yaml(STATE_DIR / "heartbeats.yaml").get("agents", [])
        }
        queue: list[dict[str, Any]] = []
        manager_identity = self.manager_git_identity()
        manager_display = (
            f"{manager_identity['name']} <{manager_identity['email']}>"
            if manager_identity["name"] and manager_identity["email"]
            else "A0 manager identity"
        )
        for worker in self.workers:
            agent = str(worker.get("agent", ""))
            runtime_entry = runtime_workers.get(agent, {})
            heartbeat = heartbeats.get(agent, {})
            git_identity = self.worker_git_identity(worker)
            worker_display = (
                f"{git_identity['name']} <{git_identity['email']}>"
                if git_identity["name"] and git_identity["email"]
                else "environment default"
            )
            queue.append(
                {
                    "agent": agent,
                    "branch": worker.get("branch", "unassigned"),
                    "submit_strategy": worker.get("submit_strategy", "patch_handoff"),
                    "merge_target": self.integration_branch(),
                    "worker_identity": worker_display,
                    "manager_identity": manager_display,
                    "status": runtime_entry.get("status", heartbeat.get("state", "not_started")),
                    "manager_action": f"A0 merges {worker.get('branch', 'unassigned')} into {self.integration_branch()}",
                }
            )
        return queue

    def render_prompt(self, worker: dict[str, Any], provider_name: str, model: str) -> Path:
        prompt_path = PROMPT_DIR / f"{worker['agent']}.md"
        task_id = worker.get("task_id", "unassigned")
        task_title = self.task_title(task_id)
        git_identity = self.worker_git_identity(worker)
        git_identity_text = (
            f"{git_identity['name']} <{git_identity['email']}>"
            if git_identity["name"] and git_identity["email"]
            else "environment default"
        )
        text = f"""# {worker['agent']} Worker Prompt

Repository name: {self.project.get('repository_name', 'supersonic-moe')}
Local workspace root: {self.project.get('local_repo_root', str(REPO_ROOT))}
Paddle reference repo: {self.project.get('paddle_repo_path', 'unassigned')}
Agent: {worker['agent']}
Task: {task_id} - {task_title}
Provider: {provider_name}
Model: {model}
Worktree: {worker['worktree_path']}
Branch: {worker['branch']}
Commit identity: {git_identity_text}
Manager merge target: {self.integration_branch()}

Mandatory rules:

1. Work only inside the assigned worktree.
2. Do not edit shared control-plane files unless the manager explicitly asks and the lock is held.
3. Update your status file in `control_plane/fp8/status/agents/` and your checkpoint in `control_plane/fp8/checkpoints/agents/`.
4. Treat `tests/reference_layers/standalone_moe_layer` and `{self.project.get('paddle_repo_path', 'unassigned')}` as reference inputs, not as the final host implementation.
5. Report blockers before widening scope.
6. Commit only on your assigned branch; A0 owns final merge or cherry-pick into `{self.integration_branch()}`.

First files to read:

- control_plane/fp8/strategy/integration_plan.md
- control_plane/fp8/strategy/baseline_trace.md
- control_plane/fp8/governance/operating_model.md
- control_plane/fp8/state/backlog.yaml
- control_plane/fp8/state/gates.yaml
- control_plane/fp8/state/agent_runtime.yaml

Primary test command:

{worker.get('test_command', 'unassigned')}
"""
        prompt_path.write_text(text, encoding="utf-8")
        return prompt_path

    def provider_runtime(
        self, pool_name: str, worker: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], str, str]:
        pool = self.resource_pools[pool_name]
        provider_name = worker.get("provider") or pool["provider"]
        provider = self.providers[provider_name]
        model = worker.get("model") or pool["model"]
        return provider, pool, provider_name, model

    def branch_exists(self, branch: str) -> bool:
        result = subprocess.run(
            ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def ensure_worktree(self, worker: dict[str, Any]) -> None:
        worktree_path = Path(worker["worktree_path"])
        if worktree_path.exists():
            return
        branch = worker["branch"]
        base_branch = self.project.get("base_branch", "main")
        if self.branch_exists(branch):
            command = ["git", "worktree", "add", str(worktree_path), branch]
        else:
            command = ["git", "worktree", "add", str(worktree_path), "-b", branch, base_branch]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    def ensure_environment(self, worker: dict[str, Any]) -> None:
        sync_command = worker.get("sync_command")
        if not sync_command or sync_command == "none":
            return
        result = run_shell(sync_command, Path(worker["worktree_path"]))
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "sync failed")

    def update_runtime_entry(
        self, worker: dict[str, Any], pool_name: str, provider_name: str, model: str, status: str
    ) -> None:
        runtime_path = STATE_DIR / "agent_runtime.yaml"
        runtime = load_yaml(runtime_path)
        workers = runtime.get("workers", [])
        target = None
        for entry in workers:
            if entry.get("agent") == worker["agent"]:
                target = entry
                break
        if target is None:
            target = {"agent": worker["agent"]}
            workers.append(target)
        target.update(
            {
                "repository_name": self.project.get("repository_name", "supersonic-moe"),
                "resource_pool": pool_name,
                "provider": provider_name,
                "model": model,
                "launch_owner": worker.get("launch_owner", "manager"),
                "local_workspace_root": self.project.get("local_repo_root", str(REPO_ROOT)),
                "repository_root": str(REPO_ROOT),
                "worktree_path": worker["worktree_path"],
                "branch": worker["branch"],
                "merge_target": self.integration_branch(),
                "environment_type": worker.get("environment_type", "uv"),
                "environment_path": worker.get("environment_path", "unassigned"),
                "sync_command": worker.get("sync_command", "uv sync"),
                "test_command": worker.get("test_command", "unassigned"),
                "submit_strategy": worker.get("submit_strategy", "patch_handoff"),
                "git_author_name": self.worker_git_identity(worker).get("name", ""),
                "git_author_email": self.worker_git_identity(worker).get("email", ""),
                "status": status,
            }
        )
        runtime["last_updated"] = now_iso()
        dump_yaml(runtime_path, runtime)

    def update_heartbeat(self, agent: str, state: str, evidence: str, escalation: str) -> None:
        heartbeats_path = STATE_DIR / "heartbeats.yaml"
        heartbeats = load_yaml(heartbeats_path)
        entries = heartbeats.get("agents", [])
        for entry in entries:
            if entry.get("agent") == agent:
                entry["state"] = state
                entry["last_seen"] = now_iso()
                entry["evidence"] = evidence
                entry["expected_next_checkin"] = "while worker process is alive"
                entry["escalation"] = escalation
                break
        heartbeats["last_updated"] = now_iso()
        dump_yaml(heartbeats_path, heartbeats)

    def launch_worker(self, worker: dict[str, Any]) -> dict[str, Any]:
        pool_name, evaluation = self.best_pool_for_worker(worker)
        provider, pool, provider_name, model = self.provider_runtime(pool_name, worker)
        prompt_path = self.render_prompt(worker, provider_name, model)
        self.ensure_worktree(worker)
        self.configure_git_identity(worker)
        self.ensure_environment(worker)
        template = worker.get("command_template") or provider.get("command_template")
        if not template:
            raise RuntimeError(f"no command template configured for provider {provider_name}")

        if not evaluation["binary_found"]:
            raise RuntimeError(f"provider binary missing for pool {pool_name}: {evaluation['binary']}")
        if not evaluation["api_key_present"]:
            raise RuntimeError(f"api key missing for pool {pool_name}")

        values = {
            "agent": worker["agent"],
            "model": model,
            "prompt_file": str(prompt_path),
            "worktree_path": worker["worktree_path"],
            "branch": worker["branch"],
            "repository_name": self.project.get("repository_name", "supersonic-moe"),
            "paddle_repo_path": self.project.get("paddle_repo_path", "unassigned"),
        }
        command = format_command(template, values)
        env = os.environ.copy()
        api_value = pool.get("api_key", "")
        api_env_name = provider.get("api_key_env_name")
        if api_env_name and api_value and api_value != "replace_me_or_use_api_key_env":
            env[api_env_name] = api_value
        env.update({str(key): str(value) for key, value in pool.get("extra_env", {}).items()})

        log_path = LOG_DIR / f"{worker['agent']}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=worker["worktree_path"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        previous = self.processes.get(worker["agent"])
        if previous and previous.process.poll() is None:
            previous.process.terminate()
        self.provider_stats[pool_name]["launch_successes"] += 1
        self.provider_stats[pool_name]["last_failure"] = ""
        self.processes[worker["agent"]] = WorkerProcess(
            agent=worker["agent"],
            resource_pool=pool_name,
            provider=provider_name,
            model=model,
            command=command,
            worktree_path=Path(worker["worktree_path"]),
            log_path=log_path,
            log_handle=log_handle,
            process=process,
            started_at=time.time(),
        )
        self.update_runtime_entry(worker, pool_name, provider_name, model, "active")
        self.update_heartbeat(worker["agent"], "healthy", "orchestrator_launch", "none")
        return {
            "agent": worker["agent"],
            "resource_pool": pool_name,
            "provider": provider_name,
            "model": model,
            "pid": process.pid,
            "command": command,
        }

    def launch_all(self, restart: bool = False) -> dict[str, Any]:
        with self.lock:
            if self.dry_run:
                return self.simulate_launch_all(restart=restart)
            errors = self.validation_errors()
            if errors:
                return {"ok": False, "errors": errors}
            if restart:
                self.stop_workers()
            launched: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            for worker in self.workers:
                if worker["agent"] in self.processes and self.processes[worker["agent"]].process.poll() is None:
                    continue
                try:
                    launched.append(self.launch_worker(worker))
                except Exception as exc:
                    candidate_pools = self.candidate_pools_for_worker(worker)
                    pool_name = candidate_pools[0] if candidate_pools else "unassigned"
                    if pool_name in self.provider_stats:
                        self.provider_stats[pool_name]["launch_failures"] += 1
                        self.provider_stats[pool_name]["last_failure"] = str(exc)
                    provider_name = worker.get("provider", "unassigned") or "unassigned"
                    model = worker.get("model", "unassigned") or "unassigned"
                    self.update_runtime_entry(worker, pool_name, provider_name, model, f"launch_failed: {exc}")
                    self.update_heartbeat(worker["agent"], "stale", "launch_failed", str(exc))
                    failures.append({"agent": worker["agent"], "error": str(exc)})
            self.last_event = f"launch:{len(launched)} workers"
            self.write_session_state()
            return {"ok": len(failures) == 0, "launched": launched, "failures": failures}

    def stop_workers(self) -> dict[str, Any]:
        with self.lock:
            stopped: list[str] = []
            for agent, worker in list(self.processes.items()):
                if worker.process.poll() is None:
                    worker.process.terminate()
                    try:
                        worker.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker.process.kill()
                        worker.process.wait(timeout=5)
                worker.log_handle.close()
                self.update_heartbeat(agent, "offline", "manager_stop", "none")
                runtime_entry = next((w for w in self.workers if w.get("agent") == agent), None)
                if runtime_entry:
                    self.update_runtime_entry(
                        runtime_entry, worker.resource_pool, worker.provider, worker.model, "stopped"
                    )
                stopped.append(agent)
                del self.processes[agent]
            for agent, worker in list(self.simulated_processes.items()):
                runtime_entry = next((w for w in self.workers if w.get("agent") == agent), None)
                if runtime_entry:
                    self.update_runtime_entry(
                        runtime_entry,
                        str(worker.get("resource_pool", "unassigned")),
                        str(worker.get("provider", "unassigned")),
                        str(worker.get("model", "unassigned")),
                        "stopped",
                    )
                self.update_heartbeat(agent, "offline", "dry_run_stop", "none")
                stopped.append(agent)
                del self.simulated_processes[agent]
            self.last_event = f"stop:{len(stopped)} workers"
            self.write_session_state()
            return {"ok": True, "stopped": stopped}

    def process_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for agent, worker in self.processes.items():
            snapshot[agent] = {
                "resource_pool": worker.resource_pool,
                "provider": worker.provider,
                "model": worker.model,
                "pid": worker.process.pid,
                "alive": worker.process.poll() is None,
                "returncode": worker.process.poll(),
                "worktree_path": str(worker.worktree_path),
                "log_path": str(worker.log_path),
                "command": worker.command,
            }
        snapshot.update(self.simulated_processes)
        return snapshot

    def simulate_launch_all(self, restart: bool = False) -> dict[str, Any]:
        if restart:
            self.stop_workers()
        launched: list[dict[str, Any]] = []
        for worker in self.workers:
            agent = str(worker.get("agent", "")).strip()
            if not agent or agent in self.simulated_processes:
                continue
            pool_name = str(worker.get("resource_pool") or "unassigned")
            pool = self.resource_pools.get(pool_name, {})
            provider_name = str(worker.get("provider") or pool.get("provider") or "unassigned")
            model = str(worker.get("model") or pool.get("model") or "unassigned")
            self.simulated_processes[agent] = {
                "resource_pool": pool_name,
                "provider": provider_name,
                "model": model,
                "pid": "dry-run",
                "alive": True,
                "returncode": None,
                "worktree_path": str(worker.get("worktree_path", "unassigned")),
                "log_path": "dry-run",
                "command": ["dry-run", agent],
                "simulated": True,
            }
            self.update_runtime_entry(worker, pool_name, provider_name, model, "active")
            self.update_heartbeat(agent, "healthy", "dry_run_launch", "none")
            launched.append(
                {
                    "agent": agent,
                    "resource_pool": pool_name,
                    "provider": provider_name,
                    "model": model,
                    "pid": "dry-run",
                    "command": ["dry-run", agent],
                }
            )
        self.last_event = f"dry_run_launch:{len(launched)} workers"
        self.write_session_state()
        return {"ok": True, "launched": launched, "failures": [], "dry_run": True}

    def build_dashboard_state(self) -> dict[str, Any]:
        config_text = self.config_path.read_text(encoding="utf-8") if self.config_path.exists() else ""
        return {
            "updated_at": now_iso(),
            "last_event": self.last_event,
            "mode": {
                "dry_run": self.dry_run,
                "reason": self.dry_run_reason,
                "config_path": str(self.config_path),
                "persist_config_path": str(self.persist_config_path),
            },
            "project": self.project,
            "commands": self.build_cli_commands(),
            "manager_report": MANAGER_REPORT.read_text(encoding="utf-8"),
            "runtime": load_yaml(STATE_DIR / "agent_runtime.yaml"),
            "heartbeats": load_yaml(STATE_DIR / "heartbeats.yaml"),
            "backlog": load_yaml(STATE_DIR / "backlog.yaml"),
            "gates": load_yaml(STATE_DIR / "gates.yaml"),
            "processes": self.process_snapshot(),
            "provider_queue": self.provider_queue(),
            "merge_queue": self.merge_queue(),
            "config": self.config,
            "config_text": config_text,
            "validation_errors": self.validation_errors(),
        }

    def monitor_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                for agent, worker in list(self.processes.items()):
                    returncode = worker.process.poll()
                    if returncode is None:
                        self.update_heartbeat(agent, "healthy", "process_running", "none")
                    else:
                        if returncode == 0:
                            self.provider_stats[worker.resource_pool]["clean_exits"] += 1
                        else:
                            self.provider_stats[worker.resource_pool]["failed_exits"] += 1
                            self.provider_stats[worker.resource_pool][
                                "last_failure"
                            ] = f"worker exited with {returncode}"
                        state = "offline" if returncode == 0 else "stale"
                        escalation = "worker exited cleanly" if returncode == 0 else f"worker exited with {returncode}"
                        self.update_heartbeat(agent, state, "process_exit", escalation)
                self.write_session_state()
            time.sleep(5)

    def handle_api_get(self, handler: BaseHTTPRequestHandler) -> bool:
        if handler.path == "/api/state":
            payload = json.dumps(self.build_dashboard_state(), default=str).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
            return True
        if handler.path == "/api/config":
            payload = json.dumps(
                {"config": self.config, "config_text": self.config_path.read_text(encoding="utf-8")}, default=str
            ).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
            return True
        return False

    def parse_request_json(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def write_json(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def handle_api_post(self, handler: BaseHTTPRequestHandler) -> bool:
        try:
            payload = self.parse_request_json(handler)
        except json.JSONDecodeError as exc:
            self.write_json(handler, {"ok": False, "error": f"invalid json: {exc}"}, status=400)
            return True

        if handler.path == "/api/config":
            raw_text = payload.get("config_text")
            if not isinstance(raw_text, str):
                self.write_json(handler, {"ok": False, "error": "config_text is required"}, status=400)
                return True
            try:
                errors = self.save_config_text(raw_text)
            except Exception as exc:
                self.write_json(handler, {"ok": False, "error": str(exc)}, status=400)
                return True
            if errors:
                self.write_json(handler, {"ok": False, "errors": errors}, status=400)
                return True
            self.write_json(handler, {"ok": True, "validation_errors": self.validation_errors()})
            return True

        if handler.path == "/api/launch":
            restart = bool(payload.get("restart", False))
            result = self.launch_all(restart=restart)
            self.write_json(handler, result, status=200 if result.get("ok") else 400)
            return True

        if handler.path == "/api/stop":
            result = self.stop_workers()
            self.write_json(handler, result)
            return True

        if handler.path == "/api/shutdown":
            stop_agents = bool(payload.get("stop_agents", True))
            result = {"ok": True, "stop_agents": stop_agents}
            self.write_json(handler, result)
            threading.Thread(target=self.shutdown, kwargs={"stop_agents": stop_agents}, daemon=True).start()
            return True

        return False

    def start_dashboard(self, open_browser: bool = False) -> None:
        host = self.host_override or self.project.get("dashboard", {}).get("host", "127.0.0.1")
        port = self.port_override or int(self.project.get("dashboard", {}).get("port", 8233))
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if service.handle_api_get(self):
                    return
                if self.path not in {"/", "/index.html"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if service.handle_api_post(self):
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        self.http_servers = create_http_servers(host, port, Handler)
        self.server_threads = []
        for server in self.http_servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.server_threads.append(thread)

        listen_endpoints = []
        for server in self.http_servers:
            listen_host, listen_port = server.server_address[:2]
            endpoint = format_endpoint(listen_host, listen_port)
            listen_endpoints.append(endpoint)
            print(f"control plane listening on {endpoint}", file=sys.stderr, flush=True)
        self.listen_host = host
        self.listen_port = listen_port
        self.listen_endpoints = listen_endpoints
        self.write_session_state()
        if host in {"0.0.0.0", "::"}:
            print(
                f"remote access URL: http://<server-hostname-or-ip>:{listen_port}",
                file=sys.stderr,
                flush=True,
            )
        if open_browser:
            webbrowser.open(f"http://{browser_open_host(host)}:{listen_port}")
        self.last_event = f"dashboard:{', '.join(listen_endpoints)}"

    def wait_forever(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(1)

    def start_monitoring(self) -> None:
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def run_up(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        if self.dry_run:
            self.last_event = f"dry_run:{self.dry_run_reason or 'dashboard only'}"
        else:
            self.launch_all()
        self.wait_forever()

    def run_serve(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        self.wait_forever()

    def shutdown(self, stop_agents: bool = True) -> None:
        self.stop_event.set()
        if stop_agents:
            self.stop_workers()
        for server in self.http_servers:
            server.shutdown()
            server.server_close()
        self.write_session_state()


DASHBOARD_HTML = """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>supersonic-moe control plane</title>
    <style>
        :root { color-scheme: dark; }
        body { font-family: "Avenir Next", "Segoe UI", sans-serif; margin: 0; background: radial-gradient(circle at top, #15345f 0%, #08111f 42%, #060b14 100%); color: #edf2ff; }
        header { padding: 30px 24px 26px; background: linear-gradient(135deg, rgba(13, 28, 51, 0.96), rgba(13, 93, 89, 0.88)); border-bottom: 1px solid rgba(157, 196, 255, 0.12); }
        main { max-width: 1380px; margin: 0 auto; padding: 18px 20px 36px; display: grid; gap: 14px; }
        .card { background: rgba(16, 25, 43, 0.84); border: 1px solid rgba(87, 118, 163, 0.32); border-radius: 16px; padding: 16px; box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2); backdrop-filter: blur(12px); }
        .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        .summary { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
        .metric { padding: 14px; border-radius: 12px; background: linear-gradient(180deg, rgba(12, 19, 34, 0.95), rgba(10, 17, 28, 0.88)); border: 1px solid rgba(72, 102, 145, 0.34); }
        .metric strong { display: block; font-size: 28px; line-height: 1.1; margin-bottom: 4px; }
        h1, h2, h3, p { margin: 0 0 10px; }
        h2 { font-size: 18px; }
        h3 { font-size: 14px; color: #c7d5f8; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #1e2b42; vertical-align: top; }
        th { color: #b6c6ea; font-weight: 600; }
        pre, textarea, code { font-family: ui-monospace, monospace; }
        pre { white-space: pre-wrap; background: #0c1322; padding: 12px; border-radius: 10px; max-height: 320px; overflow: auto; margin: 0; }
        textarea { width: 100%; min-height: 420px; background: #0c1322; color: #edf2ff; border: 1px solid #243252; border-radius: 10px; padding: 12px; box-sizing: border-box; resize: vertical; }
        button { background: #1f7a6d; color: #fff; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
        button.secondary { background: #29405f; }
        button.ghost { background: #18243a; }
        button.danger { background: #8a3d52; }
        button.nav-button { background: transparent; border: 1px solid transparent; color: #b6c6ea; }
        button.nav-button.active { background: rgba(27, 115, 101, 0.2); border-color: rgba(87, 211, 183, 0.35); color: #eefbff; }
        button:disabled { opacity: 0.45; cursor: not-allowed; }
        .small { color: #a7b5d6; font-size: 13px; }
        .muted { color: #93a7cc; }
        .status { padding: 4px 0 0; color: #9ce5c7; min-height: 22px; }
        .error { color: #ffb2c0; }
        .hero { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px; }
        .hero-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: rgba(9, 18, 33, 0.46); border: 1px solid rgba(155, 203, 255, 0.18); color: #cfe1ff; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; }
        .tagline { max-width: 760px; }
        .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
        .toolbar-group { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .toggle { display: inline-flex; gap: 8px; align-items: center; color: #b6c6ea; font-size: 13px; }
        .tab-nav { display: flex; flex-wrap: wrap; gap: 8px; }
        .tab-panel { display: none; gap: 14px; }
        .tab-panel.active { display: grid; }
        .panel-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
        .overview-hero { display: grid; gap: 14px; grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.8fr); }
        .progress-card { display: grid; gap: 12px; }
        .progress-bar { height: 12px; border-radius: 999px; background: rgba(31, 45, 70, 0.92); overflow: hidden; border: 1px solid rgba(72, 102, 145, 0.34); }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #1f7a6d, #6ad1c3); width: 0%; }
        .progress-list { display: grid; gap: 8px; }
        .progress-row { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
        .merge-board { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
        .merge-card { padding: 14px; border-radius: 14px; background: linear-gradient(180deg, rgba(10, 18, 30, 0.96), rgba(10, 22, 39, 0.84)); border: 1px solid rgba(74, 105, 149, 0.34); display: grid; gap: 10px; }
        .merge-card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
        .merge-branch { font-size: 18px; font-weight: 700; line-height: 1.2; }
        .merge-track { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #b8c8e6; }
        .merge-arrow { color: #6ad1c3; font-weight: 700; }
        .merge-meta { display: grid; gap: 6px; font-size: 13px; }
        .merge-meta strong { color: #9fbbe8; }
        .merge-note { padding-top: 4px; color: #a7b5d6; font-size: 12px; }
        .agent-wall { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
        .agent-card { padding: 14px; border-radius: 14px; background: linear-gradient(180deg, rgba(11, 18, 31, 0.96), rgba(12, 21, 37, 0.84)); border: 1px solid rgba(74, 105, 149, 0.34); min-height: 148px; display: grid; gap: 10px; }
        .agent-card header { padding: 0; background: none; border: 0; display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
        .agent-name { font-size: 20px; font-weight: 700; line-height: 1; }
        .agent-role { color: #9fb2d5; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
        .agent-meta { display: grid; gap: 6px; font-size: 13px; color: #d9e5ff; }
        .agent-meta strong { color: #9fbbe8; font-weight: 600; }
        .chip { display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.02em; }
        .state-healthy, .state-active { background: rgba(31, 122, 109, 0.22); color: #9ff0d8; border: 1px solid rgba(63, 197, 170, 0.32); }
        .state-stale, .state-launch_failed { background: rgba(160, 92, 36, 0.22); color: #ffd4a6; border: 1px solid rgba(223, 142, 76, 0.34); }
        .state-offline, .state-stopped { background: rgba(75, 92, 124, 0.22); color: #cad6ef; border: 1px solid rgba(118, 138, 175, 0.3); }
        .state-not_started, .state-not-started, .state-unassigned { background: rgba(50, 61, 88, 0.28); color: #d4def5; border: 1px solid rgba(111, 126, 163, 0.28); }
        .state-error { background: rgba(138, 61, 82, 0.26); color: #ffbfd0; border: 1px solid rgba(212, 101, 133, 0.34); }
        .section-stack { display: grid; gap: 14px; }
        .page-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
        .config-layout { display: grid; gap: 14px; grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr); }
        .helper-list { display: grid; gap: 12px; }
        .helper-card { padding: 12px; border-radius: 12px; background: rgba(12, 19, 34, 0.86); border: 1px solid rgba(72, 102, 145, 0.28); }
        .pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .key-pair { display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 999px; background: rgba(12, 19, 34, 0.9); border: 1px solid rgba(72, 102, 145, 0.28); font-size: 12px; }
        @media (max-width: 980px) {
            .overview-hero, .config-layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <div class=\"hero\">
            <div>
                <div class=\"hero-badge\">FP8 delivery orchestration</div>
                <h1>supersonic-moe control plane</h1>
                <p class=\"small tagline\">Overview stays focused on agent health and program progress. Operations and settings move into dedicated pages.</p>
            </div>
        </div>
    </header>
    <main>
        <section class=\"card\">
            <div class=\"toolbar\">
                <div class=\"toolbar-group\">
                    <button data-action onclick=\"launchWorkers(false)\">Launch</button>
                    <button data-action class=\"secondary\" onclick=\"launchWorkers(true)\">Restart</button>
                    <button data-action class=\"danger\" onclick=\"stopWorkers()\">Stop</button>
                    <button data-action class=\"ghost\" onclick=\"refresh(true)\">Refresh</button>
                </div>
                <div class=\"toolbar-group\">
                    <button class=\"ghost\" onclick=\"copyCommand('serve')\">Copy Serve</button>
                    <button class=\"ghost\" onclick=\"copyCommand('up')\">Copy Up</button>
                    <label class=\"toggle\"><input id=\"auto_refresh\" type=\"checkbox\" checked> Auto refresh</label>
                </div>
            </div>
            <div id=\"status_line\" class=\"status\"></div>
        </section>

        <section class=\"card\">
            <div class=\"toolbar\">
                <div class=\"tab-nav\" role=\"tablist\" aria-label=\"Dashboard sections\">
                    <button id=\"nav_overview\" class=\"nav-button active\" type=\"button\" onclick=\"showTab('overview')\">Overview</button>
                    <button id=\"nav_operations\" class=\"nav-button\" type=\"button\" onclick=\"showTab('operations')\">Operations</button>
                    <button id=\"nav_settings\" class=\"nav-button\" type=\"button\" onclick=\"showTab('settings')\">Settings</button>
                </div>
                <div class=\"pill-row\" id=\"top_meta\"></div>
            </div>
        </section>

        <section id=\"tab_overview\" class=\"tab-panel active\">
            <section class=\"overview-hero\">
                <section class=\"card progress-card\">
                    <div class=\"page-header\">
                        <div>
                            <h2>Overall Progress</h2>
                            <p class=\"small\">A compact view of delivery momentum and the current control-plane state.</p>
                        </div>
                        <div class=\"small muted\" id=\"progress_meta\"></div>
                    </div>
                    <div class=\"progress-bar\"><div id=\"progress_fill\" class=\"progress-fill\"></div></div>
                    <div class=\"summary\" id=\"summary\"></div>
                    <div id=\"progress_details\" class=\"progress-list\"></div>
                </section>
                <section class=\"card\">
                    <div class=\"page-header\">
                        <div>
                            <h2>Program Snapshot</h2>
                            <p class=\"small\">What is blocked, what is runnable, and which event happened last.</p>
                        </div>
                    </div>
                    <div id=\"overview_snapshot\" class=\"helper-list\"></div>
                </section>
            </section>

            <section class=\"card\">
                <div class=\"panel-title\">
                    <div>
                        <h2>Branch Merge Status</h2>
                        <p class=\"small\">A0-owned merge visibility for every worker branch, without requiring manual worktree management.</p>
                    </div>
                    <div class=\"small muted\" id=\"merge_status_meta\"></div>
                </div>
                <div id=\"overview_merge_board\" class=\"merge-board\"></div>
            </section>

            <section class=\"card\">
                <div class=\"panel-title\">
                    <div>
                        <h2>Agent Dashboards</h2>
                        <p class=\"small\">The home page only shows agent health and essential execution context.</p>
                    </div>
                    <div class=\"small muted\" id=\"agent_status_meta\"></div>
                </div>
                <div class=\"agent-wall\" id=\"agent_wall\"></div>
            </section>
        </section>

        <section id=\"tab_operations\" class=\"tab-panel\">
            <section class=\"grid\">
                <section class=\"card\">
                    <h2>Commands</h2>
                    <pre id=\"commands\"></pre>
                </section>
                <section class=\"card\">
                    <h2>Validation</h2>
                    <pre id=\"validation\"></pre>
                </section>
            </section>
            <section class=\"grid\">
                <section class=\"card\">
                    <h2>Provider Queue</h2>
                    <div id=\"provider_queue\"></div>
                </section>
                <section class=\"card\">
                    <h2>Merge Queue</h2>
                    <div id=\"merge_queue\"></div>
                </section>
            </section>
            <section class=\"grid\">
                <section class=\"card\">
                    <h2>Active Processes</h2>
                    <div id=\"processes\"></div>
                </section>
                <section class=\"card\">
                    <h2>Project</h2>
                    <div id=\"project\"></div>
                </section>
            </section>
            <section class=\"grid\">
                <section class=\"card\">
                    <h2>Runtime Topology</h2>
                    <div id=\"runtime\"></div>
                </section>
                <section class=\"card\">
                    <h2>Heartbeats</h2>
                    <div id=\"heartbeats\"></div>
                </section>
            </section>
            <section class=\"grid\">
                <section class=\"card\">
                    <h2>Backlog</h2>
                    <div id=\"backlog\"></div>
                </section>
                <section class=\"card\">
                    <h2>Gates</h2>
                    <div id=\"gates\"></div>
                </section>
            </section>
            <section class=\"card\">
                <h2>Manager Report</h2>
                <pre id=\"manager_report\"></pre>
            </section>
        </section>

        <section id=\"tab_settings\" class=\"tab-panel\">
            <section class=\"card\">
                <div class=\"page-header\">
                    <div>
                        <h2>Settings</h2>
                        <p class=\"small\">Edit API keys, provider routing, worktrees, Paddle path, and worker commands here.</p>
                    </div>
                    <button data-action onclick=\"saveConfig()\">Save Settings</button>
                </div>
                <div class=\"config-layout\">
                    <div>
                        <textarea id=\"config_editor\"></textarea>
                    </div>
                    <div class=\"helper-list\">
                        <section class=\"helper-card\">
                            <h3>Project</h3>
                            <div id=\"project\"></div>
                        </section>
                        <section class=\"helper-card\">
                            <h3>Resource Pools</h3>
                            <div id=\"resource_pools\"></div>
                        </section>
                        <section class=\"helper-card\">
                            <h3>Merge Policy</h3>
                            <div id=\"merge_policy\"></div>
                        </section>
                        <section class=\"helper-card\">
                            <h3>Worker Config</h3>
                            <div id=\"workers\"></div>
                        </section>
                    </div>
                </div>
            </section>
        </section>
    </main>
    <script>
        let editorDirty = false;
        let actionInFlight = false;
        let latestRefreshOk = false;
        let latestState = {};
        let currentCommands = { serve: '', up: '' };
        let currentTab = 'overview';

        const editor = () => document.getElementById('config_editor');
        const actionButtons = () => Array.from(document.querySelectorAll('[data-action]'));
        const autoRefresh = () => document.getElementById('auto_refresh').checked;

        function renderTable(rows, columns) {
            if (!rows || !rows.length) {
                return '<div class="small muted">No data</div>';
            }
            const head = columns.map((col) => `<th>${col}</th>`).join('');
            const body = rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ''}</td>`).join('')}</tr>`).join('');
            return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
        }

        function sortAgents(values) {
            return values.sort((left, right) => {
                const leftNum = Number(String(left.agent || '').replace(/[^0-9]/g, ''));
                const rightNum = Number(String(right.agent || '').replace(/[^0-9]/g, ''));
                return leftNum - rightNum;
            });
        }

        function displayState(value) {
            return String(value || 'unknown').replaceAll('_', ' ');
        }

        function stateClass(value) {
            return `state-${String(value || 'unknown').replace(/[^a-zA-Z0-9]+/g, '_')}`;
        }

        function buildAgentRows(data) {
            const byAgent = new Map();
            const remember = (agent, values) => {
                if (!agent) {
                    return;
                }
                byAgent.set(agent, { ...(byAgent.get(agent) || { agent }), ...values, agent });
            };

            (data.runtime?.workers || []).forEach((item) => {
                remember(item.agent, {
                    provider: item.provider,
                    model: item.model,
                    resource_pool: item.resource_pool,
                    branch: item.branch,
                    runtime_status: item.status,
                });
            });
            (data.config?.workers || []).forEach((item) => {
                remember(item.agent, {
                    task_id: item.task_id,
                    config_pool: item.resource_pool,
                    queue: (item.resource_pool_queue || []).join(' -> '),
                    branch: item.branch,
                    test_command: item.test_command,
                });
            });
            (data.heartbeats?.agents || []).forEach((item) => {
                remember(item.agent, {
                    role: item.role,
                    heartbeat_state: item.state,
                    last_seen: item.last_seen,
                    expected_next_checkin: item.expected_next_checkin,
                    evidence: item.evidence,
                    escalation: item.escalation,
                });
            });
            Object.entries(data.processes || {}).forEach(([agent, item]) => {
                remember(agent, {
                    process_alive: item.alive,
                    pid: item.pid,
                    provider: item.provider,
                    model: item.model,
                    resource_pool: item.resource_pool,
                });
            });

            return sortAgents(Array.from(byAgent.values())).map((item) => {
                const state = item.process_alive ? 'active' : (item.heartbeat_state || item.runtime_status || 'not_started');
                return {
                    ...item,
                    display_state: state,
                    provider: item.provider || 'unassigned',
                    model: item.model || 'unassigned',
                    resource_pool: item.resource_pool || item.config_pool || 'unassigned',
                    branch: item.branch || 'unassigned',
                    role: item.role || 'worker',
                };
            });
        }

        function buildProgressModel(data, agentRows) {
            const gates = data.gates?.gates || [];
            const backlog = data.backlog?.items || [];
            const passedGates = gates.filter((item) => item.status === 'passed').length;
            const progress = gates.length ? Math.round((passedGates / gates.length) * 100) : 0;
            const completedItems = backlog.filter((item) => ['done', 'completed', 'closed'].includes(String(item.status))).length;
            const blockedItems = backlog.filter((item) => String(item.status) === 'blocked').length;
            const activeAgents = agentRows.filter((item) => item.display_state === 'active' || item.display_state === 'healthy').length;
            const attentionAgents = agentRows.filter((item) => item.display_state === 'stale' || String(item.display_state).startsWith('launch_failed')).length;
            const openGate = gates.find((item) => item.status !== 'passed');
            return {
                progress,
                passedGates,
                totalGates: gates.length,
                completedItems,
                totalItems: backlog.length,
                blockedItems,
                activeAgents,
                attentionAgents,
                openGate,
            };
        }

        function renderSummaryCards(data) {
            const agentRows = buildAgentRows(data);
            const progress = buildProgressModel(data, agentRows);
            const validationCount = (data.validation_errors || []).length;
            const cards = [
                { label: 'Agents', value: agentRows.length, hint: `${progress.activeAgents} active or healthy` },
                { label: 'Overall Progress', value: `${progress.progress}%`, hint: `${progress.passedGates}/${progress.totalGates} gates passed` },
                { label: 'Attention Needed', value: progress.attentionAgents, hint: `${progress.blockedItems} backlog items blocked` },
                { label: 'Validation Issues', value: validationCount, hint: validationCount ? 'settings need cleanup' : 'settings are ready' },
            ];
            return cards.map((item) => `<div class=\"metric\"><strong>${item.value}</strong><div>${item.label}</div><div class=\"small\">${item.hint}</div></div>`).join('');
        }

        function renderOverviewSnapshot(data, progress) {
            const entries = [
                { title: 'Current gate', body: progress.openGate ? `${progress.openGate.id} · ${progress.openGate.name}` : 'All gates passed' },
                { title: 'Backlog completion', body: `${progress.completedItems}/${progress.totalItems} items complete` },
                { title: 'Last event', body: data.last_event || 'none' },
            ];
            return entries.map((item) => `<section class=\"helper-card\"><h3>${item.title}</h3><p class=\"small\">${item.body}</p></section>`).join('');
        }

        function renderProgressDetails(progress) {
            const rows = [
                { label: 'Gates', value: `${progress.passedGates}/${progress.totalGates} passed` },
                { label: 'Backlog', value: `${progress.completedItems}/${progress.totalItems} completed` },
                { label: 'Blocked work', value: `${progress.blockedItems} items` },
                { label: 'Agents needing action', value: `${progress.attentionAgents}` },
            ];
            return rows.map((item) => `<div class=\"progress-row\"><span class=\"small\">${item.label}</span><strong>${item.value}</strong></div>`).join('');
        }

        function mergeDisplayStatus(item) {
            const raw = String(item.status || 'not_started');
            if (raw === 'active' || raw === 'healthy') {
                return { label: 'In progress', className: 'state-active' };
            }
            if (raw === 'stale' || raw.startsWith('launch_failed')) {
                return { label: 'Needs attention', className: 'state-stale' };
            }
            if (raw === 'offline' || raw === 'stopped') {
                return { label: 'Ready for review', className: 'state-offline' };
            }
            return { label: 'Queued', className: 'state-not_started' };
        }

        function renderMergeBoard(items) {
            const rows = items || [];
            const reviewReady = rows.filter((item) => ['offline', 'stopped'].includes(String(item.status))).length;
            const inFlight = rows.filter((item) => ['active', 'healthy'].includes(String(item.status))).length;
            document.getElementById('merge_status_meta').textContent = `${inFlight} in progress, ${reviewReady} ready for manager review`;
            if (!rows.length) {
                return '<div class="small muted">No worker branches registered for manager merge review.</div>';
            }
            return rows.map((item) => {
                const status = mergeDisplayStatus(item);
                return `
                    <article class="merge-card">
                        <div class="merge-card-header">
                            <div>
                                <div class="merge-branch">${item.branch}</div>
                                <div class="merge-track">
                                    <span>${item.agent}</span>
                                    <span class="merge-arrow">-></span>
                                    <span>${item.merge_target}</span>
                                </div>
                            </div>
                            <span class="chip ${status.className}">${status.label}</span>
                        </div>
                        <div class="merge-meta">
                            <div><strong>Submit</strong> ${item.submit_strategy}</div>
                            <div><strong>Worker identity</strong> ${item.worker_identity}</div>
                            <div><strong>Manager</strong> ${item.manager_identity}</div>
                        </div>
                        <div class="merge-note">${item.manager_action}</div>
                    </article>
                `;
            }).join('');
        }

        function renderAgentWall(data) {
            const rows = buildAgentRows(data);
            const meta = `${rows.filter((item) => item.display_state === 'active' || item.display_state === 'healthy').length} active, ${rows.filter((item) => item.display_state === 'stale' || String(item.display_state).startsWith('launch_failed')).length} need attention`;
            document.getElementById('agent_status_meta').textContent = meta;
            return rows.map((item) => {
                const processLine = item.process_alive ? `pid ${item.pid}` : (item.last_seen || 'no heartbeat yet');
                const poolLine = `${item.resource_pool} / ${item.provider}`;
                const branchLine = item.branch || 'unassigned';
                const detailLine = item.process_alive ? 'process alive' : (item.evidence || item.expected_next_checkin || 'waiting for launch');
                return `
                    <article class="agent-card">
                        <header>
                            <div>
                                <div class="agent-name">${item.agent}</div>
                                <div class="agent-role">${item.role}</div>
                            </div>
                            <span class="chip ${stateClass(item.display_state)}">${displayState(item.display_state)}</span>
                        </header>
                        <div class="agent-meta">
                            <div><strong>Pool</strong> ${poolLine}</div>
                            <div><strong>Model</strong> ${item.model}</div>
                            <div><strong>Branch</strong> ${branchLine}</div>
                            <div><strong>Heartbeat</strong> ${processLine}</div>
                            <div class="muted">${detailLine}</div>
                        </div>
                    </article>
                `;
            }).join('');
        }

        function setStatus(message, isError = false) {
            const node = document.getElementById('status_line');
            const stamp = new Date().toLocaleTimeString();
            node.textContent = `[${stamp}] ${message}`;
            node.className = isError ? 'status error' : 'status';
        }

        function setActionState(inFlight) {
            actionInFlight = inFlight;
            actionButtons().forEach((button) => {
                button.disabled = inFlight;
            });
        }

        function showTab(tabName) {
            currentTab = tabName;
            ['overview', 'operations', 'settings'].forEach((name) => {
                document.getElementById(`tab_${name}`).classList.toggle('active', name === tabName);
                document.getElementById(`nav_${name}`).classList.toggle('active', name === tabName);
            });
        }

        async function fetchJson(path, options = {}) {
            const response = await fetch(path, options);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || (data.errors || []).join('\n') || 'request failed');
            }
            return data;
        }

        async function postJson(path, payload) {
            return fetchJson(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
        }

        async function copyCommand(mode) {
            const value = currentCommands[mode] || '';
            if (!value) {
                setStatus(`no ${mode} command available`, true);
                return;
            }
            await navigator.clipboard.writeText(value);
            setStatus(`${mode} command copied`);
        }

        async function runAction(label, action) {
            if (actionInFlight) {
                return;
            }
            setActionState(true);
            setStatus(`${label}...`);
            try {
                await action();
            } catch (error) {
                setStatus(String(error), true);
            } finally {
                setActionState(false);
            }
        }

        function renderProject(project) {
            const rows = [
                { key: 'repository_name', value: project.repository_name || '' },
                { key: 'local_repo_root', value: project.local_repo_root || '' },
                { key: 'paddle_repo_path', value: project.paddle_repo_path || '' },
                { key: 'integration_branch', value: project.integration_branch || project.base_branch || '' },
                { key: 'dashboard', value: `${project.dashboard.host}:${project.dashboard.port}` },
            ];
            return renderTable(rows, ['key', 'value']);
        }

        function renderProcessRows(processes) {
            return Object.entries(processes).map(([agent, item]) => ({
                agent,
                provider: item.provider,
                model: item.model,
                alive: item.alive,
                pid: item.pid,
                resource_pool: item.resource_pool,
                returncode: item.returncode,
            }));
        }

        function renderRuntime(runtime) {
            return renderTable(runtime.workers || [], ['agent', 'resource_pool', 'provider', 'model', 'branch', 'status']);
        }

        function renderHeartbeats(heartbeats) {
            return renderTable(heartbeats.agents || [], ['agent', 'state', 'last_seen', 'expected_next_checkin']);
        }

        function renderBacklog(backlog) {
            return renderTable(backlog.items || [], ['id', 'owner', 'status', 'gate', 'title']);
        }

        function renderGates(gates) {
            return renderTable(gates.gates || [], ['id', 'name', 'status', 'owner']);
        }

        function renderPools(config) {
            const pools = Object.entries(config.resource_pools || {}).map(([name, item]) => ({
                name,
                priority: item.priority ?? 100,
                provider: item.provider,
                model: item.model,
            }));
            return renderTable(pools, ['name', 'priority', 'provider', 'model']);
        }

        function renderWorkers(config) {
            const rows = (config.workers || []).map((item) => ({
                agent: item.agent,
                task_id: item.task_id,
                resource_pool: item.resource_pool,
                resource_pool_queue: (item.resource_pool_queue || []).join(', '),
                branch: item.branch,
                git_identity: item.git_identity ? `${item.git_identity.name || ''} <${item.git_identity.email || ''}>` : 'environment default',
                submit_strategy: item.submit_strategy,
                test_command: item.test_command,
            }));
            return renderTable(rows, ['agent', 'task_id', 'resource_pool', 'resource_pool_queue', 'branch', 'git_identity', 'submit_strategy', 'test_command']);
        }

        function renderProviderQueue(items) {
            return renderTable(items || [], ['resource_pool', 'provider', 'priority', 'binary_found', 'api_key_present', 'connection_quality', 'work_quality', 'score']);
        }

        function renderMergeQueue(items) {
            return renderTable(items || [], ['agent', 'branch', 'submit_strategy', 'worker_identity', 'merge_target', 'status', 'manager_action']);
        }

        function renderMergePolicy(project, mergeQueue) {
            const manager = project.manager_git_identity
                ? `${project.manager_git_identity.name || ''} <${project.manager_git_identity.email || ''}>`
                : 'A0 manager identity';
            const rows = [
                { key: 'integration_branch', value: project.integration_branch || project.base_branch || 'main' },
                { key: 'manager_identity', value: manager },
                { key: 'merge_owner', value: 'A0' },
                { key: 'tracked_worker_branches', value: String((mergeQueue || []).length) },
            ];
            return renderTable(rows, ['key', 'value']);
        }

        function renderTopMeta(data) {
            const mode = data.mode || {};
            const values = [
                { label: 'Mode', value: mode.dry_run ? 'dry run' : 'live' },
                { label: 'Config', value: mode.config_path || 'unknown' },
                { label: 'Last event', value: data.last_event || 'none' },
                { label: 'Updated', value: data.updated_at || 'unknown' },
            ];
            return values.map((item) => `<div class=\"key-pair\"><span class=\"muted\">${item.label}</span><strong>${item.value}</strong></div>`).join('');
        }

        async function saveConfig() {
            await runAction('saving settings', async () => {
                await postJson('/api/config', { config_text: editor().value });
                editorDirty = false;
                await refresh(true);
                setStatus('settings saved');
            });
        }

        async function launchWorkers(restart) {
            await runAction(restart ? 'restarting workers' : 'launching workers', async () => {
                const data = await postJson('/api/launch', { restart });
                setStatus(`launch complete: ${(data.launched || []).length} launched, ${(data.failures || []).length} failures`, !(data.ok ?? false));
                await refresh(true);
            });
        }

        async function stopWorkers() {
            await runAction('stopping workers', async () => {
                const data = await postJson('/api/stop', {});
                setStatus(`stopped workers: ${(data.stopped || []).join(', ') || 'none'}`);
                await refresh(true);
            });
        }

        async function refresh(forceStatus = false) {
            try {
                const data = await fetchJson('/api/state');
                latestState = data;
                latestRefreshOk = true;
                currentCommands = data.commands || { serve: '', up: '' };
                const agentRows = buildAgentRows(data);
                const progress = buildProgressModel(data, agentRows);
                if (!editorDirty) {
                    editor().value = data.config_text || '';
                }
                document.getElementById('top_meta').innerHTML = renderTopMeta(data);
                document.getElementById('summary').innerHTML = renderSummaryCards(data);
                document.getElementById('agent_wall').innerHTML = renderAgentWall(data);
                document.getElementById('progress_fill').style.width = `${progress.progress}%`;
                document.getElementById('progress_meta').textContent = `${progress.passedGates}/${progress.totalGates} gates passed`;
                document.getElementById('progress_details').innerHTML = renderProgressDetails(progress);
                document.getElementById('overview_snapshot').innerHTML = renderOverviewSnapshot(data, progress);
                document.getElementById('overview_merge_board').innerHTML = renderMergeBoard(data.merge_queue);
                document.getElementById('merge_queue').innerHTML = renderMergeQueue(data.merge_queue);
                document.getElementById('project').innerHTML = renderProject(data.project);
                document.getElementById('processes').innerHTML = renderTable(renderProcessRows(data.processes), ['agent', 'provider', 'model', 'alive', 'pid', 'resource_pool', 'returncode']);
                document.getElementById('provider_queue').innerHTML = renderProviderQueue(data.provider_queue);
                document.getElementById('runtime').innerHTML = renderRuntime(data.runtime);
                document.getElementById('heartbeats').innerHTML = renderHeartbeats(data.heartbeats);
                document.getElementById('backlog').innerHTML = renderBacklog(data.backlog);
                document.getElementById('gates').innerHTML = renderGates(data.gates);
                document.getElementById('resource_pools').innerHTML = renderPools(data.config);
                document.getElementById('workers').innerHTML = renderWorkers(data.config);
                document.getElementById('merge_policy').innerHTML = renderMergePolicy(data.project, data.merge_queue);
                document.getElementById('manager_report').textContent = data.manager_report;
                document.getElementById('commands').textContent = `serve:\n${currentCommands.serve}\n\nup:\n${currentCommands.up}`;
                document.getElementById('validation').textContent = data.validation_errors.length ? data.validation_errors.join('\n') : 'settings valid';
                if (forceStatus) {
                    setStatus(`state refreshed, last event: ${data.last_event || 'none'}`);
                }
            } catch (error) {
                if (latestRefreshOk || forceStatus) {
                    setStatus(`refresh failed: ${error}`, true);
                }
                latestRefreshOk = false;
            }
        }

        editor().addEventListener('input', () => {
            editorDirty = true;
        });

        showTab(currentTab);
        refresh(true);
        setInterval(() => {
            if (!actionInFlight && autoRefresh()) {
                refresh(false);
            }
        }, 4000);
    </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="supersonic-moe control plane runtime")
    parser.add_argument(
        "command",
        choices=["up", "serve", "stop-agents", "stop-listener", "stop-all"],
        help="launch the control plane, or stop agents/listener from a running session",
    )
    parser.add_argument("--config", type=Path, default=RUNTIME_DIR / "local_config.yaml", help="runtime config path")
    parser.add_argument("--host", default=None, help="override dashboard host")
    parser.add_argument("--port", type=int, default=None, help="override dashboard port")
    parser.add_argument("--open-browser", action="store_true", help="open the dashboard in a browser")
    parser.add_argument("--detach", action="store_true", help="start the control plane in the background and return")
    parser.add_argument(
        "--foreground", action="store_true", help="keep the control plane attached to the current shell"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the dashboard without launching workers; if the config file is missing, fall back to config_template.yaml",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=RUNTIME_DIR / "control_plane.log",
        help="log file used when --detach is enabled",
    )
    return parser.parse_args()


def apply_runtime_defaults(args: argparse.Namespace, dry_run: bool) -> None:
    if args.command == "serve" and dry_run:
        if args.host is None:
            args.host = DEFAULT_DASHBOARD_HOST
        if not args.foreground:
            args.detach = True


def stop_agents_command(args: argparse.Namespace) -> int:
    session_state = load_session_state_file()
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    result = post_control_plane(control_plane_base_url(args, session_state), "/api/stop", {})
    print(json.dumps(result, indent=2))
    return 0


def stop_listener_command(args: argparse.Namespace) -> int:
    session_state = load_session_state_file()
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    try:
        result = post_control_plane(
            control_plane_base_url(args, session_state), "/api/shutdown", {"stop_agents": False}
        )
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError:
        server_pid = int(session_state.get("server", {}).get("pid") or 0)
        if not server_pid or not pid_is_running(server_pid):
            print("listener is not running", file=sys.stderr)
            return 1
        terminate_pid(server_pid)
        print(json.dumps({"ok": True, "listener_pid": server_pid, "stop_agents": False}, indent=2))
        return 0


def stop_all_command(args: argparse.Namespace) -> int:
    session_state = load_session_state_file()
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    try:
        result = post_control_plane(
            control_plane_base_url(args, session_state), "/api/shutdown", {"stop_agents": True}
        )
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError:
        stopped_workers: list[int] = []
        for worker in session_state.get("workers", {}).values():
            pid = int(worker.get("pid") or 0)
            if pid and pid_is_running(pid):
                terminate_pid(pid)
                stopped_workers.append(pid)
        server_pid = int(session_state.get("server", {}).get("pid") or 0)
        if server_pid and pid_is_running(server_pid):
            terminate_pid(server_pid)
        print(
            json.dumps(
                {
                    "ok": True,
                    "listener_pid": server_pid,
                    "stopped_worker_pids": stopped_workers,
                    "stop_agents": True,
                },
                indent=2,
            )
        )
        return 0


def resolve_runtime_config(args: argparse.Namespace) -> tuple[Path, Path, bool, str]:
    requested_path = args.config
    persist_path = requested_path
    force_dry_run = args.dry_run
    reasons: list[str] = []
    default_local_config = (RUNTIME_DIR / "local_config.yaml").resolve()

    if not requested_path.exists():
        if requested_path.resolve() == default_local_config or args.dry_run:
            if not CONFIG_TEMPLATE_PATH.exists():
                raise FileNotFoundError(f"missing template config: {CONFIG_TEMPLATE_PATH}")
            requested_path = CONFIG_TEMPLATE_PATH
            force_dry_run = True
            reasons.append(f"booted from template because {persist_path} does not exist")
        else:
            raise FileNotFoundError(f"missing config: {persist_path}")

    if requested_path.resolve() == CONFIG_TEMPLATE_PATH.resolve():
        force_dry_run = True
        if persist_path == requested_path:
            persist_path = RUNTIME_DIR / "local_config.yaml"
        reasons.append("template-backed session keeps launch actions disabled until a real config is saved")

    return requested_path, persist_path, force_dry_run, "; ".join(dict.fromkeys(reasons))


def detach_process(args: argparse.Namespace) -> int:
    log_path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = str(Path(__file__).resolve())
    if shutil.which("uv"):
        command = ["uv", "run", "--no-project", "--with", "PyYAML>=6.0.2", "python", script_path, args.command]
    else:
        command = [sys.executable, script_path, args.command]
    command.extend(["--config", str(args.config)])
    if args.host is not None:
        command.extend(["--host", args.host])
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    if args.open_browser:
        command.append("--open-browser")
    if args.dry_run:
        command.append("--dry-run")

    env = os.environ.copy()
    env["CONTROL_PLANE_DETACHED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"control plane started in background: pid={process.pid} log={log_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "stop-agents":
        return stop_agents_command(args)
    if args.command == "stop-listener":
        return stop_listener_command(args)
    if args.command == "stop-all":
        return stop_all_command(args)

    try:
        config_path, persist_config_path, force_dry_run, dry_run_reason = resolve_runtime_config(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"use --dry-run or point --config at {CONFIG_TEMPLATE_PATH} to render the dashboard without a filled local config",
            file=sys.stderr,
        )
        return 2

    apply_runtime_defaults(args, force_dry_run)

    if args.detach and os.environ.get("CONTROL_PLANE_DETACHED") != "1":
        args.config = config_path
        args.dry_run = force_dry_run
        return detach_process(args)

    service = ControlPlaneService(
        config_path,
        host_override=args.host,
        port_override=args.port,
        persist_config_path=persist_config_path,
        force_dry_run=force_dry_run,
    )
    if service.dry_run and dry_run_reason:
        service.dry_run_reason = dry_run_reason
        service.last_event = f"dry_run:{dry_run_reason}"

    def handle_signal(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
        del signum, frame
        service.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.command == "up":
            service.run_up(open_browser=args.open_browser)
        else:
            service.run_serve(open_browser=args.open_browser)
    except OSError as exc:
        print(f"control plane startup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
