from __future__ import annotations

import argparse
import copy
import json
import mimetypes
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
from urllib.parse import unquote, urlparse


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
WEB_STATIC_DIR = RUNTIME_DIR / "web" / "static"
WEB_INDEX_FILE = WEB_STATIC_DIR / "index.html"
DEFAULT_INITIAL_PROVIDER = "copilot"
LAUNCH_STRATEGIES = {"initial_copilot", "selected_model", "elastic"}
CONFIG_SECTIONS = {"project", "merge_policy", "resource_pools", "worker_defaults", "workers"}


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


def run_command(args: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout)


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


def is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}


def path_exists_via_ls(path_value: str) -> bool:
    try:
        result = run_command(["ls", "-d", path_value])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Path(path_value).exists()
    return result.returncode == 0


def host_reachable_via_ping(host: str) -> bool:
    if is_local_host(host):
        return True
    try:
        result = run_command(["ping", "-c", "1", host], timeout=2.5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False
        return True
    return result.returncode == 0


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


@dataclass(frozen=True)
class LaunchPolicy:
    strategy: str
    provider: str | None = None
    model: str | None = None


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


def session_state_path_for_port(port: int) -> Path:
    return RUNTIME_DIR / f"session_state_{port}.json"


def load_preferred_session_state(preferred_port: int | None = None) -> dict[str, Any]:
    if preferred_port is not None:
        port_path = session_state_path_for_port(preferred_port)
        if port_path.exists():
            return json.loads(port_path.read_text(encoding="utf-8"))
    return load_session_state_file()


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_pid(pid: int, sig: int = signal.SIGTERM) -> None:
    os.kill(pid, sig)


def terminate_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    try:
        process_group = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(process_group, sig)
    except OSError:
        return


def wait_for_process_exit(pid: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.1)
    return not pid_is_running(pid)


def tcp_port_in_use(port: int, hosts: tuple[str, ...] = ("127.0.0.1", "::1")) -> bool:
    for host in hosts:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.2)
                if probe.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def wait_for_port_release(port: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not tcp_port_in_use(port):
            return True
        time.sleep(0.1)
    return not tcp_port_in_use(port)


def safe_relative_web_path(request_path: str) -> Path | None:
    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if raw_path in {"", "/"}:
        return Path("index.html")
    relative = Path(raw_path.lstrip("/"))
    if any(part in {"..", ""} for part in relative.parts):
        return None
    return relative


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
        bootstrap_requested: bool = False,
    ):
        self.config_path = config_path
        self.persist_config_path = persist_config_path or config_path
        self.host_override = host_override
        self.port_override = port_override
        self.bootstrap_requested = bootstrap_requested
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.server_threads: list[threading.Thread] = []
        self.http_servers: list[ThreadingHTTPServer] = []
        self.processes: dict[str, WorkerProcess] = {}
        self.last_event = "initialized"
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.config: dict[str, Any] = {}
        self.project: dict[str, Any] = {}
        self.providers: dict[str, Any] = {}
        self.resource_pools: dict[str, Any] = {}
        self.workers: list[dict[str, Any]] = []
        self.provider_stats: dict[str, dict[str, Any]] = {}
        self.bootstrap_mode = False
        self.bootstrap_reason = ""
        self.listen_host = ""
        self.listen_port = 0
        self.listen_endpoints: list[str] = []
        self.listener_active = False
        self.reload_config()

    def worker_defaults(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = config or self.config
        if not isinstance(cfg, dict):
            return {}
        defaults = cfg.get("worker_defaults", {})
        return defaults if isinstance(defaults, dict) else {}

    def suggested_worktree_path(self, worker: dict[str, Any], config: dict[str, Any] | None = None) -> str:
        cfg = config or self.config
        if not isinstance(cfg, dict) or not isinstance(worker, dict):
            return ""
        project = cfg.get("project", {})
        if not isinstance(project, dict):
            return ""
        agent = str(worker.get("agent", "")).strip()
        local_repo_root = str(project.get("local_repo_root", "")).strip()
        repository_name = str(project.get("repository_name", "")).strip()
        if not agent or not local_repo_root or is_placeholder_path(local_repo_root):
            return ""
        root_path = Path(local_repo_root).expanduser()
        parent = root_path.parent if root_path.parent != root_path else root_path
        base_name = repository_name or root_path.name or "supersonic-moe"
        safe_base_name = "_".join(part for part in base_name.replace("-", "_").split("_") if part) or "supersonic_moe"
        return str((parent / f"{safe_base_name}_{agent.lower()}").resolve())

    def merge_worker_config(self, worker: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(worker, dict):
            return {}

        merged = dict(worker)
        worker_defaults = defaults if isinstance(defaults, dict) else self.worker_defaults()

        inheritable_fields = (
            "resource_pool",
            "environment_type",
            "environment_path",
            "sync_command",
            "test_command",
            "submit_strategy",
        )
        for field_name in inheritable_fields:
            raw_value = merged.get(field_name)
            if raw_value in {None, ""} and worker_defaults.get(field_name) not in {None, ""}:
                merged[field_name] = worker_defaults[field_name]

        raw_queue = merged.get("resource_pool_queue")
        default_queue = worker_defaults.get("resource_pool_queue")
        if (not isinstance(raw_queue, list) or not raw_queue) and isinstance(default_queue, list) and default_queue:
            merged["resource_pool_queue"] = list(default_queue)

        raw_worktree_path = str(merged.get("worktree_path", "")).strip()
        if not raw_worktree_path:
            suggested_path = self.suggested_worktree_path(merged)
            if suggested_path:
                merged["worktree_path"] = suggested_path

        default_identity = worker_defaults.get("git_identity")
        raw_identity = merged.get("git_identity")
        if isinstance(default_identity, dict) or isinstance(raw_identity, dict):
            merged_identity: dict[str, str] = {}
            for key in ("name", "email"):
                worker_value = str((raw_identity or {}).get(key, "")).strip() if isinstance(raw_identity, dict) else ""
                default_value = (
                    str((default_identity or {}).get(key, "")).strip() if isinstance(default_identity, dict) else ""
                )
                if worker_value:
                    merged_identity[key] = worker_value
                elif default_value:
                    merged_identity[key] = default_value
            if merged_identity:
                merged["git_identity"] = merged_identity

        return merged

    def field_matches_section(self, field: str, section: str) -> bool:
        if section == "project":
            return (
                field.startswith("project.")
                and not field.startswith("project.integration_branch")
                and not field.startswith("project.manager_git_identity")
            )
        if section == "merge_policy":
            return field.startswith("project.integration_branch") or field.startswith("project.manager_git_identity")
        if section == "resource_pools":
            return field.startswith("resource_pools.")
        if section == "worker_defaults":
            return field.startswith("worker_defaults.")
        if section == "workers":
            return field.startswith("workers[")
        return False

    def filter_section_issue_text(self, values: list[str], section: str) -> list[str]:
        if section == "project":
            keywords = (
                "project.repository_name",
                "project.local_repo_root",
                "project.paddle_repo_path",
                "project.dashboard",
            )
        elif section == "merge_policy":
            keywords = ("project.integration_branch", "project.manager_git_identity", "merge")
        elif section == "resource_pools":
            keywords = ("resource_pools.", "provider", "pool")
        elif section == "worker_defaults":
            keywords = ("worker_defaults",)
        elif section == "workers":
            keywords = (
                "worker ",
                "workers[",
                "worktree_path",
                "resource_pool_queue",
                "branch",
                "submit_strategy",
                "test_command",
            )
        else:
            return values
        return [value for value in values if any(keyword in value for keyword in keywords)]

    def config_for_section(
        self, section: str, value: Any, base_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if section not in CONFIG_SECTIONS:
            raise ValueError(f"unknown config section: {section}")
        current = copy.deepcopy(base_config or self.config or {})
        if not isinstance(current, dict):
            current = {}

        if section == "project":
            project = current.get("project", {})
            if not isinstance(project, dict):
                project = {}
            payload = value if isinstance(value, dict) else {}
            project.update(
                {
                    "repository_name": payload.get("repository_name", project.get("repository_name")),
                    "local_repo_root": payload.get("local_repo_root", project.get("local_repo_root")),
                    "paddle_repo_path": payload.get("paddle_repo_path", project.get("paddle_repo_path")),
                    "dashboard": payload.get("dashboard", project.get("dashboard", {})),
                }
            )
            current["project"] = project
            return current

        if section == "merge_policy":
            project = current.get("project", {})
            if not isinstance(project, dict):
                project = {}
            payload = value if isinstance(value, dict) else {}
            project.update(
                {
                    "integration_branch": payload.get("integration_branch", project.get("integration_branch")),
                    "manager_git_identity": payload.get(
                        "manager_git_identity", project.get("manager_git_identity", {})
                    ),
                }
            )
            current["project"] = project
            return current

        current[section] = copy.deepcopy(value)
        return current

    def config_validation_issues(self, config: dict[str, Any] | None = None) -> list[dict[str, str]]:
        cfg = config or self.config
        issues: list[dict[str, str]] = []

        def add_issue(field: str, message: str) -> None:
            issues.append({"field": field, "message": message})

        if not isinstance(cfg, dict):
            add_issue("config", "top-level config must be a YAML mapping")
            return issues

        project = cfg.get("project", {})
        providers = cfg.get("providers", {})
        resource_pools = cfg.get("resource_pools", {})
        worker_defaults = cfg.get("worker_defaults", {})
        workers = cfg.get("workers", [])

        if not isinstance(project, dict):
            add_issue("project", "project must be a mapping")
            project = {}
        if not isinstance(providers, dict):
            add_issue("providers", "providers must be a mapping")
            providers = {}
        if not isinstance(resource_pools, dict):
            add_issue("resource_pools", "resource_pools must be a mapping")
            resource_pools = {}
        if not isinstance(worker_defaults, dict):
            add_issue("worker_defaults", "worker_defaults must be a mapping")
            worker_defaults = {}
        if not isinstance(workers, list):
            add_issue("workers", "workers must be a list")
            workers = []

        repository_name = str(project.get("repository_name", "")).strip()
        if not repository_name:
            add_issue("project.repository_name", "repository name is required")

        for field_name in ("local_repo_root", "paddle_repo_path"):
            raw_value = str(project.get(field_name, "")).strip()
            field_path = f"project.{field_name}"
            if not raw_value:
                add_issue(field_path, f"{field_name} is required")
            elif is_placeholder_path(raw_value):
                add_issue(field_path, f"{field_name} must be replaced with a real path")
            elif not path_exists_via_ls(raw_value):
                add_issue(field_path, f"{field_name} does not exist: {raw_value}")

        dashboard = project.get("dashboard", {})
        if not isinstance(dashboard, dict):
            add_issue("project.dashboard", "dashboard must be a mapping")
            dashboard = {}
        host = str(dashboard.get("host", "")).strip()
        if not host:
            add_issue("project.dashboard.host", "dashboard host is required")
        elif not host_reachable_via_ping(host):
            add_issue("project.dashboard.host", f"dashboard host is not reachable via ping: {host}")
        port = dashboard.get("port")
        if not isinstance(port, int) or not (1 <= int(port) <= 65535):
            add_issue("project.dashboard.port", "dashboard port must be an integer between 1 and 65535")

        seen_agents: set[str] = set()
        seen_branches: set[str] = set()
        seen_worktrees: set[str] = set()

        for pool_name, pool in resource_pools.items():
            if not isinstance(pool, dict):
                add_issue(f"resource_pools.{pool_name}", "resource pool must be a mapping")
                continue
            provider_name = str(pool.get("provider", "")).strip()
            if not provider_name:
                add_issue(f"resource_pools.{pool_name}.provider", "provider is required")
            elif provider_name not in providers:
                add_issue(f"resource_pools.{pool_name}.provider", f"unknown provider: {provider_name}")
            if not str(pool.get("model", "")).strip():
                add_issue(f"resource_pools.{pool_name}.model", "model is required")
            priority = pool.get("priority", 100)
            if not isinstance(priority, int):
                add_issue(f"resource_pools.{pool_name}.priority", "priority must be an integer")

        default_pool_name = str(worker_defaults.get("resource_pool", "")).strip()
        if default_pool_name and default_pool_name not in resource_pools:
            add_issue("worker_defaults.resource_pool", f"unknown resource pool: {default_pool_name}")
        default_pool_queue = worker_defaults.get("resource_pool_queue", [])
        if default_pool_queue and not isinstance(default_pool_queue, list):
            add_issue("worker_defaults.resource_pool_queue", "resource_pool_queue must be a list")
        if isinstance(default_pool_queue, list):
            for queue_index, candidate_pool in enumerate(default_pool_queue):
                if str(candidate_pool) not in resource_pools:
                    add_issue(
                        f"worker_defaults.resource_pool_queue[{queue_index}]",
                        f"unknown resource pool: {candidate_pool}",
                    )

        default_environment_type = str(worker_defaults.get("environment_type", "uv")).strip() or "uv"
        default_environment_path = str(worker_defaults.get("environment_path", "")).strip()
        if default_environment_type != "none" and default_environment_path:
            if is_placeholder_path(default_environment_path):
                add_issue("worker_defaults.environment_path", "environment path must be replaced with a real path")
            elif not path_exists_via_ls(default_environment_path):
                add_issue(
                    "worker_defaults.environment_path",
                    f"environment path does not exist: {default_environment_path}",
                )

        default_git_identity = worker_defaults.get("git_identity")
        if default_git_identity is not None:
            if not isinstance(default_git_identity, dict):
                add_issue("worker_defaults.git_identity", "git_identity must be a mapping")
            else:
                if default_git_identity.get("name") and not str(default_git_identity.get("email", "")).strip():
                    add_issue("worker_defaults.git_identity.email", "email is required when git_identity.name is set")
                if default_git_identity.get("email") and not str(default_git_identity.get("name", "")).strip():
                    add_issue("worker_defaults.git_identity.name", "name is required when git_identity.email is set")

        for worker_index, worker in enumerate(workers):
            field_root = f"workers[{worker_index}]"
            if not isinstance(worker, dict):
                add_issue(field_root, "worker must be a mapping")
                continue
            effective_worker = self.merge_worker_config(worker, worker_defaults)
            agent = str(worker.get("agent", "")).strip()
            if not agent:
                add_issue(f"{field_root}.agent", "agent is required")
            elif agent in seen_agents:
                add_issue(f"{field_root}.agent", f"duplicate agent: {agent}")
            else:
                seen_agents.add(agent)

            branch = str(worker.get("branch", "")).strip()
            if not branch:
                add_issue(f"{field_root}.branch", "branch is required")
            elif branch in seen_branches:
                add_issue(f"{field_root}.branch", f"duplicate branch: {branch}")
            else:
                seen_branches.add(branch)

            worktree_path = str(effective_worker.get("worktree_path", "")).strip()
            if not worktree_path:
                add_issue(f"{field_root}.worktree_path", "worktree path is required")
            elif is_placeholder_path(worktree_path):
                add_issue(f"{field_root}.worktree_path", "worktree path must be replaced with a real path")
            elif worktree_path in seen_worktrees:
                add_issue(f"{field_root}.worktree_path", f"duplicate worktree path: {worktree_path}")
            else:
                seen_worktrees.add(worktree_path)

            pool_name = str(effective_worker.get("resource_pool", "")).strip()
            pool_queue = effective_worker.get("resource_pool_queue", [])
            if not pool_name and not pool_queue:
                add_issue(f"{field_root}.resource_pool", "resource_pool or resource_pool_queue is required")
            if pool_name and pool_name not in resource_pools:
                add_issue(f"{field_root}.resource_pool", f"unknown resource pool: {pool_name}")
            if pool_queue and not isinstance(pool_queue, list):
                add_issue(f"{field_root}.resource_pool_queue", "resource_pool_queue must be a list")
            if isinstance(pool_queue, list):
                for queue_index, candidate_pool in enumerate(pool_queue):
                    if str(candidate_pool) not in resource_pools:
                        add_issue(
                            f"{field_root}.resource_pool_queue[{queue_index}]",
                            f"unknown resource pool: {candidate_pool}",
                        )

            environment_type = str(effective_worker.get("environment_type", "uv")).strip() or "uv"
            environment_path = str(effective_worker.get("environment_path", "")).strip()
            if environment_type != "none":
                if not environment_path:
                    add_issue(
                        f"{field_root}.environment_path",
                        "environment path is required when environment_type is not none",
                    )
                elif is_placeholder_path(environment_path):
                    add_issue(f"{field_root}.environment_path", "environment path must be replaced with a real path")
                elif not path_exists_via_ls(environment_path):
                    add_issue(f"{field_root}.environment_path", f"environment path does not exist: {environment_path}")

            if not str(effective_worker.get("test_command", "")).strip():
                add_issue(f"{field_root}.test_command", "test_command is required")
            if not str(effective_worker.get("submit_strategy", "")).strip():
                add_issue(f"{field_root}.submit_strategy", "submit_strategy is required")

        return issues

    def validate_config_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        issues = self.config_validation_issues(config)
        return {
            "ok": len(issues) == 0,
            "validation_issues": issues,
            "validation_errors": self.validation_errors(config),
            "launch_blockers": self.launch_blockers(config),
        }

    def validate_config_section(self, section: str, value: Any) -> dict[str, Any]:
        next_config = self.config_for_section(section, value)
        validation = self.validate_config_payload(next_config)
        validation["validation_issues"] = [
            issue for issue in validation["validation_issues"] if self.field_matches_section(issue["field"], section)
        ]
        validation["validation_errors"] = self.filter_section_issue_text(validation["validation_errors"], section)
        validation["launch_blockers"] = self.filter_section_issue_text(validation["launch_blockers"], section)
        validation["ok"] = len(validation["validation_issues"]) == 0
        return validation

    def refresh_runtime_mode(self) -> None:
        using_template = self.config_path.resolve() == CONFIG_TEMPLATE_PATH.resolve()
        self.bootstrap_mode = using_template
        reasons: list[str] = []
        if using_template:
            reasons.append(f"cold-start bootstrap loaded from template {self.config_path}")
        if self.persist_config_path != self.config_path:
            reasons.append(f"save target is {self.persist_config_path}")
        if self.bootstrap_requested and using_template:
            reasons.append("bootstrap mode was requested explicitly")
        self.bootstrap_reason = "; ".join(reasons)

    def reload_config(self) -> None:
        with self.lock:
            self.config = load_yaml(self.config_path)
            self.project = self.config.get("project", {})
            self.providers = self.config.get("providers", {})
            self.resource_pools = self.config.get("resource_pools", {})
            self.worker_defaults_config = self.worker_defaults(self.config)
            self.workers = [
                self.merge_worker_config(worker, self.worker_defaults_config)
                for worker in self.config.get("workers", [])
                if isinstance(worker, dict)
            ]
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
        if not isinstance(cfg, dict):
            return ["top-level config must be a YAML mapping"]
        errors: list[str] = []
        project = cfg.get("project", {})
        providers = cfg.get("providers", {})
        resource_pools = cfg.get("resource_pools", {})
        worker_defaults = self.worker_defaults(cfg)
        workers = cfg.get("workers", [])

        if not project.get("repository_name"):
            errors.append("project.repository_name is recommended")
        if not project.get("local_repo_root"):
            errors.append(f"project.local_repo_root is recommended; default runtime root is {REPO_ROOT}")
        elif is_placeholder_path(project.get("local_repo_root")):
            errors.append("project.local_repo_root still points at a placeholder path")
        if not project.get("paddle_repo_path"):
            errors.append("project.paddle_repo_path is recommended for reference tracing")
        elif is_placeholder_path(project.get("paddle_repo_path")):
            errors.append("project.paddle_repo_path still points at a placeholder path")
        dashboard = project.get("dashboard", {})
        if not dashboard.get("host"):
            errors.append("project.dashboard.host is recommended")
        if not dashboard.get("port"):
            errors.append("project.dashboard.port is recommended")
        if project.get("manager_git_identity"):
            manager_identity = project.get("manager_git_identity", {})
            if not str(manager_identity.get("name", "")).strip():
                errors.append("project.manager_git_identity.name should be set when manager_git_identity is present")
            if not str(manager_identity.get("email", "")).strip():
                errors.append("project.manager_git_identity.email should be set when manager_git_identity is present")

        seen_agents: set[str] = set()
        seen_branches: set[str] = set()
        seen_worktrees: set[str] = set()

        for pool_name, pool in resource_pools.items():
            provider_name = pool.get("provider")
            if provider_name not in providers:
                errors.append(f"resource_pools.{pool_name}.provider references unknown provider {provider_name}")
            if not pool.get("model"):
                errors.append(f"resource_pools.{pool_name}.model is recommended")
            priority = pool.get("priority", 100)
            if not isinstance(priority, int):
                errors.append(f"resource_pools.{pool_name}.priority must be an integer")

        for worker in workers:
            if not isinstance(worker, dict):
                errors.append("worker entries must be mappings")
                continue
            effective_worker = self.merge_worker_config(worker, worker_defaults)
            agent = str(worker.get("agent", "")).strip()
            if not agent:
                errors.append("worker.agent is required")
                continue
            if agent in seen_agents:
                errors.append(f"duplicate worker agent {agent}")
            seen_agents.add(agent)

            pool_name = effective_worker.get("resource_pool")
            pool_queue = effective_worker.get("resource_pool_queue", [])
            if pool_name and pool_name not in resource_pools:
                errors.append(f"worker {agent} references unknown resource_pool {pool_name}")
            if not pool_name and not pool_queue:
                errors.append(f"worker {agent} should define resource_pool or resource_pool_queue")
            if pool_queue and not isinstance(pool_queue, list):
                errors.append(f"worker {agent} resource_pool_queue must be a list")
            for candidate_pool in pool_queue if isinstance(pool_queue, list) else []:
                if candidate_pool not in resource_pools:
                    errors.append(f"worker {agent} resource_pool_queue references unknown pool {candidate_pool}")
            branch = str(worker.get("branch", "")).strip()
            if not branch:
                errors.append(f"worker {agent} branch is required for launch")
            elif branch in seen_branches:
                errors.append(f"duplicate worker branch {branch}")
            else:
                seen_branches.add(branch)

            worktree = str(effective_worker.get("worktree_path", "")).strip()
            if not worktree:
                errors.append(f"worker {agent} worktree_path is required for launch")
            elif is_placeholder_path(worktree):
                errors.append(f"worker {agent} worktree_path still points at a placeholder path")
            elif worktree in seen_worktrees:
                errors.append(f"duplicate worker worktree_path {worktree}")
            else:
                seen_worktrees.add(worktree)

            environment_path = effective_worker.get("environment_path")
            if effective_worker.get("environment_type") not in {"none", None} and is_placeholder_path(
                environment_path
            ):
                errors.append(f"worker {agent} environment_path still points at a placeholder path")

            if not effective_worker.get("test_command"):
                errors.append(f"worker {agent} test_command is recommended")
            if not effective_worker.get("submit_strategy"):
                errors.append(f"worker {agent} submit_strategy is recommended")
            git_identity = effective_worker.get("git_identity")
            if git_identity is not None:
                if not isinstance(git_identity, dict):
                    errors.append(f"worker {agent} git_identity must be a mapping")
                else:
                    if not str(git_identity.get("name", "")).strip():
                        errors.append(f"worker {agent} git_identity.name is required when git_identity is set")
                    if not str(git_identity.get("email", "")).strip():
                        errors.append(f"worker {agent} git_identity.email is required when git_identity is set")

        return errors

    def launch_blockers(self, config: dict[str, Any] | None = None) -> list[str]:
        cfg = config or self.config
        if not isinstance(cfg, dict):
            return ["top-level config must be a YAML mapping before launch"]

        blockers: list[str] = []
        providers = cfg.get("providers", {})
        resource_pools = cfg.get("resource_pools", {})
        worker_defaults = self.worker_defaults(cfg)
        workers = cfg.get("workers", [])

        if not isinstance(providers, dict):
            blockers.append("providers must be a mapping")
            providers = {}
        if not isinstance(resource_pools, dict):
            blockers.append("resource_pools must be a mapping")
            resource_pools = {}
        if not isinstance(workers, list):
            blockers.append("workers must be a list")
            workers = []

        if not workers:
            blockers.append("define at least one worker before launch")

        seen_agents: set[str] = set()
        seen_branches: set[str] = set()
        seen_worktrees: set[str] = set()

        for pool_name, pool in resource_pools.items():
            provider_name = pool.get("provider")
            if not provider_name:
                blockers.append(f"resource_pools.{pool_name}.provider is required")
            elif provider_name not in providers:
                blockers.append(f"resource_pools.{pool_name}.provider references unknown provider {provider_name}")
            if not pool.get("model"):
                blockers.append(f"resource_pools.{pool_name}.model is required")
            if not isinstance(pool.get("priority", 100), int):
                blockers.append(f"resource_pools.{pool_name}.priority must be an integer")

        for provider_name, provider in providers.items():
            template = provider.get("command_template")
            if not template:
                blockers.append(f"providers.{provider_name}.command_template is required")

        for worker in workers:
            if not isinstance(worker, dict):
                blockers.append("worker entries must be mappings")
                continue
            effective_worker = self.merge_worker_config(worker, worker_defaults)
            agent = str(worker.get("agent", "")).strip()
            if not agent:
                blockers.append("worker.agent is required")
                continue
            if agent in seen_agents:
                blockers.append(f"duplicate worker agent {agent}")
            seen_agents.add(agent)

            branch = str(worker.get("branch", "")).strip()
            if not branch:
                blockers.append(f"worker {agent} branch is required")
            elif branch in seen_branches:
                blockers.append(f"duplicate worker branch {branch}")
            else:
                seen_branches.add(branch)

            worktree = str(effective_worker.get("worktree_path", "")).strip()
            if not worktree:
                blockers.append(f"worker {agent} worktree_path is required")
            elif is_placeholder_path(worktree):
                blockers.append(f"worker {agent} worktree_path must be replaced with a real path")
            elif worktree in seen_worktrees:
                blockers.append(f"duplicate worker worktree_path {worktree}")
            else:
                seen_worktrees.add(worktree)

            pool_name = effective_worker.get("resource_pool")
            pool_queue = effective_worker.get("resource_pool_queue", [])
            if pool_name and pool_name not in resource_pools:
                blockers.append(f"worker {agent} references unknown resource_pool {pool_name}")
            if not pool_name and not pool_queue:
                blockers.append(f"worker {agent} must define resource_pool or resource_pool_queue")
            if pool_queue and not isinstance(pool_queue, list):
                blockers.append(f"worker {agent} resource_pool_queue must be a list")
            for candidate_pool in pool_queue if isinstance(pool_queue, list) else []:
                if candidate_pool not in resource_pools:
                    blockers.append(f"worker {agent} resource_pool_queue references unknown pool {candidate_pool}")

            environment_path = effective_worker.get("environment_path")
            if effective_worker.get("environment_type") not in {"none", None} and is_placeholder_path(
                environment_path
            ):
                blockers.append(f"worker {agent} environment_path must be replaced with a real path")
            if not effective_worker.get("test_command"):
                blockers.append(f"worker {agent} test_command is required")
            if not effective_worker.get("submit_strategy"):
                blockers.append(f"worker {agent} submit_strategy is required")

        return blockers

    def save_config_data(self, parsed: dict[str, Any]) -> list[str]:
        target_path = self.persist_config_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml_text(parsed), encoding="utf-8")
        self.config_path = target_path
        self.reload_config()
        self.last_event = f"config_saved:{now_iso()}"
        return self.validation_errors(parsed)

    def save_config_section(self, section: str, value: Any) -> list[str]:
        next_config = self.config_for_section(section, value)
        validation = self.validate_config_section(section, value)
        if validation["validation_issues"]:
            raise ValueError(f"section {section} has validation issues")
        return self.save_config_data(next_config)

    def save_config_text(self, raw_text: str) -> list[str]:
        parsed = yaml.safe_load(raw_text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("top-level config must be a YAML mapping")
        return self.save_config_data(parsed)

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

    def has_launch_history(self) -> bool:
        if self.processes:
            return True
        worker_agents = {str(worker.get("agent", "")).strip() for worker in self.workers if worker.get("agent")}
        if not worker_agents:
            return False
        runtime_workers = load_yaml(STATE_DIR / "agent_runtime.yaml").get("workers", [])
        for entry in runtime_workers:
            agent = str(entry.get("agent", "")).strip()
            if agent not in worker_agents:
                continue
            status = str(entry.get("status", "")).strip()
            if status and status not in {"not_started", "not-started", "unassigned"}:
                return True
        heartbeat_workers = load_yaml(STATE_DIR / "heartbeats.yaml").get("agents", [])
        for entry in heartbeat_workers:
            agent = str(entry.get("agent", "")).strip()
            if agent not in worker_agents:
                continue
            state = str(entry.get("state", "")).strip()
            last_seen = str(entry.get("last_seen", "")).strip()
            if state and state not in {"not_started", "not-started"}:
                return True
            if last_seen and last_seen.lower() != "none":
                return True
        return False

    def default_launch_policy(self) -> LaunchPolicy:
        if not self.has_launch_history():
            return LaunchPolicy(strategy="initial_copilot", provider=DEFAULT_INITIAL_PROVIDER)
        return LaunchPolicy(strategy="elastic")

    def parse_launch_policy(self, payload: dict[str, Any]) -> LaunchPolicy:
        default_policy = self.default_launch_policy()
        raw_strategy = str(payload.get("strategy") or default_policy.strategy).strip() or default_policy.strategy
        if raw_strategy not in LAUNCH_STRATEGIES:
            raise ValueError(f"unknown launch strategy: {raw_strategy}")

        provider = str(payload.get("provider") or "").strip() or None
        model = str(payload.get("model") or "").strip() or None

        if raw_strategy == "initial_copilot":
            provider = DEFAULT_INITIAL_PROVIDER
        elif raw_strategy == "selected_model":
            if not provider:
                raise ValueError("provider is required when strategy is selected_model")
            if provider not in self.providers:
                raise ValueError(f"unknown provider for selected_model: {provider}")
            if not model:
                raise ValueError("model is required when strategy is selected_model")

        if provider and provider not in self.providers:
            raise ValueError(f"unknown provider: {provider}")
        return LaunchPolicy(strategy=raw_strategy, provider=provider, model=model)

    def launch_policy_state(self) -> dict[str, Any]:
        default_policy = self.default_launch_policy()
        return {
            "default_strategy": default_policy.strategy,
            "default_provider": default_policy.provider,
            "default_model": default_policy.model,
            "available_strategies": sorted(LAUNCH_STRATEGIES),
            "available_providers": sorted(self.providers.keys()),
            "initial_provider": DEFAULT_INITIAL_PROVIDER,
            "has_launch_history": self.has_launch_history(),
        }

    def candidate_pools_for_worker(self, worker: dict[str, Any]) -> list[str]:
        configured_queue = worker.get("resource_pool_queue")
        if isinstance(configured_queue, list) and configured_queue:
            return configured_queue
        if worker.get("resource_pool"):
            return [str(worker["resource_pool"])]
        return [item["resource_pool"] for item in self.provider_queue()]

    def best_pool_for_provider(self, provider_name: str) -> tuple[str, dict[str, Any]]:
        ordered_candidates = [item for item in self.provider_queue() if item["provider"] == provider_name]
        if not ordered_candidates:
            raise RuntimeError(f"no eligible resource pool candidates exist for provider {provider_name}")
        for item in ordered_candidates:
            if item["binary_found"] and item["api_key_present"]:
                return item["resource_pool"], item
        return ordered_candidates[0]["resource_pool"], ordered_candidates[0]

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

    def resolve_pool_for_launch(self, worker: dict[str, Any], policy: LaunchPolicy) -> tuple[str, dict[str, Any]]:
        if policy.strategy == "elastic":
            return self.best_pool_for_worker(worker)
        provider_name = policy.provider or DEFAULT_INITIAL_PROVIDER
        return self.best_pool_for_provider(provider_name)

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
                "cold_start": self.bootstrap_mode,
                "bootstrap_reason": self.bootstrap_reason,
                "listener_active": self.listener_active,
                "alive": not self.stop_event.is_set(),
            },
            "workers": worker_payload,
        }
        encoded = json.dumps(payload, indent=2)
        SESSION_STATE.write_text(encoded, encoding="utf-8")
        if self.listen_port:
            session_state_path_for_port(self.listen_port).write_text(encoded, encoding="utf-8")

    def serve_static_asset(self, request_path: str) -> tuple[bytes, str] | None:
        relative_path = safe_relative_web_path(request_path)
        if relative_path is None:
            return None
        asset_path = (WEB_STATIC_DIR / relative_path).resolve()
        web_root = WEB_STATIC_DIR.resolve()
        try:
            asset_path.relative_to(web_root)
        except ValueError:
            return None
        if asset_path.is_dir():
            asset_path = asset_path / "index.html"
        if not asset_path.exists() or not asset_path.is_file():
            return None
        content_type, _ = mimetypes.guess_type(asset_path.name)
        return asset_path.read_bytes(), content_type or "application/octet-stream"

    def build_cli_commands(self) -> dict[str, str]:
        host = self.host_override or self.project.get("dashboard", {}).get("host", DEFAULT_DASHBOARD_HOST)
        port = self.port_override or int(self.project.get("dashboard", {}).get("port", DEFAULT_DASHBOARD_PORT))
        config = str(self.persist_config_path)
        serve_parts = [
            CONTROL_PLANE_RUNTIME,
            "control_plane/fp8/runtime/control_plane.py",
            "serve",
            "--config",
            config,
        ]
        up_parts = [
            CONTROL_PLANE_RUNTIME,
            "control_plane/fp8/runtime/control_plane.py",
            "up",
            "--config",
            config,
        ]
        if host != DEFAULT_DASHBOARD_HOST:
            serve_parts.extend(["--host", str(host)])
            up_parts.extend(["--host", str(host)])
        if port != DEFAULT_DASHBOARD_PORT:
            serve_parts.extend(["--port", str(port)])
            up_parts.extend(["--port", str(port)])
        up_parts.append("--open-browser")
        serve = " ".join(serve_parts)
        up = " ".join(up_parts)
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
        self,
        pool_name: str,
        worker: dict[str, Any],
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str, str]:
        pool = self.resource_pools[pool_name]
        provider_name = provider_override or worker.get("provider") or pool["provider"]
        provider = self.providers[provider_name]
        model = model_override or worker.get("model") or pool["model"]
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

    def launch_worker(self, worker: dict[str, Any], policy: LaunchPolicy | None = None) -> dict[str, Any]:
        resolved_policy = policy or self.default_launch_policy()
        pool_name, evaluation = self.resolve_pool_for_launch(worker, resolved_policy)
        provider, pool, provider_name, model = self.provider_runtime(
            pool_name,
            worker,
            provider_override=resolved_policy.provider,
            model_override=resolved_policy.model,
        )
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
            start_new_session=True,
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
        self.update_runtime_entry(worker, pool_name, provider_name, model, "launching")
        self.update_heartbeat(worker["agent"], "launching", "process_spawned", "waiting for first monitor check")
        return {
            "agent": worker["agent"],
            "resource_pool": pool_name,
            "provider": provider_name,
            "model": model,
            "pid": process.pid,
            "command": command,
            "launch_strategy": resolved_policy.strategy,
        }

    def launch_all(self, restart: bool = False, policy: LaunchPolicy | None = None) -> dict[str, Any]:
        with self.lock:
            errors = self.launch_blockers()
            if errors:
                return {"ok": False, "errors": errors}
            resolved_policy = policy or self.default_launch_policy()
            if restart:
                self.stop_workers()
            launched: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            for worker in self.workers:
                if worker["agent"] in self.processes and self.processes[worker["agent"]].process.poll() is None:
                    continue
                try:
                    launched.append(self.launch_worker(worker, policy=resolved_policy))
                except Exception as exc:
                    try:
                        candidate_pools = [self.resolve_pool_for_launch(worker, resolved_policy)[0]]
                    except Exception:
                        candidate_pools = self.candidate_pools_for_worker(worker)
                    pool_name = candidate_pools[0] if candidate_pools else "unassigned"
                    if pool_name in self.provider_stats:
                        self.provider_stats[pool_name]["launch_failures"] += 1
                        self.provider_stats[pool_name]["last_failure"] = str(exc)
                    provider_name = resolved_policy.provider or worker.get("provider", "unassigned") or "unassigned"
                    model = resolved_policy.model or worker.get("model", "unassigned") or "unassigned"
                    self.update_runtime_entry(worker, pool_name, provider_name, model, f"launch_failed: {exc}")
                    self.update_heartbeat(worker["agent"], "stale", "launch_failed", str(exc))
                    failures.append({"agent": worker["agent"], "error": str(exc)})
            self.last_event = f"launch:{resolved_policy.strategy}:{len(launched)} workers"
            self.write_session_state()
            return {
                "ok": len(failures) == 0,
                "launched": launched,
                "failures": failures,
                "launch_policy": {
                    "strategy": resolved_policy.strategy,
                    "provider": resolved_policy.provider,
                    "model": resolved_policy.model,
                },
            }

    def stop_workers(self) -> dict[str, Any]:
        with self.lock:
            stopped: list[str] = []
            for agent, worker in list(self.processes.items()):
                if worker.process.poll() is None:
                    terminate_process_tree(worker.process.pid, signal.SIGTERM)
                    try:
                        worker.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        terminate_process_tree(worker.process.pid, signal.SIGKILL)
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
        return snapshot

    def build_dashboard_state(self) -> dict[str, Any]:
        config_text = self.config_path.read_text(encoding="utf-8") if self.config_path.exists() else ""
        return {
            "updated_at": now_iso(),
            "last_event": self.last_event,
            "mode": {
                "state": "cold-start" if self.bootstrap_mode else "configured",
                "cold_start": self.bootstrap_mode,
                "listener_active": self.listener_active,
                "reason": self.bootstrap_reason,
                "config_path": str(self.config_path),
                "persist_config_path": str(self.persist_config_path),
            },
            "project": self.project,
            "commands": self.build_cli_commands(),
            "launch_policy": self.launch_policy_state(),
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
            "launch_blockers": self.launch_blockers(),
        }

    def monitor_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                for agent, worker in list(self.processes.items()):
                    returncode = worker.process.poll()
                    if returncode is None:
                        self.update_heartbeat(agent, "healthy", "process_running", "none")
                        runtime_entry = next((w for w in self.workers if w.get("agent") == agent), None)
                        if runtime_entry:
                            self.update_runtime_entry(
                                runtime_entry,
                                worker.resource_pool,
                                worker.provider,
                                worker.model,
                                "healthy",
                            )
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
                        runtime_entry = next((w for w in self.workers if w.get("agent") == agent), None)
                        if runtime_entry:
                            self.update_runtime_entry(
                                runtime_entry,
                                worker.resource_pool,
                                worker.provider,
                                worker.model,
                                state,
                            )
                self.write_session_state()
            time.sleep(5)

    def handle_api_get(self, handler: BaseHTTPRequestHandler) -> bool:
        if handler.path == "/api/state":
            payload = json.dumps(self.build_dashboard_state(), default=str).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            try:
                handler.wfile.write(payload)
            except BrokenPipeError:
                return True
            return True
        if handler.path == "/api/config":
            payload = json.dumps(
                {"config": self.config, "config_text": self.config_path.read_text(encoding="utf-8")}, default=str
            ).encode("utf-8")
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            try:
                handler.wfile.write(payload)
            except BrokenPipeError:
                return True
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
        try:
            handler.wfile.write(body)
        except BrokenPipeError:
            return

    def handle_api_post(self, handler: BaseHTTPRequestHandler) -> bool:
        try:
            payload = self.parse_request_json(handler)
        except json.JSONDecodeError as exc:
            self.write_json(handler, {"ok": False, "error": f"invalid json: {exc}"}, status=400)
            return True

        if handler.path == "/api/config":
            raw_config = payload.get("config")
            try:
                if isinstance(raw_config, dict):
                    validation = self.validate_config_payload(raw_config)
                    if validation["validation_issues"]:
                        self.write_json(handler, validation, status=400)
                        return True
                    errors = self.save_config_data(raw_config)
                else:
                    raw_text = payload.get("config_text")
                    if not isinstance(raw_text, str):
                        self.write_json(
                            handler, {"ok": False, "error": "config or config_text is required"}, status=400
                        )
                        return True
                    parsed = yaml.safe_load(raw_text) or {}
                    if not isinstance(parsed, dict):
                        self.write_json(
                            handler, {"ok": False, "error": "top-level config must be a YAML mapping"}, status=400
                        )
                        return True
                    validation = self.validate_config_payload(parsed)
                    if validation["validation_issues"]:
                        self.write_json(handler, validation, status=400)
                        return True
                    errors = self.save_config_data(parsed)
            except Exception as exc:
                self.write_json(handler, {"ok": False, "error": str(exc)}, status=400)
                return True
            self.write_json(
                handler,
                {
                    "ok": True,
                    "validation_issues": [],
                    "validation_errors": errors,
                    "launch_blockers": self.launch_blockers(),
                    "cold_start": self.bootstrap_mode,
                },
            )
            return True

        if handler.path == "/api/config/validate":
            raw_config = payload.get("config")
            if not isinstance(raw_config, dict):
                self.write_json(handler, {"ok": False, "error": "config is required"}, status=400)
                return True
            self.write_json(handler, self.validate_config_payload(raw_config))
            return True

        if handler.path == "/api/config/validate-section":
            section = str(payload.get("section", "")).strip()
            value = payload.get("value")
            if section not in CONFIG_SECTIONS:
                self.write_json(handler, {"ok": False, "error": "valid section is required"}, status=400)
                return True
            self.write_json(handler, self.validate_config_section(section, value))
            return True

        if handler.path == "/api/config/section":
            section = str(payload.get("section", "")).strip()
            value = payload.get("value")
            if section not in CONFIG_SECTIONS:
                self.write_json(handler, {"ok": False, "error": "valid section is required"}, status=400)
                return True
            try:
                validation = self.validate_config_section(section, value)
                if validation["validation_issues"]:
                    self.write_json(handler, validation, status=400)
                    return True
                errors = self.save_config_section(section, value)
            except Exception as exc:
                self.write_json(handler, {"ok": False, "error": str(exc)}, status=400)
                return True
            self.write_json(
                handler,
                {
                    "ok": True,
                    "validation_issues": [],
                    "validation_errors": self.filter_section_issue_text(errors, section),
                    "launch_blockers": self.filter_section_issue_text(self.launch_blockers(), section),
                    "cold_start": self.bootstrap_mode,
                },
            )
            return True

        if handler.path == "/api/launch":
            restart = bool(payload.get("restart", False))
            try:
                launch_policy = self.parse_launch_policy(payload)
            except ValueError as exc:
                self.write_json(handler, {"ok": False, "error": str(exc)}, status=400)
                return True
            result = self.launch_all(restart=restart, policy=launch_policy)
            self.write_json(handler, result, status=200 if result.get("ok") else 400)
            return True

        if handler.path == "/api/stop":
            result = self.stop_workers()
            self.write_json(handler, result)
            return True

        if handler.path == "/api/stop-all":
            stopped_workers = sorted(self.processes.keys())
            listener_port = self.listen_port
            self.write_json(
                handler,
                {
                    "ok": True,
                    "stop_agents": True,
                    "stopped_workers": stopped_workers,
                    "listener_port": listener_port,
                    "listener_released": False,
                },
            )
            threading.Thread(target=self.shutdown, kwargs={"stop_agents": True}, daemon=True).start()
            return True

        if handler.path == "/api/silent":
            self.write_json(
                handler,
                {
                    "ok": True,
                    "listener_port": self.listen_port,
                    "listener_active": False,
                    "stop_agents": False,
                },
            )
            threading.Thread(target=self.enter_silent_mode, daemon=True).start()
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
                asset = service.serve_static_asset(self.path)
                if asset is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body, content_type = asset
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    return

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
        self.listener_active = True
        self.last_event = f"dashboard:{', '.join(listen_endpoints)}"
        self.write_session_state()
        if host in {"0.0.0.0", "::"}:
            print(
                f"remote access URL: http://<server-hostname-or-ip>:{listen_port}",
                file=sys.stderr,
                flush=True,
            )
        if open_browser:
            webbrowser.open(f"http://{browser_open_host(host)}:{listen_port}")

    def wait_forever(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(1)

    def start_monitoring(self) -> None:
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def run_up(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        result = self.launch_all()
        if not result.get("ok"):
            launch_blockers = result.get("errors") or []
            if launch_blockers:
                self.last_event = f"cold_start: launch blocked by {len(launch_blockers)} issue(s)"
        self.wait_forever()

    def run_serve(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        self.wait_forever()

    def close_http_servers(self) -> bool:
        released = True
        for server in self.http_servers:
            server.shutdown()
            server.server_close()
        if self.listen_port:
            released = wait_for_port_release(self.listen_port)
        self.http_servers = []
        self.server_threads = []
        self.listen_endpoints = []
        self.listener_active = False
        return released

    def enter_silent_mode(self) -> None:
        with self.lock:
            if not self.listener_active:
                return
            released = self.close_http_servers()
            self.last_event = f"silent_mode:listener released={released}"
            self.write_session_state()

    def shutdown(self, stop_agents: bool = True) -> None:
        self.stop_event.set()
        if stop_agents:
            self.stop_workers()
        listener_released = self.close_http_servers()
        self.last_event = (
            f"shutdown:all released={listener_released}"
            if stop_agents
            else f"shutdown:listener released={listener_released}"
        )
        self.write_session_state()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="supersonic-moe control plane runtime")
    parser.add_argument(
        "command",
        choices=["up", "serve", "silent", "stop-agents", "stop-listener", "stop-all"],
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
        "--bootstrap",
        action="store_true",
        help="force template-backed cold-start bootstrap even before local_config.yaml exists",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=RUNTIME_DIR / "control_plane.log",
        help="log file used when --detach is enabled",
    )
    return parser.parse_args()


def apply_runtime_defaults(args: argparse.Namespace, cold_start: bool) -> None:
    if args.command in {"serve", "up"}:
        if args.host is None:
            args.host = DEFAULT_DASHBOARD_HOST
        if args.port is None:
            args.port = DEFAULT_DASHBOARD_PORT
    if args.command == "serve" and not args.foreground:
        args.detach = True
    elif args.command == "serve" and cold_start:
        if not args.foreground:
            args.detach = True


def stop_agents_command(args: argparse.Namespace) -> int:
    session_state = load_preferred_session_state(args.port or DEFAULT_DASHBOARD_PORT)
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    worker_pids = {
        agent: int(worker.get("pid") or 0)
        for agent, worker in session_state.get("workers", {}).items()
        if int(worker.get("pid") or 0)
    }
    try:
        result = post_control_plane(control_plane_base_url(args, session_state), "/api/stop", {})
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError:
        stopped_workers: list[str] = []
        for agent, pid in worker_pids.items():
            if pid and pid_is_running(pid):
                terminate_process_tree(pid, signal.SIGTERM)
                if not wait_for_process_exit(pid, timeout=3):
                    terminate_process_tree(pid, signal.SIGKILL)
                    wait_for_process_exit(pid, timeout=2)
            if not pid or not pid_is_running(pid):
                stopped_workers.append(agent)
        print(json.dumps({"ok": True, "stopped": sorted(stopped_workers)}, indent=2))
        return 0


def stop_listener_command(args: argparse.Namespace) -> int:
    session_state = load_preferred_session_state(args.port or DEFAULT_DASHBOARD_PORT)
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    server_pid = int(session_state.get("server", {}).get("pid") or 0)
    listener_port = int(session_state.get("server", {}).get("port") or DEFAULT_DASHBOARD_PORT)
    if not session_state.get("server", {}).get("listener_active", True):
        print(
            json.dumps(
                {
                    "ok": True,
                    "listener_port": listener_port,
                    "listener_released": True,
                    "stop_agents": False,
                },
                indent=2,
            )
        )
        return 0
    try:
        result = post_control_plane(control_plane_base_url(args, session_state), "/api/silent", {})
        listener_released = wait_for_port_release(listener_port)
        result["listener_port"] = listener_port
        result["listener_released"] = listener_released
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError:
        print("listener control plane is unreachable; cannot enter silent mode safely", file=sys.stderr)
        return 1


def stop_all_command(args: argparse.Namespace) -> int:
    session_state = load_preferred_session_state(args.port or DEFAULT_DASHBOARD_PORT)
    if not session_state:
        print(f"no active session state found at {SESSION_STATE}", file=sys.stderr)
        return 1
    server_pid = int(session_state.get("server", {}).get("pid") or 0)
    listener_port = int(session_state.get("server", {}).get("port") or DEFAULT_DASHBOARD_PORT)
    worker_pids = {
        agent: int(worker.get("pid") or 0)
        for agent, worker in session_state.get("workers", {}).items()
        if int(worker.get("pid") or 0)
    }
    try:
        result = post_control_plane(control_plane_base_url(args, session_state), "/api/stop-all", {})
        stopped_worker_names = []
        for agent, pid in worker_pids.items():
            if not pid:
                continue
            if wait_for_process_exit(pid, timeout=5):
                stopped_worker_names.append(agent)
            else:
                terminate_process_tree(pid, signal.SIGTERM)
                if wait_for_process_exit(pid, timeout=3):
                    stopped_worker_names.append(agent)
                else:
                    terminate_process_tree(pid, signal.SIGKILL)
                    if wait_for_process_exit(pid, timeout=2):
                        stopped_worker_names.append(agent)
        listener_released = wait_for_port_release(listener_port)
        if not listener_released and server_pid and pid_is_running(server_pid):
            terminate_process_tree(server_pid, signal.SIGTERM)
            listener_released = wait_for_port_release(listener_port)
        if not listener_released and server_pid and pid_is_running(server_pid):
            terminate_process_tree(server_pid, signal.SIGKILL)
            listener_released = wait_for_port_release(listener_port)
        result["listener_port"] = listener_port
        result["listener_released"] = listener_released
        result["stopped_workers"] = sorted(stopped_worker_names)
        result["warning"] = "listener port is still busy" if not listener_released else ""
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError:
        stopped_workers: list[str] = []
        for agent, pid in worker_pids.items():
            if pid and pid_is_running(pid):
                terminate_process_tree(pid, signal.SIGTERM)
                if not wait_for_process_exit(pid, timeout=3):
                    terminate_process_tree(pid, signal.SIGKILL)
                    wait_for_process_exit(pid, timeout=2)
            if not pid or not pid_is_running(pid):
                stopped_workers.append(agent)
        if server_pid and pid_is_running(server_pid):
            terminate_process_tree(server_pid, signal.SIGTERM)
            if not wait_for_port_release(listener_port):
                terminate_process_tree(server_pid, signal.SIGKILL)
        listener_released = wait_for_port_release(listener_port)
        print(
            json.dumps(
                {
                    "ok": listener_released,
                    "listener_pid": server_pid,
                    "listener_port": listener_port,
                    "listener_released": listener_released,
                    "stopped_workers": sorted(stopped_workers),
                    "stop_agents": True,
                },
                indent=2,
            )
        )
        return 0


def resolve_runtime_config(args: argparse.Namespace) -> tuple[Path, Path, bool, str]:
    requested_path = args.config
    persist_path = requested_path
    bootstrap_requested = bool(args.bootstrap)
    reasons: list[str] = []
    default_local_config = (RUNTIME_DIR / "local_config.yaml").resolve()

    if not requested_path.exists():
        if requested_path.resolve() == default_local_config or bootstrap_requested:
            if not CONFIG_TEMPLATE_PATH.exists():
                raise FileNotFoundError(f"missing template config: {CONFIG_TEMPLATE_PATH}")
            requested_path = CONFIG_TEMPLATE_PATH
            reasons.append(f"cold-start bootstrapped from template because {persist_path} does not exist")
        else:
            raise FileNotFoundError(f"missing config: {persist_path}")

    if requested_path.resolve() == CONFIG_TEMPLATE_PATH.resolve():
        if persist_path == requested_path:
            persist_path = RUNTIME_DIR / "local_config.yaml"
        reasons.append(
            "template-backed control plane will accept settings edits immediately and launch when blockers are cleared"
        )

    return (
        requested_path,
        persist_path,
        requested_path.resolve() == CONFIG_TEMPLATE_PATH.resolve(),
        "; ".join(dict.fromkeys(reasons)),
    )


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
    if args.bootstrap:
        command.append("--bootstrap")

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
    if args.command == "silent":
        return stop_listener_command(args)
    if args.command == "stop-agents":
        return stop_agents_command(args)
    if args.command == "stop-listener":
        return stop_listener_command(args)
    if args.command == "stop-all":
        return stop_all_command(args)

    try:
        config_path, persist_config_path, cold_start, bootstrap_reason = resolve_runtime_config(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"create {RUNTIME_DIR / 'local_config.yaml'} or point --config at {CONFIG_TEMPLATE_PATH} to bootstrap from the template",
            file=sys.stderr,
        )
        return 2

    apply_runtime_defaults(args, cold_start)

    if args.detach and os.environ.get("CONTROL_PLANE_DETACHED") != "1":
        args.config = config_path
        args.bootstrap = cold_start
        return detach_process(args)

    service = ControlPlaneService(
        config_path,
        host_override=args.host,
        port_override=args.port,
        persist_config_path=persist_config_path,
        bootstrap_requested=args.bootstrap,
    )
    if service.bootstrap_mode and bootstrap_reason:
        service.bootstrap_reason = bootstrap_reason
        service.last_event = f"cold_start:{bootstrap_reason}"

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
