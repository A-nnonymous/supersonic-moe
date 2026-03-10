from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
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


class ControlPlaneService:
    def __init__(self, config_path: Path, host_override: str | None = None, port_override: int | None = None):
        self.config_path = config_path
        self.host_override = host_override
        self.port_override = port_override
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.server_thread: threading.Thread | None = None
        self.httpd: ThreadingHTTPServer | None = None
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
        self.reload_config()

    def reload_config(self) -> None:
        with self.lock:
            self.config = load_yaml(self.config_path)
            self.project = self.config.get("project", {})
            self.providers = self.config.get("providers", {})
            self.resource_pools = self.config.get("resource_pools", {})
            self.workers = self.config.get("workers", [])
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

        return errors

    def save_config_text(self, raw_text: str) -> list[str]:
        parsed = yaml.safe_load(raw_text) or {}
        errors = self.validation_errors(parsed)
        if errors:
            return errors
        self.config_path.write_text(yaml_text(parsed), encoding="utf-8")
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
        payload = {
            "updated_at": now_iso(),
            "last_event": self.last_event,
            "workers": {
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
                }
                for agent, worker in self.processes.items()
            },
        }
        SESSION_STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_cli_commands(self) -> dict[str, str]:
        host = self.host_override or self.project.get("dashboard", {}).get("host", "127.0.0.1")
        port = self.port_override or int(self.project.get("dashboard", {}).get("port", 8233))
        config = str(self.config_path)
        serve = f"{CONTROL_PLANE_RUNTIME} control_plane/fp8/runtime/control_plane.py serve --config {config} --host {host} --port {port} --open-browser"
        up = f"{CONTROL_PLANE_RUNTIME} control_plane/fp8/runtime/control_plane.py up --config {config} --host {host} --port {port} --open-browser"
        return {"serve": serve, "up": up}

    def task_title(self, task_id: str) -> str:
        backlog = load_yaml(STATE_DIR / "backlog.yaml")
        for item in backlog.get("items", []):
            if item.get("id") == task_id:
                return str(item.get("title", task_id))
        return task_id

    def render_prompt(self, worker: dict[str, Any], provider_name: str, model: str) -> Path:
        prompt_path = PROMPT_DIR / f"{worker['agent']}.md"
        task_id = worker.get("task_id", "unassigned")
        task_title = self.task_title(task_id)
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

Mandatory rules:

1. Work only inside the assigned worktree.
2. Do not edit shared control-plane files unless the manager explicitly asks and the lock is held.
3. Update your status file in `control_plane/fp8/status/agents/` and your checkpoint in `control_plane/fp8/checkpoints/agents/`.
4. Treat `tests/reference_layers/standalone_moe_layer` and `{self.project.get('paddle_repo_path', 'unassigned')}` as reference inputs, not as the final host implementation.
5. Report blockers before widening scope.

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
                "environment_type": worker.get("environment_type", "uv"),
                "environment_path": worker.get("environment_path", "unassigned"),
                "sync_command": worker.get("sync_command", "uv sync"),
                "test_command": worker.get("test_command", "unassigned"),
                "submit_strategy": worker.get("submit_strategy", "patch_handoff"),
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
            "project": self.project,
            "commands": self.build_cli_commands(),
            "manager_report": MANAGER_REPORT.read_text(encoding="utf-8"),
            "runtime": load_yaml(STATE_DIR / "agent_runtime.yaml"),
            "heartbeats": load_yaml(STATE_DIR / "heartbeats.yaml"),
            "backlog": load_yaml(STATE_DIR / "backlog.yaml"),
            "gates": load_yaml(STATE_DIR / "gates.yaml"),
            "processes": self.process_snapshot(),
            "provider_queue": self.provider_queue(),
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

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")
        self.last_event = f"dashboard:{host}:{port}"

    def wait_forever(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(1)

    def start_monitoring(self) -> None:
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def run_up(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        self.launch_all()
        self.wait_forever()

    def run_serve(self, open_browser: bool = False) -> None:
        self.start_monitoring()
        self.start_dashboard(open_browser=open_browser)
        self.wait_forever()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.stop_workers()
        if self.httpd:
            self.httpd.shutdown()


DASHBOARD_HTML = """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>supersonic-moe control plane</title>
    <style>
        body { font-family: ui-sans-serif, sans-serif; margin: 0; background: #08111f; color: #edf2ff; }
        header { padding: 24px; background: linear-gradient(135deg, #0f2140, #126556); }
        main { padding: 20px 24px 32px; display: grid; gap: 16px; }
        .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
        .card { background: #10192b; border: 1px solid #26334d; border-radius: 14px; padding: 16px; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18); }
        h1, h2, h3, p { margin: 0 0 10px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #23314a; vertical-align: top; }
        pre, textarea, code { font-family: ui-monospace, monospace; }
        pre { white-space: pre-wrap; background: #0c1322; padding: 12px; border-radius: 8px; max-height: 320px; overflow: auto; }
        textarea { width: 100%; min-height: 420px; background: #0c1322; color: #edf2ff; border: 1px solid #243252; border-radius: 8px; padding: 12px; box-sizing: border-box; }
        .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
        button { background: #1f7a6d; color: #fff; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
        button.secondary { background: #29405f; }
        button.danger { background: #8a3d52; }
        button:disabled { opacity: 0.45; cursor: not-allowed; }
        .small { color: #a7b5d6; font-size: 13px; }
        .status { padding: 8px 0; color: #9ce5c7; min-height: 22px; }
        .error { color: #ffb2c0; }
        .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #16233a; color: #c7d5f8; font-size: 12px; margin-right: 6px; }
    </style>
</head>
<body>
    <header>
        <h1>supersonic-moe control plane</h1>
        <p class=\"small\">Multi-provider orchestration, runtime topology, and local progress dashboard.</p>
    </header>
    <main>
        <section class=\"card\">
            <h2>Controls</h2>
            <p class=\"small\">Save config, refresh state, and launch or stop workers from the same page.</p>
            <div class=\"controls\" id=\"action_buttons\">
                <button data-action onclick=\"saveConfig()\">Save Config</button>
                <button data-action onclick=\"launchWorkers(false)\">Launch Workers</button>
                <button data-action class=\"secondary\" onclick=\"launchWorkers(true)\">Restart Workers</button>
                <button data-action class=\"danger\" onclick=\"stopWorkers()\">Stop Workers</button>
                <button data-action class=\"secondary\" onclick=\"refresh(true)\">Refresh</button>
            </div>
            <div id=\"status_line\" class=\"status\"></div>
            <div class=\"grid\">
                <section>
                    <h3>Main Agent Commands</h3>
                    <pre id=\"commands\"></pre>
                </section>
                <section>
                    <h3>Validation</h3>
                    <pre id=\"validation\"></pre>
                </section>
            </div>
        </section>
        <section class=\"card\">
            <h2>Local Config</h2>
            <p class=\"small\">Edit resource pool keys, provider/model mapping, Paddle path, worker worktrees, branches, and environment commands here.</p>
            <textarea id=\"config_editor\"></textarea>
        </section>
        <div class=\"grid\">
            <section class=\"card\"><h2>Project</h2><div id=\"project\"></div></section>
            <section class=\"card\"><h2>Processes</h2><div id=\"processes\"></div></section>
        </div>
        <div class=\"grid\">
            <section class=\"card\"><h2>Provider Queue</h2><div id=\"provider_queue\"></div></section>
            <section class=\"card\"><h2>Resource Pools</h2><div id=\"resource_pools\"></div></section>
        </div>
        <div class=\"grid\">
            <section class=\"card\"><h2>Worker Config</h2><div id=\"workers\"></div></section>
            <section class=\"card\"><h2>Runtime Topology</h2><div id=\"runtime\"></div></section>
        </div>
        <div class=\"grid\">
            <section class=\"card\"><h2>Heartbeats</h2><div id=\"heartbeats\"></div></section>
            <section class=\"card\"><h2>Backlog</h2><div id=\"backlog\"></div></section>
        </div>
        <div class=\"grid\">
            <section class=\"card\"><h2>Gates</h2><div id=\"gates\"></div></section>
            <section class=\"card\"><h2>Manager Report</h2><pre id=\"manager_report\"></pre></section>
        </div>
    </main>
    <script>
        let editorDirty = false;
        let actionInFlight = false;
        let latestRefreshOk = false;

        const editor = () => document.getElementById('config_editor');
        const actionButtons = () => Array.from(document.querySelectorAll('[data-action]'));

        function renderTable(rows, columns) {
            const head = columns.map((col) => `<th>${col}</th>`).join('');
            const body = rows.map((row) => `<tr>${columns.map((col) => `<td>${row[col] ?? ''}</td>`).join('')}</tr>`).join('');
            return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
        }

        function setStatus(message, isError = false) {
            const node = document.getElementById('status_line');
            const stamp = new Date().toLocaleTimeString();
            node.textContent = `[${stamp}] ${message}`;
            node.className = isError ? 'status error' : 'status';
        }

        function setActionState(inFlight) {
            actionInFlight = inFlight;
            actionButtons().forEach((button) => { button.disabled = inFlight; });
        }

        async function postJson(path, payload) {
            const response = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || (data.errors || []).join('\n') || 'request failed');
            }
            return data;
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
                { key: 'dashboard', value: `${project.dashboard.host}:${project.dashboard.port}` },
            ];
            return renderTable(rows, ['key', 'value']);
        }

        function renderProcessRows(processes) {
            return Object.entries(processes).map(([agent, item]) => ({
                agent,
                resource_pool: item.resource_pool,
                provider: item.provider,
                model: item.model,
                pid: item.pid,
                alive: item.alive,
                returncode: item.returncode,
                worktree_path: item.worktree_path,
            }));
        }

        function renderRuntime(runtime) {
            return renderTable(runtime.workers || [], ['agent', 'resource_pool', 'provider', 'model', 'branch', 'worktree_path', 'environment_path', 'submit_strategy', 'status']);
        }

        function renderHeartbeats(heartbeats) {
            return renderTable(heartbeats.agents || [], ['agent', 'state', 'last_seen', 'evidence', 'expected_next_checkin']);
        }

        function renderBacklog(backlog) {
            return renderTable(backlog.items || [], ['id', 'owner', 'status', 'gate', 'priority', 'title']);
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
                api_key: item.api_key || '',
            }));
            return renderTable(pools, ['name', 'priority', 'provider', 'model', 'api_key']);
        }

        function renderWorkers(config) {
            const rows = (config.workers || []).map((item) => ({
                ...item,
                resource_pool_queue: (item.resource_pool_queue || []).join(', '),
            }));
            return renderTable(rows, ['agent', 'task_id', 'resource_pool', 'resource_pool_queue', 'branch', 'worktree_path', 'environment_path', 'test_command', 'submit_strategy']);
        }

        function renderProviderQueue(items) {
            return renderTable(items || [], ['resource_pool', 'provider', 'model', 'priority', 'binary_found', 'api_key_present', 'connection_quality', 'work_quality', 'score', 'active_workers', 'last_failure']);
        }

        async function saveConfig() {
            await runAction('saving config', async () => {
                await postJson('/api/config', { config_text: editor().value });
                editorDirty = false;
                setStatus('config saved');
                await refresh(true);
            });
        }

        async function launchWorkers(restart) {
            await runAction(restart ? 'restarting workers' : 'launching workers', async () => {
                const data = await postJson('/api/launch', { restart });
                const count = (data.launched || []).length;
                const failures = (data.failures || []).length;
                setStatus(`launch complete: ${count} launched, ${failures} failures`, failures > 0);
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
                const response = await fetch('/api/state');
                const data = await response.json();
                latestRefreshOk = true;
                if (!editorDirty) {
                    editor().value = data.config_text || '';
                }
                document.getElementById('project').innerHTML = renderProject(data.project);
                document.getElementById('processes').innerHTML = renderTable(renderProcessRows(data.processes), ['agent', 'resource_pool', 'provider', 'model', 'pid', 'alive', 'returncode', 'worktree_path']);
                document.getElementById('provider_queue').innerHTML = renderProviderQueue(data.provider_queue);
                document.getElementById('runtime').innerHTML = renderRuntime(data.runtime);
                document.getElementById('heartbeats').innerHTML = renderHeartbeats(data.heartbeats);
                document.getElementById('backlog').innerHTML = renderBacklog(data.backlog);
                document.getElementById('gates').innerHTML = renderGates(data.gates);
                document.getElementById('resource_pools').innerHTML = renderPools(data.config);
                document.getElementById('workers').innerHTML = renderWorkers(data.config);
                document.getElementById('manager_report').textContent = data.manager_report;
                document.getElementById('commands').textContent = `serve:\n${data.commands.serve}\n\nup:\n${data.commands.up}`;
                document.getElementById('validation').textContent = data.validation_errors.length ? data.validation_errors.join('\n') : 'config valid';
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

        refresh(true);
        setInterval(() => {
            if (!actionInFlight) {
                refresh(false);
            }
        }, 3000);
    </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="supersonic-moe control plane runtime")
    parser.add_argument("command", choices=["up", "serve"], help="launch server with workers, or only the server")
    parser.add_argument(
        "--config", type=Path, default=RUNTIME_DIR / "local_config.yaml", help="ignored local runtime config"
    )
    parser.add_argument("--host", default=None, help="override dashboard host")
    parser.add_argument("--port", type=int, default=None, help="override dashboard port")
    parser.add_argument("--open-browser", action="store_true", help="open the dashboard in a browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"missing config: {args.config}", file=sys.stderr)
        print(f"copy {RUNTIME_DIR / 'config_template.yaml'} to {args.config} and fill it first", file=sys.stderr)
        return 2

    service = ControlPlaneService(args.config, host_override=args.host, port_override=args.port)

    def handle_signal(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
        del signum, frame
        service.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.command == "up":
        service.run_up(open_browser=args.open_browser)
    else:
        service.run_serve(open_browser=args.open_browser)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
