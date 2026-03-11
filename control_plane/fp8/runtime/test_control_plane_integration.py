from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_FP8_ROOT = REPO_ROOT / "control_plane" / "fp8"


def read_json(url: str, payload: dict[str, object] | None = None, timeout: float = 5.0) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"request failed with status {exc.code}: {body}") from exc


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.5) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition was not satisfied before timeout")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ControlPlaneIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="fp8-control-plane-it-")
        self.root = Path(self.temp_dir.name)
        self.fp8_root = self.root / "control_plane" / "fp8"
        shutil.copytree(SOURCE_FP8_ROOT, self.fp8_root)
        self.runtime_script = self.fp8_root / "runtime" / "control_plane.py"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = self.root / "workspace"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.paddle_root = self.root / "paddle"
        self.paddle_root.mkdir(parents=True, exist_ok=True)

        self.worker_roots = {
            "A1": self.root / "workers" / "A1",
            "A2": self.root / "workers" / "A2",
        }
        for path in self.worker_roots.values():
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.write_fake_provider_binary("copilot")
        self.write_fake_provider_binary("opencode")
        self.write_fake_provider_binary("claude-code")

        self.port = find_free_port()
        self.config_path = self.root / "integration_config.yaml"
        self.config_path.write_text(self.render_config(), encoding="utf-8")
        self.base_url = f"http://127.0.0.1:{self.port}"

        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["GITHUB_TOKEN"] = "integration-token"
        env["OPENCODE_API_KEY"] = "integration-token"
        env["ANTHROPIC_API_KEY"] = "integration-token"

        self.server = subprocess.Popen(
            [
                "uv",
                "run",
                "--with",
                "PyYAML>=6.0.2",
                "python",
                str(self.runtime_script),
                "serve",
                "--config",
                str(self.config_path),
                "--foreground",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for(self.server_ready)

    def tearDown(self) -> None:
        if hasattr(self, "server") and self.server.poll() is None:
            try:
                read_json(f"{self.base_url}/api/stop-all", {})
            except Exception:
                pass
            try:
                self.server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5)
        if hasattr(self, "server") and self.server.stdout is not None:
            self.server.stdout.close()
        self.temp_dir.cleanup()

    def write_fake_provider_binary(self, binary_name: str) -> None:
        binary_path = self.bin_dir / binary_name
        binary_path.write_text(
            "#!/bin/sh\n" "while [ $# -gt 0 ]; do\n" "  shift\n" "done\n" "sleep 30\n",
            encoding="utf-8",
        )
        binary_path.chmod(0o755)

    def render_config(self) -> str:
        config = {
            "project": {
                "repository_name": "sonicmoe-fp8-it",
                "local_repo_root": str(self.project_root),
                "reference_workspace_root": str(self.paddle_root),
                "base_branch": "main",
                "integration_branch": "main",
                "manager_git_identity": {
                    "name": "Integration Manager",
                    "email": "integration-manager@example.com",
                },
                "dashboard": {
                    "host": "127.0.0.1",
                    "port": self.port,
                },
            },
            "providers": {
                "copilot": {
                    "api_key_env_name": "GITHUB_TOKEN",
                    "command_template": [
                        "copilot",
                        "--model",
                        "{model}",
                        "--prompt-file",
                        "{prompt_file}",
                        "--worktree",
                        "{worktree_path}",
                    ],
                },
                "opencode": {
                    "api_key_env_name": "OPENCODE_API_KEY",
                    "command_template": [
                        "opencode",
                        "--model",
                        "{model}",
                        "--prompt-file",
                        "{prompt_file}",
                        "--cwd",
                        "{worktree_path}",
                    ],
                },
                "claude_code": {
                    "api_key_env_name": "ANTHROPIC_API_KEY",
                    "command_template": [
                        "claude-code",
                        "--model",
                        "{model}",
                        "--prompt-file",
                        "{prompt_file}",
                        "--cwd",
                        "{worktree_path}",
                    ],
                },
            },
            "task_policies": {
                "defaults": {
                    "task_type": "default",
                    "preferred_providers": ["copilot", "claude_code", "opencode"],
                    "suggested_test_command": "uv run pytest tests/moe_test.py -k test_moe",
                },
                "types": {
                    "protocol": {
                        "preferred_providers": ["copilot", "claude_code", "opencode"],
                        "suggested_test_command": "uv run pytest tests/reference_layers/standalone_moe_layer/tests/test_imports_and_interfaces.py",
                    },
                    "audit_hopper": {
                        "preferred_providers": ["copilot", "claude_code", "opencode"],
                        "suggested_test_command": "uv run pytest tests/moe_test.py -k test_moe",
                    },
                },
                "rules": [
                    {"name": "protocol-a1", "task_type": "protocol", "task_ids": ["A1-001"]},
                    {"name": "audit-a2", "task_type": "audit_hopper", "task_ids": ["A2-001"]},
                ],
            },
            "resource_pools": {
                "copilot_pool": {
                    "priority": 100,
                    "provider": "copilot",
                    "model": "gpt-5.4",
                    "api_key": "replace_me_or_use_api_key_env",
                    "extra_env": {},
                },
                "opencode_pool": {
                    "priority": 400,
                    "provider": "opencode",
                    "model": "o4-mini",
                    "api_key": "replace_me_or_use_api_key_env",
                    "extra_env": {},
                },
                "claude_pool": {
                    "priority": 200,
                    "provider": "claude_code",
                    "model": "claude-sonnet-4-5",
                    "api_key": "replace_me_or_use_api_key_env",
                    "extra_env": {},
                },
            },
            "worker_defaults": {
                "resource_pool_queue": ["copilot_pool", "opencode_pool", "claude_pool"],
                "environment_type": "none",
                "sync_command": "none",
                "submit_strategy": "patch_handoff",
            },
            "workers": [
                {
                    "agent": "A1",
                    "task_id": "A1-001",
                    "branch": "integration-a1",
                    "worktree_path": str(self.worker_roots["A1"]),
                    "git_identity": {
                        "name": "Integration A1",
                        "email": "a1@example.com",
                    },
                },
                {
                    "agent": "A2",
                    "task_id": "A2-001",
                    "branch": "integration-a2",
                    "worktree_path": str(self.worker_roots["A2"]),
                    "git_identity": {
                        "name": "Integration A2",
                        "email": "a2@example.com",
                    },
                },
            ],
        }
        return json.dumps(config, indent=2) + "\n"

    def server_ready(self) -> bool:
        if self.server.poll() is not None:
            output = self.server.stdout.read() if self.server.stdout else ""
            raise RuntimeError(f"control plane exited early:\n{output}")
        try:
            state = read_json(f"{self.base_url}/api/state")
        except Exception:
            return False
        return bool(state.get("mode"))

    def fetch_state(self) -> dict[str, object]:
        return read_json(f"{self.base_url}/api/state")

    def wait_for_agent_state(self, expected_provider: str, expected_model: str | None = None) -> dict[str, object]:
        final_state: dict[str, object] = {}

        def predicate() -> bool:
            nonlocal final_state
            state = self.fetch_state()
            heartbeats = {item["agent"]: item for item in state["heartbeats"]["agents"]}
            runtime_workers = {item["agent"]: item for item in state["runtime"]["workers"]}
            for agent in ("A1", "A2"):
                heartbeat = heartbeats.get(agent, {})
                runtime = runtime_workers.get(agent, {})
                if heartbeat.get("state") != "healthy":
                    return False
                if runtime.get("provider") != expected_provider:
                    return False
                if expected_model is not None and runtime.get("model") != expected_model:
                    return False
            final_state = state
            return True

        wait_for(predicate, timeout=20, interval=1)
        return final_state

    def stop_workers(self) -> None:
        read_json(f"{self.base_url}/api/stop", {})
        wait_for(
            lambda: all(
                item.get("state") in {"offline", "not-started", "not_started"}
                for item in self.fetch_state()["heartbeats"]["agents"]
                if item.get("agent") in {"A1", "A2"}
            ),
            timeout=15,
            interval=1,
        )

    def test_multi_agent_launch_policies_and_heartbeats(self) -> None:
        initial_state = self.fetch_state()
        self.assertEqual(initial_state["launch_policy"]["default_strategy"], "initial_copilot")
        resolved_workers = {item["agent"]: item for item in initial_state["resolved_workers"]}
        self.assertEqual(
            resolved_workers["A1"]["test_command"],
            "uv run pytest tests/reference_layers/standalone_moe_layer/tests/test_imports_and_interfaces.py",
        )
        self.assertEqual(resolved_workers["A1"]["task_type"], "protocol")
        self.assertEqual(resolved_workers["A1"]["locked_pool"], "copilot_pool")
        self.assertEqual(resolved_workers["A2"]["test_command"], "uv run pytest tests/moe_test.py -k test_moe")
        self.assertEqual(resolved_workers["A2"]["task_type"], "audit_hopper")
        self.assertEqual(resolved_workers["A2"]["locked_pool"], "copilot_pool")

        launch_result = read_json(f"{self.base_url}/api/launch", {"restart": False})
        self.assertTrue(launch_result["ok"])
        self.assertEqual(launch_result["launch_policy"]["strategy"], "initial_copilot")

        state = self.wait_for_agent_state(expected_provider="copilot", expected_model="gpt-5.4")
        runtime_workers = {item["agent"]: item for item in state["runtime"]["workers"]}
        self.assertEqual(runtime_workers["A1"]["status"], "healthy")
        self.assertEqual(runtime_workers["A2"]["status"], "healthy")

        self.stop_workers()

        selected_result = read_json(
            f"{self.base_url}/api/launch",
            {"restart": False, "strategy": "selected_model", "provider": "opencode", "model": "gpt-5.4-mini-it"},
        )
        self.assertTrue(selected_result["ok"])
        self.assertEqual(selected_result["launch_policy"]["provider"], "opencode")

        selected_state = self.wait_for_agent_state(expected_provider="opencode", expected_model="gpt-5.4-mini-it")
        selected_runtime = {item["agent"]: item for item in selected_state["runtime"]["workers"]}
        self.assertEqual(selected_runtime["A1"]["provider"], "opencode")
        self.assertEqual(selected_runtime["A2"]["provider"], "opencode")

        self.stop_workers()

        elastic_result = read_json(f"{self.base_url}/api/launch", {"restart": False, "strategy": "elastic"})
        self.assertTrue(elastic_result["ok"])
        elastic_state = self.wait_for_agent_state(expected_provider="copilot", expected_model="gpt-5.4")
        elastic_runtime = {item["agent"]: item for item in elastic_state["runtime"]["workers"]}
        self.assertEqual(elastic_runtime["A1"]["provider"], "copilot")
        self.assertEqual(elastic_runtime["A2"]["provider"], "copilot")

        self.stop_workers()


if __name__ == "__main__":
    unittest.main()
