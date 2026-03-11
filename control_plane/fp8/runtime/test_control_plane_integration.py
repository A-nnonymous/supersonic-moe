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
from copy import deepcopy
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


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def request_json_allow_error(
    url: str, payload: dict[str, object] | None = None, timeout: float = 5.0
) -> tuple[int, dict[str, object]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        exc.close()
        return int(exc.code), body


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
        self.write_fake_provider_binary("ducc")

        self.port = find_free_port()
        self.config_path = self.root / "integration_config.yaml"
        self.config_path.write_text(self.render_config(), encoding="utf-8")
        self.base_url = f"http://127.0.0.1:{self.port}"

        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["GITHUB_TOKEN"] = "integration-token"
        env["OPENCODE_API_KEY"] = "integration-token"
        env["ANTHROPIC_API_KEY"] = "integration-token"
        self.env = env

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
        if binary_name == "ducc":
            binary_path.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "session-ok" ]; then\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$1" = "session-fail" ]; then\n'
                '  echo "ducc session unavailable" >&2\n'
                "  exit 17\n"
                "fi\n"
                "while [ $# -gt 0 ]; do\n"
                "  shift\n"
                "done\n"
                "sleep 30\n",
                encoding="utf-8",
            )
        else:
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
                "ducc": {
                    "auth_mode": "session",
                    "session_probe_command": ["ducc", "session-ok"],
                    "command_template": [
                        "ducc",
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
                    "preferred_providers": ["copilot", "claude_code", "ducc", "opencode"],
                    "suggested_test_command": "uv run pytest tests/moe_test.py -k test_moe",
                },
                "types": {
                    "protocol": {
                        "preferred_providers": ["copilot", "claude_code", "ducc", "opencode"],
                        "suggested_test_command": "uv run pytest tests/reference_layers/standalone_moe_layer/tests/test_imports_and_interfaces.py",
                    },
                    "audit_hopper": {
                        "preferred_providers": ["copilot", "claude_code", "ducc", "opencode"],
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
                "ducc_pool": {
                    "priority": 350,
                    "provider": "ducc",
                    "model": "claude-sonnet-4-5",
                    "extra_env": {},
                },
            },
            "worker_defaults": {
                "resource_pool_queue": ["copilot_pool", "ducc_pool", "opencode_pool", "claude_pool"],
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

    def wait_for_state_available(self) -> dict[str, object]:
        final_state: dict[str, object] = {}

        def predicate() -> bool:
            nonlocal final_state
            try:
                final_state = self.fetch_state()
            except Exception:
                return False
            return bool(final_state.get("mode"))

        wait_for(predicate, timeout=20, interval=0.5)
        return final_state

    def run_cli_command(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "PyYAML>=6.0.2",
                "python",
                str(self.runtime_script),
                *args,
                "--port",
                str(self.port),
            ],
            cwd=self.root,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    def session_state(self) -> dict[str, object]:
        session_state_path = self.fp8_root / "runtime" / f"session_state_{self.port}.json"
        return json.loads(session_state_path.read_text(encoding="utf-8"))

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

    def test_project_section_validate_and_save(self) -> None:
        updated_local_repo_root = self.project_root / "repo-updated"
        updated_reference_workspace_root = self.paddle_root / "ref-updated"
        updated_local_repo_root.mkdir(parents=True, exist_ok=True)
        updated_reference_workspace_root.mkdir(parents=True, exist_ok=True)
        updated_project = {
            "repository_name": "sonicmoe-fp8-it-updated",
            "local_repo_root": str(updated_local_repo_root),
            "reference_workspace_root": str(updated_reference_workspace_root),
            "dashboard": {
                "host": "127.0.0.1",
                "port": self.port,
            },
        }

        validation_result = read_json(
            f"{self.base_url}/api/config/validate-section",
            {"section": "project", "value": updated_project},
        )
        self.assertTrue(validation_result["ok"])
        self.assertEqual(validation_result["validation_issues"], [])

        save_result = read_json(
            f"{self.base_url}/api/config/section",
            {"section": "project", "value": updated_project},
        )
        self.assertTrue(save_result["ok"])
        self.assertEqual(save_result["validation_issues"], [])

        state = self.fetch_state()
        project = state["config"]["project"]
        self.assertEqual(project["repository_name"], updated_project["repository_name"])
        self.assertEqual(project["local_repo_root"], updated_project["local_repo_root"])
        self.assertEqual(project["reference_workspace_root"], updated_project["reference_workspace_root"])

        saved_config_text = self.config_path.read_text(encoding="utf-8")
        self.assertIn(updated_project["local_repo_root"], saved_config_text)
        self.assertIn(updated_project["reference_workspace_root"], saved_config_text)

    def test_unknown_api_routes_return_json_errors(self) -> None:
        request = urllib.request.Request(f"{self.base_url}/api/does-not-exist", data=b"{}")
        request.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        error = exc_info.exception
        self.assertEqual(error.code, 404)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertFalse(payload["ok"])
        self.assertIn("unknown api route", payload["error"])

    def test_stop_agents_cli_stops_workers_and_keeps_listener(self) -> None:
        launch_result = read_json(f"{self.base_url}/api/launch", {"restart": False})
        self.assertTrue(launch_result["ok"])
        self.wait_for_agent_state(expected_provider="copilot", expected_model="gpt-5.4")

        stop_result = self.run_cli_command("stop-agents")
        stop_payload = json.loads(stop_result.stdout)
        self.assertTrue(stop_payload["ok"])
        self.assertEqual(sorted(stop_payload["stopped"]), ["A1", "A2"])

        wait_for(lambda: port_is_listening(self.port), timeout=10, interval=0.5)
        state = self.fetch_state()
        heartbeats = {item["agent"]: item for item in state["heartbeats"]["agents"]}
        runtime_workers = {item["agent"]: item for item in state["runtime"]["workers"]}
        self.assertEqual(heartbeats["A1"]["state"], "offline")
        self.assertEqual(heartbeats["A2"]["state"], "offline")
        self.assertEqual(runtime_workers["A1"]["status"], "stopped")
        self.assertEqual(runtime_workers["A2"]["status"], "stopped")

        session_state = self.session_state()
        self.assertTrue(session_state["server"]["listener_active"])
        self.assertEqual(session_state["workers"], {})

    def test_detached_serve_can_start_and_stop_via_session_state(self) -> None:
        initial_stop = read_json(f"{self.base_url}/api/stop-all", {})
        self.assertTrue(initial_stop["ok"])
        wait_for(lambda: self.server.poll() is not None, timeout=15, interval=0.5)
        wait_for(lambda: not port_is_listening(self.port), timeout=10, interval=0.5)

        detached_result = self.run_cli_command(
            "serve",
            "--config",
            str(self.config_path),
            "--host",
            "127.0.0.1",
        )
        self.assertIn("control plane started in background", detached_result.stdout)
        wait_for(lambda: port_is_listening(self.port), timeout=15, interval=0.5)

        state = self.wait_for_state_available()
        self.assertEqual(state["project"]["repository_name"], "sonicmoe-fp8-it")
        session_state = self.session_state()
        self.assertTrue(session_state["server"]["listener_active"])
        detached_pid = int(session_state["server"]["pid"])
        self.assertGreater(detached_pid, 0)

        detached_stop = self.run_cli_command("stop-all")
        detached_stop_payload = json.loads(detached_stop.stdout)
        self.assertTrue(detached_stop_payload["ok"])
        self.assertTrue(detached_stop_payload["listener_released"])
        wait_for(lambda: not port_is_listening(self.port), timeout=15, interval=0.5)
        wait_for(lambda: not pid_is_running(detached_pid), timeout=15, interval=0.5)

    def test_launch_failure_surfaces_missing_provider_credentials(self) -> None:
        current_state = self.fetch_state()
        broken_config = deepcopy(current_state["config"])
        broken_config["providers"]["copilot"]["api_key_env_name"] = "MISSING_GITHUB_TOKEN"
        broken_config["resource_pools"]["copilot_pool"]["api_key"] = "replace_me_or_use_api_key_env"

        save_result = read_json(f"{self.base_url}/api/config", {"config": broken_config})
        self.assertTrue(save_result["ok"])

        status_code, failed_launch = request_json_allow_error(f"{self.base_url}/api/launch", {"restart": False})
        self.assertEqual(status_code, 400)
        self.assertFalse(failed_launch["ok"])
        self.assertEqual(len(failed_launch["failures"]), 2)
        self.assertIn("launch failed for 2 worker(s)", failed_launch["error"])
        self.assertIn("A1: api key missing for pool copilot_pool", failed_launch["error"])
        self.assertTrue(all("api key missing" in item["error"] for item in failed_launch["failures"]))

        state = self.fetch_state()
        provider_queue = {item["resource_pool"]: item for item in state["provider_queue"]}
        self.assertFalse(provider_queue["copilot_pool"]["api_key_present"])
        self.assertIn("api key missing", provider_queue["copilot_pool"]["last_failure"])

        heartbeats = {item["agent"]: item for item in state["heartbeats"]["agents"]}
        runtime_workers = {item["agent"]: item for item in state["runtime"]["workers"]}
        self.assertEqual(heartbeats["A1"]["state"], "stale")
        self.assertEqual(heartbeats["A2"]["state"], "stale")
        self.assertIn("launch_failed", runtime_workers["A1"]["status"])
        self.assertIn("launch_failed", runtime_workers["A2"]["status"])

    def test_session_backed_provider_launches_without_api_key(self) -> None:
        current_state = self.fetch_state()
        session_config = deepcopy(current_state["config"])
        session_config["worker_defaults"]["resource_pool_queue"] = ["ducc_pool"]

        save_result = read_json(f"{self.base_url}/api/config", {"config": session_config})
        self.assertTrue(save_result["ok"])

        launch_result = read_json(
            f"{self.base_url}/api/launch",
            {"restart": False, "strategy": "selected_model", "provider": "ducc", "model": "ducc-sonnet-it"},
        )
        self.assertTrue(launch_result["ok"])

        state = self.wait_for_agent_state(expected_provider="ducc", expected_model="ducc-sonnet-it")
        provider_queue = {item["resource_pool"]: item for item in state["provider_queue"]}
        self.assertEqual(provider_queue["ducc_pool"]["auth_mode"], "session")
        self.assertTrue(provider_queue["ducc_pool"]["auth_ready"])
        self.assertTrue(provider_queue["ducc_pool"]["launch_ready"])
        self.assertFalse(provider_queue["ducc_pool"]["api_key_present"])

        runtime_workers = {item["agent"]: item for item in state["runtime"]["workers"]}
        self.assertEqual(runtime_workers["A1"]["provider"], "ducc")
        self.assertEqual(runtime_workers["A2"]["provider"], "ducc")

        self.stop_workers()

    def test_session_probe_failure_surfaces_actionable_error(self) -> None:
        current_state = self.fetch_state()
        broken_config = deepcopy(current_state["config"])
        broken_config["providers"]["ducc"]["session_probe_command"] = ["ducc", "session-fail"]
        broken_config["worker_defaults"]["resource_pool_queue"] = ["ducc_pool"]

        save_result = read_json(f"{self.base_url}/api/config", {"config": broken_config})
        self.assertTrue(save_result["ok"])

        status_code, failed_launch = request_json_allow_error(
            f"{self.base_url}/api/launch",
            {"restart": False, "strategy": "selected_model", "provider": "ducc", "model": "ducc-sonnet-it"},
        )
        self.assertEqual(status_code, 400)
        self.assertFalse(failed_launch["ok"])
        self.assertIn("ducc session unavailable", failed_launch["error"])
        self.assertTrue(all("ducc session unavailable" in item["error"] for item in failed_launch["failures"]))

        state = self.fetch_state()
        provider_queue = {item["resource_pool"]: item for item in state["provider_queue"]}
        self.assertEqual(provider_queue["ducc_pool"]["auth_mode"], "session")
        self.assertFalse(provider_queue["ducc_pool"]["auth_ready"])
        self.assertFalse(provider_queue["ducc_pool"]["launch_ready"])
        self.assertIn("ducc session unavailable", provider_queue["ducc_pool"]["auth_detail"])

    def test_initial_launch_provider_falls_back_to_configured_ducc(self) -> None:
        current_state = self.fetch_state()
        ducc_only_config = deepcopy(current_state["config"])
        ducc_only_config["providers"] = {"ducc": ducc_only_config["providers"]["ducc"]}
        ducc_only_config["resource_pools"] = {"ducc_pool": ducc_only_config["resource_pools"]["ducc_pool"]}
        ducc_only_config["worker_defaults"]["resource_pool_queue"] = ["ducc_pool"]
        ducc_only_config["task_policies"]["defaults"]["preferred_providers"] = ["ducc"]
        for entry in ducc_only_config["task_policies"]["types"].values():
            entry["preferred_providers"] = ["ducc"]

        save_result = read_json(f"{self.base_url}/api/config", {"config": ducc_only_config})
        self.assertTrue(save_result["ok"])

        state = self.fetch_state()
        self.assertEqual(state["launch_policy"]["initial_provider"], "ducc")
        self.assertEqual(state["launch_policy"]["default_provider"], "ducc")

        launch_result = read_json(f"{self.base_url}/api/launch", {"restart": False})
        self.assertTrue(launch_result["ok"])
        self.assertEqual(launch_result["launch_policy"]["provider"], "ducc")

        self.wait_for_agent_state(expected_provider="ducc", expected_model="claude-sonnet-4-5")
        self.stop_workers()

    def test_worker_prompt_forbids_nested_agent_orchestration(self) -> None:
        launch_result = read_json(f"{self.base_url}/api/launch", {"restart": False})
        self.assertTrue(launch_result["ok"])
        self.wait_for_agent_state(expected_provider="copilot", expected_model="gpt-5.4")

        prompt_path = self.fp8_root / "runtime" / "generated_prompts" / "A1.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self.assertIn("Do not start nested control-plane sessions", prompt_text)
        self.assertIn("claude-code", prompt_text)
        self.assertIn("ducc", prompt_text)

        self.stop_workers()

    def test_worker_context_cannot_start_nested_control_plane(self) -> None:
        nested_port = find_free_port()
        nested_env = dict(self.env)
        nested_env["CONTROL_PLANE_WORKER_CONTEXT"] = "1"
        nested_env["CONTROL_PLANE_WORKER_AGENT"] = "A1"
        nested_env["CONTROL_PLANE_RECURSION_POLICY"] = "forbid-nested-control-plane"

        result = subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "PyYAML>=6.0.2",
                "python",
                str(self.runtime_script),
                "up",
                "--config",
                str(self.config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(nested_port),
            ],
            cwd=self.root,
            env=nested_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing to start nested control plane", result.stderr)
        self.assertFalse(port_is_listening(nested_port))

    def test_silent_and_stop_all_cli_preserve_then_release_workers(self) -> None:
        launch_result = read_json(f"{self.base_url}/api/launch", {"restart": False})
        self.assertTrue(launch_result["ok"])
        self.wait_for_agent_state(expected_provider="copilot", expected_model="gpt-5.4")

        silent_result = self.run_cli_command("silent")
        silent_payload = json.loads(silent_result.stdout)
        self.assertTrue(silent_payload["ok"])
        self.assertTrue(silent_payload["listener_released"])
        wait_for(lambda: not port_is_listening(self.port), timeout=10, interval=0.5)

        session_state = self.session_state()
        self.assertFalse(session_state["server"]["listener_active"])
        worker_pids = [
            int(worker["pid"]) for worker in session_state["workers"].values() if int(worker.get("pid") or 0)
        ]
        self.assertTrue(worker_pids)
        self.assertTrue(all(pid > 0 for pid in worker_pids))

        stop_all_result = self.run_cli_command("stop-all")
        stop_all_payload = json.loads(stop_all_result.stdout)
        self.assertTrue(stop_all_payload["ok"])
        self.assertTrue(stop_all_payload["listener_released"])
        wait_for(lambda: not port_is_listening(self.port), timeout=10, interval=0.5)
        wait_for(lambda: self.server.poll() is not None, timeout=10, interval=0.5)

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
