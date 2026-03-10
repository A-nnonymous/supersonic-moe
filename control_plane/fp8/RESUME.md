# Resume

## Purpose

Use this file after repo migration, machine migration, or interrupted sessions.

For a first-time startup on a clean machine, use `control_plane/fp8/new_machine_prompt.md`.

## Read order

1. `control_plane/fp8/checkpoints/manager/latest.md`
2. `control_plane/fp8/reports/manager_report.md`
3. `control_plane/fp8/state/backlog.yaml`
4. `control_plane/fp8/state/gates.yaml`
5. `control_plane/fp8/state/heartbeats.yaml`
6. `control_plane/fp8/state/edit_locks.yaml`
7. `control_plane/fp8/state/agent_runtime.yaml`
8. `control_plane/fp8/status/agents/`
9. `control_plane/fp8/checkpoints/agents/`
10. `control_plane/fp8/experiments/registry.yaml`
11. `control_plane/fp8/strategy/integration_plan.md`
12. `control_plane/fp8/strategy/baseline_trace.md`

## Resume prompt

Paste this into the next chat session after migration:

```text
你现在是 SonicMoE FP8 项目的总控 agent。请先不要写代码，按以下顺序恢复上下文并汇报：
1. 阅读 control_plane/fp8/checkpoints/manager/latest.md
2. 阅读 control_plane/fp8/reports/manager_report.md
3. 阅读 control_plane/fp8/state/backlog.yaml
4. 阅读 control_plane/fp8/state/gates.yaml
5. 阅读 control_plane/fp8/state/heartbeats.yaml
6. 阅读 control_plane/fp8/state/edit_locks.yaml
7. 阅读 control_plane/fp8/state/agent_runtime.yaml
8. 阅读 control_plane/fp8/status/agents/ 下全部 agent 状态
9. 阅读 control_plane/fp8/checkpoints/agents/ 下全部 agent checkpoint
10. 阅读 control_plane/fp8/experiments/registry.yaml
11. 阅读 control_plane/fp8/strategy/integration_plan.md
12. 阅读 control_plane/fp8/strategy/baseline_trace.md

恢复后请输出：
- 当前项目阶段
- 已通过和未通过的 gate
- 当前 blocker
- 当前可并行的 agent 集合
- 当前 agent 心跳与存活状态
- 当前 provider / worktree / branch / env 拓扑
- 当前高冲突文件的写锁状态
- 推荐的下一步动作

除非我明确要求，否则先不要进入真实实验或大规模代码修改。
```

## Terminal bootstrap command

If you want a quick local preflight after migration, run:

```bash
git status --short && \
sed -n '1,220p' control_plane/fp8/checkpoints/manager/latest.md && \
sed -n '1,220p' control_plane/fp8/reports/manager_report.md && \
sed -n '1,220p' control_plane/fp8/state/heartbeats.yaml && \
sed -n '1,220p' control_plane/fp8/state/agent_runtime.yaml && \
sed -n '1,220p' control_plane/fp8/state/edit_locks.yaml
```