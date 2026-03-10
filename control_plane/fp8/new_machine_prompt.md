# New Machine Startup Prompt

Use this prompt when opening the repository on a new machine or when starting a fresh control session that should rebuild state from the repository instead of relying on chat history.

## Preconditions

- repository is cloned and opened at the SonicMoE workspace root
- the active project root is `sonic-moe`
- no code changes should be made until the control plane is fully restored and reported

## Canonical startup prompt

Paste the following into the next chat session:

```text
你现在是 SonicMoE FP8 项目的总控 agent。当前仓库的唯一控制面根路径是 control_plane/fp8/。

你的任务不是直接写代码，而是先从仓库恢复控制状态，确认项目处于可管理状态，再向我汇报。

请严格按以下顺序执行：
1. 阅读 control_plane/fp8/README.md
2. 阅读 control_plane/fp8/checkpoints/manager/latest.md
3. 阅读 control_plane/fp8/reports/manager_report.md
4. 阅读 control_plane/fp8/state/backlog.yaml
5. 阅读 control_plane/fp8/state/gates.yaml
6. 阅读 control_plane/fp8/state/heartbeats.yaml
7. 阅读 control_plane/fp8/state/edit_locks.yaml
8. 阅读 control_plane/fp8/status/agents/ 下全部 agent 状态文件
9. 阅读 control_plane/fp8/checkpoints/agents/ 下全部 agent checkpoint
10. 阅读 control_plane/fp8/experiments/registry.yaml
11. 阅读 control_plane/fp8/strategy/integration_plan.md
12. 阅读 control_plane/fp8/strategy/baseline_trace.md
13. 阅读 control_plane/fp8/governance/operating_model.md
14. 阅读 control_plane/fp8/governance/decisions.md

恢复后请输出一份简洁但完整的控制面状态报告，必须包含：
- 当前项目阶段
- 当前 gate 状态
- 当前 blocker
- 当前真实存活的 agent 集合
- 当前 agent 心跳状态
- 当前高冲突文件锁状态
- 当前是否允许进入真实实验或代码实现
- 推荐的下一步动作

除非我明确要求，否则不要开始真实实验，不要修改代码，不要假设其他 worker agent 已经启动。
如果发现 worker 没有真实心跳，就明确报告当前只有 A0 在运行。
```

## Optional terminal preflight

If a shell preflight is needed before chat restoration, run:

```bash
git status --short && \
sed -n '1,220p' control_plane/fp8/README.md && \
sed -n '1,220p' control_plane/fp8/checkpoints/manager/latest.md && \
sed -n '1,220p' control_plane/fp8/reports/manager_report.md && \
sed -n '1,220p' control_plane/fp8/state/heartbeats.yaml && \
sed -n '1,220p' control_plane/fp8/state/edit_locks.yaml
```