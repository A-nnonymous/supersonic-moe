# SonicMoE FP8 Reports

Start here: [`HANDOFF.md`](fp8_upgrade/HANDOFF.md) — 包含完整项目状态、已知 bug、架构决策、环境配置。

| File | Content |
|------|---------|
| [`HANDOFF.md`](fp8_upgrade/HANDOFF.md) | 全链路状态、代码变更、已知 bug、性能分析、优先级、环境速查 |
| [`BLOCKSCALED_ALIGNMENT.md`](fp8_upgrade/BLOCKSCALED_ALIGNMENT.md) | 128-row alignment 硬件约束参考 |

## Quick Start for Next Agent

1. 读 `HANDOFF.md` — 了解全部上下文
2. 修复 Section 5 中的 **dz 有损转换 bug** (P0)
3. 等 GPU 空闲后运行 `tools/rmse_verification.py` 验证精度
4. 运行 `tools/final_benchmark.py` 采集性能数据
