# Decisions

## Frozen

### D-001

Torch is the only final implementation target.

Reason:

- matches project scope
- keeps integration boundary clear

### D-002

`tests/reference_layers/standalone_moe_layer` is the primary correctness and training baseline.

Reason:

- it is the closest working end-to-end FP8-MoE baseline already vendored inside this repository

### D-003

Paddle kernels are semantic references, not structural templates for SonicMoE.

Reason:

- Paddle solves quant, dequant, gate, and scale semantics well
- Paddle does not directly provide the final grouped fused route SonicMoE needs

### D-004

Protocol freeze happens before heavy parallel implementation.

Reason:

- avoids scale and backend contract drift

## Open

### O-001

Should the first torch protocol expose only per-tensor scales publicly and hide subchannel handling entirely inside backend adapters?

Owner:

- A1 with input from A2, A3, A6

### O-002

Should Blackwell compatibility expose packed e8m0 scale explicitly, or should it be fully normalized before entering the public MoE API?

Owner:

- A1 with input from A3 and A6