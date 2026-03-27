from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OperatorOpt:
    name: str
    env_var: str
    paper_operator: str
    engineering_scope: str
    description: str


NATIVE_FP8_UPPROJ = OperatorOpt(
    name="native-fp8-upproj",
    env_var="SONIC_MOE_OPT_NATIVE_FP8_UPPROJ",
    paper_operator="up-proj (`varlen-M grouped GEMM + act`)",
    engineering_scope="replace the current bf16-working up-proj activation path with a native fp8-capable operator contract",
    description="operator-level opt slot for native fp8 up-proj work",
)

NATIVE_FP8_DOWNPROJ = OperatorOpt(
    name="native-fp8-downproj",
    env_var="SONIC_MOE_OPT_NATIVE_FP8_DOWNPROJ",
    paper_operator="down-proj (`varlen-M grouped GEMM`)",
    engineering_scope="replace the current bf16 restored-A_e down-proj path with a native fp8-capable forward mainloop",
    description="operator-level opt slot for native fp8 down-proj forward work",
)

FP8_WEIGHT_STORAGE = OperatorOpt(
    name="fp8-weight-storage",
    env_var="SONIC_MOE_OPT_FP8_WEIGHT_STORAGE",
    paper_operator="up-proj + down-proj weights",
    engineering_scope="store and consume expert weights in fp8-compatible format without changing external routing semantics",
    description="operator-level opt slot for fp8 weight storage / consumption work",
)

MIXED_DTYPE_DOWNPROJ_DW2 = OperatorOpt(
    name="mixed-dtype-downproj-dw2",
    env_var="SONIC_MOE_OPT_MIXED_DTYPE_DOWNPROJ_DW2",
    paper_operator="down-proj weight grad (`varlen-K grouped GEMM`)",
    engineering_scope="consume bf16 dO and fp8 routed activation side product to produce dW2_e",
    description="operator-level opt slot for mixed-dtype down-proj weight-grad work",
)

RANKFLEX_VARLEN_DOWNPROJ = OperatorOpt(
    name="rankflex-varlen-downproj",
    env_var="SONIC_MOE_OPT_RANKFLEX_VARLEN_DOWNPROJ",
    paper_operator="down-proj (`varlen-M grouped GEMM`)",
    engineering_scope="support flat varlen blockscaled down-proj without grouped/static-capacity fallback",
    description="operator-level opt slot for rank-flexible blockscaled varlen down-proj work",
)


ALL_OPERATOR_OPTS = (
    NATIVE_FP8_UPPROJ,
    NATIVE_FP8_DOWNPROJ,
    FP8_WEIGHT_STORAGE,
    MIXED_DTYPE_DOWNPROJ_DW2,
    RANKFLEX_VARLEN_DOWNPROJ,
)


def is_operator_opt_enabled(*opts: OperatorOpt) -> bool:
    return any(os.getenv(opt.env_var, "").lower() in {"1", "true", "yes", "on"} for opt in opts)
